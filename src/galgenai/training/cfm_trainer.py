"""CFM trainer implementation."""

import math
from typing import Any, Dict, Optional, Tuple

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.cfm import CFM, count_parameters
from .base_trainer import BaseTrainer
from .config import CFMTrainingConfig


def cfm_scheduler_total_steps(
    config: CFMTrainingConfig, steps_per_epoch: int
) -> int:
    """Number of optimizer steps the OneCycleLR schedule runs over.

    The scheduler still advances once per batch, so the horizon
    is ``num_epochs * steps_per_epoch``. It can be set shorter via
    ``lr_converge_at_epoch`` so the schedule ramps up and anneals down
    early, then holds flat at the ``lr_min_factor`` floor for the
    remainder of training.
    """
    epochs = config.lr_converge_at_epoch or config.num_epochs
    return max(1, round(epochs * steps_per_epoch))


def build_cfm_scheduler(
    optimizer: torch.optim.Optimizer,
    config: CFMTrainingConfig,
    steps_per_epoch: int,
) -> OneCycleLR:
    """Build the OneCycleLR schedule used for CFM training: ramps up
    to ``learning_rate`` over ``warmup_epochs``, then cosine-anneals
    down to ``learning_rate * lr_min_factor``."""
    total_steps = cfm_scheduler_total_steps(config, steps_per_epoch)
    warmup_steps = config.warmup_epochs * steps_per_epoch
    return OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        total_steps=total_steps,
        pct_start=min(max(warmup_steps / total_steps, 1e-6), 0.999),
        div_factor=config.div_factor,
        final_div_factor=1.0 / (config.div_factor * config.lr_min_factor),
        anneal_strategy="cos",
    )


def _extract_cfm_batch(
    batch, device: torch.device, noiseless: bool = False
) -> Tuple[
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    torch.Tensor,
]:
    """
    Extract (x, ivar, mask, f) from a CFM batch.

    Expected batch formats (see ``data.hsc`` / ``data.cosmos_dataset``):
    - (flux, cond): conditioning only
    - (flux, ivar, mask, cond): conditioning with aux data
    """
    x, ivar, mask, x_clean, f = batch
    if noiseless:
        x = x_clean
    if ivar is not None:
        ivar = ivar.to(device)
    if mask is not None:
        mask = mask.to(device)
    return x.to(device), ivar, mask, f.to(device)


class CFMTrainer(BaseTrainer[CFMTrainingConfig]):
    """
    Trainer for CFM with epoch-based training.

    Features:
    - OneCycleLR scheduler (ramp up to max LR, then anneal down),
      advanced once per batch over the full epoch budget
    - Sample generation for visualization
    - Per-epoch validation, logging and checkpointing
    """

    def __init__(
        self,
        model: CFM,
        train_loader: DataLoader,
        config: CFMTrainingConfig,
        val_loader: Optional[DataLoader] = None,
    ):
        super().__init__(model, train_loader, config, val_loader)

        # Additional CFM-specific directories
        (self.output_dir / "samples").mkdir(exist_ok=True)

        # Print model info
        num_params = count_parameters(model)
        print("CFM Model initialized:")
        print(f"  Trainable parameters: {num_params:,}")
        print(f"  Conditioning vector dim: {model.cond_vec_dim}")
        print(f"  Learning rate: {config.learning_rate}")
        print(f"  Total training epochs: {config.num_epochs:,}")

    def _setup_optimizer(self):
        """Set up AdamW with a OneCycleLR schedule."""
        trainable_params = [
            p for p in self.model.parameters() if p.requires_grad
        ]
        self.optimizer = AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999),
        )

        steps_per_epoch = len(self.train_loader)
        self.scheduler = build_cfm_scheduler(
            self.optimizer, self.config, steps_per_epoch
        )
        self._scheduler_total_steps = cfm_scheduler_total_steps(
            self.config, steps_per_epoch
        )

    def _train_step(self, batch: Any) -> Dict[str, float]:
        """Execute single CFM training step."""
        x, ivar, mask, f = _extract_cfm_batch(
            batch, self.device, noiseless=self.config.train_on_noiseless
        )
        loss = self.model.compute_loss(x, f)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self._clip_gradients()
        self.optimizer.step()
        # OneCycleLR finishes annealing at ``total_steps - 1``; stepping
        # it beyond that wraps the cosine back up. Guarding here holds
        # the LR at the lr_min_factor floor for the rest of training.
        if self.global_step < self._scheduler_total_steps - 1:
            self.scheduler.step()

        self.global_step += 1

        return {"loss": loss.item(), "lr": self.scheduler.get_last_lr()[0]}

    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch, return average metrics."""
        loss_sum = 0.0
        num_batches = 0
        last_lr = self._get_current_lr()

        progress_bar = tqdm(
            self.train_loader, desc=f"Epoch {self.current_epoch}"
        )

        for batch in progress_bar:
            metrics = self._train_step(batch)

            loss_sum += metrics["loss"]
            last_lr = metrics["lr"]
            num_batches += 1

            progress_bar.set_postfix(
                {
                    "loss": f"{metrics['loss']:.3e}",
                    "lr": f"{metrics['lr']:.3e}",
                }
            )

        return {"loss": loss_sum / num_batches, "lr": last_lr}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Compute validation loss."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            x, ivar, mask, f = _extract_cfm_batch(
                batch, self.device, noiseless=self.config.train_on_noiseless
            )
            loss = self.model.compute_loss(x, f)  # , ivar=ivar, mask=mask)
            total_loss += loss.item()
            num_batches += 1

        self.model.train()
        return {"val_loss": total_loss / num_batches}

    @torch.no_grad()
    def generate_samples(
        self, num_samples: int = 16
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate samples for visualization.

        Pulls a batch from the validation loader (or training loader as
        fallback), takes the first ``num_samples`` conditioning vectors,
        and runs the model's Euler sampler.
        """
        self.model.eval()

        loader = (
            self.val_loader
            if self.val_loader is not None
            else self.train_loader
        )
        batch = next(iter(loader))
        _, _, _, f = _extract_cfm_batch(batch, self.device)
        f = f[:num_samples]

        raw_model = getattr(self.model, "_orig_mod", self.model)
        samples = raw_model.sample(
            batch_size=f.shape[0],
            device=self.device,
            f=f,
            num_steps=50,
        )
        self.model.train()
        return samples, f

    def train(self):
        """Main epoch-based training loop."""
        print(f"\nStarting training from epoch {self.current_epoch}")
        print(f"Training until epoch {self.config.num_epochs}")

        self.model.train()
        if self.device.type == "mps":
            print(
                "torch.compile() skipped on MPS (inductor Metal backend bug)"
            )
        else:
            try:
                self.model = torch.compile(self.model)
                print("Model compiled with torch.compile()")
            except RuntimeError:
                print("torch.compile() not available, skipping")

        # Warmup forward pass to pay the compile cost before the
        # first epoch.
        warmup_batch = next(iter(self.train_loader))
        x, _, _, f = _extract_cfm_batch(
            warmup_batch,
            self.device,
            noiseless=self.config.train_on_noiseless,
        )
        with torch.no_grad():
            self.model.compute_loss(x, f)

        start_epoch = self.current_epoch + 1

        for epoch in range(start_epoch, self.config.num_epochs + 1):
            self.current_epoch = epoch
            print(
                f"\nEpoch {epoch}/{self.config.num_epochs} "
                f"(lr: {self._get_current_lr():.3e})"
            )

            train_metrics = self._train_epoch()

            # Check for non-finite loss
            if not math.isfinite(train_metrics["loss"]):
                print("\n" + "=" * 60)
                print("[ERROR] Non-finite loss detected!")
                print("Stopping training early.")
                print("=" * 60)
                break

            print(f"Epoch {epoch} - Loss: {train_metrics['loss']:.3e}")

            # Validation
            val_metrics = {}
            if epoch % self.config.validate_every == 0:
                val_metrics = self.validate()
                if val_metrics:
                    print(f"  Val - Loss: {val_metrics['val_loss']:.3e}")
                    train_metrics.update(val_metrics)

            # Track best model on validation loss when available,
            # otherwise on training loss
            if val_metrics:
                current_loss = val_metrics["val_loss"]
            else:
                current_loss = train_metrics["loss"]
            loss_type = "val" if val_metrics else "train"

            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.best_step_or_epoch = epoch
                self.save_checkpoint(is_best=True)
                print(
                    f"  New best {loss_type} loss {current_loss:.3e} "
                    f"at epoch {epoch} — saved best.pt"
                )
            else:
                print(
                    f"  Current {loss_type} loss: {current_loss:.3e} "
                    f"at epoch {epoch} | Best: {self.best_loss:.3e} "
                    f"at epoch {self.best_step_or_epoch}"
                )

            if epoch % self.config.log_every == 0:
                self._log_metrics(train_metrics)

            # Sample generation
            if epoch % self.config.sample_every == 0:
                print(f"Generating samples at epoch {epoch}...")
                samples, conditioning = self.generate_samples(
                    self.config.num_sample_images
                )

                sample_path = (
                    self.output_dir / "samples" / f"samples_epoch_{epoch}.pt"
                )
                torch.save(
                    {
                        "samples": samples.cpu(),
                        "conditioning": conditioning.cpu(),
                    },
                    sample_path,
                )

            # Checkpointing
            if epoch % self.config.save_every == 0:
                self.save_checkpoint()

        print("\nTraining complete!")

        self.save_checkpoint()
        self.save_loss_plot()
