"""CFM trainer implementation."""

from typing import Any, Dict, Optional, Tuple

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.cfm import CFM, count_parameters
from .base_trainer import BaseTrainer
from .config import CFMTrainingConfig


def cfm_scheduler_total_steps(config: CFMTrainingConfig) -> int:
    """Number of steps over which the OneCycleLR schedule runs.

    Defaults to ``num_steps``, but can be set shorter via
    ``lr_converge_at_step`` so the schedule ramps up and anneals down
    early, then holds flat at the ``lr_min_factor`` floor for the
    remainder of training.
    """
    return config.lr_converge_at_step or config.num_steps


def build_cfm_scheduler(
    optimizer: torch.optim.Optimizer, config: CFMTrainingConfig
) -> OneCycleLR:
    """Build the OneCycleLR schedule used for CFM training: ramps up
    to ``learning_rate`` over ``warmup_steps``, then cosine-anneals
    down to ``learning_rate * lr_min_factor``."""
    total_steps = cfm_scheduler_total_steps(config)
    return OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        total_steps=total_steps,
        pct_start=config.warmup_steps / total_steps,
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
    Trainer for CFM with step-based training.

    Features:
    - OneCycleLR scheduler (ramp up to max LR, then anneal down)
    - Sample generation for visualization
    - Infinite data loader pattern
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
        print(f"  Total training steps: {config.num_steps:,}")

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

        self.scheduler = build_cfm_scheduler(self.optimizer, self.config)
        self._scheduler_total_steps = cfm_scheduler_total_steps(self.config)

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
        if self.global_step < self._scheduler_total_steps:
            self.scheduler.step()

        return {"loss": loss.item(), "lr": self.scheduler.get_last_lr()[0]}

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
        """Main step-based training loop."""
        print(f"\nStarting training from step {self.global_step}")
        print(f"Training for {self.config.num_steps - self.global_step} steps")

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

        def infinite_loader():
            while True:
                for batch in self.train_loader:
                    yield batch

        data_iter = iter(infinite_loader())

        # Running averages for logging
        running_loss = 0.0
        log_steps = 0

        # Progress bar spanning all steps
        pbar = tqdm(
            total=self.config.num_steps,
            initial=self.global_step,
            desc="Training",
            unit="step",
        )

        while self.global_step < self.config.num_steps:
            batch = next(data_iter)
            loss_dict = self._train_step(batch)

            running_loss += loss_dict["loss"]
            log_steps += 1

            self.global_step += 1
            pbar.update(1)

            pbar.set_postfix(
                {
                    "loss": f"{loss_dict['loss']:.3e}",
                    "lr": f"{loss_dict['lr']:.3e}",
                }
            )

            # Periodic logging (for metrics tracking, not display)
            if self.global_step % self.config.log_every == 0:
                avg_metrics = {
                    "loss": running_loss / log_steps,
                    "lr": loss_dict["lr"],
                }

                # Validation
                val_metrics = {}
                if self.global_step % self.config.validate_every == 0:
                    val_metrics = self.validate()
                    if val_metrics:
                        pbar.write(
                            f"  Step {self.global_step} Val"
                            f" - Loss: {val_metrics['val_loss']:.3e}"
                        )
                        avg_metrics.update(val_metrics)

                self._log_metrics(avg_metrics)

                if val_metrics:
                    current_loss = val_metrics["val_loss"]
                else:
                    current_loss = avg_metrics["loss"]

                if current_loss < self.best_loss:
                    self.best_loss = current_loss
                    self.best_step_or_epoch = self.global_step
                    loss_type = "val" if val_metrics else "train"
                    self.save_checkpoint(is_best=True)
                    pbar.write(
                        f"  New best {loss_type} loss "
                        f"{current_loss:.3e} at step "
                        f"{self.global_step} — saved best.pt"
                    )
                else:
                    loss_type = "val" if val_metrics else "train"
                    pbar.write(
                        f"  Current {loss_type} loss: {current_loss:.3e} "
                        f"at step {self.global_step} | "
                        f"Best: {self.best_loss:.3e} "
                        f"at step {self.best_step_or_epoch}"
                    )

                running_loss = 0.0
                log_steps = 0

            # Sample generation
            if self.global_step % self.config.sample_every == 0:
                pbar.write(f"Generating samples at step {self.global_step}...")
                samples, conditioning = self.generate_samples(
                    self.config.num_sample_images
                )

                sample_path = (
                    self.output_dir
                    / "samples"
                    / f"samples_step_{self.global_step}.pt"
                )
                torch.save(
                    {
                        "samples": samples.cpu(),
                        "conditioning": conditioning.cpu(),
                    },
                    sample_path,
                )

            # Checkpointing
            if self.global_step % self.config.save_every == 0:
                self.save_checkpoint()

        pbar.close()
        print("\nTraining complete!")

        self.save_checkpoint()
        self.save_loss_plot()
