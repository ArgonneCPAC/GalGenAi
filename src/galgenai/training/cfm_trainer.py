"""CFM trainer implementation."""

from typing import Any, Dict, Optional, Tuple

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.cfm import CFM, count_parameters
from .base_trainer import BaseTrainer
from .config import CFMTrainingConfig


def _extract_cfm_batch(
    batch, device: torch.device
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """
    Extract (x, f, ivar, mask) from a CFM batch.

    Expected batch formats (see ``data.hsc`` / ``data.cosmos_dataset``):
    - (flux, cond): conditioning only
    - (flux, ivar, mask, cond): conditioning with aux data
    """
    if not isinstance(batch, (tuple, list)):
        raise ValueError(
            "CFM trainer requires batches with a conditioning vector; "
            "got a bare tensor"
        )

    if len(batch) == 2:
        x, cond = batch
        ivar = mask = None
    elif len(batch) == 4:
        x, ivar, mask, cond = batch
        ivar = ivar.to(device)
        mask = mask.to(device)
    else:
        raise ValueError(
            f"Unexpected CFM batch length {len(batch)}; expected 2 or 4"
        )

    return x.to(device), cond.to(device), ivar, mask


class CFMTrainer(BaseTrainer[CFMTrainingConfig]):
    """
    Trainer for CFM with step-based training.

    Features:
    - Warmup + cosine annealing scheduler
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
        """Set up AdamW with cosine annealing (default) or custom
        scheduler."""
        trainable_params = [
            p for p in self.model.parameters() if p.requires_grad
        ]
        self.optimizer = AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999),
        )

        if self.config.scheduler_factory is not None:
            self.scheduler = self.config.scheduler_factory(self.optimizer)
        else:
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_steps - self.config.warmup_steps,
                eta_min=self.config.learning_rate * 0.01,
            )

    def _get_lr_with_warmup(self) -> float:
        """Get current LR accounting for warmup."""
        if self.global_step < self.config.warmup_steps:
            return self.config.learning_rate * (
                self.global_step / self.config.warmup_steps
            )
        if self.scheduler is not None:
            return self.scheduler.get_last_lr()[0]
        return self.config.learning_rate

    def _set_lr(self, lr: float):
        """Set learning rate for all parameter groups."""
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _train_step(self, batch: Any) -> Dict[str, float]:
        """Execute single CFM training step."""
        x, f, ivar, mask = _extract_cfm_batch(batch, self.device)

        loss = self.model.compute_loss(x, f, ivar=ivar, mask=mask)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self._clip_gradients()
        self.optimizer.step()

        # Update LR with warmup handling
        current_lr = self._get_lr_with_warmup()
        self._set_lr(current_lr)

        if (
            self.scheduler is not None
            and self.global_step >= self.config.warmup_steps
        ):
            self.scheduler.step()

        return {"loss": loss.item(), "lr": current_lr}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Compute validation loss."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            x, f, ivar, mask = _extract_cfm_batch(batch, self.device)
            loss = self.model.compute_loss(x, f, ivar=ivar, mask=mask)
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
        _, f, _, _ = _extract_cfm_batch(batch, self.device)
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
                    loss_type = "val" if val_metrics else "train"
                    self.save_checkpoint(is_best=True)
                    pbar.write(
                        f"  New best {loss_type} loss "
                        f"{current_loss:.4f} at step "
                        f"{self.global_step} — saved best.pt"
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
