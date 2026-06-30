"""VAE trainer implementation."""

import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from .base_trainer import BaseTrainer
from .config import VAETrainingConfig
from .utils import extract_batch_data, vae_loss


class VAETrainer(BaseTrainer[VAETrainingConfig]):
    """
    Trainer for VAE models with epoch-based training.

    Features:
    - Supports MSE and masked weighted MSE reconstruction loss
    - Beta-VAE training
    - Per-epoch validation and logging
    - Checkpointing
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        config: VAETrainingConfig,
        val_loader: Optional[DataLoader] = None,
        denorm_fn: Optional[Any] = None,
    ):
        super().__init__(model, train_loader, config, val_loader)
        self.denorm_fn = denorm_fn

        if config.reconstruction_loss_fn == "masked_weighted_mse":
            if self.denorm_fn is not None:
                print("Using denorm in flux for masked_weighted_mse loss")
            else:
                print(
                    "WARNING: Not using denorm in masked_weighted_mse."
                    "Units may not match with ivar"
                )

    def _setup_optimizer(self):
        """Set up Adam optimizer and optional scheduler."""
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        self.scheduler = lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.num_epochs,
            eta_min=self.config.learning_rate * self.config.lr_min_factor,
        )

    def _train_step(self, batch: Any) -> Dict[str, float]:
        """Execute single VAE training step."""
        data, ivar, mask, noiseless_flux = extract_batch_data(
            batch,
            self.device,
            extract_noiseless=self.config.compute_loss_on_noiseless,
        )

        # Forward pass
        reconstruction, mu, logvar = self.model(data)

        # Use noiseless flux as target if enabled
        if self.config.compute_loss_on_noiseless:
            target = noiseless_flux
        else:
            target = data

        # Compute loss
        total_loss, recon_loss, kl_loss = vae_loss(
            reconstruction,
            target,
            mu,
            logvar,
            reconstruction_loss_fn=self.config.reconstruction_loss_fn,
            beta=self.config.beta,
            ivar=ivar,
            mask=mask,
            denorm_fn=self.denorm_fn,
        )

        # Normalize by batch size
        batch_size = data.size(0)
        total_loss = total_loss / batch_size
        recon_loss = recon_loss / batch_size
        kl_loss = kl_loss / batch_size

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self._clip_gradients()
        self.optimizer.step()

        self.global_step += 1

        return {
            "total_loss": total_loss.item(),
            "recon_loss": recon_loss.item(),
            "kl_loss": kl_loss.item(),
            "lr": self._get_current_lr(),
        }

    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch, return average metrics."""
        total_loss_sum = 0.0
        recon_loss_sum = 0.0
        kl_loss_sum = 0.0
        num_batches = 0

        progress_bar = tqdm(
            self.train_loader, desc=f"Epoch {self.current_epoch}"
        )

        for batch in progress_bar:
            metrics = self._train_step(batch)

            total_loss_sum += metrics["total_loss"]
            recon_loss_sum += metrics["recon_loss"]
            kl_loss_sum += metrics["kl_loss"]
            num_batches += 1

            progress_bar.set_postfix(
                {
                    "loss": f"{metrics['total_loss']:.3e}",
                    "recon": f"{metrics['recon_loss']:.3e}",
                    "kl": f"{metrics['kl_loss']:.3e}",
                }
            )

        return {
            "total_loss": total_loss_sum / num_batches,
            "recon_loss": recon_loss_sum / num_batches,
            "kl_loss": kl_loss_sum / num_batches,
            "lr": self._get_current_lr(),
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Compute validation loss."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss_sum = 0.0
        recon_loss_sum = 0.0
        kl_loss_sum = 0.0
        num_batches = 0

        for batch in self.val_loader:
            data, ivar, mask, noiseless_flux = extract_batch_data(
                batch,
                self.device,
                extract_noiseless=self.config.compute_loss_on_noiseless,
            )
            reconstruction, mu, logvar = self.model(data)

            # Use noiseless flux as target if enabled
            if self.config.compute_loss_on_noiseless:
                target = noiseless_flux
            else:
                target = data

            total_loss, recon_loss, kl_loss = vae_loss(
                reconstruction,
                target,
                mu,
                logvar,
                reconstruction_loss_fn=self.config.reconstruction_loss_fn,
                beta=self.config.beta,
                ivar=ivar,
                mask=mask,
                denorm_fn=self.denorm_fn,
            )

            batch_size = data.size(0)
            total_loss_sum += (total_loss / batch_size).item()
            recon_loss_sum += (recon_loss / batch_size).item()
            kl_loss_sum += (kl_loss / batch_size).item()
            num_batches += 1

        return {
            "val_total_loss": total_loss_sum / num_batches,
            "val_recon_loss": recon_loss_sum / num_batches,
            "val_kl_loss": kl_loss_sum / num_batches,
        }

    def train(self):
        """Main epoch-based training loop."""
        print(f"Training on device: {self.device}")
        print(f"Number of epochs: {self.config.num_epochs}")
        print(f"Reconstruction loss: {self.config.reconstruction_loss_fn}")
        print(f"Beta: {self.config.beta}")
        print(
            f"Compute noiseless loss: {self.config.compute_loss_on_noiseless}"
        )
        print(f"Learning rate: {self.config.learning_rate}")
        if self.scheduler is not None:
            print(f"LR scheduler: {self.scheduler.__class__.__name__}")
        print(f"Gradient clipping: max_norm={self.config.max_grad_norm}")
        print("-" * 60)

        self.model.train()
        if self.device.type == "mps":
            print("torch.compile() skipped on MPS")
        else:
            try:
                self.model = torch.compile(self.model)
                print("Model compiled with torch.compile()")
            except RuntimeError:
                print("torch.compile() not available, skipping")

        # Warmup forward pass to pay compile cost before the
        # first epoch.
        warmup_batch = next(iter(self.train_loader))
        data, _, _, _ = extract_batch_data(
            warmup_batch,
            self.device,
            extract_noiseless=self.config.compute_loss_on_noiseless,
        )
        with torch.no_grad():
            self.model(data)

        start_epoch = self.current_epoch + 1

        for epoch in range(start_epoch, self.config.num_epochs + 1):
            self.current_epoch = epoch
            print(
                f"\nEpoch {epoch}/{self.config.num_epochs} "
                f"(lr: {self._get_current_lr():.3e})"
            )

            # Train epoch
            train_metrics = self._train_epoch()

            # Check for non-finite loss
            if not math.isfinite(train_metrics["total_loss"]):
                print("\n" + "=" * 60)
                print("[ERROR] Non-finite loss detected!")
                print("Stopping training early.")
                print("=" * 60)
                break

            # Log metrics
            print(
                f"Epoch {epoch} - "
                f"Loss: {train_metrics['total_loss']:.3e}, "
                f"Recon: {train_metrics['recon_loss']:.3e}, "
                f"KL: {train_metrics['kl_loss']:.3e}"
            )

            # Validation
            if (
                self.val_loader is not None
                and epoch % self.config.validate_every == 0
            ):
                val_metrics = self.validate()
                if val_metrics:
                    val_loss = val_metrics["val_total_loss"]
                    print(f"  Val - Loss: {val_loss:.3e}")
                    train_metrics.update(val_metrics)

                    # Track best model based on validation loss
                    if val_loss < self.best_loss:
                        self.best_loss = val_loss
                        self.best_step_or_epoch = epoch
                        self.save_checkpoint(is_best=True)
                        print(
                            "  New best validation loss:"
                            f" {self.best_loss:.3e} — saved best.pt"
                        )

                    # Print current best info
                    print(
                        f"  Best - Loss: {self.best_loss:.3e} "
                        f"(Epoch {self.best_step_or_epoch})"
                    )

            self._log_metrics(train_metrics)

            # Step learning rate scheduler (once per epoch)
            if self.scheduler is not None:
                self.scheduler.step()

            # Checkpointing (regular scheduled checkpoints)
            if epoch % self.config.save_every == 0:
                self.save_checkpoint(is_best=False)

        print("\nTraining complete!")
        self.save_checkpoint(is_best=False)
        self.save_loss_plot()
