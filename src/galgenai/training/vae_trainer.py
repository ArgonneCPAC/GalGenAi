"""VAE trainer implementation."""

import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
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
    ):
        super().__init__(model, train_loader, config, val_loader)

    def _setup_optimizer(self):
        """Set up Adam optimizer and optional scheduler."""
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        if self.config.scheduler_factory is not None:
            self.scheduler = self.config.scheduler_factory(self.optimizer)

    def _train_step(self, batch: Any) -> Dict[str, float]:
        """Execute single VAE training step."""
        data, ivar, mask = extract_batch_data(batch, self.device)

        # Forward + loss under autocast on CUDA (bf16, no GradScaler).
        with self._autocast_ctx():
            reconstruction, mu, logvar = self.model(data)
            total_loss, recon_loss, kl_loss = vae_loss(
                reconstruction,
                data,
                mu,
                logvar,
                reconstruction_loss_fn=self.config.reconstruction_loss_fn,
                beta=self.config.beta,
                ivar=ivar,
                mask=mask,
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

        if self.scheduler is not None:
            self.scheduler.step()

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
            data, ivar, mask = extract_batch_data(batch, self.device)
            with self._autocast_ctx():
                reconstruction, mu, logvar = self.model(data)
                total_loss, recon_loss, kl_loss = vae_loss(
                    reconstruction,
                    data,
                    mu,
                    logvar,
                    reconstruction_loss_fn=self.config.reconstruction_loss_fn,
                    beta=self.config.beta,
                    ivar=ivar,
                    mask=mask,
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
        data, _, _ = extract_batch_data(warmup_batch, self.device)
        with torch.no_grad(), self._autocast_ctx():
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
                    val_loss = val_metrics['val_total_loss']
                    print(f"  Val - Loss: {val_loss:.3e}")
                    train_metrics.update(val_metrics)

                    # Track best model based on validation loss
                    if val_loss < self.best_loss:
                        self.best_loss = val_loss
                        self.save_checkpoint(is_best=True)
                        print(f"  New best validation loss: {self.best_loss:.3e} — saved best.pt")

            self._log_metrics(train_metrics)

            # Checkpointing (regular scheduled checkpoints)
            if epoch % self.config.save_every == 0:
                self.save_checkpoint(is_best=False)

        print("\nTraining complete!")
        self.save_checkpoint(is_best=False)
