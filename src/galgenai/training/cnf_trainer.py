"""Conditional Normalizing Flow trainer implementation."""

from typing import Any, Dict, Optional

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.cnf import ConditionalNormalizingFlow
from ..models.vae import VAEEncoder
from .base_trainer import BaseTrainer
from .config import CNFTrainingConfig


def _count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class CNFTrainer(BaseTrainer[CNFTrainingConfig]):
    """
    Trainer for Conditional Normalizing Flow with step-based training.
    """

    def __init__(
        self,
        model: ConditionalNormalizingFlow,
        train_loader: DataLoader,
        config: CNFTrainingConfig,
        val_loader: Optional[DataLoader] = None,
        encoder: Optional[VAEEncoder] = None,
    ):
        """
        Initialize CNF trainer.

        Args:
            model: ConditionalNormalizingFlow model
            train_loader: DataLoader providing batches.
                If encoder is None: (latents, conditions)
                If encoder is provided:
                    either (flux img, ivar, mask, condition)
                    or (flux img, condition)
            config: CNFTrainingConfig
            val_loader: Optional validation DataLoader
            encoder: Optional trained encoder for on-the-fly encoding.
                Encodes images to latents during training.
        """
        super().__init__(model, train_loader, config, val_loader)

        # Store encoder and set to eval mode if provided
        self.encoder = encoder
        if self.encoder is not None:
            self.encoder.to(self.device).eval()
            for param in self.encoder.parameters():
                param.requires_grad_(False)
            print("Using on-the-fly latent encoding with frozen VAE encoder")

        # Additional CNF-specific directories
        (self.output_dir / "samples").mkdir(exist_ok=True)

        # Print model info
        num_params = _count_trainable_parameters(model)
        print("Conditional Normalizing Flow initialized:")
        print(f"  Trainable parameters: {num_params:,}")
        print(f"  Latent dimension: {model.latent_dim}")
        print(f"  Condition dimension: {model.condition_dim}")
        print(f"  Number of coupling blocks: {model.num_blocks}")
        print(f"  Learning rate: {config.learning_rate}")
        print(f"  Total training steps: {config.num_steps:,}")

    def _setup_optimizer(self):
        """Set up AdamW with cosine annealing (default) or
        custom scheduler."""
        trainable_params = [
            p for p in self.model.parameters() if p.requires_grad
        ]
        self.optimizer = AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999),
        )

        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.num_steps - self.config.warmup_steps,
            eta_min=self.config.learning_rate * self.config.lr_min_factor,
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

    @torch.no_grad()
    def _extract_latents_and_conditions(
        self, batch: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract latents and conditions from batch.

        If encoder is provided, encodes images on the fly.
        Otherwise, uses precomputed latents from batch.

        Args:
            batch: Either (latents, conditions) or
                (flux, ivar, mask, condition) tuple

        Returns:
            Tuple of (latents, conditions) tensors
        """
        if self.encoder is not None:
            # On-the-fly encoding
            # batch is (flux, ivar, mask, condition) / (flux, condition)
            flux = batch[0].to(self.device)
            conditions = batch[-1].to(self.device)

            mu, logvar = self.encoder(flux)

            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            latents = mu + eps * std

            return latents, conditions
        else:
            latents, conditions = batch
            return latents.to(self.device), conditions.to(self.device)

    def _train_step(self, batch: Any) -> Dict[str, float]:
        """
        Execute single CNF training step.

        Args:
            batch: Either (latents, conditions) or
                (flux, ivar, mask, condition) / (flux, condition)
                depending on encoder

        Returns:
            Dictionary with loss metrics
        """
        # Extract latents and conditions (handles both cases)
        latents, conditions = self._extract_latents_and_conditions(batch)

        # Compute negative log-likelihood loss
        log_probs = self.model.log_prob(latents, conditions)
        nll_loss = -log_probs.mean()

        # Backward pass
        self.optimizer.zero_grad()
        nll_loss.backward()
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

        return {
            "nll_loss": nll_loss.item(),
            "avg_log_prob": log_probs.mean().item(),
            "lr": current_lr,
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Compute validation metrics.

        Returns:
            Dictionary with validation negative log-likelihood
        """
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_nll = 0.0
        total_log_prob = 0.0
        num_batches = 0

        for batch in self.val_loader:
            latents, conditions = self._extract_latents_and_conditions(batch)

            log_probs = self.model.log_prob(latents, conditions)
            nll = -log_probs.mean()

            total_nll += nll.item()
            total_log_prob += log_probs.mean().item()
            num_batches += 1

        self.model.train()
        return {
            "val_nll_loss": total_nll / num_batches,
            "val_avg_log_prob": total_log_prob / num_batches,
        }

    @torch.no_grad()
    def generate_samples(
        self, num_samples: int = 64
    ) -> Dict[str, torch.Tensor]:
        """
        Generate latent samples for visualization.

        Args:
            num_samples: Number of samples to generate

        Returns:
            Dictionary with samples and conditioning used
        """
        self.model.eval()

        # Get conditioning from validation set
        # (or training if no val set)
        loader = self.val_loader if self.val_loader else self.train_loader
        batch = next(iter(loader))
        _, conditions = self._extract_latents_and_conditions(batch)

        # Take subset of conditions
        conditions = conditions[:num_samples]

        # Sample latents given conditioning
        samples = self.model.sample(conditions, num_samples=1)

        self.model.train()
        return {
            "latent_samples": samples.cpu(),
            "conditions": conditions.cpu(),
        }

    @torch.no_grad()
    def compute_log_det_statistics(self) -> Dict[str, float]:
        """
        Compute statistics of log determinant Jacobian.

        Useful for monitoring numerical stability during training.

        Returns:
            Dictionary with log_det statistics
        """
        self.model.eval()

        # Get a batch from training data
        batch = next(iter(self.train_loader))
        latents, conditions = self._extract_latents_and_conditions(batch)

        # Compute log determinants
        _, log_dets = self.model.forward(latents, conditions)

        self.model.train()
        return {
            "log_det_mean": log_dets.mean().item(),
            "log_det_std": log_dets.std().item(),
            "log_det_min": log_dets.min().item(),
            "log_det_max": log_dets.max().item(),
        }

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
            """Infinite data loader generator."""
            while True:
                for batch in self.train_loader:
                    yield batch

        data_iter = iter(infinite_loader())

        # Running averages for logging
        running_nll = 0.0
        running_log_prob = 0.0
        log_steps = 0

        # Progress bar spanning all steps
        pbar = tqdm(
            total=self.config.num_steps,
            initial=self.global_step,
            desc="Training CNF",
            unit="step",
        )

        while self.global_step < self.config.num_steps:
            batch = next(data_iter)
            loss_dict = self._train_step(batch)

            running_nll += loss_dict["nll_loss"]
            running_log_prob += loss_dict["avg_log_prob"]
            log_steps += 1

            self.global_step += 1
            pbar.update(1)

            # Update progress bar with current metrics
            pbar.set_postfix(
                {
                    "nll": f"{loss_dict['nll_loss']:.3e}",
                    "log_p": f"{loss_dict['avg_log_prob']:.3f}",
                    "lr": f"{loss_dict['lr']:.3e}",
                }
            )

            # Periodic logging (for metrics tracking, not display)
            if self.global_step % self.config.log_every == 0:
                avg_metrics = {
                    "nll_loss": running_nll / log_steps,
                    "avg_log_prob": running_log_prob / log_steps,
                    "lr": loss_dict["lr"],
                }

                # Validation
                val_metrics = {}
                if self.global_step % self.config.validate_every == 0:
                    val_metrics = self.validate()
                    if val_metrics:
                        pbar.write(
                            f"  Step {self.global_step} Val"
                            f" - NLL: {val_metrics['val_nll_loss']:.3e}"
                            f", Log P: {val_metrics['val_avg_log_prob']:.3f}"
                        )
                        avg_metrics.update(val_metrics)

                    # Also log determinant statistics during validation
                    log_det_stats = self.compute_log_det_statistics()
                    pbar.write(
                        f"  Log det Jacobian: "
                        f"mean={log_det_stats['log_det_mean']:.2f}, "
                        f"std={log_det_stats['log_det_std']:.2f}"
                    )
                    avg_metrics.update(log_det_stats)

                self._log_metrics(avg_metrics)

                # Track best loss (use validation if available,
                # otherwise training)
                if val_metrics:
                    current_loss = val_metrics["val_nll_loss"]
                else:
                    current_loss = avg_metrics["nll_loss"]

                if current_loss < self.best_loss:
                    self.best_loss = current_loss
                    self.best_step_or_epoch = self.global_step
                    loss_type = "val" if val_metrics else "train"
                    self.save_checkpoint(is_best=True)
                    pbar.write(
                        f"  New best {loss_type} loss"
                        f" {current_loss:.4f} at step"
                        f" {self.global_step} — saved best.pt"
                    )
                else:
                    loss_type = "val" if val_metrics else "train"
                    pbar.write(
                        f"  Current {loss_type} loss: {current_loss:.4f} "
                        f"at step {self.global_step} | "
                        f"Best: {self.best_loss:.4f} "
                        f"at step {self.best_step_or_epoch}"
                    )

                # Reset running stats
                running_nll = 0.0
                running_log_prob = 0.0
                log_steps = 0

            # Sample generation
            if self.global_step % self.config.sample_every == 0:
                pbar.write(
                    f"Generating {self.config.num_sample_latents} "
                    f"latent samples at step {self.global_step}..."
                )
                sample_dict = self.generate_samples(
                    self.config.num_sample_latents
                )

                sample_path = (
                    self.output_dir
                    / "samples"
                    / f"latent_samples_step_{self.global_step}.pt"
                )
                torch.save(sample_dict, sample_path)

            # Checkpointing
            if self.global_step % self.config.save_every == 0:
                self.save_checkpoint()

        pbar.close()
        print("\nTraining complete")

        # Save final checkpoint (best.pt already saved
        # whenever a new best was found)
        self.save_checkpoint()
        self.save_loss_plot()
