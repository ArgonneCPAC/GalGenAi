"""Abstract base trainer class."""

from abc import ABC, abstractmethod
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Generic, List, Optional, TypeVar

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..utils import get_device
from .config import BaseTrainingConfig

ConfigT = TypeVar("ConfigT", bound=BaseTrainingConfig)


class BaseTrainer(ABC, Generic[ConfigT]):
    """
    Abstract base class for model trainers.

    Provides shared functionality:
    - Device management
    - Checkpointing (save/load)
    - Gradient clipping
    - Loss history tracking
    - Output directory setup

    Subclasses must implement:
    - _train_step(): Single training step logic
    - _setup_optimizer(): Optimizer/scheduler creation
    - train(): Main training loop
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        config: ConfigT,
        val_loader: Optional[DataLoader] = None,
    ):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Device setup
        if config.device is None:
            self.device = get_device()
        else:
            self.device = torch.device(config.device)
        self.model = model.to(self.device)

        # Output directories
        self.output_dir = Path(config.output_dir)
        self._setup_directories()

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_loss = float("inf")
        self.loss_history: List[Dict[str, Any]] = []

        # Optimizer and scheduler (set by subclass)
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[Any] = None

        # Setup optimizer
        self._setup_optimizer()

        # Resume from checkpoint if specified
        if config.checkpoint_path is not None:
            self.load_checkpoint(config.checkpoint_path)

    def _setup_directories(self):
        """Create output directories."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)

    @abstractmethod
    def _train_step(self, batch: Any) -> Dict[str, float]:
        """
        Execute single training step.

        Args:
            batch: Input batch from dataloader.

        Returns:
            Dict with loss components (must include 'total_loss').
        """
        pass

    @abstractmethod
    def _setup_optimizer(self):
        """Set up optimizer and optional scheduler."""
        pass

    @abstractmethod
    def train(self):
        """Main training loop."""
        pass

    def _clip_gradients(self):
        """Clip gradients by max norm."""
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.max_grad_norm
        )

    def _autocast_ctx(self):
        """
        Mixed-precision context manager.

        bf16 autocast on Ampere+ (sm_80+): ~1.5-2x speedup at no
        quality cost, no GradScaler needed (bf16 has fp32-equivalent
        exponent range).
        """
        if (
            self.device.type == "cuda"
            and torch.cuda.is_bf16_supported(including_emulation=False)
        ):
            return torch.amp.autocast(
                device_type="cuda", dtype=torch.bfloat16
            )
        return nullcontext()

    def _get_current_lr(self) -> float:
        """Get current learning rate from optimizer."""
        if self.optimizer is None:
            return 0.0
        return self.optimizer.param_groups[0]["lr"]

    def save_checkpoint(
        self, path: Optional[str] = None, is_best: bool = False
    ):
        """Save model checkpoint."""
        if path is None:
            path = (
                self.output_dir / "checkpoints" / f"step_{self.global_step}.pt"
            )

        # Unwrap torch.compile() wrapper so state_dict keys are clean
        raw_model = getattr(self.model, "_orig_mod", self.model)

        checkpoint = {
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": (
                self.optimizer.state_dict() if self.optimizer else None
            ),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler else None
            ),
            "config": self.config,
            "loss_history": self.loss_history,
            "best_loss": self.best_loss,
        }

        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")

        if is_best:
            best_path = self.output_dir / "checkpoints" / "best.pt"
            torch.save(checkpoint, best_path)
            print(f"Saved best checkpoint to {best_path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])

        if self.optimizer is not None and checkpoint.get(
            "optimizer_state_dict"
        ):
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler is not None and checkpoint.get(
            "scheduler_state_dict"
        ):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.global_step = checkpoint.get("global_step", 0)
        self.current_epoch = checkpoint.get("current_epoch", 0)
        self.loss_history = checkpoint.get("loss_history", [])
        self.best_loss = checkpoint.get("best_loss", float("inf"))

        print(f"Loaded checkpoint from {path} (step {self.global_step})")

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Run validation loop. Override in subclass for custom validation.

        Returns:
            Dict with validation metrics (empty if no val_loader).
        """
        if self.val_loader is None:
            return {}

        self.model.eval()
        return {}

    def _log_metrics(self, metrics: Dict[str, float], prefix: str = ""):
        """Store metrics in loss history."""
        entry = {"step": self.global_step, "epoch": self.current_epoch}
        entry.update({f"{prefix}{k}": v for k, v in metrics.items()})
        self.loss_history.append(entry)
