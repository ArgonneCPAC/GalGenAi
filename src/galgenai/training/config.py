"""Training configuration dataclasses."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from galgenai.config import load_config


@dataclass
class BaseTrainingConfig:
    """Base training configuration with shared parameters."""

    # Optimization
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0

    # Scheduler (callable that takes optimizer and returns scheduler)
    scheduler_factory: Optional[Callable[[Any], Any]] = None

    # Logging & checkpointing
    log_every: int = 100
    save_every: int = 1000

    # Paths
    output_dir: str = "./output"
    checkpoint_path: Optional[str] = None

    # Device (auto-detect if None)
    device: Optional[str] = None


@dataclass
class VAETrainingConfig(BaseTrainingConfig):
    """VAE-specific training configuration."""

    # VAE-specific parameters
    reconstruction_loss_fn: str = "mse"
    beta: float = 1.0

    # Epoch-based training
    num_epochs: int = 10

    # Validation
    validate_every: int = 1

    # Override defaults for epoch-based training
    log_every: int = 1
    save_every: int = 10


@dataclass
class LCFMTrainingConfig(BaseTrainingConfig):
    """LCFM-specific training configuration."""

    # LCFM-specific loss
    beta: float = 0.001

    # Step-based training
    num_steps: int = 100_000
    warmup_steps: int = 1000

    # Sampling during training
    sample_every: int = 5000
    num_sample_images: int = 16

    # Validation (should be a multiple of log_every)
    validate_every: int = 500

    # Override defaults for step-based training
    log_every: int = 100
    save_every: int = 10_000
    learning_rate: float = 2e-4
    weight_decay: float = 0.01


@dataclass
class CFMTrainingConfig(BaseTrainingConfig):
    """CFM-specific training configuration."""

    # Step-based training
    num_steps: int = 100_000
    warmup_steps: int = 1000

    # Sampling during training
    sample_every: int = 5000
    num_sample_images: int = 16

    # Validation (should be a multiple of log_every)
    validate_every: int = 500

    # Override defaults for step-based training
    log_every: int = 100
    save_every: int = 10_000
    learning_rate: float = 2e-4
    weight_decay: float = 0.01


@dataclass
class CNFTrainingConfig(BaseTrainingConfig):
    """CNF training config."""

    # Step-based training
    num_steps: int = 50_000
    warmup_steps: int = 1000

    # Sampling during training
    sample_every: int = 5000
    num_sample_latents: int = 64

    # Validation (should be a multiple of log_every)
    validate_every: int = 500

    # Override defaults for step-based training
    log_every: int = 100
    save_every: int = 5_000


# ============================================================================
# Config loading functions
# ============================================================================


def load_vae_training_config(
    config_path: Optional[str] = None,
) -> VAETrainingConfig:
    """
    Load VAE training config from YAML file.

    Parameters
    ----------
    config_path : str, optional
        Path to config file. If None, uses default galgenai_config.yaml

    Returns
    -------
    VAETrainingConfig
        Configuration instance loaded from file
    """
    config = load_config(config_path)
    training_config = config["training"]
    vae_config = training_config["vae"]

    # Automatically append /vae to output directory
    output_dir = Path(training_config["output_dir"]) / "vae"

    return VAETrainingConfig(
        # VAE-specific
        reconstruction_loss_fn=vae_config["reconstruction_loss_fn"],
        beta=vae_config["beta"],
        num_epochs=vae_config["epochs"],
        validate_every=vae_config["validate_every"],
        # Base config
        learning_rate=vae_config["lr"],
        weight_decay=vae_config["weight_decay"],
        max_grad_norm=vae_config["max_grad_norm"],
        log_every=vae_config["log_every"],
        save_every=vae_config["save_every"],
        output_dir=str(output_dir),
        checkpoint_path=None,
        device=None,
        scheduler_factory=None,
    )


def load_lcfm_training_config(
    config_path: Optional[str] = None,
) -> LCFMTrainingConfig:
    """
    Load LCFM training config from YAML file.

    Parameters
    ----------
    config_path : str, optional
        Path to config file. If None, uses default galgenai_config.yaml

    Returns
    -------
    LCFMTrainingConfig
        Configuration instance loaded from file
    """
    config = load_config(config_path)
    training_config = config["training"]
    lcfm_config = training_config["lcfm"]

    # Automatically append /lcfm to output directory
    output_dir = Path(training_config["output_dir"]) / "lcfm"

    return LCFMTrainingConfig(
        # LCFM-specific
        beta=lcfm_config["beta"],
        num_steps=lcfm_config["steps"],
        warmup_steps=lcfm_config["warmup"],
        sample_every=lcfm_config["sample_every"],
        num_sample_images=lcfm_config["num_sample_images"],
        validate_every=lcfm_config["validate_every"],
        # Base config
        learning_rate=lcfm_config["lr"],
        weight_decay=lcfm_config["weight_decay"],
        max_grad_norm=lcfm_config["max_grad_norm"],
        log_every=lcfm_config["log_every"],
        save_every=lcfm_config["save_every"],
        output_dir=str(output_dir),
        checkpoint_path=None,
        device=None,
        scheduler_factory=None,
    )


def load_cnf_training_config(
    config_path: Optional[str] = None,
) -> CNFTrainingConfig:
    """
    Load CNF training config from YAML file.

    Parameters
    ----------
    config_path : str, optional
        Path to config file. If None, uses default galgenai_config.yaml

    Returns
    -------
    CNFTrainingConfig
        Configuration instance loaded from file
    """
    config = load_config(config_path)
    training_config = config["training"]
    cnf_config = training_config["cnf"]

    # Automatically append /cnf to output directory
    output_dir = Path(training_config["output_dir"]) / "cnf"

    return CNFTrainingConfig(
        # CNF-specific
        num_steps=cnf_config["steps"],
        warmup_steps=cnf_config["warmup"],
        sample_every=cnf_config["sample_every"],
        num_sample_latents=cnf_config["num_sample_latents"],
        validate_every=cnf_config["validate_every"],
        # Base config
        learning_rate=cnf_config["lr"],
        weight_decay=cnf_config["weight_decay"],
        max_grad_norm=cnf_config["max_grad_norm"],
        log_every=cnf_config["log_every"],
        save_every=cnf_config["save_every"],
        output_dir=str(output_dir),
        checkpoint_path=None,
        device=None,
        scheduler_factory=None,
    )
