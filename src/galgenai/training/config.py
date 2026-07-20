"""Training configuration dataclasses."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from galgenai.config import load_config


@dataclass
class BaseTrainingConfig:
    """Base training configuration with shared parameters."""

    # Optimization
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0

    # Scheduler parameters
    lr_min_factor: float = 0.01

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

    reconstruction_loss_fn: str = "mse"
    beta: float = 1.0
    compute_loss_on_noiseless: bool = False

    num_epochs: int = 10
    validate_every: int = 1

    log_every: int = 1
    save_every: int = 10


@dataclass
class CFMTrainingConfig(BaseTrainingConfig):
    """CFM-specific training configuration."""

    num_epochs: int = 100
    warmup_epochs: float = 1.0

    # OneCycleLR: initial_lr = learning_rate / div_factor
    div_factor: float = 25.0
    # If set, the schedule ramps up + anneals down within this many
    # epochs (< num_epochs) instead of the full run, then holds flat at
    # the lr_min_factor floor for the remaining epochs.
    lr_converge_at_epoch: Optional[float] = None

    sample_every: int = 5
    num_sample_images: int = 16
    validate_every: int = 1

    log_every: int = 1
    save_every: int = 10
    learning_rate: float = 2e-4
    weight_decay: float = 0.01

    train_on_noiseless: bool = False


@dataclass
class CNFTrainingConfig(BaseTrainingConfig):
    """CNF training config."""

    num_epochs: int = 50
    warmup_epochs: float = 1.0

    sample_every: int = 5
    num_sample_latents: int = 64
    validate_every: int = 1

    log_every: int = 1
    save_every: int = 5


def _model_output_dir(config: dict, model_name: str) -> str:
    """Return the model artifact directory."""
    return str(Path(config["results_dir"]) / model_name)


def load_vae_training_config(
    config_path: Optional[str] = None,
) -> VAETrainingConfig:
    """Load VAE training config from YAML."""
    config = load_config(config_path)
    vae_config = config["training"]["vae"]

    return VAETrainingConfig(
        reconstruction_loss_fn=vae_config["reconstruction_loss_fn"],
        beta=vae_config["beta"],
        compute_loss_on_noiseless=vae_config["compute_loss_on_noiseless"],
        num_epochs=vae_config["epochs"],
        validate_every=vae_config["validate_every"],
        learning_rate=vae_config["lr"],
        weight_decay=vae_config["weight_decay"],
        max_grad_norm=vae_config["max_grad_norm"],
        lr_min_factor=vae_config.get("lr_min_factor", 0.01),
        log_every=vae_config["log_every"],
        save_every=vae_config["save_every"],
        output_dir=_model_output_dir(config, "vae"),
    )


def load_cfm_training_config(
    config_path: Optional[str] = None,
) -> CFMTrainingConfig:
    """Load CFM training config from YAML."""
    config = load_config(config_path)
    cfm_config = config["training"]["cfm"]

    return CFMTrainingConfig(
        num_epochs=cfm_config["epochs"],
        warmup_epochs=cfm_config["warmup_epochs"],
        div_factor=cfm_config.get("div_factor", 25.0),
        lr_converge_at_epoch=cfm_config.get("lr_converge_at_epoch"),
        sample_every=cfm_config["sample_every"],
        num_sample_images=cfm_config["num_sample_images"],
        validate_every=cfm_config["validate_every"],
        learning_rate=cfm_config["lr"],
        weight_decay=cfm_config["weight_decay"],
        max_grad_norm=cfm_config["max_grad_norm"],
        lr_min_factor=cfm_config.get("lr_min_factor", 0.01),
        log_every=cfm_config["log_every"],
        save_every=cfm_config["save_every"],
        output_dir=_model_output_dir(config, "cfm"),
        train_on_noiseless=cfm_config["train_on_noiseless"],
    )


def load_cnf_training_config(
    config_path: Optional[str] = None,
) -> CNFTrainingConfig:
    """Load CNF training config from YAML."""
    config = load_config(config_path)
    cnf_config = config["training"]["cnf"]

    return CNFTrainingConfig(
        num_epochs=cnf_config["epochs"],
        warmup_epochs=cnf_config["warmup_epochs"],
        sample_every=cnf_config["sample_every"],
        num_sample_latents=cnf_config["num_sample_latents"],
        validate_every=cnf_config["validate_every"],
        learning_rate=cnf_config["lr"],
        weight_decay=cnf_config["weight_decay"],
        max_grad_norm=cnf_config["max_grad_norm"],
        lr_min_factor=cnf_config.get("lr_min_factor", 0.01),
        log_every=cnf_config["log_every"],
        save_every=cnf_config["save_every"],
        output_dir=_model_output_dir(config, "cnf"),
    )
