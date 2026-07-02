"""
VAE + CNF Training Pipeline for COSMOS FITS Dataset.

Run with:
    uv run python scripts/train_cnf_cosmos.py
    uv run python scripts/train_cnf_cosmos.py \
        --config path/to/config.yaml
    uv run python scripts/train_cnf_cosmos.py --skip-vae
"""

import argparse
from pathlib import Path

import torch

from galgenai import get_device
from galgenai.config import (
    copy_config_to_results,
    load_config,
    resolve_config_path,
)
from galgenai.data.cosmos_dataset import load_fits_dataset, make_loaders
from galgenai.data.normalization import (
    get_conditional_norm_fn,
    get_image_norm_fn,
    save_conditional_stats,
    save_image_norm_stats,
)
from galgenai.models import ConditionalNormalizingFlow, VAE
from galgenai.training import (
    CNFTrainer,
    VAETrainer,
    load_cnf_training_config,
    load_vae_training_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train VAE + CNF on a COSMOS FITS dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(resolve_config_path()),
        help="Training config file",
    )
    parser.add_argument(
        "--skip-vae",
        action="store_true",
        help="Skip VAE training and load from existing checkpoint",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    cfg = load_config(args.config)

    data_cfg = cfg["data"]
    cosmos_cfg = cfg["datasets"]["cosmos"]
    vae_model_cfg = cfg["models"]["vae"]
    cnf_model_cfg = cfg["models"]["cnf"]
    vae_train_cfg = cfg["training"]["vae"]

    results_dir = Path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    copied_config = copy_config_to_results(args.config, results_dir)

    nx = data_cfg["image_size"]
    batch_size = data_cfg["batch_size"]
    in_channels = data_cfg["in_channels"]
    condition_cols = data_cfg["condition_cols"]
    condition_dim = len(condition_cols)
    latent_dim = vae_model_cfg["latent_dim"]
    image_norm_type = data_cfg["image_norm_type"]

    print(f"Using device: {device}")
    print(f"Results directory: {results_dir}")
    print(f"Copied config to: {copied_config}")

    print(f"\nLoading FITS dataset from: {cosmos_cfg['path']}")
    dataset_raw = load_fits_dataset(
        cosmos_cfg["path"],
        metadata_file=cosmos_cfg.get("metadata_file", "metadata.csv"),
        mag_cols=cosmos_cfg["mag_cols"],
        redshift_col=cosmos_cfg["redshift_col"],
        mag_sentinel=cosmos_cfg.get("mag_sentinel", 999.0),
        redshift_sentinel=cosmos_cfg.get("redshift_sentinel", -99.0),
        nx=nx,
        load_noiseless=vae_train_cfg["compute_loss_on_noiseless"],
    )

    n_total = len(dataset_raw)
    train_ratio = cosmos_cfg["train_ratio"]
    val_ratio = cosmos_cfg["val_ratio"]
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = int(n_total * (1 - train_ratio - val_ratio))
    print(
        f"Dataset sizes: {n_train} train / {n_val} val / {n_test} test "
        f"(total: {n_total})"
    )

    norm_cfg = cosmos_cfg["normalization"]
    print(f"\nImage normalization: {image_norm_type}")
    image_norm_fn, image_denorm_fn, norm_stats = get_image_norm_fn(
        img_norm_type=image_norm_type,
        config=norm_cfg["image"],
        return_denorm=True,
    )
    norm_stats_path = results_dir / "norm_stats.yaml"
    save_image_norm_stats(norm_stats, norm_stats_path)
    print(f"Image normalization stats saved to: {norm_stats_path}")

    print(f"\nConditioning columns ({condition_dim}): {condition_cols}")
    conditional_norm_fn, cond_stats = get_conditional_norm_fn(
        config=norm_cfg["conditions"],
    )
    if condition_cols != cond_stats.cols:
        raise ValueError(
            f"Configured condition_cols {condition_cols} do not match "
            f"normalization.conditions.cols {cond_stats.cols}"
        )
    cond_stats_path = results_dir / "cond_stats.yaml"
    save_conditional_stats(cond_stats, cond_stats_path)
    print(f"Conditional stats saved to: {cond_stats_path}")

    print("\nCreating data loaders")
    train_loader, val_loader, test_loader = make_loaders(
        dataset_raw,
        nx=nx,
        batch_size=batch_size,
        num_workers=cosmos_cfg["num_workers"],
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        random_seed=cosmos_cfg["split_seed"],
        image_norm_fn=image_norm_fn,
        return_aux_data=True,
        return_noiseless_flux=vae_train_cfg["compute_loss_on_noiseless"],
        condition_cols=condition_cols,
        conditional_norm_fn=conditional_norm_fn,
        invert_mask=cosmos_cfg.get("invert_mask", False),
        augment_train=cosmos_cfg.get("augment_train", True),
    )
    print(f"Crop size: {nx}x{nx} px")
    print(f"Batches: {len(train_loader)} train / {len(val_loader)} val")
    if test_loader is not None:
        print(f"         {len(test_loader)} test")

    if not args.skip_vae:
        print("\n" + "=" * 60)
        print("STAGE 1: TRAINING VAE")
        print("=" * 60)

        vae = VAE(
            in_channels=in_channels,
            latent_dim=latent_dim,
            input_size=nx,
        )
        print(f"VAE parameters: {sum(p.numel() for p in vae.parameters()):,}")

        vae_config = load_vae_training_config(args.config)
        vae_trainer = VAETrainer(
            model=vae,
            train_loader=train_loader,
            config=vae_config,
            val_loader=val_loader,
            denorm_fn=image_denorm_fn,
        )
        vae_trainer.train()

        del vae, vae_trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        print("\n" + "=" * 60)
        print("STAGE 1: SKIPPED - VAE training not required")
        print("=" * 60)

    print("\n" + "=" * 60)
    print("STAGE 2: TRAINING CONDITIONAL NORMALIZING FLOW")
    print("=" * 60)

    vae_ckpt_path = results_dir / "vae" / "checkpoints" / "best.pt"
    if not vae_ckpt_path.exists():
        raise FileNotFoundError(
            f"No VAE checkpoint found at {vae_ckpt_path}. Train VAE first."
        )

    vae = VAE(
        in_channels=in_channels,
        latent_dim=latent_dim,
        input_size=nx,
    )
    checkpoint = torch.load(
        vae_ckpt_path, map_location=device, weights_only=False
    )
    vae.load_state_dict(checkpoint["model_state_dict"])

    encoder = vae.encoder
    encoder.to(device).eval()
    for param in encoder.parameters():
        param.requires_grad_(False)
    print(f"Loaded frozen encoder from: {vae_ckpt_path}")

    del vae, checkpoint
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    cnf = ConditionalNormalizingFlow(
        latent_dim=latent_dim,
        condition_dim=condition_dim,
        num_blocks=cnf_model_cfg["num_blocks"],
        hidden_dim=cnf_model_cfg["hidden_dim"],
    ).to(device)
    print(f"CNF parameters: {sum(p.numel() for p in cnf.parameters()):,}")

    cnf_config = load_cnf_training_config(args.config)
    cnf_trainer = CNFTrainer(
        model=cnf,
        train_loader=train_loader,
        config=cnf_config,
        val_loader=val_loader,
        encoder=encoder,
    )
    cnf_trainer.train()

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"""
Output layout:
  Config file        : {copied_config}
  Normalization stats: {norm_stats_path}
  Conditional stats  : {cond_stats_path}
  VAE checkpoints    : {results_dir / "vae" / "checkpoints"}
  VAE loss plot      : {results_dir / "vae" / "loss_history.png"}
  CNF checkpoints    : {results_dir / "cnf" / "checkpoints"}
  CNF samples        : {results_dir / "cnf" / "samples"}
  CNF loss plot      : {results_dir / "cnf" / "loss_history.png"}
""")


if __name__ == "__main__":
    main()
