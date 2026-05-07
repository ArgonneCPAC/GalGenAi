"""
VAE + DL-CFM Training Pipeline for COSMOS HuggingFace Dataset

Trains a two-stage generative model on simulated galaxy images:

  Stage 1  Train a VAE on COSMOS multi-band images.
  Stage 2  Freeze the VAE encoder and train an LCFM model with the
           DL-CFM disentanglement loss (Ganguli et al. 2025).

Run with:
    uv run python scripts/train_dlcfm_cosmos.py
    uv run python scripts/train_dlcfm_cosmos.py --skip-vae
"""

import argparse
import shutil
from pathlib import Path

import torch

from galgenai import get_device
from galgenai.config import load_config
from galgenai.data.normalization import (
    get_image_norm_fn,
    get_conditional_norm_fn,
    save_image_norm_stats,
    save_conditional_stats,
)
from galgenai.data.cosmos_dataset import (
    load_fits_dataset,
    make_loaders,
)
from galgenai.models import VAE, VAEEncoder, LCFM
from galgenai.training import (
    DLCFMTrainer,
    VAETrainer,
    load_vae_training_config,
    load_dlcfm_training_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Train VAE + DL-CFM on the COSMOS HuggingFace dataset"),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
    cfg = load_config()

    print(f"Using device: {device}")

    # --------------------------------------------------------------
    # Load all required config values
    # --------------------------------------------------------------
    try:
        cosmos_cfg = cfg["cosmos"]
        train_cfg = cfg["training"]
        model_cfg = train_cfg["model"]
        vae_cfg = train_cfg["vae"]
        dlcfm_cfg = train_cfg["dlcfm"]
        norm_cfg = cosmos_cfg["normalization"]
        run_name = cfg["run_name"]

        # Dataset config
        dataset_path = cosmos_cfg["hf_dataset_path"]
        train_ratio = cosmos_cfg["train_ratio"]
        val_ratio = cosmos_cfg["val_ratio"]
        num_workers = cosmos_cfg["num_workers"]
        split_seed = cosmos_cfg["split_seed"]

        # Training config
        output_dir = Path(train_cfg["output_dir"]) / run_name
        nx = train_cfg["nx"]
        batch_size = train_cfg["batch_size"]

        # Model config
        in_channels = model_cfg["in_channels"]
        latent_dim = model_cfg["latent_dim"]
        base_channels = model_cfg["base_channels"]

        # VAE config (only non-training params)
        vae_image_norm_type = vae_cfg["norm_type"]

        # DL-CFM config
        condition_cols = dlcfm_cfg["condition_cols"]

    except KeyError as e:
        raise ValueError(
            f"Missing required config value: {e}. "
            "Please ensure all required values are present "
            "in the config file."
        ) from e

    print(f"Run name: {run_name}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy the full config file to output directory
    config_file_path = (
        Path(__file__).parent.parent
        / "src"
        / "galgenai"
        / "galgenai_config.yaml"
    )
    if config_file_path.exists():
        shutil.copy(
            config_file_path,
            output_dir / "galgenai_config.yaml",
        )
        print(f"Copied config to: {output_dir / 'galgenai_config.yaml'}")

    cond_stats_save_path = output_dir / "cond_stats.yaml"
    norm_stats_save_path = output_dir / "norm_stats.yaml"
    condition_dim = len(condition_cols)

    # --------------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------------
    print(f"\nLoading FITS dataset from: {dataset_path}")

    catalog_cols = cosmos_cfg["catalog_columns"]
    mag_cols = catalog_cols["mag_cols"]
    redshift_col = catalog_cols["redshift_col"]

    dataset_raw = load_fits_dataset(
        dataset_path,
        metadata_file="metadata.csv",
        mag_cols=mag_cols,
        redshift_col=redshift_col,
        mag_sentinel=cosmos_cfg.get("mag_sentinel", 999.0),
        redshift_sentinel=cosmos_cfg.get("redshift_sentinel", -99.0),
    )

    n_total = len(dataset_raw)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val
    print(
        f"Dataset sizes: {n_train} train / {n_val} val "
        f"/ {n_test} test (total: {n_total})"
    )

    # --------------------------------------------------------------
    # Normalization
    # --------------------------------------------------------------
    print("\n\nLoading Norm Config\n")

    print(f"\nImage normalisation: {vae_image_norm_type}")
    print("  Loading stats from config file")

    image_norm_fn, norm_stats = get_image_norm_fn(
        img_norm_type=vae_image_norm_type,
        config=norm_cfg["image"],
    )

    save_image_norm_stats(norm_stats, norm_stats_save_path)
    print(f"  Image normalization stats saved to: {norm_stats_save_path}")

    # Conditional normalization
    print(f"\nConditioning columns ({condition_dim}): {condition_cols}")
    print("  Loading conditional stats from config file")
    conditional_norm_fn, cond_stats = get_conditional_norm_fn(
        config=norm_cfg["conditions"],
    )

    if condition_cols != cond_stats.cols:
        raise ValueError(
            f"Mismatch between DL-CFM condition_cols "
            f"{condition_cols} and config "
            f"normalization.conditions.cols "
            f"{cond_stats.cols}"
        )

    save_conditional_stats(cond_stats, cond_stats_save_path)
    print(f"  Conditional stats saved to: {cond_stats_save_path}")

    # --------------------------------------------------------------
    # Create data loaders (shared by VAE and DL-CFM)
    # --------------------------------------------------------------
    print("\n\nCreating data loaders:\n")

    train_loader, val_loader, test_loader = make_loaders(
        dataset_raw,
        nx=nx,
        batch_size=batch_size,
        num_workers=num_workers,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        random_seed=split_seed,
        image_norm_fn=image_norm_fn,
        return_aux_data=True,
        condition_cols=condition_cols,
        conditional_norm_fn=conditional_norm_fn,
    )
    print(f"Crop size  : {nx}x{nx} px")
    n_train_batches = (n_train + batch_size - 1) // batch_size
    n_val_batches = (n_val + batch_size - 1) // batch_size
    print(f"Batches    : {n_train_batches} train / {n_val_batches} val")
    if test_loader is not None:
        n_test_batches = (n_test + batch_size - 1) // batch_size
        print(f"           : {n_test_batches} test")

    # --------------------------------------------------------------
    # Stage 1: VAE
    # --------------------------------------------------------------
    encoder_path = output_dir / "encoder.pt"

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

        vae_config = load_vae_training_config()

        vae_trainer = VAETrainer(
            model=vae,
            train_loader=train_loader,
            config=vae_config,
            val_loader=val_loader,
        )
        vae_trainer.train()

        # Save encoder
        torch.save(
            vae_trainer.model.encoder.state_dict(),
            encoder_path,
        )
        print(f"Encoder saved to: {encoder_path}")

        del vae, vae_trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    else:
        print("\n" + "=" * 60)
        print("STAGE 1: SKIPPED — loading VAE from checkpoint")
        print("=" * 60)

        vae_ckpt_path = output_dir / "vae" / "checkpoints" / "best.pt"
        vae = VAE(
            in_channels=in_channels,
            latent_dim=latent_dim,
            input_size=nx,
        )
        checkpoint = torch.load(
            vae_ckpt_path,
            map_location=device,
            weights_only=False,
        )
        vae.load_state_dict(checkpoint["model_state_dict"])
        torch.save(vae.encoder.state_dict(), encoder_path)
        print(f"Encoder saved to: {encoder_path}")

        del vae, checkpoint
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --------------------------------------------------------------
    # Stage 2: DL-CFM
    # --------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STAGE 2: TRAINING DL-CFM")
    print("=" * 60)

    # Load frozen encoder
    encoder = VAEEncoder(
        in_channels=in_channels,
        latent_dim=latent_dim,
        input_size=nx,
    )
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    encoder.to(device).eval()
    print(f"Loaded encoder from: {encoder_path}")

    # Build LCFM model with beta=0 (KL handled by DL-CFM loss)
    lcfm = LCFM(
        vae_encoder=encoder,
        latent_dim=latent_dim,
        in_channels=in_channels,
        input_size=nx,
        base_channels=base_channels,
        beta=0.0,
    ).to(device)
    print(f"LCFM parameters: {sum(p.numel() for p in lcfm.parameters()):,}")
    print(
        "  (encoder backbone frozen; fc_mu/fc_logvar and flow network trained)"
    )

    # Configure and run DL-CFM training
    dlcfm_config = load_dlcfm_training_config()

    dlcfm_trainer = DLCFMTrainer(
        model=lcfm,
        train_loader=train_loader,
        config=dlcfm_config,
        val_loader=val_loader,
    )
    dlcfm_trainer.train()

    print("DL-CFM training complete!")
    print(f"Checkpoints saved to: {output_dir / 'dlcfm' / 'checkpoints'}")

    # --------------------------------------------------------------
    # Summary
    # --------------------------------------------------------------
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)

    vae_ckpt = output_dir / "vae" / "checkpoints"
    dlcfm_ckpt = output_dir / "dlcfm" / "checkpoints"

    print(f"""
Output layout:
  Config file        : {output_dir / "galgenai_config.yaml"}
  Image normalisation: {vae_image_norm_type}
  Normalization stats: {norm_stats_save_path}
  Conditional stats  : {cond_stats_save_path}
  Frozen encoder     : {encoder_path}
  VAE checkpoints    : {vae_ckpt}
  DL-CFM checkpoints : {dlcfm_ckpt}
  DL-CFM samples     : {output_dir / "dlcfm" / "samples"}
""")


if __name__ == "__main__":
    main()
