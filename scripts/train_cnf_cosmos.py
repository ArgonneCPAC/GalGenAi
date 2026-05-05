"""
VAE + CNF Training Pipeline for COSMOS HuggingFace Dataset

Trains a two-stage generative model on simulated galaxy images produced
by generate_hf_dataset.py (or generate_fits_dataset.py):

  Stage 1  Train a VAE on COSMOS multi-band images.
  Stage 2  Freeze the VAE encoder, precompute latent codes from the
           training and validation sets, then train a Conditional
           Normalizing Flow (CNF). 

sampling:

    condition = normalize_mags(magnitudes, mag_stats)
    z         = cnf.sample(condition)          # shape (1, latent_dim)
    image     = vae_decoder(z)                 # shape (5, nx, nx)
    raw_image = arcsinh_denorm(image, norm_stats)

Run with:
    uv run python scripts/train_cnf_cosmos.py --run-name my_run
    uv run python scripts/train_cnf_cosmos.py --run-name my_run --skip-vae
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
    precompute_latents,
)
from galgenai.models import VAE, ConditionalNormalizingFlow
from galgenai.training import (
    CNFTrainer,
    VAETrainer,
    load_vae_training_config,
    load_cnf_training_config,
)

# CUDA-wide perf knobs: TF32 matmuls + cuDNN autotune for static shapes.
# No-ops on non-CUDA backends. Must be set before the first CUDA op.
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train VAE + CNF on the COSMOS HuggingFace dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--skip-vae",
        action="store_true",
        help="Skip VAE training and load from existing checkpoint"
    )

    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    cfg = load_config()

    print(f"Using device: {device}")
    

    # ------------------------------------------------------------------
    # Load all required config values
    # ------------------------------------------------------------------
    try:
        # Top-level sections
        cosmos_cfg = cfg["cosmos"]
        train_cfg = cfg["training"]
        model_cfg = train_cfg["model"]
        vae_cfg = train_cfg["vae"]
        cnf_cfg = train_cfg["cnf"]
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

        # VAE config (only non-training params)
        vae_image_norm_type = vae_cfg["norm_type"]

        # CNF config (only non-training params)
        condition_cols = cnf_cfg["condition_cols"]
        cnf_num_blocks = cnf_cfg["num_blocks"]
        cnf_hidden_dim = cnf_cfg["hidden_dim"]

    except KeyError as e:
        raise ValueError(
            f"Missing required config value: {e}. "
            "Please ensure all required values are present in the config file."
        )
    
    print(f"Run name: {run_name}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy the full config file to output directory
    config_file_path = Path(__file__).parent.parent / "src" / "galgenai" / "galgenai_config.yaml"
    if config_file_path.exists():
        shutil.copy(config_file_path, output_dir / "galgenai_config.yaml")
        print(f"Copied config to: {output_dir / 'galgenai_config.yaml'}")

    cond_stats_save_path = output_dir / "cond_stats.yaml"
    norm_stats_save_path = output_dir / "norm_stats.yaml"
    condition_dim = len(condition_cols)

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    print(f"\nLoading FITS dataset from: {dataset_path}")

    # Extract magnitude columns and redshift column from config
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
    n_test = int(n_total * (1 - train_ratio - val_ratio))
    print(f"Dataset sizes: {n_train} train / {n_val} val / {n_test} test (total: {n_total})")

    # ------------------------------------------------------------------
    # Load normalization configuration
    # ------------------------------------------------------------------
    print("\n\nLoading Norm Config\n")

    # Load image normalization stats from config
    print(f"\nImage normalisation: {vae_image_norm_type}")
    print(f"  Loading stats from config file")

    image_norm_fn, norm_stats = get_image_norm_fn(
        img_norm_type=vae_image_norm_type,
        config=norm_cfg["image"],
    )

    save_image_norm_stats(norm_stats, norm_stats_save_path)
    print(f"  Image normalization stats saved to: {norm_stats_save_path}")

    # ------------------------------------------------------------------
    # Load conditional normalization from config
    # ------------------------------------------------------------------
    print(f"\nConditioning columns ({condition_dim}): {condition_cols}")

    print(f"  Loading conditional stats from config file")
    conditional_norm_fn, cond_stats = get_conditional_norm_fn(
        config=norm_cfg["conditions"],
    )

    # Verify that condition_cols matches the config
    if condition_cols != cond_stats.cols:
        raise ValueError(
            f"Mismatch between CNF condition_cols {condition_cols} and "
            f"config normalization.conditions.cols {cond_stats.cols}"
        )

    # Save conditional stats
    save_conditional_stats(cond_stats, cond_stats_save_path)
    print(f"  Conditional stats saved to: {cond_stats_save_path}")

    # ------------------------------------------------------------------
    # Create data loaders (shared by VAE and CNF)
    # ------------------------------------------------------------------
    print("\n\nCreating data loaders:\n")

    # Create loaders with both image and conditional normalization
    # Returns (flux, ivar, mask, condition) tuples when return_aux_data=True and condition_cols are set
    # VAE will use (flux, ivar, mask), CNF will use (flux, condition)
    train_loader, val_loader, test_loader = make_loaders(
        dataset_raw,
        nx=nx,
        batch_size=batch_size,
        num_workers=num_workers,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        random_seed=split_seed,
        image_norm_fn=image_norm_fn,
        return_aux_data=True,  # Return (flux, ivar, mask, condition)
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

    # ------------------------------------------------------------------
    # Stage 1: VAE
    # ------------------------------------------------------------------
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

        # Configure VAE training (loads from config)
        vae_config = load_vae_training_config()

        vae_trainer = VAETrainer(
            model=vae,
            train_loader=train_loader,
            config=vae_config,
            val_loader=val_loader,
        )
        vae_trainer.train()

        del vae, vae_trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    else:
        print("\n" + "=" * 60)
        print("STAGE 1: SKIPPED — VAE training not required")
        print("=" * 60)

    # ------------------------------------------------------------------
    # Stage 2: CNF
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STAGE 2: TRAINING CONDITIONAL NORMALIZING FLOW")
    print("=" * 60)

    # ---- Load frozen encoder from VAE checkpoint ----------------
    vae_ckpt_path = output_dir / "vae" / "checkpoints" / "best.pt"
    if not vae_ckpt_path.exists():
        raise FileNotFoundError(
            f"No VAE checkpoint found at {vae_ckpt_path}. "
            "Train VAE first (Stage 1)."
        )

    vae = VAE(
        in_channels=in_channels,
        latent_dim=latent_dim,
        input_size=nx,
    )
    checkpoint = torch.load(vae_ckpt_path, map_location=device, weights_only=False)
    vae.load_state_dict(checkpoint["model_state_dict"])

    # Extract encoder and freeze it
    encoder = vae.encoder
    encoder.to(device).eval()
    for param in encoder.parameters():
        param.requires_grad_(False)
    print(f"Loaded frozen encoder from: {vae_ckpt_path}")

    del vae, checkpoint
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- Precompute latents + conditions -------------------------
    print("\nPrecomputing VAE latent codes ...")
    # train_loader and val_loader already have (flux, ivar, mask, condition) tuples
    # precompute_latents will use flux and condition from these loaders

    latent_cache_dir = output_dir / "latent_cache"
    train_cache_path = latent_cache_dir / "train.pt"
    val_cache_path = latent_cache_dir / "val.pt"

    print("  Encoding training set ...")
    cnf_train_loader = precompute_latents(
        encoder, train_loader, device,
        cache_path=train_cache_path,
        return_dataloader=True,
        batch_size=batch_size,
        shuffle=True
    )
    print("  Encoding validation set ...")
    cnf_val_loader = precompute_latents(
        encoder, val_loader, device,
        cache_path=val_cache_path,
        return_dataloader=True,
        batch_size=batch_size,
        shuffle=False
    )

    # Get dataset info from the dataloaders
    n_train = len(cnf_train_loader.dataset)
    n_val = len(cnf_val_loader.dataset)
    latent_dim_actual = cnf_train_loader.dataset.latent_dim
    print(f"  Train latents : {n_train:,} x {latent_dim_actual}")
    print(f"  Val latents   : {n_val:,} x {latent_dim_actual}")

    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- Build CNF model -----------------------------------------
    cnf = ConditionalNormalizingFlow(
        latent_dim=latent_dim_actual,
        condition_dim=condition_dim,
        num_blocks=cnf_num_blocks,
        hidden_dim=cnf_hidden_dim,
    ).to(device)
    print(
        f"\nCNF parameters: "
        f"{sum(p.numel() for p in cnf.parameters()):,}"
    )
    print(f"  latent_dim    : {latent_dim_actual}")
    print(f"  condition_dim : {condition_dim}  {condition_cols}")
    print(f"  num_blocks    : {cnf_num_blocks}")
    print(f"  hidden_dim    : {cnf_hidden_dim}")

    # ---- Train CNF -----------------------------------------------
    # Configure CNF training (loads from config)
    cnf_config = load_cnf_training_config()

    cnf_trainer = CNFTrainer(
        model=cnf,
        train_loader=cnf_train_loader,
        config=cnf_config,
        val_loader=cnf_val_loader,
    )
    cnf_trainer.train()

    print("CNF training complete!")
    print(
        f"Checkpoints saved to: {output_dir / 'cnf' / 'checkpoints'}"
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)

    vae_ckpt = output_dir / "vae" / "checkpoints"
    cnf_ckpt = output_dir / "cnf" / "checkpoints"

    print(f"""
Output layout:
  Config file        : {output_dir / "galgenai_config.yaml"}
  Image normalisation: {vae_image_norm_type}
  Normalization stats: {norm_stats_save_path}
  Conditional stats  : {cond_stats_save_path}
  VAE checkpoints    : {vae_ckpt}
  CNF checkpoints    : {cnf_ckpt}
  CNF samples        : {output_dir / "cnf" / "samples"}
""")


if __name__ == "__main__":
    main()
