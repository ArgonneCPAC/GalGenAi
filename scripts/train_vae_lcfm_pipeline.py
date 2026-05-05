"""
VAE + LCFM Training Pipeline

Demonstrates the complete workflow for training generative
models on galaxy images:
1. Train a VAE on galaxy data (COSMOS or HSC-MMU)
2. Extract and save the frozen encoder
3. Train an LCFM model using the encoder

Run with:
    uv run python scripts/train_vae_lcfm_pipeline.py --dataset cosmos
    uv run python scripts/train_vae_lcfm_pipeline.py --dataset hsc_mmu
    uv run python scripts/train_vae_lcfm_pipeline.py --dataset hsc_mmu --skip-vae
"""

import argparse
import torch
from pathlib import Path
from datasets import load_from_disk

from galgenai.models import VAE, VAEEncoder, LCFM

from galgenai.training import (
    VAETrainer,
    LCFMTrainer,
    load_vae_training_config,
    load_lcfm_training_config,
)

from galgenai.data.hsc import get_dataset_and_loaders
from galgenai.data.cosmos_dataset import load_fits_dataset, make_loaders
from galgenai.data.normalization import (
    get_image_norm_fn,
    save_image_norm_stats,
)

from galgenai import get_device
from galgenai.config import load_config

# CUDA-wide perf knobs: TF32 matmuls + cuDNN autotune for static shapes.
# No-ops on non-CUDA backends. Must be set before the first CUDA op.
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train VAE + LCFM pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["cosmos", "hsc_mmu"],
        help="Dataset to use: 'cosmos' or 'hsc_mmu'",
    )
    parser.add_argument(
        "--skip-vae",
        action="store_true",
        help="Skip VAE training and load from existing checkpoint",
    )
    return parser.parse_args()


if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_args()

    # Get the best available device
    device = get_device()
    print(f"Using device: {device}")

    # Load configuration
    cfg = load_config()
    print("Loaded configuration from galgenai_config.yaml")
    
    # Select dataset configuration based on argument
    dataset_type = args.dataset
    train_vae = not args.skip_vae
    
    print(f"Dataset type: {dataset_type}")
    
    # ------------------------------------------------------------------
    # Load all required config values
    # ------------------------------------------------------------------
    try:
        # Get dataset-specific configuration
        if dataset_type == "cosmos":
            dataset_cfg = cfg["cosmos"]
            data_path = dataset_cfg["hf_dataset_path"]
            num_workers = dataset_cfg["num_workers"]
            train_ratio = dataset_cfg["train_ratio"]
            val_ratio = dataset_cfg["val_ratio"]
            split_seed = dataset_cfg["split_seed"]
        elif dataset_type == "hsc_mmu":
            dataset_cfg = cfg["hsc_mmu"]
            data_path = dataset_cfg["hf_dataset_path"]
            num_workers = dataset_cfg["num_workers"]
            train_ratio = dataset_cfg["train_ratio"]
            val_ratio = dataset_cfg["val_ratio"]
            split_seed = dataset_cfg["split_seed"]
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
    
        # Top-level sections
        train_cfg = cfg["training"]
        model_cfg = train_cfg["model"]
        vae_cfg = train_cfg["vae"]
        norm_cfg = dataset_cfg["normalization"]

        # Training config
        output_dir = Path(train_cfg["output_dir"]) / "pipeline"
        nx = train_cfg["nx"]
        batch_size = train_cfg["batch_size"]

        # Model config
        in_channels = model_cfg["in_channels"]
        latent_dim = model_cfg["latent_dim"]
        base_channels = model_cfg["base_channels"]

        # VAE config (only non-training params)
        vae_image_norm_type = vae_cfg["norm_type"]

        # LCFM config (model parameter - beta is passed to LCFM constructor)
        lcfm_cfg = train_cfg["lcfm"]
        lcfm_beta = lcfm_cfg["beta"]
    
    except KeyError as e:
        raise ValueError(
            f"Missing required config value: {e}. "
            "Please ensure all required values are present in the config file."
        )
    
    # Create output directory structure
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir.absolute()}")
    
    
    # =================================================================
    # ----  NORMALIZATION SETUP  -------------------------------------
    # =================================================================
    
    print("\n" + "=" * 60)
    print("SETTING UP NORMALIZATION")
    print("=" * 60)
    
    print(f"Image normalisation: {vae_image_norm_type}")
    image_norm_fn, norm_stats = get_image_norm_fn(
        img_norm_type=vae_image_norm_type,
        config=norm_cfg["image"],
    )
    
    # Save normalization stats
    norm_stats_save_path = output_dir / "norm_stats.yaml"
    save_image_norm_stats(norm_stats, norm_stats_save_path)
    print(f"Normalization stats saved to: {norm_stats_save_path}")
    
    
    # =================================================================
    # ----  DATA LOADING  --------------------------------------------
    # =================================================================
    
    print("\n" + "=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    
    if dataset_type == "cosmos":
        catalog_cols = dataset_cfg["catalog_columns"]
        mag_cols = catalog_cols["mag_cols"]
        redshift_col = catalog_cols["redshift_col"]
    
        dataset_raw = load_fits_dataset(
            data_dir=data_path,
            metadata_file="metadata.csv",
            format="torch",
            filter_invalid_mags=True,
            mag_sentinel=dataset_cfg["mag_sentinel"],
            mag_cols=mag_cols,
            filter_invalid_redshift=True,
            redshift_sentinel=dataset_cfg["redshift_sentinel"],
            redshift_col=redshift_col,
        )
        print(f"Loaded COSMOS dataset from: {data_path}")
    
        # Create PyTorch dataset and data loaders for COSMOS
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
        )
    
        n_total = len(dataset_raw)
        n_train = len(train_loader.dataset)
        n_val = len(val_loader.dataset)
        print(f"Dataset sizes: {n_train} train / {n_val} val (total: {n_total})")
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    elif dataset_type == "hsc_mmu":
        # Load HSC-MMU HuggingFace dataset
        dataset_raw = load_from_disk(data_path)
        print(f"Loaded HSC-MMU dataset from: {data_path}")
    
        # Create PyTorch dataset and data loaders for HSC-MMU
        dataset, train_loader, val_loader = get_dataset_and_loaders(
            dataset_raw,
            nx=nx,
            batch_size=batch_size,
            num_workers=num_workers,
            image_norm_fn=image_norm_fn,
        )
        print(f"Dataset size: {len(dataset)} samples")
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    
    # =================================================================
    # ----  VAE TRAINING  ---------------------------------------------
    # =================================================================
    
    encoder_path = output_dir / "encoder.pt"
    
    if train_vae:
        print("\n" + "=" * 60)
        print("TRAINING VAE")
        print("=" * 60)
    
        # Create VAE model
        vae = VAE(
            in_channels=in_channels,
            latent_dim=latent_dim,
            input_size=nx,
        )
        print(f"VAE parameters: {sum(p.numel() for p in vae.parameters()):,}")
    
        # Configure VAE training (loads from config)
        vae_config = load_vae_training_config()
    
        # Create trainer and run training
        vae_trainer = VAETrainer(
            model=vae,
            train_loader=train_loader,
            config=vae_config,
            val_loader=val_loader,
        )
        vae_trainer.train()
    
        print("VAE training complete!")
        print(f"Checkpoints saved to: {output_dir / 'vae' / 'checkpoints'}")
    
        # =============================================================
        # ----  EXTRACT & SAVE ENCODER  -------------------------------
        # =============================================================
    
        print("\n" + "=" * 60)
        print("SAVING FROZEN ENCODER")
        print("=" * 60)
    
        # The VAE has separate encoder and decoder components
        # We save just the encoder for use with LCFM
        torch.save(vae_trainer.model.encoder.state_dict(), encoder_path)
        print(f"Encoder saved to: {encoder_path}")
    
        # Clean up VAE to free memory before LCFM training
        del vae, vae_trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    else:
        print("\n" + "=" * 60)
        print("LOADING VAE FROM CHECKPOINT (train_vae=False)")
        print("=" * 60)
    
        # Load trained VAE weights from best checkpoint
        vae_ckpt_path = output_dir / "vae" / "checkpoints" / "best.pt"
        vae = VAE(
            in_channels=in_channels,
            latent_dim=latent_dim,
            input_size=nx,
        )
        checkpoint = torch.load(vae_ckpt_path, map_location=device)
        vae.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded VAE from: {vae_ckpt_path}")
    
        # Save the encoder for LCFM
        torch.save(vae.encoder.state_dict(), encoder_path)
        print(f"Encoder saved to: {encoder_path}")
    
        # Clean up VAE to free memory before LCFM training
        del vae, checkpoint
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    
    # =================================================================
    # ----  LCFM TRAINING  --------------------------------------------
    # =================================================================
    
    print("\n" + "=" * 60)
    print("TRAINING LCFM")
    print("=" * 60)
    
    # Create a fresh encoder and load the trained weights
    # Must match the VAE encoder architecture
    encoder = VAEEncoder(
        in_channels=in_channels,
        latent_dim=latent_dim,
        input_size=nx,
    )
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    encoder.to(device).eval()
    print(f"Loaded encoder from: {encoder_path}")
    
    # Create LCFM model
    lcfm = LCFM(
        vae_encoder=encoder,
        latent_dim=latent_dim,
        in_channels=in_channels,
        input_size=nx,
        base_channels=base_channels,
        beta=lcfm_beta,
    ).to(device)
    print(f"LCFM parameters: {sum(p.numel() for p in lcfm.parameters()):,}")
    print("  (encoder backbone frozen; fc_mu/fc_logvar and flow network trained)")
    
    # Configure LCFM training (loads from config)
    lcfm_config = load_lcfm_training_config()
    
    # Create trainer and run training
    lcfm_trainer = LCFMTrainer(
        model=lcfm,
        train_loader=train_loader,
        config=lcfm_config,
        val_loader=val_loader,
    )
    lcfm_trainer.train()
    
    print("LCFM training complete!")
    print(f"Checkpoints saved to: {output_dir / 'lcfm' / 'checkpoints'}")
    
    
    # =================================================================
    # ----  SUMMARY  --------------------------------------------------
    # =================================================================
    
    print("\n" + "=" * 60)
    print("TRAINING PIPELINE COMPLETE")
    print("=" * 60)
    vae_ckpt = output_dir / "vae" / "checkpoints"
    lcfm_ckpt = output_dir / "lcfm" / "checkpoints"
    best_model = lcfm_ckpt / "best.pt"
    
    print(f"""
    Output files:
      - VAE checkpoints:  {vae_ckpt}
      - Frozen encoder:   {encoder_path}
      - LCFM checkpoints: {lcfm_ckpt}
      - LCFM samples:     {output_dir / "lcfm" / "samples"}
    
    To load the trained LCFM model:
        encoder = VAEEncoder(in_channels={in_channels}, latent_dim={latent_dim},
                             input_size={nx})
        encoder.load_state_dict(torch.load("{encoder_path}"))
        lcfm = LCFM(encoder, latent_dim={latent_dim}, in_channels={in_channels},
                    input_size={nx}, base_channels={base_channels})
        lcfm.load_state_dict(
            torch.load("{best_model}")["model_state_dict"]
        )
    """)
