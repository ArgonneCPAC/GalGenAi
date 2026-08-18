"""
CFM Training Pipeline for the HSC MultiModalUniverse dataset.

Trains on real HSC images (PDR3 Deep/UltraDeep, i < 22.5) rather than
the simulated COSMOS FITS cutouts used by ``train_cfm_cosmos.py``. The
two scripts share everything downstream of dataset loading.

Run with:
    uv run python scripts/train_cfm_hsc_mmu.py
    uv run python scripts/train_cfm_hsc_mmu.py \
        --config path/to/config.yaml
"""

import argparse
from pathlib import Path

from galgenai import get_device
from galgenai.config import (
    REPO_ROOT,
    copy_config_to_results,
    load_config,
)
from galgenai.data.cosmos_dataset import make_loaders
from galgenai.data.hsc import load_hsc_mmu_dataset
from galgenai.data.normalization import (
    get_conditional_norm_fn,
    get_image_norm_fn,
    save_conditional_stats,
    save_image_norm_stats,
)
from galgenai.models import CFM
from galgenai.training import CFMTrainer, load_cfm_training_config

DEFAULT_CONFIG = REPO_ROOT / "configs" / "hsc_mmu_config.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train CFM on the HSC MultiModalUniverse dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG),
        help="Training config file",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    cfg = load_config(args.config)

    data_cfg = cfg["data"]
    hsc_cfg = cfg["datasets"]["hsc_mmu"]
    cfm_model_cfg = cfg["models"]["cfm"]
    cfm_train_cfg = cfg["training"]["cfm"]

    # Real HSC images have no noiseless counterpart. Fail loudly here
    # rather than inside the trainer, where the missing tensor surfaces
    # as an opaque AttributeError on None.
    if cfm_train_cfg["train_on_noiseless"]:
        raise ValueError(
            "training.cfm.train_on_noiseless must be false for HSC MMU: "
            "the dataset contains only observed images, with no noiseless "
            "field to train against."
        )

    results_dir = Path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    copied_config = copy_config_to_results(args.config, results_dir)

    nx = data_cfg["image_size"]
    batch_size = data_cfg["batch_size"]
    in_channels = data_cfg["in_channels"]
    condition_cols = data_cfg["condition_cols"]
    image_norm_type = data_cfg["image_norm_type"]
    condition_dim = len(condition_cols)

    print(f"Using device: {device}")
    print(f"Results directory: {results_dir}")
    print(f"Copied config to: {copied_config}")

    dataset_path = hsc_cfg["path"]
    print()
    dataset_raw = load_hsc_mmu_dataset(
        dataset_path,
        split=hsc_cfg.get("split", "train"),
        condition_cols=condition_cols,
    )

    n_total = len(dataset_raw)
    train_ratio = hsc_cfg["train_ratio"]
    val_ratio = hsc_cfg["val_ratio"]
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = int(n_total * (1 - train_ratio - val_ratio))
    print(
        f"Dataset sizes: {n_train} train / {n_val} val / {n_test} test "
        f"(total: {n_total})"
    )

    norm_cfg = hsc_cfg["normalization"]
    print(f"\nImage normalization: {image_norm_type}")
    image_norm_fn, _, norm_stats = get_image_norm_fn(
        img_norm_type=image_norm_type,
        config=norm_cfg["image"],
        return_denorm=True,
    )
    norm_stats_path = results_dir / "norm_stats.yaml"
    save_image_norm_stats(norm_stats, norm_stats_path)
    print(f"Image normalization stats saved to: {norm_stats_path}")

    print(f"\nConditioning columns ({condition_dim}): {condition_cols}")
    conditional_norm_fn, conditional_denorm_fn, cond_stats = (
        get_conditional_norm_fn(
            config=norm_cfg["conditions"],
            return_denorm=True,
        )
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
        num_workers=hsc_cfg["num_workers"],
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        random_seed=hsc_cfg["split_seed"],
        image_norm_fn=image_norm_fn,
        return_aux_data=True,
        condition_cols=condition_cols,
        conditional_norm_fn=conditional_norm_fn,
        invert_mask=hsc_cfg.get("invert_mask", False),
        augment_train=hsc_cfg.get("augment_train", False),
        return_noiseless_flux=False,
    )
    print(f"Crop size: {nx}x{nx} px")
    print(f"Batches: {len(train_loader)} train / {len(val_loader)} val")
    if test_loader is not None:
        print(f"         {len(test_loader)} test")

    print("\n" + "=" * 60)
    print("TRAINING CFM")
    print("=" * 60)

    cfm = CFM(
        cond_vec_dim=condition_dim,
        in_channels=in_channels,
        input_size=nx,
        base_channels=cfm_model_cfg["base_channels"],
    ).to(device)
    print(f"CFM parameters: {sum(p.numel() for p in cfm.parameters()):,}")

    cfm_config = load_cfm_training_config(args.config)
    cfm_trainer = CFMTrainer(
        model=cfm,
        train_loader=train_loader,
        config=cfm_config,
        val_loader=val_loader,
        condition_labels=condition_cols,
        condition_denorm_fn=conditional_denorm_fn,
    )
    cfm_trainer.train()

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"""
Output layout:
  Config file        : {copied_config}
  Normalization stats: {norm_stats_path}
  Conditional stats  : {cond_stats_path}
  CFM checkpoints    : {results_dir / "cfm" / "checkpoints"}
  CFM samples        : {results_dir / "cfm" / "samples"}
  CFM loss plot      : {results_dir / "cfm" / "loss_history.png"}
""")


if __name__ == "__main__":
    main()
