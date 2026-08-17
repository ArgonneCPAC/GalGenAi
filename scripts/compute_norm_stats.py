"""
Compute and save normalization statistics.

For galaxy datasets (COSMOS or HSC MMU).

This script:
1. Reads dataset path, crop size (nx), and condition columns from
   config file
2. Loads the entire dataset (COSMOS FITS or HSC MMU Arrow format)
3. Computes image normalization statistics using min-max normalization:
   - Linear: Direct min-max normalization on raw flux
   - Arcsinh: Arcsinh stretch (per-band scale) + min-max normalization
4. Computes conditional normalization statistics (min-max)
5. Saves all stats as YAML files (rounded to 2 decimal places)
6. Optionally updates the config file with the computed statistics

By default uses a subset of 2000 objects to compute the statistics.
--n-samples can be used to modify this.

The script reads these parameters from the config file:
- datasets.<name>.path: Path to dataset directory
- data.image_size: Crop size for images
- data.condition_cols: List of condition columns

Run with:
# Minimal example:
uv run python scripts/compute_norm_stats.py --dataset-type cosmos

# COSMOS dataset:
uv run python scripts/compute_norm_stats.py --dataset-type cosmos \
        --n-samples 2000 --config-path ./my_config.yaml

# HSC MMU dataset:
uv run python scripts/compute_norm_stats.py --dataset-type hsc_mmu \
        --config-path configs/hsc_mmu_config.yaml
"""

import argparse
from pathlib import Path
import yaml

from galgenai.config import load_config
from galgenai.data.cosmos_dataset import load_fits_dataset
from galgenai.data.hsc import HSCDataset, load_hsc_mmu_dataset
from galgenai.data.normalization import (
    compute_arcsinh_norm_stats,
    compute_linear_norm_stats,
    compute_conditional_stats,
    save_image_norm_stats,
    save_conditional_stats,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute normalization statistics for galaxy datasets. "
        "All dataset parameters are read from config file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--dataset-type",
        type=str,
        choices=["cosmos", "hsc_mmu"],
        required=True,
        help="Dataset type: cosmos or hsc_mmu",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=2000,
        help="Number of samples for stats (default: entire dataset)",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Path to config file (default: galgenai_config.yaml)",
    )
    parser.add_argument(
        "--write-to-config",
        action="store_true",
        default=True,
        help="Write stats to config file (default: True)",
    )
    parser.add_argument(
        "--skip-conditions",
        action="store_true",
        default=False,
        help="Skip computing conditional stats",
    )

    return parser.parse_args()


def update_config_file(config_path: Path, stats_dict: dict, dataset_type: str):
    """Update the config file with computed normalization statistics."""
    # Load existing config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if "datasets" not in config:
        config["datasets"] = {}
    if dataset_type not in config["datasets"]:
        config["datasets"][dataset_type] = {}

    # Merge into the existing normalization sub-tree so that keys we did
    # not recompute (e.g. an existing conditions block when running with
    # --skip-conditions) are preserved rather than wiped.
    existing = config["datasets"][dataset_type].setdefault("normalization", {})
    existing.update(stats_dict)

    # Write back to file
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"\n✓ Config file updated: {config_path}")


def main():
    args = parse_args()

    # Load config file first
    cfg = load_config(args.config_path)

    if not cfg:
        raise ValueError(
            "Config file is empty or not found. "
            "Please provide a valid config file path."
        )

    # Extract parameters from config
    dataset_cfg = cfg.get("datasets", {}).get(args.dataset_type, {})
    data_cfg = cfg.get("data", {})

    data_dir = dataset_cfg.get("path")
    if not data_dir:
        raise ValueError(
            f"datasets.{args.dataset_type}.path not found in config file"
        )

    nx = data_cfg.get("image_size", 32)
    condition_cols = data_cfg.get("condition_cols", [])

    # Create output directory
    output_dir = Path("./normalization_stats")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    ds_type = args.dataset_type.upper()
    print(f"COMPUTING NORMALIZATION STATISTICS FOR {ds_type}")
    # Continue to next line
    print("=" * 70)

    print(
        f"\nConfig file: {args.config_path if args.config_path else 'default'}"
    )
    print(f"Dataset type: {args.dataset_type}")
    print(f"Dataset path: {data_dir}")
    print(f"Output directory: {output_dir}")
    n_samples = args.n_samples if args.n_samples else "entire dataset"
    print(f"Number of samples: {n_samples}")
    print(f"Crop size: {nx}x{nx}")
    print(f"Condition columns: {condition_cols}")
    print(f"Skip conditions: {args.skip_conditions}")

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Loading dataset...")
    print("-" * 70)

    if args.dataset_type == "hsc_mmu":
        dataset_raw = load_hsc_mmu_dataset(
            data_dir,
            split=dataset_cfg.get("split", "train"),
            condition_cols=condition_cols,
        )
    else:
        print(f"Loading COSMOS FITS dataset from: {data_dir}")
        dataset_raw = load_fits_dataset(
            data_dir=data_dir,
            metadata_file="metadata.csv",
        )

    print(f"Dataset size: {len(dataset_raw)}")

    # ------------------------------------------------------------------
    # Compute image normalization stats
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Computing image normalization statistics...")
    print("-" * 70)

    # Wrap dataset with HSCDataset (no splitting, entire dataset)
    print("\nWrapping dataset for image normalization...")
    wrapped_dataset = HSCDataset(
        dataset_raw,
        nx=nx,
        image_norm_fn=None,  # No normalization
        return_aux_data=False,  # Only need flux
    )
    print(f"Wrapped dataset size: {len(wrapped_dataset)}")

    # Linear normalization
    print("\n[1/3] Computing linear normalization stats...")
    linear_stats = compute_linear_norm_stats(
        wrapped_dataset,
        n_samples=args.n_samples,
    )

    linear_stats_path = output_dir / "linear_norm_stats.yaml"
    save_image_norm_stats(linear_stats, linear_stats_path)

    print(f"  Min: {[round(x, 2) for x in linear_stats.min.tolist()]}")
    print(f"  Max: {[round(x, 2) for x in linear_stats.max.tolist()]}")
    print(f"  Saved to: {linear_stats_path}")

    # Arcsinh normalization
    print("\n[2/3] Computing arcsinh normalization stats...")
    arcsinh_stats = compute_arcsinh_norm_stats(
        wrapped_dataset, n_samples=args.n_samples, scale_quantile=0.98
    )

    arcsinh_stats_path = output_dir / "arcsinh_norm_stats.yaml"
    save_image_norm_stats(arcsinh_stats, arcsinh_stats_path)

    print(f"  Min: {[round(x, 2) for x in arcsinh_stats.min.tolist()]}")
    print(f"  Max: {[round(x, 2) for x in arcsinh_stats.max.tolist()]}")
    print(f"  Scale: {[round(x, 2) for x in arcsinh_stats.scale.tolist()]}")
    print(f"  Saved to: {arcsinh_stats_path}")

    # ------------------------------------------------------------------
    # Compute conditional normalization stats
    # ------------------------------------------------------------------
    print("\n[3/3] Computing conditional normalization stats...")

    if args.skip_conditions:
        print("  Skipping conditional stats (--skip-conditions flag set).")
        cond_stats = None
        cond_stats_path = None
    elif condition_cols:
        cond_stats = compute_conditional_stats(
            dataset_raw,
            cols=condition_cols,
            n_samples=args.n_samples,
        )

        cond_stats_path = output_dir / "conditional_stats.yaml"
        save_conditional_stats(cond_stats, cond_stats_path)

        print(f"  Columns: {cond_stats.cols}")
        print(f"  Min: {[round(x, 2) for x in cond_stats.min.tolist()]}")
        print(f"  Max: {[round(x, 2) for x in cond_stats.max.tolist()]}")
        print(f"  Saved to: {cond_stats_path}")
    else:
        print("  No condition columns specified, skipping conditional stats.")
        cond_stats = None
        cond_stats_path = None

    # ------------------------------------------------------------------
    # Write to config file if requested
    # ------------------------------------------------------------------
    if args.write_to_config:
        print("\n" + "-" * 70)
        print("Updating config file...")
        print("-" * 70)

        # Determine config path
        if args.config_path:
            config_path = Path(args.config_path)
        else:
            # Use default path
            from galgenai import __file__ as galgenai_init

            config_path = Path(galgenai_init).parent / "galgenai_config.yaml"

        # Build stats dictionary (values already rounded to 2
        # decimal places)
        stats_dict = {
            "image": {
                "linear": {
                    "min": [round(x, 2) for x in linear_stats.min.tolist()],
                    "max": [round(x, 2) for x in linear_stats.max.tolist()],
                },
                "arcsinh": {
                    "min": [round(x, 2) for x in arcsinh_stats.min.tolist()],
                    "max": [round(x, 2) for x in arcsinh_stats.max.tolist()],
                    "scale": [
                        round(x, 2) for x in arcsinh_stats.scale.tolist()
                    ],
                },
            },
        }

        # Only set the conditions block when we actually computed it, so
        # a skipped run does not clobber an existing populated block.
        if cond_stats:
            stats_dict["conditions"] = {
                "cols": cond_stats.cols,
                "min": [round(x, 2) for x in cond_stats.min.tolist()],
                "max": [round(x, 2) for x in cond_stats.max.tolist()],
            }

        update_config_file(config_path, stats_dict, args.dataset_type)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)

    print(f"""
Statistics saved to: {output_dir}

Files created:
  - {linear_stats_path.name}
  - {arcsinh_stats_path.name}
""")

    if cond_stats:
        print(f"  - {cond_stats_path.name}")

    if args.write_to_config:
        print("\nStatistics written to config file:")
        print(f"  datasets.{args.dataset_type}.normalization.image.linear")
        print(f"  datasets.{args.dataset_type}.normalization.image.arcsinh")
        if cond_stats:
            print(f"  datasets.{args.dataset_type}.normalization.conditions")


if __name__ == "__main__":
    main()
