"""Configuration loading helpers."""

import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


REPO_ROOT = Path(__file__).parent.parent.parent
DEFAULT_CONFIG_PATH = Path(__file__).with_name("galgenai_config.yaml")


def resolve_config_path(config_path: Optional[str | Path] = None) -> Path:
    """Return the explicit config path, or the packaged default."""
    if config_path is None:
        return DEFAULT_CONFIG_PATH
    return Path(config_path)


def _resolve_path(path_value: str | Path) -> str:
    """Resolve relative config paths from the repository root."""
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str(REPO_ROOT / path)


def load_config(config_path: Optional[str | Path] = None) -> Dict[str, Any]:
    """Load a YAML config file and resolve paths used by training."""
    path = resolve_config_path(config_path)
    if not path.exists():
        if config_path is not None:
            raise FileNotFoundError(f"Config file not found: {path}")
        return {}

    try:
        with open(path, "r") as f:
            config = yaml.safe_load(f) or {}
    except Exception as exc:
        raise ValueError(f"Failed to load config from {path}: {exc}") from exc

    if "results_dir" in config:
        config["results_dir"] = _resolve_path(config["results_dir"])

    for dataset_cfg in config.get("datasets", {}).values():
        if "path" in dataset_cfg:
            dataset_cfg["path"] = _resolve_path(dataset_cfg["path"])

    gen_cfg = config.get("dataset_generation", {})
    for key in ("catalog_path", "output_dir"):
        if key in gen_cfg:
            gen_cfg[key] = _resolve_path(gen_cfg[key])

    return config


def copy_config_to_results(
    config_path: Optional[str | Path], results_dir: str | Path
) -> Path:
    """Copy the config file used for a run into ``results_dir``."""
    source = resolve_config_path(config_path)
    target_dir = Path(results_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "config.yaml"
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)
    return target
