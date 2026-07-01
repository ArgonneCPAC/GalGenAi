"""
Plot the learning-rate schedule specified by a training config.

For each training section present in the config (``vae``, ``cfm``,
``cnf``), reproduces the exact LR trajectory the corresponding trainer
would apply, driving the same scheduler objects/builders the trainers
use: ``build_cfm_scheduler`` (OneCycleLR) for CFM, and a linear
warmup + ``CosineAnnealingLR`` for CNF.

Run with:
# Plot every training section found in the config:
uv run python scripts/plot_lr_schedule.py \
        --config-path ./galgenai_config.yaml

# Plot a single model's schedule and save elsewhere:
uv run python scripts/plot_lr_schedule.py \
        --config-path ./galgenai_config.yaml \
        --model cfm --output ./lr_schedule.png
"""

import argparse

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from galgenai.config import load_config
from galgenai.training.cfm_trainer import (
    build_cfm_scheduler,
    cfm_scheduler_total_steps,
)
from galgenai.training.config import (
    load_cfm_training_config,
    load_cnf_training_config,
    load_vae_training_config,
)

_STEP_BASED_LOADERS = {
    "cfm": load_cfm_training_config,
    "cnf": load_cnf_training_config,
}


def _cfm_schedule(config) -> tuple[list[int], list[float]]:
    """Replay the OneCycleLR schedule built by build_cfm_scheduler(),
    holding flat at the final LR past ``lr_converge_at_step`` (if
    set) for the remainder of ``num_steps``."""
    dummy_param = torch.nn.Parameter(torch.zeros(1))
    dummy_param.grad = torch.zeros_like(dummy_param)
    optimizer = torch.optim.AdamW([dummy_param], lr=config.learning_rate)
    scheduler = build_cfm_scheduler(optimizer, config)
    total_steps = cfm_scheduler_total_steps(config)

    steps, lrs = [], []
    for global_step in range(config.num_steps):
        steps.append(global_step)
        lrs.append(scheduler.get_last_lr()[0])
        optimizer.step()
        if global_step < total_steps:
            scheduler.step()

    return steps, lrs


def _cnf_schedule(config) -> tuple[list[int], list[float]]:
    """Replay the warmup + cosine schedule used by the CNF trainer."""
    dummy_param = torch.nn.Parameter(torch.zeros(1))
    dummy_param.grad = torch.zeros_like(dummy_param)
    optimizer = torch.optim.AdamW([dummy_param], lr=config.learning_rate)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.num_steps - config.warmup_steps,
        eta_min=config.learning_rate * config.lr_min_factor,
    )

    steps, lrs = [], []
    global_step = 0
    while global_step < config.num_steps:
        if global_step < config.warmup_steps:
            current_lr = config.learning_rate * (
                global_step / config.warmup_steps
            )
        else:
            current_lr = scheduler.get_last_lr()[0]

        steps.append(global_step)
        lrs.append(current_lr)

        optimizer.step()
        global_step += 1
        if global_step >= config.warmup_steps:
            scheduler.step()

    return steps, lrs


_STEP_BASED_SCHEDULES = {
    "cfm": _cfm_schedule,
    "cnf": _cnf_schedule,
}


def _epoch_based_schedule(config) -> tuple[list[int], list[float]]:
    """Replay the per-epoch cosine schedule used by the VAE trainer."""
    dummy_param = torch.nn.Parameter(torch.zeros(1))
    dummy_param.grad = torch.zeros_like(dummy_param)
    optimizer = torch.optim.Adam([dummy_param], lr=config.learning_rate)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.num_epochs,
        eta_min=config.learning_rate * config.lr_min_factor,
    )

    epochs, lrs = [], []
    for epoch in range(1, config.num_epochs + 1):
        epochs.append(epoch)
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

    return epochs, lrs


def build_schedules(
    config_path: str, models: list[str]
) -> dict[str, tuple[list[int], list[float]]]:
    """Build {model_name: (x, lr)} for each requested/present model."""
    raw_config = load_config(config_path)
    available = set(raw_config.get("training", {}).keys())

    if models:
        missing = set(models) - available
        if missing:
            raise ValueError(
                f"Model(s) {sorted(missing)} not found in "
                f"config's training section (found: {sorted(available)})"
            )
        selected = models
    else:
        selected = sorted(available & {"vae", "cfm", "cnf"})
        if not selected:
            raise ValueError("No vae/cfm/cnf training section found in config")

    schedules = {}
    for model in selected:
        if model == "vae":
            config = load_vae_training_config(config_path)
            schedules["vae"] = _epoch_based_schedule(config)
        else:
            config = _STEP_BASED_LOADERS[model](config_path)
            schedules[model] = _STEP_BASED_SCHEDULES[model](config)
    return schedules


def plot_schedules(
    schedules: dict[str, tuple[list[int], list[float]]],
    output_path: str,
):
    """Plot and save the LR-vs-step (or epoch) curves."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for model, (x, lrs) in schedules.items():
        x_label = "epoch" if model == "vae" else "step"
        ax.plot(x, lrs, label=f"{model} (lr vs {x_label})", lw=1.5)

    ax.set_xlabel("step / epoch")
    ax.set_ylabel("learning rate")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved LR schedule plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description=("Plot the LR schedule(s) defined by a training config."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Path to the YAML config file (default: packaged config)",
    )
    parser.add_argument(
        "--model",
        type=str,
        nargs="+",
        choices=["vae", "cfm", "cnf"],
        default=None,
        help=(
            "Which training section(s) to plot. Defaults to every "
            "vae/cfm/cnf section present in the config."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./lr_schedule.png",
        help="Path to save the resulting plot",
    )
    args = parser.parse_args()

    schedules = build_schedules(args.config_path, args.model)
    plot_schedules(schedules, args.output)


if __name__ == "__main__":
    main()
