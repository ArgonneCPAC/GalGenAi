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
        --model cfm --steps-per-epoch 200 --output ./lr_schedule.png
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

_PER_BATCH_LOADERS = {
    "cfm": load_cfm_training_config,
    "cnf": load_cnf_training_config,
}


def _cfm_schedule(
    config, steps_per_epoch: int
) -> tuple[list[float], list[float]]:
    """Replay the OneCycleLR schedule built by build_cfm_scheduler(),
    holding flat at the final LR past ``lr_converge_at_epoch`` (if
    set) for the remainder of ``num_epochs``."""
    dummy_param = torch.nn.Parameter(torch.zeros(1))
    dummy_param.grad = torch.zeros_like(dummy_param)
    optimizer = torch.optim.AdamW([dummy_param], lr=config.learning_rate)
    scheduler = build_cfm_scheduler(optimizer, config, steps_per_epoch)
    total_steps = cfm_scheduler_total_steps(config, steps_per_epoch)

    epochs, lrs = [], []
    for global_step in range(config.num_epochs * steps_per_epoch):
        epochs.append(global_step / steps_per_epoch)
        lrs.append(scheduler.get_last_lr()[0])
        optimizer.step()
        if global_step < total_steps - 1:
            scheduler.step()

    return epochs, lrs


def _cnf_schedule(
    config, steps_per_epoch: int
) -> tuple[list[float], list[float]]:
    """Replay the warmup + cosine schedule used by the CNF trainer."""
    dummy_param = torch.nn.Parameter(torch.zeros(1))
    dummy_param.grad = torch.zeros_like(dummy_param)
    optimizer = torch.optim.AdamW([dummy_param], lr=config.learning_rate)
    warmup_steps = max(1, round(config.warmup_epochs * steps_per_epoch))
    total_steps = config.num_epochs * steps_per_epoch
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_steps - warmup_steps),
        eta_min=config.learning_rate * config.lr_min_factor,
    )

    epochs, lrs = [], []
    global_step = 0
    while global_step < total_steps:
        if global_step < warmup_steps:
            current_lr = config.learning_rate * (global_step / warmup_steps)
        else:
            current_lr = scheduler.get_last_lr()[0]

        epochs.append(global_step / steps_per_epoch)
        lrs.append(current_lr)

        optimizer.step()
        global_step += 1
        if global_step >= warmup_steps:
            scheduler.step()

    return epochs, lrs


_PER_BATCH_SCHEDULES = {
    "cfm": _cfm_schedule,
    "cnf": _cnf_schedule,
}


def _vae_schedule(config) -> tuple[list[int], list[float]]:
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
    config_path: str, models: list[str], steps_per_epoch: int
) -> dict[str, tuple[list[float], list[float]]]:
    """Build {model_name: (epoch, lr)} for each requested/present model.

    CFM and CNF advance their scheduler once per batch, so their curves
    depend on ``steps_per_epoch`` (i.e. ``len(train_loader)``), which is
    a property of the data rather than the config.
    """
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
            schedules["vae"] = _vae_schedule(config)
        else:
            config = _PER_BATCH_LOADERS[model](config_path)
            schedules[model] = _PER_BATCH_SCHEDULES[model](
                config, steps_per_epoch
            )
    return schedules


def plot_schedules(
    schedules: dict[str, tuple[list[float], list[float]]],
    output_path: str,
):
    """Plot and save the LR-vs-epoch curves."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for model, (x, lrs) in schedules.items():
        ax.plot(x, lrs, label=model, lw=1.5)

    ax.set_xlabel("epoch")
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
        "--steps-per-epoch",
        type=int,
        default=100,
        help=(
            "Batches per epoch (i.e. len(train_loader)). CFM and CNF "
            "step their scheduler once per batch, so their curves "
            "depend on this."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./lr_schedule.png",
        help="Path to save the resulting plot",
    )
    args = parser.parse_args()

    schedules = build_schedules(
        args.config_path, args.model, args.steps_per_epoch
    )
    plot_schedules(schedules, args.output)


if __name__ == "__main__":
    main()
