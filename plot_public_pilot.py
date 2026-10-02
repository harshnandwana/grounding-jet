"""Render pilot metrics without embedding third-party COCO photos."""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def main() -> None:
    train = json.loads((DATA / "train_smoke_metrics.json").read_text())
    evaluation = json.loads((DATA / "choice_eval.json").read_text())
    by_task = evaluation["by_task"]
    labels = ["Box choice", "Spatial decision"]
    tasks = [by_task["box_choice"], by_task["spatial_boolean"]]
    base = [task["base_correct"] / task["n"] for task in tasks]
    adapter = [task["adapter_correct"] / task["n"] for task in tasks]
    x = np.arange(2)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(range(1, len(train["train_losses"]) + 1), train["train_losses"],
                 color="#4068a8", alpha=0.5, linewidth=1)
    smooth = np.convolve(train["train_losses"], np.ones(5) / 5, mode="valid")
    axes[0].plot(range(5, len(train["train_losses"]) + 1), smooth,
                 color="#154173", linewidth=2, label="5-step mean")
    axes[0].set(title="Training loss", xlabel="Optimizer step", ylabel="Loss")
    axes[0].grid(alpha=0.2)
    axes[0].legend(frameon=False)

    width = 0.35
    axes[1].bar(x - width / 2, base, width, label="Base", color="#9cacbd")
    axes[1].bar(x + width / 2, adapter, width, label="Adapter", color="#168f80")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylim(0, 1)
    axes[1].set(title="Held-out pilot choice accuracy", ylabel="Accuracy")
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].legend(frameon=False)
    fig.suptitle("Visual Jev pilot · 60 training steps on NVIDIA L4", weight="bold")
    fig.text(0.01, 0.01, "97 validation records; pilot evidence only.", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(DATA / "pilot_metrics.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
