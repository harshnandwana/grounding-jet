"""Plot a photo-free comparison of the two published held-out test runs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
OUT = RESULTS / "40k_vs_120k_test.png"
BLUE = "#6A7D95"
TEAL = "#159E92"
INK = "#18263A"
MUTED = "#607188"
BG = "#F3F6FA"


def main() -> None:
    first = json.loads((RESULTS / "budget40k_metrics.json").read_text())
    second = json.loads((RESULTS / "budget120k_metrics.json").read_text())
    cost = json.loads((RESULTS / "budget120k_cost.json").read_text())
    assert first["train_records_unique"] == 40_000
    assert second["train_records_unique"] == 120_000
    assert first["dataset_revision"] == second["dataset_revision"]
    assert first["test_records"] == second["test_records"] == 1_000
    tasks = ["box_choice", "spatial_boolean", "attribute_text", "relation_text"]
    labels = ["Box choice", "Spatial decision", "Color attribute", "Relation text"]
    score1 = first["full_test_generation_scores"]
    score2 = second["full_test_generation_scores"]
    assert all(score1[t]["n"] == score2[t]["n"] == 200 for t in ["ground_bbox", *tasks])
    assert all(abs(score1[t]["base_mean"] - score2[t]["base_mean"]) < 1e-10 for t in ["ground_bbox", *tasks])

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig = plt.figure(figsize=(15, 9.2), facecolor=BG)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.15, 5.8], width_ratios=[3.2, 1.3],
                           left=0.155, right=0.96, top=0.91, bottom=0.16, wspace=0.20, hspace=0.18)
    header = fig.add_subplot(grid[0, :])
    header.axis("off")
    header.text(0, 0.92, "VISUAL JEV  /  HELD-OUT TEST", fontsize=24, fontweight="bold", color=INK,
                transform=header.transAxes, va="top")
    header.text(0, 0.45, "Same 1,000 test records  ·  200 per task  ·  two L4 GPUs per run",
                fontsize=13, color=MUTED, transform=header.transAxes, va="top")
    header.text(0.98, 0.75, "40k → 120k training rows", fontsize=15, fontweight="bold",
                color=TEAL, ha="right", transform=header.transAxes)
    header.text(0.98, 0.40,
                f"120k Modal cost: USD {float(cost['metered_cost']):.2f} metered · "
                f"USD {float(cost['net_out_of_pocket_for_this_run_to_date']):.2f} billed after credits",
                fontsize=11, color=MUTED, ha="right", transform=header.transAxes)

    ax = fig.add_subplot(grid[1, 0], facecolor="white")
    y = np.arange(len(tasks))[::-1]
    width = 0.31
    vals1 = [score1[t]["adapter_mean"] * 100 for t in tasks]
    vals2 = [score2[t]["adapter_mean"] * 100 for t in tasks]
    ax.barh(y + width / 2, vals1, height=width, color=BLUE, label="40k adapter")
    ax.barh(y - width / 2, vals2, height=width, color=TEAL, label="120k adapter")
    for pos, v1, v2 in zip(y, vals1, vals2):
        ax.text(v1 + 0.8, pos + width / 2, f"{v1:.1f}%", va="center", color=BLUE, fontsize=11,
                fontweight="bold")
        ax.text(v2 + 0.8, pos - width / 2, f"{v2:.1f}%", va="center", color=TEAL, fontsize=11,
                fontweight="bold")
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 111)
    ax.set_xticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100%"])
    ax.set_title("Answer accuracy  ·  exact match", loc="left", fontsize=15,
                 fontweight="bold", color=INK, pad=20)
    ax.grid(axis="x", color="#DDE5EE")
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False, ncol=2, bbox_to_anchor=(1, 1.04))
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", length=0, pad=10, colors=INK)

    ground = fig.add_subplot(grid[1, 1], facecolor="white")
    ground_vals = [score1["ground_bbox"]["adapter_mean"], score2["ground_bbox"]["adapter_mean"]]
    bars = ground.bar([0, 1], ground_vals, color=[BLUE, TEAL], width=0.56)
    ground.set_ylim(0, 0.75)
    ground.set_xticks([0, 1], ["40k", "120k"])
    ground.set_yticks([0, 0.25, 0.50, 0.75])
    ground.set_title("Grounding  ·  mean IoU", loc="left", fontsize=15,
                     fontweight="bold", color=INK, pad=20)
    ground.grid(axis="y", color="#DDE5EE")
    ground.set_axisbelow(True)
    for bar, value in zip(bars, ground_vals):
        ground.text(bar.get_x() + bar.get_width() / 2, value + 0.017, f"{value:.3f}",
                    ha="center", va="bottom", fontsize=13, color=INK, fontweight="bold")
    for spine in ground.spines.values():
        spine.set_visible(False)
    ground.tick_params(axis="both", length=0, pad=10, colors=INK)

    fig.text(0.155, 0.072,
             "Source: published metrics.json and predictions.jsonl for both adapters. "
             "Grounding uses mean IoU; other tasks use greedy-generation exact match.",
             color=MUTED, fontsize=10)
    fig.text(0.155, 0.041,
             "Selected image-disjoint subset only. Cost: Modal billing report for 120k app; shared storage and later card edits excluded. No photos.",
             color=MUTED, fontsize=10)
    fig.savefig(OUT, dpi=150, facecolor=BG)
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
