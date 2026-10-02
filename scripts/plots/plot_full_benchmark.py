"""Render the held-out L40S benchmark as a shareable PNG."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "data" / "model_full" / "visual-jev-v1-l40s-lora" / "metrics.json"
OUTPUT = ROOT / "data" / "model_full" / "visual-jev-v1-l40s-lora" / "benchmark.png"


def main():
    metrics = json.loads(METRICS.read_text())
    scores = metrics["test_generation_scores"]
    names = ["Box choice", "Spatial decision", "Color attribute", "Relation"]
    keys = ["box_choice", "spatial_boolean", "attribute_text", "relation_text"]
    base = [scores[key]["base_mean"] for key in keys]
    tuned = [scores[key]["adapter_mean"] for key in keys]
    x = np.arange(len(keys))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), gridspec_kw={"width_ratios": [3.7, 1]})
    width = 0.37
    axes[0].bar(x - width / 2, base, width, label="Base", color="#aab7c4")
    axes[0].bar(x + width / 2, tuned, width, label="Trained adapter", color="#177e89")
    axes[0].set_xticks(x, names, rotation=18, ha="right")
    axes[0].set_ylim(0, 1.08)
    axes[0].set_ylabel("Exact match")
    axes[0].set_title("Held-out answers · 60 examples per task")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.2)
    axes[0].set_axisbelow(True)
    for group, positions in ((base, x - width / 2), (tuned, x + width / 2)):
        for pos, value in zip(positions, group):
            axes[0].text(pos, value + 0.02, f"{value:.2f}", ha="center", fontsize=8)
    grounding = scores["ground_bbox"]
    axes[1].bar([0, 1], [grounding["base_mean"], grounding["adapter_mean"]],
                color=["#aab7c4", "#177e89"], width=0.55)
    axes[1].set_xticks([0, 1], ["Base", "Adapter"])
    axes[1].set_ylim(0, 1.08)
    axes[1].set_ylabel("Mean IoU")
    axes[1].set_title("Grounding · 60 examples")
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].set_axisbelow(True)
    for at, value in enumerate((grounding["base_mean"], grounding["adapter_mean"])):
        axes[1].text(at, value + 0.02, f"{value:.3f}", ha="center", fontsize=9)
    fig.suptitle("Visual Jev v1 · Qwen3.5-0.8B LoRA on L40S", fontsize=14, weight="bold")
    fig.text(0.02, 0.01, "Sampled training: 2,500 records across five tasks. Image-disjoint test: 300 records. Exact match uses greedy generation.", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    fig.savefig(OUTPUT, dpi=170, bbox_inches="tight")
    print(OUTPUT)


if __name__ == "__main__":
    main()
