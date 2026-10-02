"""Validate and publish the complete-data LoRA adapter with benchmark documentation."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1] if (ROOT.parents[1] / "pyproject.toml").is_file() else ROOT
SOURCE = Path(os.environ.get("VISUAL_JEV_FULL_SOURCE", PROJECT_ROOT / "data" / "model_full" / "visual-jev-full-h100-lora"))
DEST = Path(os.environ.get("VISUAL_JEV_FULL_RELEASE", PROJECT_ROOT / "data" / "hf_full_model_release"))
REPO = os.environ.get("VISUAL_JEV_MODEL_REPO", "harshnandwana/visual-jev-full-qwen35-0.8b-lora")
DATASET = os.environ.get("VISUAL_JEV_DATASET_REPO", "harshnandwana/visual-jev-decisions-v1")
MANIFEST = Path(os.environ.get("VISUAL_JEV_FULL_MANIFEST", PROJECT_ROOT / "data" / "hf_release" / "manifest.json"))
TASKS = ["ground_bbox", "box_choice", "spatial_boolean", "attribute_text", "relation_text"]


def make_plot(metrics: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    scores = metrics["full_test_generation_scores"]
    labels = ["Box choice", "Spatial", "Attribute", "Relation"]
    keys = TASKS[1:]
    x = np.arange(4)
    base = [scores[key]["base_mean"] for key in keys]
    tuned = [scores[key]["adapter_mean"] for key in keys]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [3.7, 1]})
    width = 0.36
    axes[0].bar(x - width / 2, base, width, label="Base", color="#aab7c4")
    axes[0].bar(x + width / 2, tuned, width, label="Full-data adapter", color="#177e89")
    axes[0].set_xticks(x, labels, rotation=16, ha="right")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Exact match")
    axes[0].set_title("Full held-out test split")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.2)
    axes[0].set_axisbelow(True)
    for group, positions in ((base, x - width / 2), (tuned, x + width / 2)):
        for at, value in zip(positions, group):
            axes[0].text(at, value + 0.012, f"{value:.2f}", ha="center", fontsize=8)
    grounding = scores["ground_bbox"]
    axes[1].bar([0, 1], [grounding["base_mean"], grounding["adapter_mean"]],
                color=["#aab7c4", "#177e89"], width=0.55)
    axes[1].set_xticks([0, 1], ["Base", "Adapter"])
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Mean IoU")
    axes[1].set_title("Grounding")
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].set_axisbelow(True)
    for at, value in enumerate((grounding["base_mean"], grounding["adapter_mean"])):
        axes[1].text(at, value + 0.012, f"{value:.3f}", ha="center", fontsize=8)
    fig.suptitle("Visual Jev · full-data H100 training", fontsize=14, weight="bold")
    fig.text(0.02, 0.01, f"One epoch on {metrics['train_records_unique']:,} records. Full validation: {metrics['validation_records']:,}; image-disjoint test: {metrics['test_records']:,}.", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    fig.savefig(DEST / "benchmark.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def card(metrics: dict) -> str:
    scores = metrics["full_test_generation_scores"]
    val_base = metrics["baseline_full_validation_nll"]
    val_tuned = metrics["adapter_full_validation_nll"]
    rows = []
    names = {"ground_bbox": "Grounding (mean IoU)", "box_choice": "Box choice (exact match)",
             "spatial_boolean": "Spatial decision (exact match)",
             "attribute_text": "Color attribute (exact match)",
             "relation_text": "Relation (exact match)"}
    for task in TASKS:
        score = scores[task]
        rows.append(f"| {names[task]} | {score['n']:,} | {score['base_mean']:.3f} | {score['adapter_mean']:.3f} |")
    loss_rows = [f"| {task} | {val_base[task]['n']:,} | {val_base[task]['mean_nll']:.3f} | {val_tuned[task]['mean_nll']:.3f} |"
                 for task in TASKS]
    return f"""---
language: en
license: apache-2.0
library_name: peft
base_model: Qwen/Qwen3.5-0.8B-Base
datasets:
  - {DATASET}
pipeline_tag: image-text-to-text
tags:
  - visual-grounding
  - lora
  - qwen3.5
---

# Visual Jev full-data LoRA

This is a PEFT LoRA adapter for [Qwen3.5-0.8B-Base](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base). It was trained for one complete pass over the **entire published training split** of [Visual Jev decisions v1](https://huggingface.co/datasets/{DATASET}) at commit `{metrics['dataset_revision']}`. The model processes an image and a task-specific question, and answers with a choice, short text, or normalized box.

![Full held-out benchmark](benchmark.png)

## Training

- Training rows: {metrics['train_records_unique']:,} unique records, one epoch; {metrics['train_records_processed_including_padding']:,} examples processed including distributed padding.
- Validation: all {metrics['validation_records']:,} rows; test: all {metrics['test_records']:,} rows, disjoint by image.
- Hardware: {metrics['gpu_count']} × {metrics['gpu']}; elapsed {metrics['elapsed_seconds']/3600:.2f} hours.
- LoRA: rank {metrics['lora_rank']}, alpha 32, `q_proj` and `v_proj`; AdamW learning rate {metrics['learning_rate']}; gradient accumulation {metrics['gradient_accumulation_per_gpu']} per GPU.
- Photos resized to fit within 512 × 512 pixels. The prompt and target serialization are implemented in [`full_worker.py`](full_worker.py), and the Modal launch in [`full_modal.py`](full_modal.py).
- The separate `luna_candidates.jsonl` shard was excluded because its generated labels still await human audit.

## Full validation loss

Mean teacher-forced negative log-likelihood per row over every validation example:

| Task | Rows | Base | Adapter |
|---|---:|---:|---:|
{chr(10).join(loss_rows)}

## Full held-out test benchmark

Greedy generation, scored on every test example. Grounding is mean intersection-over-union of the predicted and annotated normalized boxes; all other tasks use case-insensitive exact match after trimming terminal punctuation.

| Task | Rows | Base | Adapter |
|---|---:|---:|---:|
{chr(10).join(rows)}

The exact metrics are in [`metrics.json`](metrics.json); per-example targets and predictions are in [`predictions.jsonl`](predictions.jsonl). These are internal dataset benchmarks, not an external VQA leaderboard.

## Load the adapter

```python
import torch
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
from peft import PeftModel

base_id = "Qwen/Qwen3.5-0.8B-Base"
adapter_id = "{REPO}"
processor = AutoProcessor.from_pretrained(base_id)
base = Qwen3_5ForConditionalGeneration.from_pretrained(base_id, dtype=torch.bfloat16).to("cuda")
model = PeftModel.from_pretrained(base, adapter_id).eval()
# Build the image + question message with the task-specific box and choice format
# in full_worker.py, then call processor.apply_chat_template and model.generate.
```

## Scope and limits

Visual Genome attribute and relation labels can be noisy. The COCO geometric tasks assume the annotated instances are visible and do not infer object absence. This model has no validated OCR, exhaustive counting, calibrated UNKNOWN, or out-of-distribution claims. COCO photos remain under their individual image owners' rights; the dataset repository links to the official source and includes per-image license metadata.
"""


def main() -> None:
    if not (SOURCE / "metrics.json").is_file():
        raise FileNotFoundError("download the completed full-run artifacts from Modal first")
    metrics = json.loads((SOURCE / "metrics.json").read_text())
    manifest = json.loads(MANIFEST.read_text())
    for split, key in (("train", "train_records_unique"), ("validation", "validation_records"), ("test", "test_records")):
        if metrics[key] != manifest["splits"][split]["records"]:
            raise ValueError(f"incomplete {split} use: {metrics[key]}")
    for task in TASKS:
        expected = manifest["splits"]["validation"]["tasks"][task]
        actual = metrics["adapter_full_validation_nll"][task]["n"]
        if actual != expected:
            raise ValueError(f"incomplete validation for {task}: {actual}/{expected}")
        expected_test = manifest["splits"]["test"]["tasks"][task]
        actual_test = metrics["full_test_generation_scores"][task]["n"]
        if actual_test != expected_test:
            raise ValueError(f"incomplete test for {task}: {actual_test}/{expected_test}")
    if sum(1 for _ in (SOURCE / "predictions.jsonl").open()) != metrics["test_records"]:
        raise ValueError("incomplete prediction log")
    DEST.mkdir(parents=True, exist_ok=True)
    for name in ("adapter_model.safetensors", "adapter_config.json"):
        shutil.copy2(SOURCE / "adapter" / name, DEST / name)
    shutil.copytree(SOURCE / "processor", DEST / "processor", dirs_exist_ok=True)
    for name in ("metrics.json", "predictions.jsonl"):
        shutil.copy2(SOURCE / name, DEST / name)
    for name in ("full_worker.py", "full_modal.py"):
        shutil.copy2(ROOT / name, DEST / name)
    make_plot(metrics)
    (DEST / "README.md").write_text(card(metrics), encoding="utf-8")
    print(f"Prepared {DEST}")
    token = os.environ.get("HF_TOKEN")
    if token:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        print(api.create_repo(REPO, repo_type="model", exist_ok=True, private=False))
        print(api.upload_folder(folder_path=DEST, repo_id=REPO, repo_type="model",
                                commit_message="Release full-data Visual Jev LoRA and complete benchmark"))


if __name__ == "__main__":
    main()
