"""Validate and release the budget-limited Visual Jev adapter and benchmark."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent.parent if (ROOT.parent.parent / "pyproject.toml").is_file() else ROOT
SOURCE = Path(os.environ.get("VISUAL_JEV_BUDGET_SOURCE", PROJECT_ROOT / "data" / "budget20" / "run"))
MANIFEST = Path(os.environ.get("VISUAL_JEV_BUDGET_MANIFEST", PROJECT_ROOT / "data" / "budget20" / "manifest.json"))
DEST = Path(os.environ.get("VISUAL_JEV_BUDGET_RELEASE", PROJECT_ROOT / "data" / "budget20" / "release"))
REPO = os.environ.get("VISUAL_JEV_MODEL_REPO", "harshnandwana/visual-jev-budget20-qwen35-0.8b-lora")
TASKS = ("ground_bbox", "box_choice", "spatial_boolean", "attribute_text", "relation_text")


def validate(metrics: dict, manifest: dict) -> None:
    if metrics["dataset_revision"] != manifest["dataset_revision"]:
        raise ValueError("model and subset dataset revisions differ")
    selection = metrics.get("dataset_selection")
    if selection is None:
        interim = json.loads((SOURCE / "validation_interim.json").read_text())
        if interim["dataset_revision"] != manifest["dataset_revision"]:
            raise ValueError("interim validation and subset dataset revisions differ")
        selection = interim["dataset_selection"]
    if selection != manifest["selection"]:
        raise ValueError("model and subset selection methods differ")
    for split, key in (("train", "train_records_unique"),
                       ("validation", "validation_records"), ("test", "test_records")):
        if metrics[key] != manifest["splits"][split]["records"]:
            raise ValueError(f"incomplete {split} run")
    for task in TASKS:
        if metrics["baseline_full_validation_nll"][task]["n"] != manifest["splits"]["validation"]["tasks"][task]:
            raise ValueError(f"incomplete baseline validation for {task}")
        if metrics["adapter_full_validation_nll"][task]["n"] != manifest["splits"]["validation"]["tasks"][task]:
            raise ValueError(f"incomplete adapter validation for {task}")
        if metrics["full_test_generation_scores"][task]["n"] != manifest["splits"]["test"]["tasks"][task]:
            raise ValueError(f"incomplete test generation for {task}")
    if sum(1 for _ in (SOURCE / "predictions.jsonl").open(encoding="utf-8")) != metrics["test_records"]:
        raise ValueError("incomplete prediction log")
    if manifest["selection"]["photo_files_on_hf"]:
        raise ValueError("photos must not be uploaded to Hugging Face")


def plot(metrics: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    scores = metrics["full_test_generation_scores"]
    names = ["Box choice", "Spatial", "Attribute", "Relation"]
    x = np.arange(4)
    base = [scores[task]["base_mean"] for task in TASKS[1:]]
    tuned = [scores[task]["adapter_mean"] for task in TASKS[1:]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7),
                             gridspec_kw={"width_ratios": [3.7, 1]})
    axes[0].bar(x - 0.18, base, 0.36, label="Base", color="#aab7c4")
    axes[0].bar(x + 0.18, tuned, 0.36, label="Adapter", color="#177e89")
    axes[0].set_xticks(x, names, rotation=16, ha="right")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Exact match")
    axes[0].set_title("Sampled held-out test")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.2)
    axes[0].set_axisbelow(True)
    ground = scores["ground_bbox"]
    axes[1].bar([0, 1], [ground["base_mean"], ground["adapter_mean"]],
                color=["#aab7c4", "#177e89"], width=0.55)
    axes[1].set_xticks([0, 1], ["Base", "Adapter"])
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Mean IoU")
    axes[1].set_title("Grounding")
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].set_axisbelow(True)
    fig.suptitle("Visual Jev · budget-limited training", fontsize=14, weight="bold")
    fig.text(0.02, 0.01,
             f"Train: {metrics['train_records_unique']:,}; validation: {metrics['validation_records']:,}; "
             f"test: {metrics['test_records']:,}. Stratified subset of the published dataset.",
             fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    fig.savefig(DEST / "benchmark.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def card(metrics: dict, manifest: dict) -> str:
    score_rows = []
    loss_rows = []
    for task in TASKS:
        score = metrics["full_test_generation_scores"][task]
        base = metrics["baseline_full_validation_nll"][task]
        tuned = metrics["adapter_full_validation_nll"][task]
        score_rows.append(f"| {task} | {score['n']:,} | {score['base_mean']:.3f} | {score['adapter_mean']:.3f} |")
        loss_rows.append(f"| {task} | {base['n']:,} | {base['mean_nll']:.3f} | {tuned['mean_nll']:.3f} |")
    return f"""---
language: en
license: apache-2.0
library_name: peft
base_model: Qwen/Qwen3.5-0.8B-Base
datasets:
  - {manifest['dataset_repo']}
pipeline_tag: image-text-to-text
tags:
  - visual-grounding
  - lora
  - qwen3.5
---

# Visual Jev sampled LoRA ({metrics['train_records_unique']:,} train records)

This PEFT adapter was trained on a **{metrics['train_records_unique']:,}-row stratified subset**, not the entire training split, of [Visual Jev decisions v1](https://huggingface.co/datasets/{manifest['dataset_repo']}) at revision `{manifest['dataset_revision']}`. It answers a task-specific question about an image with a normalized box, choice, or short text.

![Held-out subset benchmark](benchmark.png)

## Dataset and training

- Selection: per-task reservoir sampling with seed {manifest['selection']['seed']} from the original image-disjoint train, validation, and test splits. Each task contributes {manifest['selection']['per_task']['train']:,} training, {manifest['selection']['per_task']['validation']:,} validation, and {manifest['selection']['per_task']['test']:,} test records.
- Training: {metrics['train_records_unique']:,} unique rows in one epoch; {metrics['train_records_processed_including_padding']:,} processed including distributed padding.
- Evaluation: all {metrics['validation_records']:,} rows of the **selected validation subset** and all {metrics['test_records']:,} rows of the **selected test subset**. No full-split benchmark is claimed.
- Hardware: {metrics['gpu_count']} × {metrics['gpu']}; elapsed {metrics['elapsed_seconds']/3600:.2f} hours.
- Model: Qwen3.5-0.8B-Base, rank-{metrics['lora_rank']} LoRA on `q_proj` and `v_proj`, AdamW learning rate {metrics['learning_rate']}, gradient accumulation {metrics['gradient_accumulation_per_gpu']} per GPU. Images fit within 512 × 512 pixels.
- The generated-label `luna_candidates.jsonl` shard was excluded because it awaits human audit. No photo bytes are hosted in the dataset or model repository.

## Selected validation loss

Mean teacher-forced negative log-likelihood per row:

| Task | Rows | Base | Adapter |
|---|---:|---:|---:|
{chr(10).join(loss_rows)}

## Selected held-out test generation

Grounding uses mean intersection-over-union; the other tasks use case-insensitive exact match after trimming terminal punctuation.

| Task | Rows | Base | Adapter |
|---|---:|---:|---:|
{chr(10).join(score_rows)}

See [`metrics.json`](metrics.json) for exact values, [`predictions.jsonl`](predictions.jsonl) for per-record targets and predictions, [`selection_manifest.json`](selection_manifest.json) for sample counts and seed, and [`full_worker.py`](full_worker.py) for prompt serialization.

## Load

```python
import torch
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
from peft import PeftModel

base_id = "Qwen/Qwen3.5-0.8B-Base"
processor = AutoProcessor.from_pretrained(base_id)
base = Qwen3_5ForConditionalGeneration.from_pretrained(base_id, dtype=torch.bfloat16).to("cuda")
model = PeftModel.from_pretrained(base, "{REPO}").eval()
```

The adapter is a research demonstration. Visual Genome attributes and relations can be noisy; COCO geometric questions assume annotated objects are visible. Results do not establish OCR, exhaustive counting, calibrated UNKNOWN, or out-of-distribution performance.
"""


def main() -> None:
    metrics = json.loads((SOURCE / "metrics.json").read_text())
    manifest = json.loads(MANIFEST.read_text())
    validate(metrics, manifest)
    DEST.mkdir(parents=True, exist_ok=True)
    for name in ("adapter_model.safetensors", "adapter_config.json"):
        shutil.copy2(SOURCE / "adapter" / name, DEST / name)
    shutil.copytree(SOURCE / "processor", DEST / "processor", dirs_exist_ok=True)
    for name in ("metrics.json", "predictions.jsonl"):
        shutil.copy2(SOURCE / name, DEST / name)
    shutil.copy2(MANIFEST, DEST / "selection_manifest.json")
    for name in ("full_worker.py", "budget_modal.py"):
        shutil.copy2(ROOT / name, DEST / name)
    plot(metrics)
    (DEST / "README.md").write_text(card(metrics, manifest), encoding="utf-8")
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required for the promised model release")
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(REPO, repo_type="model", exist_ok=True, private=False)
    result = api.upload_folder(folder_path=DEST, repo_id=REPO, repo_type="model",
                               commit_message="Release budget-limited Visual Jev LoRA and sampled benchmark")
    print(f"published_model={result}")


if __name__ == "__main__":
    main()
