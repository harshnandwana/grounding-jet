"""Bounded, photo-free HF subset training with a configurable train size."""

import os
from pathlib import Path

import modal


REPO_ID = os.environ.get("VISUAL_JEV_DATASET_REPO", "harshnandwana/visual-jev-decisions-v1")
DATASET_REVISION = os.environ.get("VISUAL_JEV_DATASET_REVISION", "687c745c34846d104ee85af802b9fd444a854f5d")
MODEL_REPO_ID = os.environ.get("VISUAL_JEV_MODEL_REPO", "harshnandwana/visual-jev-budget20-qwen35-0.8b-lora")
APP_NAME = os.environ.get("VISUAL_JEV_BUDGET_APP", "visual-jev-budget20")
RUN_NAME = os.environ.get("VISUAL_JEV_RUN_NAME", "budget20")
if not RUN_NAME.isidentifier():
    raise ValueError("VISUAL_JEV_RUN_NAME must be an identifier")
TRAIN_TIMEOUT = int(os.environ.get("VISUAL_JEV_TRAIN_TIMEOUT", "21600"))
VOLUME_NAME = os.environ.get("VISUAL_JEV_MODAL_VOLUME", "visual-jev-full-v1")
TASKS = ("ground_bbox", "box_choice", "spatial_boolean", "attribute_text", "relation_text")
PER_TASK = {"train": int(os.environ.get("VISUAL_JEV_TRAIN_PER_TASK", "8000")),
            "validation": 200, "test": 200}
SEED = 43801

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME)
prep_image = modal.Image.debian_slim(python_version="3.11").pip_install("huggingface_hub>=0.35,<2")
gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "torchvision", "transformers>=5.6,<6", "peft", "accelerate",
                 "pillow", "safetensors", "huggingface_hub>=0.35,<2")
    .add_local_file(Path(__file__).resolve().parent / "full_worker.py", "/root/full_worker.py")
    .add_local_file(Path(__file__).resolve().parent / "batch_probe.py", "/root/batch_probe.py")
)
publish_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub>=0.35,<2", "matplotlib", "numpy")
    .add_local_file(Path(__file__).resolve().parent / "publish_budget_model.py",
                    "/root/publish_budget_model.py")
    .add_local_file(Path(__file__).resolve().parent / "full_worker.py", "/root/full_worker.py")
    .add_local_file(Path(__file__).resolve().parent / "budget_modal.py", "/root/budget_modal.py")
)


@app.function(image=prep_image, cpu=4, memory=8192, timeout=3600,
              volumes={"/volume": volume})
def prepare_budget_dataset(run_name: str = "budget20", per_task: dict | None = None) -> dict:
    import hashlib
    import json
    import random
    import tarfile
    import time
    from collections import Counter

    from huggingface_hub import snapshot_download

    started = time.time()
    source = Path("/volume/full_dataset")
    per_task = per_task or {"train": 8000, "validation": 200, "test": 200}
    if not run_name.isidentifier():
        raise ValueError("run_name must be an identifier")
    target = Path(f"/volume/{run_name}")
    target.mkdir(parents=True, exist_ok=True)
    ready = target / "manifest.json"
    archive_path = target / "images.tar"
    if ready.is_file() and archive_path.is_file():
        prior = json.loads(ready.read_text())
        if (prior.get("dataset_revision") == DATASET_REVISION
                and prior.get("selection", {}).get("per_task") == per_task):
            return prior

    metadata = Path("/tmp/budget20_hf")
    metadata.mkdir(exist_ok=True)
    snapshot_download(repo_id=REPO_ID, repo_type="dataset", revision=DATASET_REVISION,
                      allow_patterns=["train.jsonl", "validation.jsonl", "test.jsonl", "manifest.json"],
                      local_dir=metadata, max_workers=8)
    original = json.loads((metadata / "manifest.json").read_text())
    if original["image_files_included"]:
        raise ValueError("photo bytes must stay off Hugging Face")
    staged = json.loads((source / "prepare_report.json").read_text())
    if (staged["dataset_repo"] != REPO_ID
            or staged["dataset_revision"] != DATASET_REVISION
            or staged["images"] != original["image_manifest_records"]):
        raise ValueError("the complete photo shard set is not staged")

    selected_paths = set()
    split_reports = {}
    for split_index, split in enumerate(per_task):
        rng = random.Random(SEED + split_index)
        seen = Counter()
        selected = {task: [] for task in TASKS}
        with (metadata / f"{split}.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                task = row["task"]
                if task not in selected:
                    continue
                seen[task] += 1
                bucket = selected[task]
                limit = per_task[split]
                if len(bucket) < limit:
                    bucket.append(row)
                else:
                    at = rng.randrange(seen[task])
                    if at < limit:
                        bucket[at] = row
        rows = []
        for task in TASKS:
            if len(selected[task]) != per_task[split]:
                raise ValueError(f"insufficient rows for {split}/{task}")
            rows.extend(selected[task])
        random.Random(SEED + 100 + split_index).shuffle(rows)
        split_reports[split] = {
            "records": len(rows),
            "tasks": dict(Counter(row["task"] for row in rows)),
            "source_records": original["splits"][split]["records"],
        }
        with (target / f"{split}.jsonl").open("w", encoding="utf-8") as output:
            for row in rows:
                selected_paths.add(row["image"]["path"])
                output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        print(f"selected_{split}={len(rows)}", flush=True)

    temporary = target / "images.tar.part"
    temporary.unlink(missing_ok=True)
    copied = set()
    with tarfile.open(temporary, "w") as output:
        for shard in staged["image_shards"]:
            with tarfile.open(source / shard["file"], "r") as archive:
                for member in archive:
                    if member.name in selected_paths:
                        if member.name in copied:
                            raise ValueError(f"duplicate staged image: {member.name}")
                        with archive.extractfile(member) as photo:
                            output.addfile(member, photo)
                        copied.add(member.name)
            print(f"scanned_shard={shard['index']} selected_images={len(copied)}/{len(selected_paths)}",
                  flush=True)
    if copied != selected_paths:
        raise ValueError(f"missing {len(selected_paths - copied)} selected images")
    temporary.replace(archive_path)
    digest = hashlib.sha256()
    with archive_path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    manifest = {
        "dataset_repo": REPO_ID,
        "dataset_revision": DATASET_REVISION,
        "splits": split_reports,
        "selection": {"method": "per-task reservoir sampling", "seed": SEED,
                      "per_task": per_task, "photo_files_on_hf": False},
        "unique_images": len(selected_paths),
        "image_archive_bytes": archive_path.stat().st_size,
        "image_archive_sha256": digest.hexdigest(),
        "preparation_seconds": time.time() - started,
    }
    ready.write_text(json.dumps(manifest, indent=2) + "\n")
    volume.commit()
    print(f"budget_dataset_ready train={split_reports['train']['records']} "
          f"validation={split_reports['validation']['records']} "
          f"test={split_reports['test']['records']} images={len(selected_paths)}", flush=True)
    return manifest


@app.local_entrypoint()
def stage():
    import json
    result = prepare_budget_dataset.remote(RUN_NAME, PER_TASK)
    print(json.dumps({"splits": result["splits"], "unique_images": result["unique_images"],
                      "preparation_seconds": result["preparation_seconds"]}, indent=2))


@app.function(image=gpu_image, gpu="L4", cpu=4, memory=16384, timeout=900,
              volumes={"/volume": volume})
def probe_microbatch() -> None:
    import subprocess
    import sys
    subprocess.run([sys.executable, "/root/batch_probe.py"], check=True)


@app.local_entrypoint()
def probe():
    probe_microbatch.remote()


@app.function(image=gpu_image, gpu="L4:2", cpu=8, memory=32768, timeout=1800,
              volumes={"/volume": volume})
def smoke_budget_batch() -> dict:
    import json
    import os
    import shutil
    import subprocess
    import sys
    import tarfile

    root = Path("/tmp/visual_jev_budget_smoke")
    root.mkdir(exist_ok=True)
    with tarfile.open("/volume/bundle.tar") as archive:
        archive.extractall(root)
    shutil.copyfile(root / "selection_manifest.json", root / "manifest.json")
    env = os.environ.copy()
    env.update({
        "VISUAL_JEV_DATA_DIR": str(root),
        "VISUAL_JEV_OUTPUT_DIR": "/volume/budget20/smoke",
        "VISUAL_JEV_SMOKE_LIMIT": "128",
        "VISUAL_JEV_DATASET_REPO": REPO_ID,
        "VISUAL_JEV_DATASET_REVISION": "sample-smoke",
        "VISUAL_JEV_ACCUMULATION": "16",
        "VISUAL_JEV_MICROBATCH": "2",
    })
    try:
        subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone",
                        "--nproc_per_node=2", "/root/full_worker.py"], env=env, check=True)
    finally:
        volume.commit()
    return json.loads(Path("/volume/budget20/smoke/metrics.json").read_text())


@app.local_entrypoint()
def smoke():
    import json
    print(json.dumps(smoke_budget_batch.remote(), indent=2))


@app.function(image=gpu_image, gpu="L4:2", cpu=8, memory=32768, timeout=TRAIN_TIMEOUT,
              volumes={"/volume": volume})
def train_budget(run_name: str = "budget20", per_task: dict | None = None,
                 model_repo: str = "harshnandwana/visual-jev-budget20-qwen35-0.8b-lora") -> dict:
    import hashlib
    import json
    import os
    import shutil
    import subprocess
    import sys
    import tarfile
    import time

    from huggingface_hub import snapshot_download

    started = time.time()
    per_task = per_task or {"train": 8000, "validation": 200, "test": 200}
    if not run_name.isidentifier():
        raise ValueError("run_name must be an identifier")
    run_dir = f"/volume/{run_name}"
    staged = Path(run_dir)
    selection = json.loads((staged / "manifest.json").read_text())
    if selection["dataset_revision"] != DATASET_REVISION:
        raise ValueError("budget subset belongs to another HF dataset revision")
    if selection["selection"]["per_task"] != per_task:
        raise ValueError("training subset selection differs from requested counts")
    if selection["splits"]["train"]["records"] != per_task["train"] * len(TASKS):
        raise ValueError("budget training subset is incomplete")
    metadata = Path("/tmp/budget20_hf_manifest")
    snapshot_download(repo_id=REPO_ID, repo_type="dataset", revision=DATASET_REVISION,
                      allow_patterns=["manifest.json"], local_dir=metadata)
    if json.loads((metadata / "manifest.json").read_text())["image_files_included"]:
        raise ValueError("photos unexpectedly present on Hugging Face")
    archive_path = staged / "images.tar"
    digest = hashlib.sha256()
    with archive_path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != selection["image_archive_sha256"]:
        raise ValueError("budget photo archive checksum mismatch")
    root = Path(f"/tmp/visual_jev_{run_name}")
    root.mkdir(exist_ok=True)
    for split in per_task:
        shutil.copyfile(staged / f"{split}.jsonl", root / f"{split}.jsonl")
    shutil.copyfile(staged / "manifest.json", root / "manifest.json")
    with tarfile.open(archive_path, "r") as archive:
        archive.extractall(root)
    print(f"phase=budget_dataset_ready images={selection['unique_images']} "
          f"elapsed_s={time.time()-started:.0f}", flush=True)
    env = os.environ.copy()
    env.update({
        "VISUAL_JEV_DATA_DIR": str(root),
        "VISUAL_JEV_OUTPUT_DIR": f"{run_dir}/run",
        "VISUAL_JEV_DATASET_REPO": REPO_ID,
        "VISUAL_JEV_DATASET_REVISION": DATASET_REVISION,
        "VISUAL_JEV_ACCUMULATION": "16",
        "VISUAL_JEV_MICROBATCH": "2",
        "VISUAL_JEV_CHECKPOINT_EVERY": "100",
    })
    try:
        subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone",
                        "--nproc_per_node=2", "/root/full_worker.py"], env=env, check=True)
    finally:
        volume.commit()
    metrics = json.loads((staged / "run" / "metrics.json").read_text())
    publication = publish_budget_model.spawn(run_name, model_repo)
    print(f"publication_function_call_id={publication.object_id}", flush=True)
    return metrics


@app.local_entrypoint()
def train():
    import json
    # Spawn on the deployed app so the call survives this entrypoint's app shutdown.
    call = modal.Function.from_name(APP_NAME, "train_budget").spawn(RUN_NAME, PER_TASK, MODEL_REPO_ID)
    print(json.dumps({"training_function_call_id": call.object_id}))


@app.function(image=publish_image, cpu=2, memory=4096, timeout=1800,
              secrets=[modal.Secret.from_name("visual-jev-hf-publish")],
              volumes={"/volume": volume})
def publish_budget_model(run_name: str = "budget20",
                         model_repo: str = "harshnandwana/visual-jev-budget20-qwen35-0.8b-lora") -> dict:
    import json
    import os
    import subprocess
    import sys

    env = os.environ.copy()
    if not run_name.isidentifier():
        raise ValueError("run_name must be an identifier")
    run_dir = f"/volume/{run_name}"
    env["VISUAL_JEV_BUDGET_SOURCE"] = f"{run_dir}/run"
    env["VISUAL_JEV_BUDGET_MANIFEST"] = f"{run_dir}/manifest.json"
    env["VISUAL_JEV_BUDGET_RELEASE"] = f"/tmp/visual_jev_{run_name}_release"
    env["VISUAL_JEV_MODEL_REPO"] = model_repo
    subprocess.run([sys.executable, "/root/publish_budget_model.py"], env=env, check=True)
    metrics = json.loads(Path(f"{run_dir}/run/metrics.json").read_text())
    return {"model_repo": model_repo,
            "train_records": metrics["train_records_unique"],
            "validation_records": metrics["validation_records"],
            "test_records": metrics["test_records"]}
