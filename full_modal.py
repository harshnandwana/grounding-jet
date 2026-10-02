"""Fetch HF annotations and checkpoint COCO photos on Modal before training.

Photos are deliberately absent from Hugging Face. Start the remote function with
`modal run --detach full_modal.py::prepare_full_data`. Every completed photo
shard is committed to the Volume, so a cancelled call can resume.
"""

from pathlib import Path

import modal


REPO_ID = "harshnandwana/visual-jev-decisions-v1"
DATASET_REVISION = "687c745c34846d104ee85af802b9fd444a854f5d"
app = modal.App("visual-jev-full-data")
volume = modal.Volume.from_name("visual-jev-full-v1", create_if_missing=True)
prep_image = modal.Image.debian_slim(python_version="3.11").pip_install("huggingface_hub>=0.35,<2")
gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "torchvision", "transformers>=5.6,<6", "peft", "accelerate",
                 "pillow", "safetensors", "huggingface_hub>=0.35,<2")
    .add_local_file(Path(__file__).resolve().parent / "full_worker.py", "/root/full_worker.py")
)
publish_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub>=0.35,<2", "matplotlib", "numpy")
    .add_local_file(Path(__file__).resolve().parent / "publish_full_model.py", "/root/publish_full_model.py")
    .add_local_file(Path(__file__).resolve().parent / "full_worker.py", "/root/full_worker.py")
    .add_local_file(Path(__file__).resolve().parent / "full_modal.py", "/root/full_modal.py")
    .add_local_file(Path(__file__).resolve().parent / "data/hf_release/manifest.json",
                    "/root/data/hf_release/manifest.json")
)


@app.function(image=prep_image, cpu=16, memory=32768,
              timeout=43200, volumes={"/volume": volume})
def prepare_full_data() -> dict:
    import hashlib
    import json
    import os
    import shutil
    import tarfile
    import time
    from collections import Counter
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from urllib.request import urlopen

    from huggingface_hub import snapshot_download

    start = time.time()
    local = Path("/tmp/visual_jev_full")
    metadata_dir = local / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=REPO_ID, repo_type="dataset", revision=DATASET_REVISION,
        allow_patterns=["train.jsonl", "validation.jsonl", "test.jsonl", "manifest.json", "images.jsonl"],
        local_dir=metadata_dir, max_workers=8,
    )
    manifest = json.loads((metadata_dir / "manifest.json").read_text())
    for name, expected in manifest["files_sha256"].items():
        if name not in {"train.jsonl", "validation.jsonl", "test.jsonl", "images.jsonl"}:
            continue
        digest = hashlib.sha256()
        with (metadata_dir / name).open("rb") as handle:
            for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f"Hugging Face checksum mismatch: {name}")
    images = [json.loads(line) for line in (metadata_dir / "images.jsonl").open()]
    if len(images) != manifest["image_manifest_records"]:
        raise ValueError("image manifest record count mismatch")
    if manifest["image_files_included"]:
        raise ValueError("photo bytes must stay off Hugging Face")

    target_dir = Path("/volume/full_dataset")
    shard_dir = target_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = target_dir / "shards_manifest.json"
    checkpoint = json.loads(checkpoint_path.read_text()) if checkpoint_path.is_file() else {
        "dataset_revision": DATASET_REVISION, "shard_size": 5000, "shards": []}
    if checkpoint["dataset_revision"] != DATASET_REVISION or checkpoint["shard_size"] != 5000:
        raise ValueError("staged shard manifest belongs to another dataset revision")
    completed = {item["index"]: item for item in checkpoint["shards"]}

    def fetch(image: dict) -> str:
        path = local / image["path"]
        if path.is_file() and path.stat().st_size > 0:
            return "existing"
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".part")
        last_error = None
        for attempt in range(5):
            try:
                with urlopen(image["url"], timeout=90) as response, temp.open("wb") as output:
                    shutil.copyfileobj(response, output, length=1024 * 1024)
                with temp.open("rb") as handle:
                    signature = handle.read(8)
                if not (signature.startswith(b"\xff\xd8\xff") or signature == b"\x89PNG\r\n\x1a\n"):
                    raise ValueError("invalid JPEG/PNG signature")
                temp.replace(path)
                return "downloaded"
            except Exception as exc:
                last_error = exc
                temp.unlink(missing_ok=True)
                time.sleep(min(2 ** attempt, 16))
        raise RuntimeError(f"{image['id']}: {last_error}")

    counters = Counter()
    shard_size = 5000
    for offset in range(0, len(images), shard_size):
        index = offset // shard_size
        group = images[offset:offset + shard_size]
        filename = f"images-{index:05d}.tar"
        staged = shard_dir / filename
        prior = completed.get(index)
        if prior and staged.is_file() and prior["images"] == len(group):
            digest = hashlib.sha256()
            with staged.open("rb") as handle:
                for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() == prior["sha256"]:
                print(f"reused_shard={index} images={offset+len(group)}/{len(images)}", flush=True)
                continue
        errors = []
        with ThreadPoolExecutor(max_workers=64) as pool:
            futures = {pool.submit(fetch, image): image["id"] for image in group}
            for future in as_completed(futures):
                try:
                    counters[future.result()] += 1
                except Exception as exc:
                    errors.append(str(exc))
        if errors:
            raise RuntimeError(f"shard {index}: {len(errors)} image downloads failed; first: {errors[:5]}")
        archive_path = local / filename
        with tarfile.open(archive_path, "w") as archive:
            for image in group:
                archive.add(local / image["path"], arcname=image["path"])
        digest = hashlib.sha256()
        with archive_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
        temp_staged = shard_dir / (filename + ".part")
        shutil.copyfile(archive_path, temp_staged)
        temp_staged.replace(staged)
        item = {"index": index, "file": f"shards/{filename}",
                "images": len(group), "bytes": staged.stat().st_size,
                "sha256": digest.hexdigest()}
        completed[index] = item
        checkpoint["shards"] = [completed[key] for key in sorted(completed)]
        checkpoint_path.write_text(json.dumps(checkpoint, indent=2) + "\n")
        volume.commit()
        archive_path.unlink()
        for image in group:
            (local / image["path"]).unlink()
        print(f"committed_shard={index} images={offset+len(group)}/{len(images)} "
              f"failed=0 elapsed_s={time.time()-start:.0f}", flush=True)
    if sum(item["images"] for item in completed.values()) != len(images):
        raise ValueError("staged shard image count mismatch")
    for name in ("train.jsonl", "validation.jsonl", "test.jsonl", "manifest.json", "images.jsonl"):
        shutil.copyfile(metadata_dir / name, target_dir / name)
    prep_report = {
        "dataset_repo": REPO_ID, "dataset_revision": DATASET_REVISION,
        "images": len(images), "download": dict(counters),
        "image_shards": checkpoint["shards"],
        "elapsed_seconds": time.time() - start,
    }
    (target_dir / "prepare_report.json").write_text(json.dumps(prep_report, indent=2) + "\n")
    volume.commit()
    print(json.dumps(prep_report, indent=2), flush=True)
    launch_path = target_dir / "train_launch.json"
    if not launch_path.is_file():
        call = train_complete_dataset.spawn()
        launch_path.write_text(json.dumps({"training_function_call_id": call.object_id,
                                           "dataset_revision": DATASET_REVISION}, indent=2) + "\n")
        volume.commit()
        print(f"training_function_call_id={call.object_id}", flush=True)
    return prep_report


@app.local_entrypoint()
def stage():
    import json
    call = prepare_full_data.spawn()
    print(json.dumps({"staging_function_call_id": call.object_id}, indent=2))


@app.function(image=gpu_image, gpu="L40S:8", cpu=32, memory=131072,
              timeout=86400, volumes={"/volume": volume})
def train_complete_dataset() -> dict:
    import json
    import os
    import subprocess
    import sys
    import tarfile
    import time

    from huggingface_hub import snapshot_download

    staged = Path("/volume/full_dataset")
    report = json.loads((staged / "prepare_report.json").read_text())
    if report["dataset_revision"] != DATASET_REVISION:
        raise ValueError("staged image revision differs from the pinned dataset")
    start = time.time()
    root = Path("/tmp/visual_jev_full")
    root.mkdir(exist_ok=True)
    snapshot_download(
        repo_id=REPO_ID, repo_type="dataset", revision=DATASET_REVISION,
        allow_patterns=["train.jsonl", "validation.jsonl", "test.jsonl", "manifest.json"],
        local_dir=root, max_workers=8,
    )
    snapshot_download(repo_id="Qwen/Qwen3.5-0.8B-Base", max_workers=8)
    for item in report["image_shards"]:
        with tarfile.open(staged / item["file"]) as archive:
            archive.extractall(root)
    env = os.environ.copy()
    env.update({
        "VISUAL_JEV_DATA_DIR": str(root),
        "VISUAL_JEV_OUTPUT_DIR": "/volume/full_run",
        "VISUAL_JEV_DATASET_REPO": REPO_ID,
        "VISUAL_JEV_DATASET_REVISION": DATASET_REVISION,
    })
    print(f"staged_images={report['images']} elapsed_s={time.time()-start:.0f}", flush=True)
    try:
        subprocess.run(
            [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=8",
             "/root/full_worker.py"],
            env=env, check=True,
        )
    finally:
        volume.commit()
    metrics = json.loads(Path("/volume/full_run/metrics.json").read_text())
    publication = publish_complete_model.spawn()
    print(f"publication_function_call_id={publication.object_id}", flush=True)
    return metrics


@app.local_entrypoint()
def train():
    import json
    metrics = train_complete_dataset.remote()
    Path("data/full/full_train_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")


@app.function(image=publish_image, cpu=2, memory=8192, timeout=3600,
              secrets=[modal.Secret.from_name("visual-jev-hf-publish")],
              volumes={"/volume": volume})
def publish_complete_model() -> dict:
    """Verify all split counts and benchmark rows before public model upload."""
    import json
    import os
    import subprocess
    import sys

    env = os.environ.copy()
    env["VISUAL_JEV_FULL_SOURCE"] = "/volume/full_run"
    env["VISUAL_JEV_FULL_RELEASE"] = "/tmp/visual_jev_model_release"
    subprocess.run([sys.executable, "/root/publish_full_model.py"],
                   env=env, check=True)
    metrics = json.loads(Path("/volume/full_run/metrics.json").read_text())
    return {"model_repo": "harshnandwana/visual-jev-full-qwen35-0.8b-lora",
            "dataset_revision": metrics["dataset_revision"],
            "train_records": metrics["train_records_unique"],
            "validation_records": metrics["validation_records"],
            "test_records": metrics["test_records"]}


@app.function(image=gpu_image, gpu="L4:2", cpu=8, memory=32768,
              timeout=3600, volumes={"/volume": volume})
def ddp_smoke() -> dict:
    import json
    import os
    import shutil
    import subprocess
    import sys
    import tarfile

    root = Path("/tmp/visual_jev_ddp_smoke")
    root.mkdir(exist_ok=True)
    with tarfile.open("/volume/bundle.tar") as archive:
        archive.extractall(root)
    shutil.copyfile(root / "selection_manifest.json", root / "manifest.json")
    env = os.environ.copy()
    env.update({
        "VISUAL_JEV_DATA_DIR": str(root),
        "VISUAL_JEV_OUTPUT_DIR": "/volume/ddp_smoke_l4",
        "VISUAL_JEV_SMOKE_LIMIT": "32",
        "VISUAL_JEV_DATASET_REPO": REPO_ID,
        "VISUAL_JEV_DATASET_REVISION": "sample-smoke",
    })
    try:
        subprocess.run(
            [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
             "/root/full_worker.py"],
            env=env, check=True,
        )
    finally:
        volume.commit()
    return json.loads(Path("/volume/ddp_smoke_l4/metrics.json").read_text())


@app.local_entrypoint()
def smoke():
    import json
    print(json.dumps(ddp_smoke.remote(), indent=2))
