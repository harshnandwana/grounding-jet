"""Publish the photo-free 40k/120k test graphic and cost audit to the 120k model."""

import modal


REPO_ID = "harshnandwana/visual-jev-120k-qwen35-0.8b-lora"
app = modal.App("visual-jev-test-comparison-publisher")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub>=0.35,<2")
    .add_local_file("scripts/modal/publish_budget_model.py", "/root/publish_budget_model.py")
    .add_local_file("results/40k_vs_120k_test.png", "/root/comparison_test.png")
    .add_local_file("results/budget120k_cost.json", "/root/cost.json")
)


@app.function(image=image, cpu=1, memory=1024, timeout=300,
              secrets=[modal.Secret.from_name("visual-jev-hf-publish")])
def publish() -> dict:
    import io
    import json
    import os
    from pathlib import Path

    from huggingface_hub import CommitOperationAdd, HfApi
    from publish_budget_model import card

    api = HfApi(token=os.environ["HF_TOKEN"])
    info = api.model_info(REPO_ID)
    if info.private:
        raise ValueError("expected a public model")
    metrics = json.loads(Path(api.hf_hub_download(REPO_ID, "metrics.json")).read_text())
    manifest = json.loads(Path(api.hf_hub_download(REPO_ID, "selection_manifest.json")).read_text())
    config = json.loads(Path(api.hf_hub_download(REPO_ID, "adapter_config.json")).read_text())
    cost = json.loads(Path("/root/cost.json").read_text())
    if (metrics["train_records_unique"], metrics["test_records"]) != (120_000, 1_000):
        raise ValueError("unexpected training or test count")
    if config.get("task_type") != "CAUSAL_LM":
        raise ValueError("repair PEFT config before publishing comparison")
    if cost["metered_cost"] != "8.23251352":
        raise ValueError("unexpected cost audit")
    png = Path("/root/comparison_test.png").read_bytes()
    if not png.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("comparison artifact is not a PNG")
    readme = card(metrics, manifest, REPO_ID)
    commit = api.create_commit(
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="Add 40k vs 120k test graphic and measured Modal cost",
        operations=[
            CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=io.BytesIO(readme.encode())),
            CommitOperationAdd(path_in_repo="comparison_test.png", path_or_fileobj=io.BytesIO(png)),
            CommitOperationAdd(path_in_repo="cost.json", path_or_fileobj=io.BytesIO((json.dumps(cost, indent=2) + "\n").encode())),
        ],
    )
    return {"repo": REPO_ID, "commit": commit.oid, "image_bytes": len(png),
            "metered_cost": cost["metered_cost"]}


@app.local_entrypoint()
def main():
    print(publish.remote())
