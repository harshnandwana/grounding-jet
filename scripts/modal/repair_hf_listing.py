"""Repair a published model card and string PEFT task type without retraining."""

import modal


app = modal.App("visual-jev-hf-listing-repair")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub>=0.35,<2")
    .add_local_file("scripts/modal/publish_budget_model.py", "/root/publish_budget_model.py")
)


@app.function(image=image, cpu=1, memory=1024, timeout=300,
              secrets=[modal.Secret.from_name("visual-jev-hf-publish")])
def repair(repo_id: str, expected_train: int) -> dict:
    import io
    import json
    import os
    from pathlib import Path

    from huggingface_hub import CommitOperationAdd, HfApi
    from publish_budget_model import card

    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)
    info = api.model_info(repo_id)
    if info.private:
        raise ValueError("expected a public model repository")
    metrics = json.loads(Path(api.hf_hub_download(repo_id, "metrics.json")).read_text())
    manifest = json.loads(Path(api.hf_hub_download(repo_id, "selection_manifest.json")).read_text())
    config = json.loads(Path(api.hf_hub_download(repo_id, "adapter_config.json")).read_text())
    if metrics["train_records_unique"] != expected_train:
        raise ValueError("model has an unexpected training size")
    if metrics["dataset_revision"] != manifest["dataset_revision"]:
        raise ValueError("model and dataset revisions differ")
    if config.get("task_type") not in (None, "CAUSAL_LM"):
        raise ValueError("unexpected PEFT task type")
    config["task_type"] = "CAUSAL_LM"
    readme = card(metrics, manifest, repo_id)
    commit = api.create_commit(
        repo_id=repo_id,
        repo_type="model",
        commit_message="Fix PEFT task type and document image inference",
        operations=[
            CommitOperationAdd(path_in_repo="adapter_config.json",
                               path_or_fileobj=io.BytesIO((json.dumps(config, indent=2) + "\n").encode())),
            CommitOperationAdd(path_in_repo="README.md",
                               path_or_fileobj=io.BytesIO(readme.encode())),
        ],
    )
    return {"repo": repo_id, "commit": commit.oid, "task_type": config["task_type"]}


@app.local_entrypoint()
def main(repo_id: str = "harshnandwana/visual-jev-budget20-qwen35-0.8b-lora",
         expected_train: int = 40000):
    print(repair.remote(repo_id, expected_train))
