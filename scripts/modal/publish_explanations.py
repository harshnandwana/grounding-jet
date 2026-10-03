"""Update public HF cards with GitHub-hosted, photo-based explanation figures."""

import modal


DATASET = "harshnandwana/visual-jev-decisions-v1"
MODELS = (
    ("harshnandwana/visual-jev-120k-qwen35-0.8b-lora", 120_000),
    ("harshnandwana/visual-jev-budget20-qwen35-0.8b-lora", 40_000),
)
app = modal.App("visual-jev-card-explanations")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub>=0.35,<2")
    .add_local_file("scripts/modal/publish_budget_model.py", "/root/publish_budget_model.py")
    .add_local_file("docs/hf_dataset_card_addendum.md", "/root/dataset_addendum.md")
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
    results = {}
    for repo, expected in MODELS:
        if api.model_info(repo).private:
            raise ValueError(f"expected public model: {repo}")
        metrics = json.loads(Path(api.hf_hub_download(repo, "metrics.json")).read_text())
        manifest = json.loads(Path(api.hf_hub_download(repo, "selection_manifest.json")).read_text())
        if metrics["train_records_unique"] != expected or metrics["test_records"] != 1000:
            raise ValueError(f"unexpected sample size: {repo}")
        if manifest["dataset_revision"] != metrics["dataset_revision"]:
            raise ValueError(f"dataset revision mismatch: {repo}")
        readme = card(metrics, manifest, repo)
        if "raw.githubusercontent.com/harshnandwana/grounding-jet/main/results/one_image_dataset.png" not in readme:
            raise ValueError("model card is missing external photo figure")
        commit = api.create_commit(
            repo_id=repo, repo_type="model",
            commit_message="Explain dataset creation and show photo-based test cases",
            operations=[CommitOperationAdd("README.md", io.BytesIO(readme.encode()))],
        )
        results[repo] = commit.oid

    if api.dataset_info(DATASET).private:
        raise ValueError("expected a public dataset")
    current = Path(api.hf_hub_download(DATASET, "README.md", repo_type="dataset")).read_text()
    addendum = Path("/root/dataset_addendum.md").read_text()
    heading = "## How the questions were created"
    if heading in current:
        start = current.index(heading)
        end = current.index("## Training targets", start)
        updated = current[:start] + addendum + current[end:]
    else:
        marker = "## Training targets"
        if current.count(marker) != 1:
            raise ValueError("dataset card structure changed")
        updated = current.replace(marker, addendum + marker, 1)
    commit = api.create_commit(
        repo_id=DATASET, repo_type="dataset",
        commit_message="Document question templates and one-photo dataset example",
        operations=[CommitOperationAdd("README.md", io.BytesIO(updated.encode()))],
    )
    results[DATASET] = commit.oid
    return results


@app.local_entrypoint()
def main():
    print(publish.remote())
