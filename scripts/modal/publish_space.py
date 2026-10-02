"""Publish only the Gradio app's source files to the public Hugging Face Space."""

import modal


SPACE_ID = "harshnandwana/visual-jev-demo"
app = modal.App("visual-jev-space-publisher")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub>=0.35,<2")
    .add_local_dir("space", "/root/space")
)


@app.function(image=image, cpu=1, memory=1024, timeout=300,
              secrets=[modal.Secret.from_name("visual-jev-hf-publish")])
def publish(create_only: bool = False) -> dict:
    import os

    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo_id=SPACE_ID, repo_type="space", space_sdk="gradio",
                    space_hardware="cpu-basic", private=False, exist_ok=True)
    if create_only:
        return {"space": SPACE_ID, "runtime": api.get_space_runtime(SPACE_ID).stage}
    commit = api.upload_folder(
        repo_id=SPACE_ID,
        repo_type="space",
        folder_path="/root/space",
        allow_patterns=["README.md", "app.py", "requirements.txt"],
        commit_message="Publish Visual Jev Gradio demo with synthetic presets",
    )
    return {"space": SPACE_ID, "commit": commit.oid,
            "runtime": api.get_space_runtime(SPACE_ID).stage}


@app.local_entrypoint()
def main(create_only: bool = False):
    print(publish.remote(create_only))
