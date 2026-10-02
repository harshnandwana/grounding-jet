"""Smoke-test a Hub adapter with a string PEFT task type and an image prompt."""

import modal


app = modal.App("visual-jev-peft-config-check")
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch", "torchvision", "transformers>=5.6,<6", "peft>=0.21.2", "accelerate",
    "pillow", "safetensors", "huggingface_hub>=0.35,<2",
)


@app.function(image=image, gpu="L4", cpu=4, memory=16384, timeout=900)
def check() -> dict:
    import json
    from pathlib import Path

    import torch
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from PIL import Image
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    base_id = "Qwen/Qwen3.5-0.8B-Base"
    adapter_id = "harshnandwana/visual-jev-budget20-qwen35-0.8b-lora"
    local = Path(snapshot_download(adapter_id, allow_patterns=[
        "adapter_config.json", "adapter_model.safetensors",
    ], local_dir="/tmp/visual_jev_adapter_check"))
    config_path = local / "adapter_config.json"
    config = json.loads(config_path.read_text())
    if config.get("task_type") is not None:
        raise ValueError("expected the published pre-fix task_type to be null")
    config["task_type"] = "CAUSAL_LM"
    config_path.write_text(json.dumps(config, indent=2) + "\n")

    base = Qwen3_5ForConditionalGeneration.from_pretrained(
        base_id, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to("cuda")
    model = PeftModel.from_pretrained(base, str(local)).eval()
    processor = AutoProcessor.from_pretrained(base_id)
    user = {"role": "user", "content": [
        {"type": "image", "image": Image.new("RGB", (224, 224), "white")},
        {"type": "text", "text": "Is the square white?\nChoices: YES, NO, UNKNOWN. Answer with one choice only."},
    ]}
    inputs = processor.apply_chat_template(
        [user], chat_template=processor.tokenizer.chat_template,
        tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to("cuda")
    with torch.inference_mode():
        tokens = model.generate(**inputs, max_new_tokens=8, do_sample=False)
    output = processor.batch_decode(tokens[:, inputs["input_ids"].shape[1]:],
                                    skip_special_tokens=True)[0]
    return {"task_type": config["task_type"], "wrapper": type(model).__name__,
            "generated": output}


@app.local_entrypoint()
def main():
    print(check.remote())
