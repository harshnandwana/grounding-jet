"""Bounded Qwen3.5 vision LoRA smoke test on one Modal L4.

Run: modal run scripts/modal/train_modal.py::main
Evaluate saved adapter: modal run scripts/modal/train_modal.py::eval_main
"""

from pathlib import Path

import modal


ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "Qwen/Qwen3.5-0.8B-Base"
RUN_NAME = "qwen35-08b-coco-pilot-l4"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "torchvision", "transformers>=5.6,<6", "peft", "accelerate", "pillow", "safetensors")
    .add_local_file(ROOT / "data/train.jsonl", "/dataset/train.jsonl")
    .add_local_file(ROOT / "data/validation.jsonl", "/dataset/validation.jsonl")
    .add_local_dir(ROOT / "data/coco/train2017", "/dataset/train2017")
    .add_local_dir(ROOT / "data/coco/val2017", "/dataset/val2017")
)

app = modal.App("visual-jev-l4-smoke")
outputs = modal.Volume.from_name("visual-jev-smoke-artifacts", create_if_missing=True)


@app.function(image=image, gpu="L4", timeout=2700, volumes={"/outputs": outputs})
def train(max_steps: int = 60, train_limit: int = 160, val_limit: int = 20) -> dict:
    import json
    import random
    import time

    import torch
    from PIL import Image
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    torch.manual_seed(42)
    random.seed(42)
    assert torch.cuda.is_available(), "L4 GPU was not attached"
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    start = time.time()

    def load(path: str, limit: int) -> list[dict]:
        with open(path, encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle]
        random.Random(42).shuffle(records)
        return records[:limit]

    train_records = load("/dataset/train.jsonl", train_limit)
    val_records = load("/dataset/validation.jsonl", val_limit)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    chat_template = processor.tokenizer.chat_template
    if not chat_template:
        raise RuntimeError("Qwen checkpoint did not provide a tokenizer chat template")
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to("cuda")
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, target_modules=["q_proj", "v_proj"], bias="none"),
    )
    model.print_trainable_parameters()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=2e-4)

    def prepare(record: dict) -> dict:
        path = Path("/dataset") / record["image"]["source_split"] / Path(record["image"]["path"]).name
        with Image.open(path) as raw:
            photo = raw.convert("RGB")
            photo.thumbnail((512, 512))
        prompt = record["question"] + "\n"
        if record["task"] == "box_choice":
            prompt += "Candidate boxes (normalized x1,y1,x2,y2):\n"
            for candidate in record["candidates"]:
                coords = ", ".join(f"{v:.3f}" for v in candidate["bbox_xyxy"])
                prompt += f"{candidate['choice']}: [{coords}]\n"
        prompt += "Choices: " + ", ".join(record["choices"]) + "\nAnswer with exactly one choice."
        user = {"role": "user", "content": [{"type": "image", "image": photo}, {"type": "text", "text": prompt}]}
        answer = record["choices"][record["answer_index"]]
        prompt_inputs = processor.apply_chat_template(
            [user], chat_template=chat_template, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt"
        )
        full_inputs = processor.apply_chat_template(
            [user, {"role": "assistant", "content": answer}],
            chat_template=chat_template, tokenize=True, add_generation_prompt=False,
            return_dict=True, return_tensors="pt",
        )
        prefix_length = prompt_inputs["input_ids"].shape[1]
        if not torch.equal(full_inputs["input_ids"][0, :prefix_length], prompt_inputs["input_ids"][0]):
            raise ValueError("chat-template prompt is not a prefix of the labeled example")
        labels = full_inputs["input_ids"].clone()
        labels[:, :prefix_length] = -100
        assert (labels != -100).sum().item() > 0
        full_inputs["labels"] = labels
        return {key: value.to("cuda") if isinstance(value, torch.Tensor) else value for key, value in full_inputs.items()}

    def evaluate(records: list[dict]) -> float:
        model.eval()
        losses = []
        with torch.no_grad():
            for record in records:
                batch = prepare(record)
                losses.append(model(**batch).loss.float().item())
        return sum(losses) / len(losses)

    baseline_val_loss = evaluate(val_records)
    print(f"baseline_val_loss={baseline_val_loss:.4f}", flush=True)
    train_losses = []
    for step in range(max_steps):
        model.train()
        batch = prepare(train_records[step % len(train_records)])
        optimizer.zero_grad(set_to_none=True)
        loss = model(**batch).loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step + 1}: {loss.item()}")
        loss.backward()
        optimizer.step()
        train_losses.append(loss.float().item())
        if (step + 1) % 10 == 0 or step == 0:
            print(f"step={step + 1} train_loss={train_losses[-1]:.4f} elapsed_s={time.time()-start:.1f}", flush=True)

    final_val_loss = evaluate(val_records)
    metrics = {
        "model": MODEL_ID,
        "gpu": torch.cuda.get_device_name(0),
        "max_steps": max_steps,
        "train_records_available": 483,
        "validation_records_available": 97,
        "train_records_sampled": len(train_records),
        "train_records_seen": min(max_steps, len(train_records)),
        "validation_records_sampled": len(val_records),
        "baseline_val_loss": baseline_val_loss,
        "final_val_loss": final_val_loss,
        "first_10_train_loss_mean": sum(train_losses[:10]) / min(10, len(train_losses)),
        "last_10_train_loss_mean": sum(train_losses[-10:]) / min(10, len(train_losses)),
        "train_losses": train_losses,
        "elapsed_seconds": time.time() - start,
        "peak_cuda_memory_gb": torch.cuda.max_memory_allocated() / 1e9,
        "note": "Small smoke test; loss movement checks trainability, not general capability.",
    }
    output_dir = Path("/outputs") / RUN_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir / "adapter")
    processor.save_pretrained(output_dir / "processor")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    outputs.commit()
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics


@app.local_entrypoint()
def main():
    result = train.remote()
    Path("data/train_smoke_metrics.json").write_text(__import__("json").dumps(result, indent=2) + "\n")
    print("Saved data/train_smoke_metrics.json")


@app.function(image=image, gpu="L4", timeout=1800, volumes={"/outputs": outputs})
def evaluate_adapter() -> dict:
    """Compare base and adapter on held-out choice accuracy, plus a blank-image control."""
    import json
    from collections import defaultdict

    import torch
    from PIL import Image
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    with open("/dataset/validation.jsonl", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle]
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    chat_template = processor.tokenizer.chat_template
    base = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to("cuda")
    model = PeftModel.from_pretrained(base, f"/outputs/{RUN_NAME}/adapter").eval()
    result = defaultdict(lambda: {"n": 0, "base_correct": 0, "adapter_correct": 0})
    blank_result = {"n": 0, "real_image_correct": 0, "blank_image_correct": 0}
    predictions = []

    def prepare_prompt(record: dict, blank: bool = False) -> dict:
        path = Path("/dataset") / record["image"]["source_split"] / Path(record["image"]["path"]).name
        with Image.open(path) as raw:
            photo = raw.convert("RGB")
            photo.thumbnail((512, 512))
        if blank:
            photo = Image.new("RGB", photo.size, (127, 127, 127))
        prompt = record["question"] + "\n"
        if record["task"] == "box_choice":
            prompt += "Candidate boxes (normalized x1,y1,x2,y2):\n"
            for candidate in record["candidates"]:
                coords = ", ".join(f"{v:.3f}" for v in candidate["bbox_xyxy"])
                prompt += f"{candidate['choice']}: [{coords}]\n"
        prompt += "Choices: " + ", ".join(record["choices"]) + "\nAnswer with exactly one choice."
        user = {"role": "user", "content": [{"type": "image", "image": photo}, {"type": "text", "text": prompt}]}
        batch = processor.apply_chat_template(
            [user], chat_template=chat_template, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        )
        return {key: value.to("cuda") if isinstance(value, torch.Tensor) else value for key, value in batch.items()}

    with torch.no_grad():
        for index, record in enumerate(records):
            candidate_ids = [processor.tokenizer.encode(choice, add_special_tokens=False) for choice in record["choices"]]
            if any(len(ids) != 1 for ids in candidate_ids):
                raise RuntimeError(f"Candidate choice is not one token: {record['choices']} -> {candidate_ids}")
            ids = [tokens[0] for tokens in candidate_ids]
            batch = prepare_prompt(record)
            with model.disable_adapter():
                base_logits = model(**batch).logits[0, -1, ids]
            tuned_logits = model(**batch).logits[0, -1, ids]
            group = result[record["task"]]
            group["n"] += 1
            group["base_correct"] += int(base_logits.argmax().item() == record["answer_index"])
            group["adapter_correct"] += int(tuned_logits.argmax().item() == record["answer_index"])
            predictions.append({
                "id": record["id"],
                "task": record["task"],
                "image_path": record["image"]["path"],
                "question": record["question"],
                "choices": record["choices"],
                "candidates": record["candidates"],
                "evidence": record["evidence"],
                "answer_index": record["answer_index"],
                "base_pred_index": base_logits.argmax().item(),
                "adapter_pred_index": tuned_logits.argmax().item(),
                "base_choice_probs": torch.softmax(base_logits.float(), dim=0).tolist(),
                "adapter_choice_probs": torch.softmax(tuned_logits.float(), dim=0).tolist(),
            })
            if index < 20:
                blank_logits = model(**prepare_prompt(record, blank=True)).logits[0, -1, ids]
                blank_result["n"] += 1
                blank_result["real_image_correct"] += int(tuned_logits.argmax().item() == record["answer_index"])
                blank_result["blank_image_correct"] += int(blank_logits.argmax().item() == record["answer_index"])
            if (index + 1) % 20 == 0:
                print(f"evaluated={index + 1}/{len(records)}", flush=True)

    summary = {"model": MODEL_ID, "validation_records": len(records), "by_task": dict(result), "blank_image_first_20": blank_result}
    path = Path("/outputs") / RUN_NAME / "choice_eval.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    (path.parent / "predictions.json").write_text(json.dumps(predictions, indent=2) + "\n")
    outputs.commit()
    print(json.dumps(summary, indent=2), flush=True)
    return {"summary": summary, "predictions": predictions}


@app.local_entrypoint()
def eval_main():
    result = evaluate_adapter.remote()
    Path("data/choice_eval.json").write_text(__import__("json").dumps(result["summary"], indent=2) + "\n")
    Path("data/predictions.json").write_text(__import__("json").dumps(result["predictions"], indent=2) + "\n")
    print("Saved data/choice_eval.json and data/predictions.json")
