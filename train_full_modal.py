"""Train and benchmark the five-task Visual Jev v1 sample on one Modal L40S.

Upload data/train_full_v1/bundle.tar to Modal Volume visual-jev-full-v1 at
/bundle.tar, then run: modal run train_full_modal.py::main
"""

from pathlib import Path

import modal


ROOT = Path(__file__).resolve().parent
BASE_MODEL = "Qwen/Qwen3.5-0.8B-Base"
RUN_NAME = "visual-jev-v1-l40s-lora"

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch", "torchvision", "transformers>=5.6,<6", "peft", "accelerate", "pillow", "safetensors"
)
app = modal.App("visual-jev-full-v1")
volume = modal.Volume.from_name("visual-jev-full-v1", create_if_missing=True)


@app.function(image=image, gpu="L40S", timeout=14400, volumes={"/volume": volume})
def train_and_benchmark() -> dict:
    import json
    import math
    import random
    import re
    import tarfile
    import time
    from collections import defaultdict

    import torch
    from PIL import Image
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    torch.manual_seed(43801)
    random.seed(43801)
    start = time.time()
    data_dir = Path("/tmp/visual-jev-v1")
    data_dir.mkdir(exist_ok=True)
    with tarfile.open("/volume/bundle.tar") as archive:
        archive.extractall(data_dir)

    def load(split):
        with (data_dir / f"{split}.jsonl").open() as handle:
            return [json.loads(line) for line in handle]

    train_rows, val_rows, test_rows = load("train"), load("validation"), load("test")
    random.shuffle(train_rows)
    processor = AutoProcessor.from_pretrained(BASE_MODEL)
    chat_template = processor.tokenizer.chat_template
    if not chat_template:
        raise RuntimeError("base model has no chat template")
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to("cuda")
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"], bias="none"
    ))
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)

    def text_and_target(row):
        task = row["task"]
        prompt = row["question"].strip() + "\n"
        if task == "ground_bbox":
            answer = "[" + ", ".join(f"{v:.3f}" for v in row["answer_box_xyxy"]) + "]"
            prompt += "Answer with one normalized box [x1, y1, x2, y2], with three decimal places."
        elif task == "box_choice":
            prompt += "Candidate boxes (normalized x1,y1,x2,y2):\n"
            for item in row["candidates"]:
                box = ", ".join(f"{v:.3f}" for v in item["bbox_xyxy"])
                prompt += f"{item['choice']}: [{box}]\n"
            prompt += "Choices: " + ", ".join(row["choices"]) + ". Answer with one choice only."
            answer = row["choices"][row["answer_index"]]
        elif task == "spatial_boolean":
            prompt += "Choices: YES, NO, UNKNOWN. Answer with one choice only."
            answer = row["choices"][row["answer_index"]]
        elif task == "attribute_text":
            box = ", ".join(f"{v:.3f}" for v in row["region_box_xyxy"])
            prompt += f"Indicated box (normalized): [{box}]. Answer with the color only."
            answer = row["answer_text"]
        elif task == "relation_text":
            a = ", ".join(f"{v:.3f}" for v in row["subject_box_xyxy"])
            b = ", ".join(f"{v:.3f}" for v in row["object_box_xyxy"])
            prompt += f"Box 1: [{a}]. Box 2: [{b}]. Answer with a short relation only."
            answer = row["answer_text"]
        else:
            raise ValueError(task)
        return prompt, answer

    def prepare(row, with_answer=True):
        path = data_dir / row["image"]["path"]
        with Image.open(path) as source:
            photo = source.convert("RGB")
            photo.thumbnail((512, 512))
        prompt, answer = text_and_target(row)
        user = {"role": "user", "content": [{"type": "image", "image": photo}, {"type": "text", "text": prompt}]}
        messages = [user] + ([{"role": "assistant", "content": answer}] if with_answer else [])
        batch = processor.apply_chat_template(
            messages, chat_template=chat_template, tokenize=True,
            add_generation_prompt=not with_answer, return_dict=True, return_tensors="pt"
        )
        if with_answer:
            prefix = processor.apply_chat_template(
                [user], chat_template=chat_template, tokenize=True,
                add_generation_prompt=True, return_dict=True, return_tensors="pt"
            )["input_ids"]
            n = prefix.shape[1]
            if not torch.equal(batch["input_ids"][0, :n], prefix[0]):
                raise RuntimeError("chat template target alignment failed")
            labels = batch["input_ids"].clone()
            labels[:, :n] = -100
            batch["labels"] = labels
        return {key: value.to("cuda") if isinstance(value, torch.Tensor) else value for key, value in batch.items()}

    @torch.no_grad()
    def losses(rows, disable_adapter=False):
        model.eval()
        totals = defaultdict(list)
        context = model.disable_adapter() if disable_adapter else __import__("contextlib").nullcontext()
        with context:
            for row in rows:
                loss = model(**prepare(row)).loss.float().item()
                totals[row["task"]].append(loss)
        return {task: sum(values) / len(values) for task, values in totals.items()}

    baseline_val = losses(val_rows, disable_adapter=True)
    print("baseline_val_nll", baseline_val, flush=True)
    train_losses = []
    for step, row in enumerate(train_rows, 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = model(**prepare(row)).loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite loss at step {step}")
        loss.backward()
        optimizer.step()
        train_losses.append(loss.float().item())
        if step % 100 == 0 or step == 1:
            print(f"step={step}/{len(train_rows)} loss={train_losses[-1]:.4f} elapsed_s={time.time()-start:.1f}", flush=True)

    tuned_val = losses(val_rows)
    baseline_test = losses(test_rows, disable_adapter=True)
    tuned_test = losses(test_rows)
    predictions = []
    model.config.use_cache = True

    @torch.no_grad()
    def predict(row, disable_adapter=False):
        model.eval()
        batch = prepare(row, with_answer=False)
        context = model.disable_adapter() if disable_adapter else __import__("contextlib").nullcontext()
        with context:
            generated = model.generate(**batch, max_new_tokens=36, do_sample=False, pad_token_id=processor.tokenizer.eos_token_id)
        tail = generated[0, batch["input_ids"].shape[1]:]
        return processor.tokenizer.decode(tail, skip_special_tokens=True).strip()

    def score(row, prediction):
        task = row["task"]
        expected = text_and_target(row)[1]
        if task == "ground_bbox":
            numbers = re.findall(r"(?<!\w)(?:0(?:\.\d+)?|1(?:\.0+)?)(?!\w)", prediction)
            if len(numbers) < 4:
                return 0.0
            box = [float(x) for x in numbers[:4]]
            truth = row["answer_box_xyxy"]
            if not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1):
                return 0.0
            ix = max(0, min(box[2], truth[2]) - max(box[0], truth[0]))
            iy = max(0, min(box[3], truth[3]) - max(box[1], truth[1]))
            intersection = ix * iy
            area1 = (box[2]-box[0]) * (box[3]-box[1])
            area2 = (truth[2]-truth[0]) * (truth[3]-truth[1])
            return intersection / (area1 + area2 - intersection) if area1 + area2 > intersection else 0.0
        answer = prediction.strip().strip(". ").casefold()
        return float(answer == expected.casefold())

    scores = defaultdict(lambda: {"base": [], "adapter": []})
    for index, row in enumerate(test_rows, 1):
        base_pred = predict(row, disable_adapter=True)
        adapter_pred = predict(row)
        base_score, adapter_score = score(row, base_pred), score(row, adapter_pred)
        scores[row["task"]]["base"].append(base_score)
        scores[row["task"]]["adapter"].append(adapter_score)
        predictions.append({"id": row["id"], "task": row["task"],
                            "target": text_and_target(row)[1], "base_prediction": base_pred,
                            "adapter_prediction": adapter_pred, "base_score": base_score,
                            "adapter_score": adapter_score})
        if index % 50 == 0:
            print(f"benchmarked={index}/{len(test_rows)}", flush=True)

    metrics = {
        "base_model": BASE_MODEL, "gpu": torch.cuda.get_device_name(0),
        "train_records": len(train_rows), "validation_records": len(val_rows), "test_records": len(test_rows),
        "steps": len(train_rows), "learning_rate": 1e-4, "lora_rank": 16,
        "baseline_validation_nll": baseline_val, "adapter_validation_nll": tuned_val,
        "baseline_test_nll": baseline_test, "adapter_test_nll": tuned_test,
        "test_generation_scores": {
            task: {"n": len(values["base"]),
                   "base_mean": sum(values["base"])/len(values["base"]),
                   "adapter_mean": sum(values["adapter"])/len(values["adapter"])}
            for task, values in scores.items()
        },
        "score_definition": "ground_bbox: mean IoU; all other tasks: exact match",
        "first_100_train_loss_mean": sum(train_losses[:100])/min(100,len(train_losses)),
        "last_100_train_loss_mean": sum(train_losses[-100:])/min(100,len(train_losses)),
        "elapsed_seconds": time.time()-start,
        "peak_cuda_memory_gb": torch.cuda.max_memory_allocated()/1e9,
    }
    output = Path("/volume") / RUN_NAME
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output / "adapter")
    processor.save_pretrained(output / "processor")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (output / "predictions.jsonl").write_text("".join(json.dumps(p) + "\n" for p in predictions))
    (output / "train_losses.json").write_text(json.dumps(train_losses))
    volume.commit()
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics


@app.local_entrypoint()
def main():
    import json
    result = train_and_benchmark.remote()
    out = ROOT / "data" / "train_full_v1" / "metrics.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {out}")
