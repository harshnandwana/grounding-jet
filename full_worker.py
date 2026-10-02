"""Distributed full-dataset Qwen3.5 visual LoRA training and evaluation worker."""

from __future__ import annotations

from collections import defaultdict
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import random
import re
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from PIL import Image
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration


BASE_MODEL = "Qwen/Qwen3.5-0.8B-Base"
TASKS = ("ground_bbox", "box_choice", "spatial_boolean", "attribute_text", "relation_text")
DATA_DIR = Path(os.environ.get("VISUAL_JEV_DATA_DIR", "/tmp/visual_jev_full"))
OUTPUT_DIR = Path(os.environ.get("VISUAL_JEV_OUTPUT_DIR", "/volume/full_run"))
ACCUMULATION = int(os.environ.get("VISUAL_JEV_ACCUMULATION", "8"))
LEARNING_RATE = 5e-5
SEED = 43801


def load_rank_rows(split: str, rank: int, world: int) -> list[dict]:
    result = []
    with (DATA_DIR / f"{split}.jsonl").open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index % world == rank:
                result.append(json.loads(line))
    return result


def text_and_target(row: dict) -> tuple[str, str]:
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


def prepare(row: dict, processor, device, labeled: bool = True) -> dict:
    path = DATA_DIR / row["image"]["path"]
    with Image.open(path) as source:
        photo = source.convert("RGB")
        photo.thumbnail((512, 512))
    prompt, answer = text_and_target(row)
    user = {"role": "user", "content": [{"type": "image", "image": photo}, {"type": "text", "text": prompt}]}
    messages = [user] + ([{"role": "assistant", "content": answer}] if labeled else [])
    batch = processor.apply_chat_template(
        messages, chat_template=processor.tokenizer.chat_template,
        tokenize=True, add_generation_prompt=not labeled,
        return_dict=True, return_tensors="pt",
    )
    if labeled:
        prefix = processor.apply_chat_template(
            [user], chat_template=processor.tokenizer.chat_template,
            tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        )["input_ids"]
        n = prefix.shape[1]
        if not torch.equal(batch["input_ids"][0, :n], prefix[0]):
            raise RuntimeError(f"target alignment failed: {row['id']}")
        labels = batch["input_ids"].clone()
        labels[:, :n] = -100
        batch["labels"] = labels
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()}


@torch.no_grad()
def full_loss(model, processor, rows, device, base: bool) -> dict:
    model.eval()
    totals = torch.zeros((len(TASKS), 2), device=device, dtype=torch.float64)
    context = model.disable_adapter() if base else nullcontext()
    with context:
        for index, row in enumerate(rows, 1):
            loss = model(**prepare(row, processor, device)).loss.float().item()
            task_index = TASKS.index(row["task"])
            totals[task_index, 0] += loss
            totals[task_index, 1] += 1
            if index % 1000 == 0 and dist.get_rank() == 0:
                print(f"{'base' if base else 'adapter'}_loss_evaluated={index}/{len(rows)}", flush=True)
    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    return {task: {"n": int(totals[i, 1].item()),
                   "mean_nll": (totals[i, 0] / totals[i, 1]).item()}
            for i, task in enumerate(TASKS)}


def score(row: dict, prediction: str) -> float:
    if row["task"] == "ground_bbox":
        numbers = re.findall(r"(?<!\w)(?:0(?:\.\d+)?|1(?:\.0+)?)(?!\w)", prediction)
        if len(numbers) < 4:
            return 0.0
        box = [float(x) for x in numbers[:4]]
        truth = row["answer_box_xyxy"]
        if not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1):
            return 0.0
        intersection = max(0, min(box[2], truth[2]) - max(box[0], truth[0])) * max(
            0, min(box[3], truth[3]) - max(box[1], truth[1]))
        area_a = (box[2] - box[0]) * (box[3] - box[1])
        area_b = (truth[2] - truth[0]) * (truth[3] - truth[1])
        return intersection / (area_a + area_b - intersection) if area_a + area_b > intersection else 0.0
    expected = text_and_target(row)[1]
    return float(prediction.strip().strip(". ").casefold() == expected.casefold())


@torch.no_grad()
def generate_one(model, processor, row, device, base: bool) -> str:
    model.eval()
    batch = prepare(row, processor, device, labeled=False)
    limit = 40 if row["task"] == "ground_bbox" else (8 if row["task"] in ("box_choice", "spatial_boolean") else 16)
    context = model.disable_adapter() if base else nullcontext()
    with context:
        generated = model.generate(
            **batch, max_new_tokens=limit, do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
    tail = generated[0, batch["input_ids"].shape[1]:]
    return processor.tokenizer.decode(tail, skip_special_tokens=True).strip()


def full_generation_benchmark(model, processor, rows, device, rank) -> dict:
    model.config.use_cache = True
    totals = torch.zeros((len(TASKS), 3), device=device, dtype=torch.float64)
    path = Path(f"/tmp/full_predictions_rank_{rank}.jsonl")
    with path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows, 1):
            base_prediction = generate_one(model, processor, row, device, base=True)
            adapter_prediction = generate_one(model, processor, row, device, base=False)
            base_score = score(row, base_prediction)
            adapter_score = score(row, adapter_prediction)
            at = TASKS.index(row["task"])
            totals[at, 0] += base_score
            totals[at, 1] += adapter_score
            totals[at, 2] += 1
            handle.write(json.dumps({
                "id": row["id"], "task": row["task"], "target": text_and_target(row)[1],
                "base_prediction": base_prediction, "adapter_prediction": adapter_prediction,
                "base_score": base_score, "adapter_score": adapter_score,
            }, ensure_ascii=False) + "\n")
            if index % 200 == 0 and rank == 0:
                print(f"test_generated={index}/{len(rows)} per rank", flush=True)
    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    return {task: {"n": int(totals[i, 2].item()),
                   "base_mean": (totals[i, 0] / totals[i, 2]).item(),
                   "adapter_mean": (totals[i, 1] / totals[i, 2]).item()}
            for i, task in enumerate(TASKS)}


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl")
    torch.manual_seed(SEED + rank)
    random.seed(SEED + rank)
    start = time.time()
    manifest = json.loads((DATA_DIR / "manifest.json").read_text())
    train_total = manifest["splits"]["train"]["records"]
    val_total = manifest["splits"]["validation"]["records"]
    test_total = manifest["splits"]["test"]["records"]
    train_rows = load_rank_rows("train", rank, world)
    val_rows = load_rank_rows("validation", rank, world)
    test_rows = load_rank_rows("test", rank, world)
    smoke_limit = int(os.environ.get("VISUAL_JEV_SMOKE_LIMIT", "0"))
    if smoke_limit:
        train_rows = train_rows[:smoke_limit]
        val_rows = val_rows[:smoke_limit]
        test_rows = test_rows[:smoke_limit]
        train_total = len(train_rows) * world
        val_total = len(val_rows) * world
        test_total = len(test_rows) * world
    if not train_rows:
        raise RuntimeError("empty train partition")
    if rank == 0:
        print(f"world={world} full_train={train_total} full_validation={val_total} full_test={test_total}", flush=True)
    processor = AutoProcessor.from_pretrained(BASE_MODEL)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"], bias="none",
    ))
    if rank == 0:
        print(f"gpu_name={torch.cuda.get_device_name(device)} "
              f"gpu_total_gb={torch.cuda.get_device_properties(device).total_memory / 1e9:.2f}", flush=True)
    ddp = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    optimizer = torch.optim.AdamW((p for p in ddp.parameters() if p.requires_grad), lr=LEARNING_RATE)
    random.Random(SEED + rank).shuffle(train_rows)
    rows_per_rank = math.ceil(train_total / world)
    padded_rows = math.ceil(rows_per_rank / ACCUMULATION) * ACCUMULATION
    if len(train_rows) < padded_rows:
        train_rows.extend(train_rows[:padded_rows - len(train_rows)])
    if len(train_rows) != padded_rows:
        raise RuntimeError("rank training row alignment failed")
    train_sum = 0.0
    optimizer.zero_grad(set_to_none=True)
    if rank == 0:
        print("phase=training", flush=True)
    for index, row in enumerate(train_rows, 1):
        ddp.train()
        sync = index % ACCUMULATION == 0
        context = nullcontext() if sync else ddp.no_sync()
        with context:
            loss = ddp(**prepare(row, processor, device)).loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite loss rank={rank} row={index}")
            (loss / ACCUMULATION).backward()
        train_sum += loss.float().item()
        if sync:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step = index // ACCUMULATION
            if step == 1:
                torch.cuda.synchronize(device)
                print(f"rank={rank} first_step_allocated_gb="
                      f"{torch.cuda.max_memory_allocated(device) / 1e9:.2f} "
                      f"first_step_reserved_gb={torch.cuda.max_memory_reserved(device) / 1e9:.2f}",
                      flush=True)
            if rank == 0 and (step == 1 or step % 500 == 0):
                print(f"optimizer_step={step}/{padded_rows // ACCUMULATION} elapsed_s={time.time()-start:.0f}", flush=True)
            if step % 2000 == 0:
                dist.barrier()
                if rank == 0:
                    checkpoint = OUTPUT_DIR / "checkpoints" / f"step-{step:05d}"
                    checkpoint.mkdir(parents=True, exist_ok=True)
                    ddp.module.save_pretrained(checkpoint)
                    print(f"saved_checkpoint={step}", flush=True)
                dist.barrier()
    train_stats = torch.tensor([train_sum, len(train_rows)], device=device, dtype=torch.float64)
    dist.all_reduce(train_stats, op=dist.ReduceOp.SUM)
    memory_stats = torch.tensor([
        torch.cuda.max_memory_allocated(device),
        torch.cuda.max_memory_reserved(device),
        torch.cuda.get_device_properties(device).total_memory,
    ], device=device, dtype=torch.float64)
    dist.all_reduce(memory_stats, op=dist.ReduceOp.MAX)
    if rank == 0:
        print("phase=baseline_validation", flush=True)
    baseline_validation = full_loss(model, processor, val_rows, device, base=True)
    if rank == 0:
        print("full_baseline_validation_nll", baseline_validation, flush=True)
        print("phase=adapter_validation", flush=True)
    tuned_validation = full_loss(model, processor, val_rows, device, base=False)
    if rank == 0:
        print("full_adapter_validation_nll", tuned_validation, flush=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(OUTPUT_DIR / "adapter")
        processor.save_pretrained(OUTPUT_DIR / "processor")
        (OUTPUT_DIR / "validation_interim.json").write_text(json.dumps({
            "dataset_revision": os.environ.get("VISUAL_JEV_DATASET_REVISION"),
            "baseline_validation_nll": baseline_validation,
            "adapter_validation_nll": tuned_validation,
        }, indent=2) + "\n")
    dist.barrier()
    test_scores = full_generation_benchmark(model, processor, test_rows, device, rank)
    dist.barrier()
    if rank == 0:
        with (OUTPUT_DIR / "predictions.jsonl").open("w", encoding="utf-8") as target:
            for other_rank in range(world):
                target.write(Path(f"/tmp/full_predictions_rank_{other_rank}.jsonl").read_text())
        metrics = {
            "base_model": BASE_MODEL,
            "dataset_repo": os.environ.get("VISUAL_JEV_DATASET_REPO"),
            "dataset_revision": os.environ.get("VISUAL_JEV_DATASET_REVISION"),
            "gpu": torch.cuda.get_device_name(local_rank), "gpu_count": world,
            "train_records_unique": train_total, "train_records_processed_including_padding": int(train_stats[1].item()),
            "validation_records": val_total, "test_records": test_total,
            "epochs": 1, "optimizer_steps": padded_rows // ACCUMULATION,
            "gradient_accumulation_per_gpu": ACCUMULATION,
            "learning_rate": LEARNING_RATE, "lora_rank": 16,
            "mean_train_nll": (train_stats[0] / train_stats[1]).item(),
            "baseline_full_validation_nll": baseline_validation,
            "adapter_full_validation_nll": tuned_validation,
            "full_test_generation_scores": test_scores,
            "score_definition": "ground_bbox: mean IoU; all other tasks: greedy-generation exact match",
            "elapsed_seconds": time.time() - start,
            "peak_cuda_memory_gb": memory_stats[0].item() / 1e9,
            "peak_cuda_reserved_gb": memory_stats[1].item() / 1e9,
            "gpu_total_memory_gb": memory_stats[2].item() / 1e9,
        }
        (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        print(json.dumps(metrics, indent=2), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
