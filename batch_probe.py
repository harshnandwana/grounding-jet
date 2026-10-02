"""Measure safe visual microbatch sizes on one L4 before paid training."""

from __future__ import annotations

import json
from pathlib import Path
import tarfile
import time

from PIL import Image
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from full_worker import text_and_target


ROOT = Path("/tmp/visual_jev_batch_probe")


def prepare_batch(rows: list[dict], processor, device) -> dict:
    conversations = []
    users = []
    for row in rows:
        with Image.open(ROOT / row["image"]["path"]) as source:
            photo = source.convert("RGB")
            photo.thumbnail((512, 512))
        prompt, answer = text_and_target(row)
        user = {"role": "user", "content": [{"type": "image", "image": photo},
                                            {"type": "text", "text": prompt}]}
        users.append(user)
        conversations.append([user, {"role": "assistant", "content": answer}])
    batch = processor.apply_chat_template(
        conversations, chat_template=processor.tokenizer.chat_template,
        tokenize=True, add_generation_prompt=False,
        processor_kwargs={"padding": True},
        return_dict=True, return_tensors="pt",
    )
    labels = batch["input_ids"].clone()
    labels[batch["attention_mask"] == 0] = -100
    for index, user in enumerate(users):
        prefix = processor.apply_chat_template(
            [user], chat_template=processor.tokenizer.chat_template,
            tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        )["input_ids"][0]
        length = prefix.shape[0]
        if not torch.equal(batch["input_ids"][index, :length], prefix):
            raise RuntimeError(f"target alignment failed for {rows[index]['id']}")
        labels[index, :length] = -100
    batch["labels"] = labels
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()}


def main() -> None:
    ROOT.mkdir(exist_ok=True)
    with tarfile.open("/volume/bundle.tar") as archive:
        archive.extractall(ROOT)
    with (ROOT / "train.jsonl").open(encoding="utf-8") as handle:
        rows = [json.loads(next(handle)) for _ in range(16)]
    device = torch.device("cuda:0")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-0.8B-Base")
    processor.tokenizer.padding_side = "right"
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3.5-0.8B-Base", dtype=torch.bfloat16,
        attn_implementation="sdpa").to(device)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"], bias="none",
    ))
    print(f"gpu={torch.cuda.get_device_name(device)} total_gb="
          f"{torch.cuda.get_device_properties(device).total_memory/1e9:.2f}", flush=True)
    for size in (1, 2, 4, 8):
        model.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        batch = prepare_batch(rows[:size], processor, device)
        loss = model(**batch).loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite batch loss at size {size}")
        loss.backward()
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        print(json.dumps({"microbatch": size, "seconds": elapsed,
                          "examples_per_second": size / elapsed,
                          "peak_allocated_gb": torch.cuda.max_memory_allocated(device) / 1e9,
                          "peak_reserved_gb": torch.cuda.max_memory_reserved(device) / 1e9,
                          "loss": float(loss.item())}), flush=True)


if __name__ == "__main__":
    main()
