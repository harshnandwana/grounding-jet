"""Run a published Visual Jev LoRA on one local image and task question.

Install requirements-inference.txt, then try:
    python scripts/infer.py --image example.jpg --task spatial_boolean \
        --question "Is the person to the left of the backpack?"
"""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_MODEL = "harshnandwana/visual-jev-120k-qwen35-0.8b-lora"
BASE_MODEL = "Qwen/Qwen3.5-0.8B-Base"
TASKS = ("ground_bbox", "box_choice", "spatial_boolean", "attribute_text", "relation_text")


def box_text(values: list[float]) -> str:
    if len(values) != 4 or any(not 0 <= value <= 1 for value in values):
        raise ValueError("boxes must contain four normalized coordinates in [0, 1]")
    x1, y1, x2, y2 = values
    if x1 >= x2 or y1 >= y2:
        raise ValueError("box coordinates must satisfy x1 < x2 and y1 < y2")
    return "[" + ", ".join(f"{value:.3f}" for value in values) + "]"


def prompt_for(args: argparse.Namespace) -> str:
    prompt = args.question.strip() + "\n"
    if args.task == "ground_bbox":
        return prompt + "Answer with one normalized box [x1, y1, x2, y2], with three decimal places."
    if args.task == "spatial_boolean":
        return prompt + "Choices: YES, NO, UNKNOWN. Answer with one choice only."
    if args.task == "box_choice":
        if not args.candidate:
            raise ValueError("box_choice requires at least one --candidate LABEL X1 Y1 X2 Y2")
        labels = []
        prompt += "Candidate boxes (normalized x1,y1,x2,y2):\n"
        for raw in args.candidate:
            label, *coords = raw
            if label in labels:
                raise ValueError(f"duplicate candidate label: {label}")
            labels.append(label)
            prompt += f"{label}: {box_text([float(value) for value in coords])}\n"
        return prompt + "Choices: " + ", ".join(labels) + ". Answer with one choice only."
    if args.task == "attribute_text":
        if not args.region_box:
            raise ValueError("attribute_text requires --region-box X1 Y1 X2 Y2")
        return prompt + f"Indicated box (normalized): {box_text(args.region_box)}. Answer with the color only."
    if args.task == "relation_text":
        if not args.subject_box or not args.object_box:
            raise ValueError("relation_text requires --subject-box and --object-box")
        return (prompt + f"Box 1: {box_text(args.subject_box)}. "
                f"Box 2: {box_text(args.object_box)}. Answer with a short relation only.")
    raise ValueError(f"unsupported task: {args.task}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face LoRA repo ID")
    parser.add_argument("--image", type=Path, required=True, help="local photo path")
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--candidate", action="append", nargs=5, metavar=("LABEL", "X1", "Y1", "X2", "Y2"))
    parser.add_argument("--region-box", nargs=4, type=float, metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--subject-box", nargs=4, type=float, metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--object-box", nargs=4, type=float, metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f"image does not exist: {args.image}")
    try:
        question = prompt_for(args)
    except ValueError as error:
        parser.error(str(error))

    import torch
    from peft import PeftModel
    from PIL import Image
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(BASE_MODEL)
    base = Qwen3_5ForConditionalGeneration.from_pretrained(BASE_MODEL, dtype=dtype).to(device)
    model = PeftModel.from_pretrained(base, args.model).eval()
    with Image.open(args.image) as source:
        photo = source.convert("RGB")
        photo.thumbnail((512, 512))
    messages = [{"role": "user", "content": [
        {"type": "image", "image": photo}, {"type": "text", "text": question},
    ]}]
    inputs = processor.apply_chat_template(
        messages, chat_template=processor.tokenizer.chat_template,
        tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
    ).to(device)
    with torch.inference_mode():
        tokens = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    answer = processor.batch_decode(
        tokens[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
    )[0].strip()
    print(answer)


if __name__ == "__main__":
    main()
