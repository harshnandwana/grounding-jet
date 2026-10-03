"""Gradio demo for the published Visual Jev vision adapter."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import gradio as gr
from PIL import Image, ImageDraw


BASE_MODEL = "Qwen/Qwen3.5-0.8B-Base"
ADAPTER = os.getenv("VISUAL_JEV_ADAPTER", "harshnandwana/visual-jev-120k-qwen35-0.8b-lora")
TASKS = ("spatial_boolean", "ground_bbox", "box_choice", "attribute_text", "relation_text")


def parse_box(raw: str) -> str:
    try:
        values = [float(value.strip()) for value in raw.split(",")]
    except ValueError as error:
        raise gr.Error("Use four comma-separated numbers for each box.") from error
    if len(values) != 4 or any(not 0 <= value <= 1 for value in values):
        raise gr.Error("Each box needs four coordinates between 0 and 1.")
    x1, y1, x2, y2 = values
    if x1 >= x2 or y1 >= y2:
        raise gr.Error("Box coordinates must satisfy x1 < x2 and y1 < y2.")
    return "[" + ", ".join(f"{value:.3f}" for value in values) + "]"


def make_prompt(task: str, question: str, boxes: str) -> str:
    if not question or not question.strip():
        raise gr.Error("Enter a question.")
    prompt = question.strip() + "\n"
    if task == "spatial_boolean":
        return prompt + "Choices: YES, NO, UNKNOWN. Answer with one choice only."
    if task == "ground_bbox":
        return prompt + "Answer with one normalized box [x1, y1, x2, y2], with three decimal places."
    if task == "attribute_text":
        return prompt + f"Indicated box (normalized): {parse_box(boxes)}. Answer with the color only."
    if task == "relation_text":
        pieces = [part.strip() for part in boxes.split(";")]
        if len(pieces) != 2:
            raise gr.Error("Relation needs two boxes separated by a semicolon.")
        return (prompt + f"Box 1: {parse_box(pieces[0])}. "
                f"Box 2: {parse_box(pieces[1])}. Answer with a short relation only.")
    if task == "box_choice":
        lines = [line.strip() for line in boxes.splitlines() if line.strip()]
        if not lines:
            raise gr.Error("Add candidate boxes, one LABEL: x1,y1,x2,y2 per line.")
        labels = []
        prompt += "Candidate boxes (normalized x1,y1,x2,y2):\n"
        for line in lines:
            if ":" not in line:
                raise gr.Error("Use LABEL: x1,y1,x2,y2 for each candidate.")
            label, raw = (part.strip() for part in line.split(":", 1))
            if not label or label in labels:
                raise gr.Error("Candidate labels must be unique and nonempty.")
            labels.append(label)
            prompt += f"{label}: {parse_box(raw)}\n"
        return prompt + "Choices: " + ", ".join(labels) + ". Answer with one choice only."
    raise gr.Error("Choose a supported task.")


@lru_cache(maxsize=1)
def load_model():
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if device == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(BASE_MODEL)
    base = Qwen3_5ForConditionalGeneration.from_pretrained(BASE_MODEL, dtype=dtype).to(device)
    model = PeftModel.from_pretrained(base, ADAPTER).eval()
    return processor, model, device


def predict(image: Image.Image | None, task: str, question: str, boxes: str) -> str:
    if image is None:
        raise gr.Error("Upload an image or choose a synthetic example.")
    prompt = make_prompt(task, question, boxes)
    import torch

    processor, model, device = load_model()
    photo = image.convert("RGB")
    photo.thumbnail((512, 512))
    messages = [{"role": "user", "content": [
        {"type": "image", "image": photo}, {"type": "text", "text": prompt},
    ]}]
    inputs = processor.apply_chat_template(
        messages, chat_template=processor.tokenizer.chat_template,
        tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
    ).to(device)
    with torch.inference_mode():
        tokens = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    return processor.batch_decode(
        tokens[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
    )[0].strip()


def synthetic_examples() -> list[list[str]]:
    """Create simple diagrams at runtime; never copy photos into the Space."""
    folder = Path("/tmp/visual_jev_examples")
    folder.mkdir(exist_ok=True)
    image = Image.new("RGB", (512, 320), "#f6f7fb")
    draw = ImageDraw.Draw(image)
    draw.rectangle((60, 80, 210, 240), fill="#e94b4b", outline="#303746", width=3)
    draw.ellipse((315, 80, 465, 230), fill="#407be7", outline="#303746", width=3)
    draw.text((100, 255), "RED BOX", fill="#303746")
    draw.text((340, 255), "BLUE BALL", fill="#303746")
    path = folder / "shapes.png"
    image.save(path)
    return [
        [str(path), "spatial_boolean", "Is the red box to the left of the blue ball?", ""],
        [str(path), "ground_bbox", "Where is the blue ball?", ""],
        [str(path), "box_choice", "Which box contains the blue ball?", "A: 0.100,0.250,0.420,0.770\nB: 0.600,0.240,0.920,0.750"],
        [str(path), "attribute_text", "What color is the object in the indicated box?", "0.590,0.230,0.920,0.750"],
        [str(path), "relation_text", "What is the relation between the two objects?", "0.100,0.250,0.420,0.770; 0.590,0.230,0.920,0.750"],
    ]


with gr.Blocks(title="Visual Jev") as demo:
    gr.Markdown("# Visual Jev · image grounding demo\nChoose a synthetic preset or upload an image. Presets show how to format each task; outputs on these diagrams have not been benchmarked. First run downloads and loads the base model and adapter, so CPU inference may be slow.")
    with gr.Row():
        image = gr.Image(label="Image", type="pil", image_mode="RGB", sources=["upload", "webcam"])
        with gr.Column():
            task = gr.Dropdown(choices=list(TASKS), value="spatial_boolean", label="Task")
            question = gr.Textbox(label="Question", value="Is the red box to the left of the blue ball?")
            boxes = gr.Textbox(label="Boxes (normalized coordinates, if needed)", lines=3,
                               info="Color: one x1,y1,x2,y2 box. Relation: two boxes separated by ;. Box choice: one LABEL: x1,y1,x2,y2 per line.")
            run = gr.Button("Run model", variant="primary")
            output = gr.Textbox(label="Model answer", interactive=False)
    gr.Examples(examples=synthetic_examples(), inputs=[image, task, question, boxes],
                example_labels=["Left/right", "Find ball", "Choose box", "Color", "Relation"],
                cache_examples=False, run_on_click=False)
    run.click(predict, inputs=[image, task, question, boxes], outputs=output, api_name="predict")
    gr.Markdown("[Model card](https://huggingface.co/harshnandwana/visual-jev-budget20-qwen35-0.8b-lora) · [Source code and local CLI](https://github.com/harshnandwana/grounding-jet)")


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch()
