"""Render a GitHub-only composite of real held-out cases with COCO photos.

Run from the repository root after installing requirements.txt:
    python scripts/plots/plot_test_cases.py

The script uses local JSONL files when present and otherwise downloads the
published text-only predictions and annotations from Hugging Face.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps
from huggingface_hub import hf_hub_download

from visual_jev.hf import DATASET_REPO, DATASET_REVISION


ROOT = Path(__file__).resolve().parents[2]
MODEL_REPO = "harshnandwana/visual-jev-budget20-qwen35-0.8b-lora"
OUTPUT = ROOT / "results" / "test_case_comparison.png"
EXAMPLES = ROOT / "results" / "test_case_examples.json"
SELECTED = (
    ("adapter_only", "coco2017-val2017-76547-spatial-188853-1166503"),
    ("adapter_only", "coco2017-val2017-576566-ground-451097"),
    ("adapter_only", "coco2017-val2017-64718-locate-656133"),
    ("both_failed", "coco2017-val2017-30828-spatial-2042941-461443"),
    ("both_failed", "vg-2356182-attribute-3563557-brown"),
    ("both_failed", "vg-2353066-relation-4235104"),
)

BACKGROUND = "#F3F6FA"
INK = "#14213B"
MUTED = "#5C6D83"
TEAL = "#0F766E"
TEAL_PALE = "#E1F4EF"
CORAL = "#B64C46"
CORAL_PALE = "#FFF0ED"
BORDER = "#DBE3EC"
BLUE = "#4279DA"
YELLOW = "#FFB443"
PURPLE = "#B762EA"


def wrap(draw: ImageDraw.ImageDraw, value: str, face, width: int,
         max_lines: int = 2) -> list[str]:
    words = " / ".join(part.strip() for part in str(value).splitlines() if part.strip()).split()
    lines: list[str] = []
    current = ""
    for word in words:
        trial = f"{current} {word}".strip()
        if current and draw.textlength(trial, font=face) > width:
            lines.append(current)
            current = word
        else:
            current = trial
    if current:
        lines.append(current)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = clipped(draw, lines[-1] + "…", face, width)
    return lines


def photo(canvas: Image.Image, x: int, y: int, annotation: dict,
          row: dict, width: int = 455, height: int = 368) -> None:
    path = ROOT / annotation["image"]["path"]
    if not path.is_file():
        raise FileNotFoundError(f"COCO photo required for the composite: {path}")
    with Image.open(path) as original:
        fitted = ImageOps.contain(original.convert("RGB"), (width, height), Image.Resampling.LANCZOS)
    px = x + (width - fitted.width) // 2
    py = y + (height - fitted.height) // 2
    canvas.paste(fitted, (px, py))
    d = ImageDraw.Draw(canvas)
    boxes: list[tuple[str, list[float], str]] = []
    task = row["task"]
    if task == "box_choice":
        boxes = [(candidate["choice"], candidate["bbox_xyxy"], color)
                 for candidate, color in zip(annotation["candidates"], (BLUE, YELLOW, PURPLE))]
    elif task == "ground_bbox":
        boxes.append(("Target", annotation["answer_box_xyxy"], TEAL))
        # The base response uses pixel coordinates outside [0, 1], so it is not a valid overlay.
        try:
            answer = json.loads(row["adapter_prediction"])
            if len(answer) == 4 and all(0 <= float(v) <= 1 for v in answer):
                boxes.append(("Adapter", answer, BLUE))
        except (ValueError, TypeError):
            pass
    elif task == "attribute_text":
        boxes = [("Coat", annotation["region_box_xyxy"], YELLOW)]
    else:
        boxes = [(str(index), box, color)
                 for index, (box, color) in enumerate(
                     zip(annotation["evidence"]["boxes_xyxy"], (BLUE, CORAL)), start=1)]
    for label, box, color in boxes:
        x1, y1, x2, y2 = box
        rect = (round(px + x1 * fitted.width), round(py + y1 * fitted.height),
                round(px + x2 * fitted.width), round(py + y2 * fitted.height))
        d.rectangle(rect, outline=color, width=5)
        label_width = round(d.textlength(label, font=font(19, True))) + 18
        label_y = (min(py + fitted.height - 28, rect[3] + 3)
                   if task == "ground_bbox" and label == "Adapter"
                   else max(py, rect[1] - 29))
        d.rounded_rectangle((rect[0], label_y, rect[0] + label_width, label_y + 28),
                            radius=5, fill=color)
        d.text((rect[0] + 8, label_y + 3), label, font=font(19, True), fill="white")


def answer_line(draw: ImageDraw.ImageDraw, x: int, y: int, label: str,
                value: str, color: str, width: int) -> None:
    draw.text((x, y), label, font=font(21, True), fill=MUTED)
    for index, line in enumerate(wrap(draw, value, font(23, label == "ADAPTER"), width - 125, 2)):
        draw.text((x + 125, y + index * 28), line,
                  font=font(23, label == "ADAPTER"), fill=color)


def source_file(local: Path, repo: str, filename: str, revision: str | None = None,
                repo_type: str = "model") -> Path:
    if local.is_file():
        return local
    return Path(hf_hub_download(repo_id=repo, filename=filename,
                                revision=revision, repo_type=repo_type))


def category(row: dict) -> str:
    threshold = 0.5 if row["task"] == "ground_bbox" else 1.0
    base = row["base_score"] >= threshold
    adapter = row["adapter_score"] >= threshold
    if base and adapter:
        return "both_correct"
    if adapter:
        return "adapter_only"
    if base:
        return "base_only"
    return "both_failed"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = (
        Path("/System/Library/Fonts/Supplemental") / ("Arial Bold.ttf" if bold else "Arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu") / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"),
    )
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size=size)


def clipped(draw: ImageDraw.ImageDraw, value: str, face, width: int) -> str:
    display = " / ".join(part.strip() for part in value.splitlines() if part.strip())
    if draw.textlength(display, font=face) <= width:
        return display
    while display and draw.textlength(display + "…", font=face) > width:
        display = display[:-1]
    return display.rstrip() + "…"


def card(canvas: Image.Image, x: int, y: int, row: dict, annotation: dict,
         accent: str, tint: str) -> None:
    draw = ImageDraw.Draw(canvas)
    width, height = 1190, 490
    draw.rounded_rectangle((x, y, x + width, y + height), radius=24,
                           fill="white", outline=BORDER, width=2)
    draw.rounded_rectangle((x + 25, y + 23, x + 278, y + 67), radius=15, fill=tint)
    task = row["task"].replace("_", " ").title()
    draw.text((x + 43, y + 30), task, font=font(23, True), fill=accent)
    draw.rounded_rectangle((x + 26, y + 84, x + 481, y + 452), radius=11, fill=BACKGROUND)
    photo(canvas, x + 26, y + 84, annotation, row)
    tx = x + 510
    for index, line in enumerate(wrap(draw, annotation["question"], font(27, True), 638, 3)):
        draw.text((tx, y + 90 + index * 34), line, font=font(27, True), fill=INK)
    answer_line(draw, tx, y + 225, "TARGET", row["target"], INK, 645)
    answer_line(draw, tx, y + 278, "BASE", row["base_prediction"], MUTED, 645)
    answer_line(draw, tx, y + 359, "ADAPTER", row["adapter_prediction"], accent, 645)
    metric = "IoU" if row["task"] == "ground_bbox" else "exact match"
    score = f"{metric}: base {row['base_score']:.2f}   |   adapter {row['adapter_score']:.2f}"
    draw.text((tx, y + 445), score, font=font(19), fill=MUTED)


def main() -> None:
    predictions_path = source_file(ROOT / "data" / "budget20" / "predictions.jsonl",
                                   MODEL_REPO, "predictions.jsonl")
    questions_path = source_file(ROOT / "data" / "hf_release" / "test.jsonl",
                                 DATASET_REPO, "test.jsonl", DATASET_REVISION, "dataset")
    predictions = [json.loads(line) for line in predictions_path.open(encoding="utf-8")]
    if len(predictions) != 1000:
        raise ValueError("expected all 1,000 selected held-out predictions")
    counts = Counter(category(row) for row in predictions)
    by_id = {row["id"]: row for row in predictions}
    if len(by_id) != len(predictions):
        raise ValueError("duplicate prediction IDs")
    wanted = {identifier for _, identifier in SELECTED}
    annotations = {}
    with questions_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["id"] in wanted:
                annotations[row["id"]] = row
    if set(annotations) != wanted:
        raise ValueError("selected annotations missing from pinned dataset test split")
    selected = []
    for expected, identifier in SELECTED:
        row = by_id[identifier]
        annotation = annotations[identifier]
        actual = category(row)
        if actual != expected:
            raise ValueError(f"classification changed for {identifier}: {actual}")
        if row["task"] == "ground_bbox":
            target = "[" + ", ".join(f"{value:.3f}" for value in annotation["answer_box_xyxy"]) + "]"
        elif row["task"] in ("box_choice", "spatial_boolean"):
            target = annotation["choices"][annotation["answer_index"]]
        else:
            target = annotation["answer_text"]
        if target != row["target"]:
            raise ValueError(f"published target differs from pinned annotation: {identifier}")
        selected.append(({"category": actual, "question": annotation["question"], **row}, annotation))

    canvas = Image.new("RGB", (2550, 2140), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    draw.text((75, 53), "VISUAL JEV / HELD-OUT CASES", font=font(55, True), fill=INK)
    draw.text((78, 123), "Qwen3.5-0.8B-Base + LoRA  ·  40,000 training records  ·  two L4 GPUs",
              font=font(28), fill=MUTED)
    draw.rounded_rectangle((72, 190, 2478, 370), radius=24, fill="white")
    draw.text((106, 219), "Selected test split: 1,000 cases", font=font(30, True), fill=INK)
    draw.text((107, 279), f"{counts['adapter_only']}", font=font(51, True), fill=TEAL)
    draw.text((220, 298), "adapter correct · base wrong", font=font(25), fill=MUTED)
    draw.text((975, 279), f"{counts['both_failed']}", font=font(51, True), fill=CORAL)
    draw.text((1080, 298), "both wrong", font=font(25), fill=MUTED)
    draw.text((1650, 286), f"{counts['both_correct']} both correct  ·  {counts['base_only']} base only",
              font=font(25), fill=MUTED)
    draw.text((80, 407), "Adapter recovered", font=font(36, True), fill=TEAL)
    draw.text((1340, 407), "Both models missed", font=font(36, True), fill=CORAL)
    for index in range(3):
        top = 464 + index * 512
        left_row, left_annotation = selected[index]
        right_row, right_annotation = selected[index + 3]
        card(canvas, 75, top, left_row, left_annotation, TEAL, TEAL_PALE)
        card(canvas, 1285, top, right_row, right_annotation, CORAL, CORAL_PALE)
    draw.line((81, 2023, 2468, 2023), fill=BORDER, width=2)
    notes = (
        "Exact match for text and choice tasks; grounding is counted as correct when IoU ≥ 0.50. Long responses are shortened; JSON has the full outputs.",
        "COCO photo thumbnails appear in this GitHub figure only; the Hugging Face dataset and model repositories remain photo-free.",
    )
    for index, note in enumerate(notes):
        draw.text((83, 2045 + index * 35), note, font=font(20), fill=MUTED)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUTPUT, optimize=True)
    EXAMPLES.write_text(json.dumps({
        "model_repo": MODEL_REPO,
        "dataset_repo": DATASET_REPO,
        "dataset_revision": DATASET_REVISION,
        "test_records": len(predictions),
        "classification": "IoU >= 0.5 for ground_bbox; score == 1 for other tasks",
        "counts": dict(counts),
        "examples": [dict(row, image_source=annotation["image"]["url"])
                     for row, annotation in selected],
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT} and {EXAMPLES}")


if __name__ == "__main__":
    main()
