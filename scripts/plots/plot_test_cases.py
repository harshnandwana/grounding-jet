"""Render real held-out prediction cases without distributing COCO photos.

Run from the repository root after installing requirements.txt:
    python scripts/plots/plot_test_cases.py

The script uses local JSONL files when present and otherwise downloads the
published text-only predictions and annotations from Hugging Face.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
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


def card(draw: ImageDraw.ImageDraw, x: int, y: int, row: dict, question: str,
         accent: str, tint: str) -> None:
    width, height = 1010, 356
    draw.rounded_rectangle((x, y, x + width, y + height), radius=24,
                           fill="white", outline=BORDER, width=2)
    draw.rounded_rectangle((x + 25, y + 23, x + 263, y + 65), radius=15, fill=tint)
    task = row["task"].replace("_", " ").title()
    draw.text((x + 43, y + 29), task, font=font(23, True), fill=accent)
    identifier = row["id"]
    draw.text((x + 980, y + 34), identifier, font=font(18), fill=MUTED, anchor="ra")
    draw.text((x + 29, y + 89), clipped(draw, question, font(27, True), 950),
              font=font(27, True), fill=INK)
    fields = (
        ("TARGET", row["target"], INK),
        ("BASE", row["base_prediction"], MUTED),
        ("ADAPTER", row["adapter_prediction"], accent),
    )
    for index, (label, value, color) in enumerate(fields):
        top = y + 144 + index * 57
        draw.text((x + 31, top), label, font=font(19, True), fill=MUTED)
        draw.text((x + 192, top - 2), clipped(draw, str(value), font(23, index == 2), 780),
                  font=font(23, index == 2), fill=color)
    metric = "IoU" if row["task"] == "ground_bbox" else "exact match"
    score = f"{metric}: base {row['base_score']:.2f}   |   adapter {row['adapter_score']:.2f}"
    draw.text((x + 31, y + 315), score, font=font(18), fill=MUTED)


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
        selected.append({"category": actual, "question": annotation["question"], **row})

    canvas = Image.new("RGB", (2200, 1730), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    draw.text((74, 54), "When the adapter helps — and when it still misses",
              font=font(55, True), fill=INK)
    draw.text((76, 127), "Six real cases from the 1,000-record selected held-out test split",
              font=font(29), fill=MUTED)
    draw.rounded_rectangle((70, 194, 1080, 278), radius=22, fill=TEAL_PALE)
    draw.rounded_rectangle((1120, 194, 2130, 278), radius=22, fill=CORAL_PALE)
    draw.text((99, 216), f"ADAPTER CORRECT · BASE WRONG   {counts['adapter_only']} / 1,000",
              font=font(27, True), fill=TEAL)
    draw.text((1149, 216), f"BOTH WRONG   {counts['both_failed']} / 1,000",
              font=font(27, True), fill=CORAL)
    for index in range(3):
        top = 311 + index * 379
        left = selected[index]
        right = selected[index + 3]
        card(draw, 70, top, left, left["question"], TEAL, TEAL_PALE)
        card(draw, 1120, top, right, right["question"], CORAL, CORAL_PALE)
    draw.line((77, 1487, 2123, 1487), fill=BORDER, width=2)
    notes = (
        "Scoring: exact match for text and choice tasks; grounding counted as correct here when IoU ≥ 0.50.",
        "Long raw outputs are shortened on the cards; the companion JSON retains every full prediction.",
        "A verbose or differently formatted answer can score 0 even when it contains the right word. No photos are included.",
        f"Sources: {DATASET_REPO} @ {DATASET_REVISION[:12]} · {MODEL_REPO}",
    )
    for index, note in enumerate(notes):
        draw.text((78, 1514 + index * 38), note, font=font(20), fill=MUTED)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUTPUT, optimize=True)
    EXAMPLES.write_text(json.dumps({
        "model_repo": MODEL_REPO,
        "dataset_repo": DATASET_REPO,
        "dataset_revision": DATASET_REVISION,
        "test_records": len(predictions),
        "classification": "IoU >= 0.5 for ground_bbox; score == 1 for other tasks",
        "counts": dict(counts),
        "examples": selected,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT} and {EXAMPLES}")


if __name__ == "__main__":
    main()
