"""Show five published test records derived from one COCO photo.

Run from the repository root with local COCO test photos and test.jsonl:
    python scripts/plots/plot_one_image_dataset.py
The generated composite is for GitHub. Do not upload it to Hugging Face.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[2]
TEST = ROOT / "data/hf_release/test.jsonl"
OUTPUT = ROOT / "results/one_image_dataset.png"
EXAMPLES = ROOT / "results/one_image_dataset.json"
IMAGE_ID = "76547"
IDS = (
    "coco2017-val2017-76547-locate-125428",
    "coco2017-val2017-76547-ground-125428",
    "coco2017-val2017-76547-spatial-125428-188853",
    "vg-2344605-attribute-2767180-black",
    "vg-2344605-relation-3884368",
)
COLORS = {"ground_bbox": "#0F766E", "box_choice": "#426ED5",
          "spatial_boolean": "#C46B2C", "attribute_text": "#9B4DC8",
          "relation_text": "#BA4F50"}
INK, MUTED, BG = "#14213B", "#596A80", "#F3F6FA"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    names = ("Arial Bold.ttf" if bold else "Arial.ttf",)
    for name in names:
        path = Path("/System/Library/Fonts/Supplemental") / name
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    path = Path("/usr/share/fonts/truetype/dejavu") / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size)


def fit_lines(draw: ImageDraw.ImageDraw, value: str, width: int, size: int,
              max_lines: int = 3) -> list[str]:
    words = value.split()
    lines, current = [], ""
    for word in words:
        proposed = f"{current} {word}".strip()
        if current and draw.textlength(proposed, font=font(size)) > width:
            lines.append(current)
            current = word
        else:
            current = proposed
    if current:
        lines.append(current)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] += "…"
    return lines


def boxes(row: dict) -> list[tuple[str, list[float], str]]:
    task = row["task"]
    if task == "ground_bbox":
        return [("Bicycle", row["answer_box_xyxy"], COLORS[task])]
    if task == "box_choice":
        return [(candidate["choice"], candidate["bbox_xyxy"], COLORS[task])
                for candidate in row["candidates"]]
    if task == "attribute_text":
        return [("Backpack", row["region_box_xyxy"], COLORS[task])]
    return [(str(i), box, COLORS[task])
            for i, box in enumerate(row["evidence"]["boxes_xyxy"], 1)]


def image_panel(canvas: Image.Image, source: Image.Image, x: int, y: int,
                width: int, height: int, outlines: list[tuple[str, list[float], str]]) -> None:
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((x, y, x + width, y + height), radius=18, fill="#E7ECF3")
    fitted = ImageOps.contain(source, (width, height), Image.Resampling.LANCZOS)
    px, py = x + (width - fitted.width) // 2, y + (height - fitted.height) // 2
    canvas.paste(fitted, (px, py))
    draw = ImageDraw.Draw(canvas)
    for label, (x1, y1, x2, y2), color in outlines:
        rect = (round(px+x1*fitted.width), round(py+y1*fitted.height),
                round(px+x2*fitted.width), round(py+y2*fitted.height))
        draw.rectangle(rect, outline=color, width=5)
        label_width = round(draw.textlength(label, font=font(20, True))) + 14
        ly = max(py, rect[1]-29)
        draw.rounded_rectangle((rect[0], ly, rect[0]+label_width, ly+28), radius=5, fill=color)
        draw.text((rect[0]+7, ly+3), label, font=font(20, True), fill="white")


def answer(row: dict) -> str:
    if row["task"] == "ground_bbox":
        return "[" + ", ".join(f"{v:.3f}" for v in row["answer_box_xyxy"]) + "]"
    if row["task"] in ("box_choice", "spatial_boolean"):
        return row["choices"][row["answer_index"]]
    return row["answer_text"]


def main() -> None:
    selected, counts = {}, Counter()
    for line in TEST.open(encoding="utf-8"):
        row = json.loads(line)
        if row["image"]["id"] == IMAGE_ID:
            counts[row["task"]] += 1
            if row["id"] in IDS:
                selected[row["id"]] = row
    if set(selected) != set(IDS) or sum(counts.values()) != 61:
        raise ValueError("pinned test rows for image 76547 changed")
    rows = [selected[i] for i in IDS]
    if len({row["image"]["path"] for row in rows}) != 1:
        raise ValueError("selected rows have different photos")
    source_path = ROOT / rows[0]["image"]["path"]
    with Image.open(source_path) as im:
        source = im.convert("RGB")

    canvas = Image.new("RGB", (2500, 1660), BG)
    draw = ImageDraw.Draw(canvas)
    draw.text((70, 48), "ONE PHOTO  /  MANY SUPERVISED QUESTIONS", font=font(52, True), fill=INK)
    draw.text((74, 120), "Published held-out COCO photo 000000076547  ·  61 records  ·  five tasks", font=font(27), fill=MUTED)
    draw.rounded_rectangle((60, 190, 1230, 1460), radius=24, fill="white")
    image_panel(canvas, source, 95, 230, 1100, 830, [
        ("Bicycle", rows[0]["answer_box_xyxy"], COLORS["ground_bbox"]),
        ("Person", rows[2]["evidence"]["boxes_xyxy"][1], COLORS["spatial_boolean"]),
        ("Backpack", rows[3]["region_box_xyxy"], COLORS["attribute_text"]),
        ("Laptop", rows[4]["subject_box_xyxy"], COLORS["relation_text"]),
    ])
    draw = ImageDraw.Draw(canvas)
    draw.text((96, 1100), "Records generated from this photo", font=font(31, True), fill=INK)
    task_labels = (("ground_bbox", "Ground box"), ("box_choice", "Box choice"),
                   ("spatial_boolean", "Spatial"), ("attribute_text", "Color"),
                   ("relation_text", "Relation"))
    for i, (task, label) in enumerate(task_labels):
        cx = 98 + (i % 3) * 365
        cy = 1170 + (i // 3) * 105
        draw.rounded_rectangle((cx, cy, cx+340, cy+83), radius=14, fill="#EDF2F8")
        draw.text((cx+17, cy+16), f"{counts[task]}", font=font(29, True), fill=COLORS[task])
        draw.text((cx+76, cy+23), label, font=font(22), fill=INK)

    names = ("01  GROUNDING", "02  BOX CHOICE", "03  SPATIAL DECISION",
             "04  COLOR ATTRIBUTE", "05  OBJECT RELATION")
    for i, row in enumerate(rows):
        x, y, w, h = 1260, 190 + i*253, 1170, 234
        color = COLORS[row["task"]]
        draw.rounded_rectangle((x, y, x+w, y+h), radius=22, fill="white")
        image_panel(canvas, source, x+18, y+23, 290, 189, boxes(row))
        draw = ImageDraw.Draw(canvas)
        tx = x+335
        draw.text((tx, y+20), names[i], font=font(22, True), fill=color)
        for j, line in enumerate(fit_lines(draw, row["question"], 790, 25, 2)):
            draw.text((tx, y+60+j*31), line, font=font(25, True), fill=INK)
        draw.text((tx, y+136), "Target:", font=font(23), fill=MUTED)
        draw.text((tx+92, y+136), answer(row), font=font(23, True), fill=color)
        source_label = "COCO instances" if i < 3 else "Visual Genome annotation"
        draw.text((tx, y+184), source_label, font=font(19), fill=MUTED)
    draw.line((70, 1520, 2430, 1520), fill="#D6DEE8", width=2)
    draw.text((76, 1540), "Question wording is largely templated; objects, boxes, labels, and answers vary by photo.",
              font=font(24), fill=INK)
    draw.text((76, 1580), "These five examples are test annotations, not new model predictions. Visual Genome labels await human audit.",
              font=font(21), fill=MUTED)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUTPUT, optimize=True)
    EXAMPLES.write_text(json.dumps({
        "image_id": IMAGE_ID, "image_source": rows[0]["image"]["url"],
        "split": "test", "record_count": sum(counts.values()),
        "task_counts": dict(counts),
        "selected": [{"id": row["id"], "task": row["task"],
                      "question": row["question"], "target": answer(row)} for row in rows],
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT} and {EXAMPLES}")


if __name__ == "__main__":
    main()
