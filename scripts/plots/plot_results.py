"""Render a static PNG dashboard of the Modal L4 smoke-test results.

Requires Pillow and local pilot photos. From the repository root:
    python3 scripts/plots/plot_results.py
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT = DATA / "visual_jev_l4_results.png"
WIDTH, HEIGHT = 2000, 1690

BG = "#F3F5F8"
INK = "#152238"
MUTED = "#66758A"
BLUE = "#4979D8"
TEAL = "#159E8A"
RED = "#D95F69"
GRID = "#DCE3EB"
WHITE = "#FFFFFF"

FONT_DIR = Path("/System/Library/Fonts/Supplemental")
FALLBACK_FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")


def font(size: int, bold: bool = False):
    mac_name = "Arial Bold.ttf" if bold else "Arial.ttf"
    linux_name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for path in (FONT_DIR / mac_name, FALLBACK_FONT_DIR / linux_name):
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size=size)


F12, F16, F18, F20, F23, F28, F34, F44 = [font(s) for s in (12, 16, 18, 20, 23, 28, 34, 44)]
B16, B20, B23, B28, B34, B44 = [font(s, True) for s in (16, 20, 23, 28, 34, 44)]


def rounded(draw, box, fill, radius=20, outline=None, width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def text_wrap(draw, text: str, face, max_width: int) -> list[str]:
    words = text.split()
    lines = []
    line = ""
    for word in words:
        trial = word if not line else line + " " + word
        if draw.textlength(trial, font=face) <= max_width:
            line = trial
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


def label(draw, xy, text, face, fill=INK):
    draw.text(xy, text, font=face, fill=fill)


def accuracy_box(draw, x, y, title, base_correct, tuned_correct, n):
    rounded(draw, (x, y, x + 540, y + 75), WHITE, radius=14)
    label(draw, (x + 18, y + 10), title, B20)
    label(draw, (x + 18, y + 40), f"Base {base_correct}/{n}  ·  {base_correct/n:.0%}", F18, MUTED)
    label(draw, (x + 285, y + 40), f"Adapter {tuned_correct}/{n}  ·  {tuned_correct/n:.0%}", B20, TEAL)


def draw_chart(draw, losses):
    x0, y0, x1, y1 = 76, 355, 925, 526
    for val in (0.0, 0.5, 1.0, 1.5, 2.0):
        y = y1 - (val / 2.1) * (y1 - y0)
        draw.line((x0, y, x1, y), fill=GRID, width=2)
        label(draw, (x0 - 54, y - 9), f"{val:.1f}", F16, MUTED)
    for step in (1, 20, 40, 60):
        x = x0 + ((step - 1) / 59) * (x1 - x0)
        label(draw, (x - 12, y1 + 12), str(step), F16, MUTED)
    raw = [(x0 + i / 59 * (x1 - x0), y1 - min(v, 2.1) / 2.1 * (y1 - y0)) for i, v in enumerate(losses)]
    draw.line(raw, fill="#9CB6E8", width=3, joint="curve")
    rolling = [sum(losses[max(0, i - 4):i + 1]) / (i - max(0, i - 4) + 1) for i in range(len(losses))]
    points = [(x0 + i / 59 * (x1 - x0), y1 - min(v, 2.1) / 2.1 * (y1 - y0)) for i, v in enumerate(rolling)]
    draw.line(points, fill=BLUE, width=5, joint="curve")
    label(draw, (765, 335), "5-step average", F16, BLUE)


def fit_image(image: Image.Image, box: tuple[int, int, int, int]):
    x0, y0, x1, y1 = box
    fitted = ImageOps.contain(image, (x1 - x0, y1 - y0), Image.Resampling.LANCZOS)
    px = x0 + (x1 - x0 - fitted.width) // 2
    py = y0 + (y1 - y0 - fitted.height) // 2
    return fitted, px, py


BOX_COLORS = {"A": "#48A7FF", "B": "#FFB84D", "C": "#C580FF"}


def image_with_boxes(canvas, prediction, box):
    image = Image.open(ROOT / prediction["image_path"]).convert("RGB")
    fitted, px, py = fit_image(image, box)
    canvas.paste(fitted, (px, py))
    overlay = ImageDraw.Draw(canvas)
    boxes = prediction["candidates"] if prediction["task"] == "box_choice" else [
        {"choice": "1", "bbox_xyxy": prediction["evidence"]["boxes_xyxy"][0]},
        {"choice": "2", "bbox_xyxy": prediction["evidence"]["boxes_xyxy"][1]},
    ]
    for index, candidate in enumerate(boxes):
        name = candidate["choice"]
        color = BOX_COLORS[name] if name in BOX_COLORS else (BLUE, RED)[index]
        bx = candidate["bbox_xyxy"]
        rect = (px + bx[0] * fitted.width, py + bx[1] * fitted.height,
                px + bx[2] * fitted.width, py + bx[3] * fitted.height)
        overlay.rectangle(rect, outline=color, width=4)
        tag_x = max(px, min(rect[0], px + fitted.width - 36))
        tag_y = max(py, rect[1] - 26)
        overlay.rounded_rectangle((tag_x, tag_y, tag_x + 34, tag_y + 25), radius=5, fill=color)
        overlay.text((tag_x + 10, tag_y + 3), name, font=B16, fill=INK)


def example_card(canvas, draw, prediction, xy, status):
    x, y = xy
    w, h = 915, 475
    rounded(draw, (x, y, x + w, y + h), WHITE, radius=20)
    pill = TEAL if status == "CORRECT" else RED
    rounded(draw, (x + 18, y + 18, x + 135, y + 48), pill, radius=14)
    label(draw, (x + 35, y + 23), status, B16, WHITE)
    task_name = "Box choice" if prediction["task"] == "box_choice" else "Spatial decision"
    label(draw, (x + 154, y + 22), task_name, B20)
    image_box = (x + 18, y + 67, x + 445, y + 384)
    rounded(draw, image_box, "#E8EDF3", radius=10)
    image_with_boxes(canvas, prediction, image_box)
    if prediction["task"] == "spatial_boolean":
        label(draw, (x + 20, y + 393), "1 / 2: annotation boxes shown for inspection", F16, MUTED)
    else:
        label(draw, (x + 20, y + 393), "A / B / C: candidate boxes shown to model", F16, MUTED)

    tx = x + 475
    for i, line in enumerate(text_wrap(draw, prediction["question"], B23, 405)[:3]):
        label(draw, (tx, y + 75 + i * 31), line, B23)
    expected = prediction["choices"][prediction["answer_index"]]
    predicted = prediction["choices"][prediction["adapter_pred_index"]]
    base = prediction["choices"][prediction["base_pred_index"]]
    label(draw, (tx, y + 187), f"Expected   {expected}", B20, INK)
    label(draw, (tx, y + 218), f"Adapter    {predicted}", B20, TEAL if status == "CORRECT" else RED)
    label(draw, (tx, y + 249), f"Base          {base}", F20, MUTED)
    label(draw, (tx, y + 293), "Adapter option scores", F18, MUTED)
    for i, (choice, prob) in enumerate(zip(prediction["choices"], prediction["adapter_choice_probs"])):
        yy = y + 322 + i * 35
        label(draw, (tx, yy - 2), choice, B16, INK)
        rounded(draw, (tx + 42, yy, tx + 315, yy + 15), "#E7ECF3", radius=7)
        if prob > 0.005:
            rounded(draw, (tx + 42, yy, tx + 42 + max(4, 273 * prob), yy + 15), BLUE if i == prediction["adapter_pred_index"] else "#AAB8C9", radius=7)
        label(draw, (tx + 330, yy - 5), f"{prob:.1%}", F16, MUTED)


def select(predictions, task, correct, preferred_id):
    return next(item for item in predictions if item["id"] == preferred_id and item["task"] == task
                and (item["adapter_pred_index"] == item["answer_index"]) == correct)


def main():
    training = json.loads((DATA / "train_smoke_metrics.json").read_text())
    summary = json.loads((DATA / "choice_eval.json").read_text())
    predictions = json.loads((DATA / "predictions.json").read_text())
    canvas = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(canvas)

    label(draw, (68, 50), "VISUAL JEV  /  L4 SMOKE TEST", B44)
    label(draw, (70, 110), "Qwen3.5-0.8B-Base + LoRA  ·  60 training steps on 60 COCO decision records", F23, MUTED)
    rounded(draw, (68, 164, 1932, 570), WHITE, radius=22)
    label(draw, (96, 186), "Held-out choice accuracy", B28)
    groups = summary["by_task"]
    total_base = sum(g["base_correct"] for g in groups.values())
    total_tuned = sum(g["adapter_correct"] for g in groups.values())
    rounded(draw, (96, 235, 648, 317), "#EAF4F1", radius=15)
    label(draw, (116, 246), f"{total_base}/97", B34, MUTED)
    label(draw, (304, 252), "→", B28, TEAL)
    label(draw, (360, 246), f"{total_tuned}/97", B34, TEAL)
    label(draw, (116, 285), "47.4% base", F16, MUTED)
    label(draw, (360, 285), "73.2% adapter", F16, TEAL)
    accuracy_box(draw, 679, 235, "Box choice", groups["box_choice"]["base_correct"], groups["box_choice"]["adapter_correct"], groups["box_choice"]["n"])
    accuracy_box(draw, 1260, 235, "Spatial decisions", groups["spatial_boolean"]["base_correct"], groups["spatial_boolean"]["adapter_correct"], groups["spatial_boolean"]["n"])
    label(draw, (97, 332), "Training loss across 60 steps", B20)
    draw_chart(draw, training["train_losses"])
    draw.line((1000, 337, 1000, 535), fill=GRID, width=2)
    label(draw, (1043, 354), "Held-out loss (20 records)", B20)
    label(draw, (1043, 399), f"{training['baseline_val_loss']:.2f}", B34, MUTED)
    label(draw, (1177, 405), "→", B28, TEAL)
    label(draw, (1240, 399), f"{training['final_val_loss']:.2f}", B34, TEAL)
    label(draw, (1043, 451), "Lower is better", F18, MUTED)
    label(draw, (1500, 354), "Image dependence check", B20)
    control = summary["blank_image_first_20"]
    label(draw, (1500, 399), f"{control['real_image_correct']}/20", B34, TEAL)
    label(draw, (1668, 406), "real images", F18, MUTED)
    label(draw, (1500, 447), f"{control['blank_image_correct']}/20", B34, MUTED)
    label(draw, (1668, 454), "blank images", F18, MUTED)

    label(draw, (72, 598), "Held-out examples", B34)
    label(draw, (74, 641), "Option scores are normalized over the listed choices; they are not calibrated probabilities.", F20, MUTED)
    ids = [
        ("box_choice", True, "coco2017-val2017-5037-ground-169115"),
        ("spatial_boolean", True, "coco2017-val2017-1503-spatial-1100052-1115262"),
        ("box_choice", False, "coco2017-val2017-2473-ground-2201205"),
        ("spatial_boolean", False, "coco2017-val2017-139-spatial-1647285-1666628"),
    ]
    positions = [(68, 688), (1017, 688), (68, 1190), (1017, 1190)]
    for (task, correct, record_id), xy in zip(ids, positions):
        example_card(canvas, draw, select(predictions, task, correct, record_id), xy, "CORRECT" if correct else "MISSED")
    canvas.save(OUT, quality=95)
    print(OUT)


if __name__ == "__main__":
    main()
