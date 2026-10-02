"""Conservative COCO-to-visual-decision dataset converter. Standard library only."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
from typing import Any
from urllib.request import urlopen


SCHEMA_VERSION = "1.0"


def norm_box(box: list[float], width: int, height: int) -> list[float]:
    x, y, w, h = map(float, box)
    if width <= 0 or height <= 0 or w <= 0 or h <= 0:
        raise ValueError("invalid image or box dimensions")
    result = [x / width, y / height, (x + w) / width, (y + h) / height]
    if not (0 <= result[0] < result[2] <= 1 and 0 <= result[1] < result[3] <= 1):
        raise ValueError("box outside image")
    return [round(v, 6) for v in result]


def box_iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersection / (area_a + area_b - intersection)


def image_ref(image: dict[str, Any], source_split: str, images_dir: Path) -> dict[str, Any]:
    return {
        "source": "coco2017",
        "source_split": source_split,
        "id": str(image["id"]),
        "path": str(images_dir / source_split / image["file_name"]),
        "width": image["width"],
        "height": image["height"],
    }


def record_base(image: dict[str, Any], source_split: str, images_dir: Path) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "split": "train" if source_split == "train2017" else "validation",
        "image": image_ref(image, source_split, images_dir),
        "provenance": {
            "method": "deterministic",
            "source": "coco_instances",
            "review_status": "annotation_derived",
        },
    }


def make_records(image: dict[str, Any], annotations: list[dict[str, Any]], categories: dict[int, str], source_split: str, images_dir: Path) -> list[dict[str, Any]]:
    valid = []
    for ann in sorted(annotations, key=lambda a: a["id"]):
        if ann.get("iscrowd", 0) or ann.get("category_id") not in categories:
            continue
        try:
            box = norm_box(ann["bbox"], image["width"], image["height"])
        except (KeyError, ValueError, TypeError):
            continue
        valid.append((ann, box, categories[ann["category_id"]]))

    by_category = Counter(label for _, _, label in valid)
    result = []
    base = record_base(image, source_split, images_dir)
    # A box choice is unambiguous only when its named category occurs once.
    for ann, box, label in valid:
        if by_category[label] != 1:
            continue
        distractors = [(other, other_box) for other, other_box, other_label in valid
                       if other["id"] != ann["id"] and other_label != label and box_iou(box, other_box) < 0.1]
        if len(distractors) < 2:
            continue
        distractors = distractors[:2]
        # Rotate the correct choice deterministically to prevent answer-position leakage.
        answer_index = ann["id"] % 3
        candidate_boxes = [other_box for _, other_box in distractors]
        candidate_boxes.insert(answer_index, box)
        choices = ["A", "B", "C"]
        result.append({
            **base,
            "id": f"coco2017-{source_split}-{image['id']}-ground-{ann['id']}",
            "task": "box_choice",
            "question": f"Which box contains the {label}?",
            "choices": choices,
            "answer_index": answer_index,
            "candidates": [{"choice": choice, "bbox_xyxy": candidate_box} for choice, candidate_box in zip(choices, candidate_boxes)],
            "evidence": {"annotation_ids": [ann["id"]], "boxes_xyxy": [box]},
        })

    # Compare boxes that are clearly separated on the horizontal axis.
    for i, (left_ann, left_box, left_label) in enumerate(valid):
        if by_category[left_label] != 1:
            continue
        for right_ann, right_box, right_label in valid[i + 1:]:
            if left_label == right_label or by_category[right_label] != 1:
                continue
            clearly_left = left_box[2] + 0.02 < right_box[0]
            clearly_right = right_box[2] + 0.02 < left_box[0]
            if not (clearly_left or clearly_right):
                continue
            for subject, subject_box, subject_label, obj, obj_box, obj_label, answer_index in (
                (left_ann, left_box, left_label, right_ann, right_box, right_label, 0 if clearly_left else 1),
                (right_ann, right_box, right_label, left_ann, left_box, left_label, 1 if clearly_left else 0),
            ):
                result.append({
                    **base,
                    "id": f"coco2017-{source_split}-{image['id']}-spatial-{subject['id']}-{obj['id']}",
                    "task": "spatial_boolean",
                    "question": f"Is the {subject_label} to the left of the {obj_label}?",
                    "choices": ["YES", "NO", "UNKNOWN"],
                    "answer_index": answer_index,
                    "candidates": [],
                    "evidence": {"annotation_ids": [subject["id"], obj["id"]], "boxes_xyxy": [subject_box, obj_box]},
                })
    return result


def build_coco(args: argparse.Namespace) -> None:
    with args.annotations.open(encoding="utf-8") as handle:
        source = json.load(handle)
    categories = {item["id"]: item["name"] for item in source["categories"]}
    annotations = defaultdict(list)
    for ann in source["annotations"]:
        annotations[ann["image_id"]].append(ann)
    images = sorted(source["images"], key=lambda item: item["id"])
    if args.limit_images is not None:
        images = images[:args.limit_images]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for image in images:
            for record in make_records(image, annotations[image["id"]], categories, args.source_split, args.images_dir):
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                count += 1
    print(f"Wrote {count} records from {len(images)} images to {args.output}")


def validate_box(box: Any) -> bool:
    return (isinstance(box, list) and len(box) == 4 and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in box)
            and 0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1)


def validate_record(item: dict[str, Any], check_images: bool = False) -> list[str]:
    errors = []
    for key in ("schema_version", "id", "split", "image", "task", "question", "choices", "answer_index", "candidates", "evidence", "provenance"):
        if key not in item:
            errors.append(f"missing {key}")
    if errors:
        return errors
    if item["schema_version"] not in (SCHEMA_VERSION, "1.1"):
        errors.append("unsupported schema_version")
    if item["split"] not in ("train", "validation", "test"):
        errors.append("invalid split")
    image = item["image"]
    if not isinstance(image, dict) or not all(key in image for key in ("source", "source_split", "id", "path", "width", "height")):
        errors.append("invalid image reference")
    elif check_images and not Path(image["path"]).is_file():
        errors.append("image file missing")
    choices = item["choices"]
    evidence = item["evidence"]
    if item["task"] in ("attribute_text", "relation_text"):
        if choices != [] or item["answer_index"] is not None or item["candidates"] != [] or not isinstance(item.get("answer_text"), str) or not item["answer_text"].strip():
            errors.append("invalid text target")
        if item["task"] == "attribute_text" and not validate_box(item.get("region_box_xyxy")):
            errors.append("invalid attribute region")
        if item["task"] == "relation_text" and (not validate_box(item.get("subject_box_xyxy")) or not validate_box(item.get("object_box_xyxy"))):
            errors.append("invalid relation regions")
    elif item["task"] == "ground_bbox":
        if choices != [] or item["answer_index"] is not None or item["candidates"] != [] or not validate_box(item.get("answer_box_xyxy")):
            errors.append("invalid grounding answer")
        elif evidence.get("boxes_xyxy") and item["answer_box_xyxy"] != evidence["boxes_xyxy"][0]:
            errors.append("grounding answer does not match evidence")
    elif not isinstance(choices, list) or len(choices) < 2 or len(choices) != len(set(choices)):
        errors.append("invalid choices")
    elif not isinstance(item["answer_index"], int) or isinstance(item["answer_index"], bool) or not 0 <= item["answer_index"] < len(choices):
        errors.append("answer_index outside choices")
    if not isinstance(evidence, dict) or not evidence.get("annotation_ids") or not all(validate_box(box) for box in evidence.get("boxes_xyxy", [])):
        errors.append("invalid evidence")
    candidates = item["candidates"]
    if item["task"] == "box_choice":
        if not isinstance(candidates, list) or len(candidates) != len(choices) or any(c.get("choice") != choice or not validate_box(c.get("bbox_xyxy")) for c, choice in zip(candidates, choices)):
            errors.append("invalid box candidates")
        elif evidence.get("boxes_xyxy") and candidates[item["answer_index"]]["bbox_xyxy"] != evidence["boxes_xyxy"][0]:
            errors.append("correct candidate does not match evidence")
    elif item["task"] == "spatial_boolean":
        if choices != ["YES", "NO", "UNKNOWN"] or candidates != [] or item["answer_index"] == 2:
            errors.append("invalid spatial decision")
    elif item["task"] in ("attribute_choice", "referring_choice", "relation_choice"):
        if candidates != []:
            errors.append("model-generated choices must not have box candidates")
    elif item["task"] in ("ground_bbox", "attribute_text", "relation_text"):
        pass
    else:
        errors.append("unsupported task")
    provenance = item["provenance"]
    if not isinstance(provenance, dict) or provenance.get("review_status") not in ("annotation_derived", "model_verified_pending_audit"):
        errors.append("invalid provenance")
    return errors


def read_records(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                yield line_number, json.loads(line)


def validate(args: argparse.Namespace) -> None:
    seen_ids = set()
    image_splits = {}
    errors = []
    count = 0
    for line_number, item in read_records(args.path):
        count += 1
        for error in validate_record(item, args.check_images):
            errors.append(f"line {line_number}: {error}")
        record_id = item.get("id")
        if record_id in seen_ids:
            errors.append(f"line {line_number}: duplicate id")
        seen_ids.add(record_id)
        image = item.get("image", {})
        image_key = (image.get("source"), image.get("id")) if isinstance(image, dict) else None
        if image_key in image_splits and image_splits[image_key] != item.get("split"):
            errors.append(f"line {line_number}: image split leakage")
        image_splits[image_key] = item.get("split")
    if errors:
        print("\n".join(errors[:30]))
        raise SystemExit(f"Validation failed: {len(errors)} errors")
    print(f"Valid: {count} records, {len(image_splits)} images")


def stats(args: argparse.Namespace) -> None:
    tasks = Counter()
    answers = Counter()
    images = set()
    for _, item in read_records(args.path):
        tasks[item["task"]] += 1
        if item["answer_index"] is not None:
            answers[(item["task"], item["choices"][item["answer_index"]])] += 1
        images.add((item["image"]["source"], item["image"]["id"]))
    print(json.dumps({"records": sum(tasks.values()), "images": len(images), "tasks": dict(tasks), "answers": {f"{task}:{answer}": n for (task, answer), n in sorted(answers.items())}}, indent=2))


def download_images(args: argparse.Namespace) -> None:
    """Fetch only images referenced by the selected dataset records."""
    images = {}
    for _, item in read_records(args.path):
        image = item["image"]
        images[image["path"]] = image

    def fetch(image: dict[str, Any]) -> str:
        path = Path(image["path"])
        if path.is_file() and path.stat().st_size > 0:
            return "existing"
        url = f"https://s3.amazonaws.com/images.cocodataset.org/{image['source_split']}/{path.name}"
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".part")
        try:
            with urlopen(url, timeout=60) as response, temp.open("wb") as handle:
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
            with temp.open("rb") as handle:
                signature = handle.read(8)
                # A small number of COCO URLs have a .jpg name but PNG bytes.
                if not (signature.startswith(b"\xff\xd8\xff") or signature == b"\x89PNG\r\n\x1a\n"):
                    raise ValueError("download is not a JPEG or PNG")
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)
        return "downloaded"

    counts = Counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch, image): image for image in images.values()}
        for future in as_completed(futures):
            try:
                counts[future.result()] += 1
            except Exception as exc:
                counts["failed"] += 1
                print(f"Failed {futures[future]['path']}: {exc}")
    print(dict(counts))
    if counts["failed"]:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-coco")
    build.add_argument("--annotations", type=Path, required=True)
    build.add_argument("--images-dir", type=Path, required=True)
    build.add_argument("--source-split", choices=("train2017", "val2017"), required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--limit-images", type=int)
    build.set_defaults(func=build_coco)
    check = sub.add_parser("validate")
    check.add_argument("path", type=Path)
    check.add_argument("--check-images", action="store_true")
    check.set_defaults(func=validate)
    show = sub.add_parser("stats")
    show.add_argument("path", type=Path)
    show.set_defaults(func=stats)
    download = sub.add_parser("download-images")
    download.add_argument("path", type=Path)
    download.add_argument("--workers", type=int, default=8)
    download.set_defaults(func=download_images)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
