"""Build the complete annotation-derived COCO curriculum with image-level splits.

This stage deliberately uses only positive object evidence and clearly separated
spatial pairs. Attribute, absence, and UNKNOWN labels need image review.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from dataset import make_records, norm_box


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = DATA / "full"


def grounding_records(image, annotations, categories, source_split, images_dir):
    valid = []
    for ann in sorted(annotations, key=lambda item: item["id"]):
        if ann.get("iscrowd", 0) or ann.get("category_id") not in categories:
            continue
        try:
            box = norm_box(ann["bbox"], image["width"], image["height"])
        except (KeyError, ValueError, TypeError):
            continue
        valid.append((ann, box))
    category_counts = Counter(ann["category_id"] for ann, _ in valid)
    for ann, box in valid:
        if category_counts[ann["category_id"]] != 1:
            continue
        label = categories[ann["category_id"]]
        yield {
            "schema_version": "1.1",
            "id": f"coco2017-{source_split}-{image['id']}-locate-{ann['id']}",
            "split": "train" if source_split == "train2017" else "validation",
            "image": {
                "source": "coco2017",
                "source_split": source_split,
                "id": str(image["id"]),
                "path": str(images_dir / source_split / image["file_name"]),
                "width": image["width"],
                "height": image["height"],
            },
            "task": "ground_bbox",
            "target_label": label,
            "question": f"Where is the {label}? Return its bounding box as normalized x1,y1,x2,y2 coordinates.",
            "choices": [],
            "answer_index": None,
            "answer_box_xyxy": box,
            "candidates": [],
            "evidence": {"annotation_ids": [ann["id"]], "boxes_xyxy": [box]},
            "provenance": {"method": "deterministic", "source": "coco_instances", "review_status": "annotation_derived"},
        }


def load_source(split):
    path = DATA / "coco" / "annotations" / f"instances_{split}.json"
    with path.open(encoding="utf-8") as handle:
        source = json.load(handle)
    categories = {item["id"]: item["name"] for item in source["categories"]}
    anns = defaultdict(list)
    for ann in source["annotations"]:
        anns[ann["image_id"]].append(ann)
    return sorted(source["images"], key=lambda item: item["id"]), anns, categories


def test_image(image_id, pilot_validation_ids):
    if str(image_id) in pilot_validation_ids:
        return False
    digest = hashlib.sha256(f"visual-jev-test-v1:{image_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % 5 == 0


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    pilot_validation_ids = set()
    with (DATA / "validation.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            pilot_validation_ids.add(json.loads(line)["image"]["id"])
    paths = {name: OUT / f"{name}.jsonl" for name in ("train", "validation", "test")}
    counts = {name: Counter() for name in paths}
    image_ids = {name: set() for name in paths}
    with paths["train"].open("w", encoding="utf-8") as train_file, paths["validation"].open("w", encoding="utf-8") as val_file, paths["test"].open("w", encoding="utf-8") as test_file:
        handles = {"train": train_file, "validation": val_file, "test": test_file}
        for source_split in ("train2017", "val2017"):
            images, annotations, categories = load_source(source_split)
            for image in images:
                split = ("train" if source_split == "train2017" else
                         "test" if test_image(image["id"], pilot_validation_ids) else "validation")
                records = list(grounding_records(image, annotations[image["id"]], categories, source_split, DATA / "coco"))
                records.extend(make_records(image, annotations[image["id"]], categories, source_split, DATA / "coco"))
                for record in records:
                    record["schema_version"] = "1.1"
                    record["split"] = split
                    handles[split].write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                    counts[split][record["task"]] += 1
                    image_ids[split].add(image["id"])
            del images, annotations, categories
    manifest = {
        "schema_version": "1.1",
        "source": "COCO 2017 instances annotations",
        "build_script": "build_full_dataset.py",
        "scope": "all source images that yield supported annotation-derived questions",
        "tasks": ["ground_bbox", "box_choice", "spatial_boolean"],
        "split_policy": "COCO train2017 -> train; val2017 -> 80% validation / 20% test by SHA-256 image ID, with pilot validation images excluded from test",
        "annotation_derived_only": True,
        "image_download_complete": False,
        "counts": {name: {"records": sum(counts[name].values()), "images": len(image_ids[name]), "tasks": dict(counts[name])} for name in paths},
        "limitations": [
            "COCO boxes and category labels are annotations, not proof of complete object inventory.",
            "No absence, exact counting, attributes, OCR, or UNKNOWN labels are inferred from missing annotations.",
            "Box grounding uses only categories with one valid annotation in an image to avoid ambiguous targets.",
            "Images have individual license metadata in the COCO source annotations; usage rights must be reviewed before commercial training.",
        ],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    build()
