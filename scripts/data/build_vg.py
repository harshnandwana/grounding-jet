"""Build conservative Visual Genome attribute/relation targets on COCO train images.

Run after downloading official Visual Genome image_data, attributes and
relationships archives. Requires ijson for streaming the large JSON arrays.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
import zipfile

import ijson

from visual_jev.dataset import validate_record


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
VG = DATA / "vg"
OUT_DIR = DATA / "full"
COLORS = {"red", "blue", "green", "yellow", "orange", "purple", "pink", "white",
          "black", "brown", "gray", "grey", "gold", "silver", "beige", "tan"}
HUMAN_NAMES = {"person", "man", "woman", "boy", "girl", "child", "people", "men", "women"}
RELATIONS = {"on", "under", "next to", "behind", "in front of", "holding", "wearing",
             "riding", "sitting on", "standing on", "looking at", "carrying", "eating",
             "inside", "above", "below", "to the left of", "to the right of"}
PREDICATE_MAP = {"holds": "holding", "wears": "wearing", "rides": "riding",
                 "sits on": "sitting on", "stands on": "standing on", "carries": "carrying",
                 "looks at": "looking at", "eats": "eating", "in": "inside"}


def coco_images():
    images = {}
    for split in ("train", "validation", "test"):
        with (OUT_DIR / f"{split}.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                image = json.loads(line)["image"]
                images[image["id"]] = (image, split)
    return images


def metadata(coco_images):
    result = {}
    with (VG / "image_data.json").open(encoding="utf-8") as handle:
        for item in json.load(handle):
            coco_id = str(item.get("coco_id"))
            image_and_split = coco_images.get(coco_id)
            if not image_and_split:
                continue
            image, split = image_and_split
            if item["width"] <= 0 or item["height"] <= 0:
                continue
            vg_ratio = item["width"] / item["height"]
            coco_ratio = image["width"] / image["height"]
            if abs(vg_ratio / coco_ratio - 1) > 0.015:
                continue
            result[item["image_id"]] = (image, split, item["width"], item["height"])
    return result


def valid_name(value):
    if not isinstance(value, str):
        return None
    value = re.sub(r"\s+", " ", value.strip().casefold())
    return value if re.fullmatch(r"[a-z][a-z -]{1,28}", value) else None


def get_object_name(item):
    value = item.get("name")
    if value is None and isinstance(item.get("names"), list) and item["names"]:
        value = item["names"][0]
    return valid_name(value)


def box(item, width, height):
    try:
        x, y = float(item["x"]), float(item["y"])
        w, h = float(item["w"]), float(item["h"])
    except (KeyError, TypeError, ValueError):
        return None
    if min(w, h) <= 0:
        return None
    value = [max(0.0, x / width), max(0.0, y / height),
             min(1.0, (x + w) / width), min(1.0, (y + h) / height)]
    if not (0 <= value[0] < value[2] <= 1 and 0 <= value[1] < value[3] <= 1):
        return None
    if (value[2] - value[0]) * (value[3] - value[1]) < 0.012:
        return None
    return [round(v, 6) for v in value]


def base_record(record_id, image, split, task, question, answer, ids, boxes, source):
    return {
        "schema_version": "1.1", "id": record_id, "split": split, "image": image,
        "task": task, "question": question, "choices": [], "answer_index": None,
        "answer_text": answer, "candidates": [],
        "evidence": {"annotation_ids": ids, "boxes_xyxy": boxes},
        "provenance": {"method": "human_annotation_conversion", "source": source,
                       "review_status": "annotation_derived"},
    }


def attributes_for_image(item, info):
    image, split, width, height = info
    valid = []
    for obj in item.get("attributes", []):
        if not isinstance(obj.get("object_id"), int):
            continue
        name = get_object_name(obj)
        coords = box(obj, width, height)
        if name is None or coords is None:
            continue
        # The object name must not give away its color, and broad person labels
        # often carry ambiguous clothing, hair, and skin attributes.
        if set(name.split()) & COLORS or name in HUMAN_NAMES:
            continue
        colors = {str(a).casefold().strip() for a in obj.get("attributes", [])} & COLORS
        if len(colors) != 1:
            continue
        valid.append((obj, name, coords, next(iter(colors))))
    name_counts = Counter(name for _, name, _, _ in valid)
    emitted = 0
    for obj, name, coords, color in sorted(valid, key=lambda x: x[0]["object_id"]):
        if name_counts[name] != 1:
            continue
        record = base_record(
            f"vg-{item['image_id']}-attribute-{obj['object_id']}-{color}", image, split,
            "attribute_text", f"What color is the {name} in the indicated box?",
            color, [obj["object_id"]], [coords], "visual_genome_attributes_v1.2",
        )
        record["region_box_xyxy"] = coords
        yield record
        emitted += 1
        if emitted >= 4:
            break


def relations_for_image(item, info):
    image, split, width, height = info
    emitted = 0
    seen = set()
    for relation in sorted(item.get("relationships", []), key=lambda x: x["relationship_id"]):
        raw_predicate = str(relation.get("predicate", "")).casefold().strip()
        predicate = PREDICATE_MAP.get(raw_predicate, raw_predicate)
        if predicate not in RELATIONS:
            continue
        subject, obj = relation.get("subject"), relation.get("object")
        if not isinstance(subject, dict) or not isinstance(obj, dict):
            continue
        if not isinstance(subject.get("object_id"), int) or not isinstance(obj.get("object_id"), int):
            continue
        subject_name, object_name = get_object_name(subject), get_object_name(obj)
        subject_box, object_box = box(subject, width, height), box(obj, width, height)
        if not subject_name or not object_name or subject_box is None or object_box is None:
            continue
        if subject_name == object_name and subject_box == object_box:
            continue
        key = (subject.get("object_id"), object_name, predicate)
        if key in seen:
            continue
        seen.add(key)
        record = base_record(
            f"vg-{item['image_id']}-relation-{relation['relationship_id']}", image, split,
            "relation_text", f"How is the {subject_name} in box 1 related to the {object_name} in box 2?",
            predicate, [relation["relationship_id"], subject["object_id"], obj["object_id"]],
            [subject_box, object_box], "visual_genome_relationships_v1.2",
        )
        record["subject_box_xyxy"] = subject_box
        record["object_box_xyxy"] = object_box
        yield record
        emitted += 1
        if emitted >= 3:
            break


def iter_archive(path, member):
    with zipfile.ZipFile(path) as archive, archive.open(member) as handle:
        yield from ijson.items(handle, "item")


def main():
    coco = coco_images()
    metas = metadata(coco)
    print(f"{len(metas)} Visual Genome photos map to COCO splits", flush=True)
    counts = {split: Counter() for split in ("train", "validation", "test")}
    images = {split: set() for split in counts}
    from contextlib import ExitStack
    with ExitStack() as stack:
        handles = {split: stack.enter_context((OUT_DIR / f"vg_{split}.jsonl").open("w", encoding="utf-8")) for split in counts}
        for archive_name, member, converter in (
            ("attributes.json.zip", "attributes.json", attributes_for_image),
            ("relationships.json.zip", "relationships.json", relations_for_image),
        ):
            for item in iter_archive(VG / archive_name, member):
                info = metas.get(item["image_id"])
                if info is None:
                    continue
                for record in converter(item, info):
                    errors = validate_record(record)
                    if errors:
                        raise ValueError(f"{record['id']}: {errors}")
                    split = record["split"]
                    handles[split].write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                    counts[split][record["task"]] += 1
                    images[split].add(record["image"]["id"])
    report = {"splits": {split: {"records": sum(counts[split].values()), "images": len(images[split]),
                                   "tasks": dict(counts[split])} for split in counts},
              "source": "Visual Genome v1.2 attributes and relationships on overlapping COCO images",
              "note": "Human annotation conversions; visually audit a sample before final training."}
    (OUT_DIR / "vg_manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
