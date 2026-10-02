"""Build a portable Hugging Face release from the validated source shards."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

import ijson

from dataset import validate_record


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data" / "full"
DEST = ROOT / "data" / "hf_release"


def portable(record: dict, tier: str) -> dict:
    record = dict(record)
    image = dict(record["image"])
    filename = Path(image["path"]).name
    split = image["source_split"]
    image["path"] = f"data/coco/{split}/{filename}"
    image["url"] = f"https://s3.amazonaws.com/images.cocodataset.org/{split}/{filename}"
    record["image"] = image
    record["quality_tier"] = tier
    provenance = dict(record["provenance"])
    provenance.pop("generation_turn_id", None)
    provenance.pop("verification_turn_id", None)
    record["provenance"] = provenance
    return record


def stream_file(source: Path, target, tier: str, counts: Counter, ids: set, images: set) -> None:
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            record = portable(json.loads(line), tier)
            errors = validate_record(record)
            if errors:
                raise ValueError(f"{record['id']}: {errors}")
            if record["id"] in ids:
                raise ValueError(f"duplicate record ID: {record['id']}")
            ids.add(record["id"])
            images.add(record["image"]["id"])
            counts[record["task"]] += 1
            target.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    report = {"splits": {}, "files_sha256": {}, "image_files_included": False}
    split_images = {}
    all_ids = set()
    for split in ("train", "validation", "test"):
        counts = Counter()
        images = set()
        output = DEST / f"{split}.jsonl"
        with output.open("w", encoding="utf-8") as target:
            stream_file(SOURCE / f"{split}.jsonl", target, "geometry_annotation", counts, all_ids, images)
            stream_file(SOURCE / f"vg_{split}.jsonl", target, "visual_genome_annotation_pending_audit", counts, all_ids, images)
        report["splits"][split] = {"records": sum(counts.values()), "images": len(images), "tasks": dict(counts)}
        split_images[split] = images
        report["files_sha256"][output.name] = sha256(output)
    if any(split_images[a] & split_images[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("image leakage across splits")
    image_output = DEST / "images.jsonl"
    with image_output.open("w", encoding="utf-8") as target:
        for source_split, ids in (("train2017", split_images["train"]),
                                  ("val2017", split_images["validation"] | split_images["test"])):
            annotation_path = ROOT / "data" / "coco" / "annotations" / f"instances_{source_split}.json"
            with annotation_path.open("rb") as handle:
                licenses = {item["id"]: item for item in ijson.items(handle, "licenses.item")}
            found = set()
            with annotation_path.open("rb") as handle:
                for image in ijson.items(handle, "images.item"):
                    image_id = str(image["id"])
                    if image_id not in ids:
                        continue
                    license_info = licenses.get(image["license"], {})
                    target.write(json.dumps({
                        "id": image_id, "source_split": source_split,
                        "path": f"data/coco/{source_split}/{image['file_name']}",
                        "url": f"https://s3.amazonaws.com/images.cocodataset.org/{source_split}/{image['file_name']}",
                        "flickr_url": image.get("flickr_url"),
                        "license_id": image["license"],
                        "license_name": license_info.get("name"),
                        "license_url": license_info.get("url"),
                        "width": image["width"], "height": image["height"],
                    }, separators=(",", ":")) + "\n")
                    found.add(image_id)
            if found != ids:
                raise ValueError(f"missing {len(ids - found)} {source_split} image metadata records")
    report["image_manifest_records"] = sum(len(ids) for ids in split_images.values())
    report["files_sha256"][image_output.name] = sha256(image_output)
    counts = Counter()
    images = set()
    output = DEST / "luna_candidates.jsonl"
    with output.open("w", encoding="utf-8") as target:
        stream_file(SOURCE / output.name, target, "model_verified_pending_human_audit", counts, all_ids, images)
    if not images <= split_images["train"]:
        raise ValueError("Luna candidate image outside train split")
    report["luna_candidates"] = {"records": sum(counts.values()), "images": len(images), "tasks": dict(counts)}
    report["files_sha256"][output.name] = sha256(output)
    shutil.copy2(ROOT / "dataset.py", DEST / "dataset.py")
    shutil.copy2(ROOT / "build_full_dataset.py", DEST / "build_full_dataset.py")
    shutil.copy2(ROOT / "build_vg.py", DEST / "build_vg.py")
    card = f"""---
language: en
license: other
task_categories:
  - visual-question-answering
  - object-detection
tags:
  - coco
  - visual-genome
  - multimodal
  - grounding
---

# Visual Jev decisions v1

This release contains {report['splits']['train']['records']:,} training, {report['splits']['validation']['records']:,} validation, and {report['splits']['test']['records']:,} test records. An additional {report['luna_candidates']['records']:,} `gpt-6-luna` candidate records are in `luna_candidates.jsonl` and **are excluded from training and evaluation splits** pending human audit.

## Data

Each JSONL row has a `task`, `question`, `image`, answer fields, evidence, provenance, and `quality_tier`. Tasks are `ground_bbox`, `box_choice`, `spatial_boolean`, `attribute_text`, and `relation_text`. Bounding boxes are normalized `[x1,y1,x2,y2]` in image coordinates. The complete task counts and SHA-256 hashes are in `manifest.json`.

The `geometry_annotation` rows are derived from COCO 2017 instance boxes. `visual_genome_annotation_pending_audit` rows use Visual Genome v1.2 color attributes and selected relationships for COCO-overlapping images. Visual Genome labels are noisy and should be filtered or audited for high-confidence training. The Luna candidate shard was generated and independently checked by the same model family, so it also needs human review.

Image files are **not bundled** because COCO says image rights belong to their individual owners. `images.jsonl` records each photo's upstream URL, Flickr URL, and source license. `image.path` is relative to the checkout root, and `image.url` points to the corresponding official COCO image. From the repository root, download only referenced images with:

```bash
python3 dataset.py download-images train.jsonl --workers 12
python3 dataset.py download-images validation.jsonl --workers 8
python3 dataset.py download-images test.jsonl --workers 8
python3 dataset.py validate train.jsonl --check-images
```

All records for an image stay in one split. COCO train2017 maps to train; COCO val2017 is separated into validation and test by image. The pilot validation images were excluded from test.

## Training targets

Use `answer_box_xyxy` for `ground_bbox`; `choices[answer_index]` for choice tasks; `answer_text` for text tasks. Give the model the image, `question`, `choices`, and, where present, candidate or indicated boxes. **Never include** `answer_index`, `answer_box_xyxy`, `answer_text`, `evidence`, `quality_tier`, or `provenance` in the input prompt.

## Sources and limitations

- COCO 2017: https://cocodataset.org/#download
- Visual Genome v1.2: https://visualgenome.org/api/v0/api_readme

COCO photos have image-level rights separate from the derived labels. This repository does not grant rights to the photos. Check the upstream image licenses for your use. The dataset has no reliable object-absence, exhaustive counts, OCR, or calibrated UNKNOWN examples. Visual Genome relations in particular can be ambiguous or erroneous. Annotation-derived labels are not equivalent to individually reviewed ground truth.

This is a research dataset for visual grounding and decision training, not a general-purpose visual benchmark.
"""
    (DEST / "README.md").write_text(card, encoding="utf-8")
    (DEST / "manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
