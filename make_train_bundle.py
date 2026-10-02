"""Sample a reproducible five-task training and benchmark bundle for Modal."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import tarfile


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data" / "full"
OUT = ROOT / "data" / "train_full_v1"
TASKS = ("ground_bbox", "box_choice", "spatial_boolean", "attribute_text", "relation_text")
LIMITS = {"train": 500, "validation": 60, "test": 60}


def reservoir(split: str) -> dict[str, list[dict]]:
    rng = random.Random({"train": 43801, "validation": 43802, "test": 43803}[split])
    result = defaultdict(list)
    seen = Counter()
    for file in (SOURCE / f"{split}.jsonl", SOURCE / f"vg_{split}.jsonl"):
        with file.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                path = Path(row["image"]["path"])
                if not path.is_file():
                    continue
                task = row["task"]
                if task not in TASKS:
                    continue
                seen[task] += 1
                bucket = result[task]
                limit = LIMITS[split]
                if len(bucket) < limit:
                    bucket.append(row)
                else:
                    at = rng.randrange(seen[task])
                    if at < limit:
                        bucket[at] = row
    for task in TASKS:
        if len(result[task]) != LIMITS[split]:
            raise ValueError(f"{split}/{task}: only {len(result[task])} rows")
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    selected = {}
    image_paths = set()
    report = {"sampling_seed": 43801, "per_task_limits": LIMITS, "splits": {}}
    for split in LIMITS:
        groups = reservoir(split)
        rows = [row for task in TASKS for row in groups[task]]
        random.Random(45000 + len(split)).shuffle(rows)
        selected[split] = rows
        report["splits"][split] = {"records": len(rows), "tasks": dict(Counter(row["task"] for row in rows)),
                                   "images": len({row["image"]["id"] for row in rows})}
        for row in rows:
            image_paths.add(Path(row["image"]["path"]))
        with (OUT / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                row = dict(row)
                image = dict(row["image"])
                image["path"] = f"images/{image['source_split']}/{Path(image['path']).name}"
                row["image"] = image
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    report["total_unique_images"] = len(image_paths)
    (OUT / "selection_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    tar_path = OUT / "bundle.tar"
    with tarfile.open(tar_path, "w") as archive:
        for split in LIMITS:
            archive.add(OUT / f"{split}.jsonl", arcname=f"{split}.jsonl")
        archive.add(OUT / "selection_manifest.json", arcname="selection_manifest.json")
        for path in sorted(image_paths):
            split = path.parent.name
            archive.add(path, arcname=f"images/{split}/{path.name}")
    print(json.dumps(report, indent=2))
    print(f"Wrote {tar_path} ({tar_path.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
