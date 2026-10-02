"""Export model-rechecked Luna candidates from the resumable run ledger."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from dataset import validate_record


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "full"


def main():
    source = DATA / "luna_runs.jsonl"
    target = DATA / "luna_candidates.jsonl"
    seen = set()
    counts = Counter()
    images = set()
    with source.open(encoding="utf-8") as input_file, target.open("w", encoding="utf-8") as output_file:
        for line in input_file:
            run = json.loads(line)
            counts[f"run_{run['status']}"] += 1
            if run["status"] != "ok":
                continue
            for record in run["records"]:
                if record["id"] in seen:
                    continue
                errors = validate_record(record)
                if errors:
                    raise ValueError(f"{record['id']}: {errors}")
                seen.add(record["id"])
                images.add(record["image"]["id"])
                counts[record["task"]] += 1
                output_file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps({"records": len(seen), "images": len(images), "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
