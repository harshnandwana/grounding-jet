"""Fetch and verify the published annotation files from Hugging Face.

This module downloads metadata only. It never downloads or publishes photos.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


DATASET_REPO = "harshnandwana/visual-jev-decisions-v1"
DATASET_REVISION = "687c745c34846d104ee85af802b9fd444a854f5d"
REQUIRED_FILES = ("train.jsonl", "validation.jsonl", "test.jsonl", "images.jsonl")
OPTIONAL_FILES = ("luna_candidates.jsonl",)


def verify_download(directory: Path, include_candidates: bool = False) -> dict:
    """Check the split hashes and counts declared by the pinned release."""
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("image_files_included") is not False:
        raise ValueError("expected an annotation-only dataset without photo files")
    names = REQUIRED_FILES + (OPTIONAL_FILES if include_candidates else ())
    for name in names:
        path = directory / name
        expected = manifest["files_sha256"][name]
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError(f"SHA-256 mismatch: {path}")
    for split in ("train", "validation", "test"):
        path = directory / f"{split}.jsonl"
        with path.open("rb") as handle:
            count = sum(1 for _ in handle)
        expected = manifest["splits"][split]["records"]
        if count != expected:
            raise ValueError(f"{split} row count mismatch: {count} != {expected}")
    with (directory / "images.jsonl").open("rb") as handle:
        image_count = sum(1 for _ in handle)
    if image_count != manifest["image_manifest_records"]:
        raise ValueError("image manifest row count mismatch")
    if include_candidates:
        with (directory / "luna_candidates.jsonl").open("rb") as handle:
            candidate_count = sum(1 for _ in handle)
        if candidate_count != manifest["luna_candidates"]["records"]:
            raise ValueError("Luna candidate row count mismatch")
    return manifest


def fetch_dataset(directory: Path, include_candidates: bool = False,
                  repo_id: str = DATASET_REPO, revision: str = DATASET_REVISION) -> dict:
    from huggingface_hub import snapshot_download

    directory.mkdir(parents=True, exist_ok=True)
    names = ("manifest.json",) + REQUIRED_FILES + (OPTIONAL_FILES if include_candidates else ())
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        allow_patterns=list(names),
        local_dir=directory,
    )
    return verify_download(directory, include_candidates=include_candidates)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/hf_release"))
    parser.add_argument("--repo-id", default=DATASET_REPO,
                        help="Hugging Face dataset repository, for example your-name/your-dataset")
    parser.add_argument("--revision", default=DATASET_REVISION,
                        help="immutable dataset commit SHA to fetch")
    parser.add_argument("--include-candidates", action="store_true",
                        help="also fetch the unreviewed Luna candidate file")
    args = parser.parse_args()
    manifest = fetch_dataset(args.output, include_candidates=args.include_candidates,
                             repo_id=args.repo_id, revision=args.revision)
    print(json.dumps({"directory": str(args.output.resolve()),
                      "repo_id": args.repo_id,
                      "revision": args.revision,
                      "splits": {name: item["records"] for name, item in manifest["splits"].items()},
                      "photos_included": False}, indent=2))


if __name__ == "__main__":
    main()
