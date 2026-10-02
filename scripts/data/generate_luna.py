"""Resumable gpt-6-luna visual candidate generation via the official Codex Python SDK.

Only independently rechecked examples are written to luna_verified.jsonl. A
separate audit is still required before mixing these model labels into SFT.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import threading

from openai_codex import ApprovalMode, Codex, LocalImageInput, Sandbox, TextInput


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "full"
MODEL = "gpt-6-luna"
STOP = threading.Event()
TASKS = {"attribute_choice", "referring_choice", "relation_choice"}
GEN_SCHEMA = {
    "type": "object",
    "properties": {"examples": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "task": {"type": "string", "enum": sorted(TASKS)},
            "question": {"type": "string"},
            "choices": {"type": "array", "items": {"type": "string"}},
            "answer": {"type": "string"},
            "evidence_annotation_ids": {"type": "array", "items": {"type": "integer"}},
        },
        "required": ["task", "question", "choices", "answer", "evidence_annotation_ids"],
        "additionalProperties": False,
    }}},
    "required": ["examples"],
    "additionalProperties": False,
}
VERIFY_SCHEMA = {
    "type": "object",
    "properties": {"verdicts": {"type": "array", "items": {
        "type": "object",
        "properties": {"index": {"type": "integer"}, "keep": {"type": "boolean"}, "reason": {"type": "string"}},
        "required": ["index", "keep", "reason"],
        "additionalProperties": False,
    }}},
    "required": ["verdicts"],
    "additionalProperties": False,
}
GEN_INSTRUCTIONS = """You produce candidate visual decision questions for a grounded image model.
Inspect the attached image and use the verified COCO object annotations below.
Return at most {limit} diverse examples across attributes, referring expressions,
and relations when they are clearly visible. Use more than one question per
object only when each asks about a distinct visual property. Every answer must be visible in
the image and supported by one or two listed annotation IDs. Prefer distinctive
colors, material, or visual appearance. Do not ask about absence, exact counts,
hidden parts, intent, actions, occlusion, subtle depth, OCR, or uncertain facts.
For each question give 2-4 plausible short choices and exactly one answer.
Do not simply restate a COCO category or ask where an object is; those tasks
already exist in the deterministic dataset. Return fewer examples if needed.
Verified annotations: {objects}"""
VERIFY_INSTRUCTIONS = """You are an independent visual data reviewer.
Inspect the attached image. Assess each candidate below on its own. Keep only
if the question has exactly one visually supported answer, its cited objects
are visible, and no other offered choice could reasonably be right. Reject
ambiguous colors, subjective attributes, unprovable relations, category-only
questions, and any answer needing assumptions beyond the image. Be strict.
Return one verdict for every index. Candidates: {candidates}"""


def grouped_train_images():
    current_id = None
    objects = []
    image = None
    with (DATA / "train.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            next_id = record["image"]["id"]
            if current_id is not None and next_id != current_id:
                yield current_id, image, objects
                objects = []
            current_id = next_id
            image = record["image"]
            if record["task"] == "ground_bbox":
                box = record["answer_box_xyxy"]
                if (box[2] - box[0]) * (box[3] - box[1]) >= 0.015:
                    objects.append({"annotation_id": record["evidence"]["annotation_ids"][0],
                                    "label": record["target_label"], "bbox_xyxy": box})
    if current_id is not None:
        yield current_id, image, objects


def validate_candidate(item, allowed_ids):
    if not isinstance(item, dict) or item.get("task") not in TASKS:
        return False
    question, choices, answer = item.get("question"), item.get("choices"), item.get("answer")
    ids = item.get("evidence_annotation_ids")
    if not isinstance(question, str) or not 12 <= len(question) <= 220 or "?" not in question:
        return False
    if not isinstance(choices, list) or not 2 <= len(choices) <= 4 or not all(isinstance(c, str) and 1 <= len(c) <= 60 for c in choices):
        return False
    if len({c.casefold().strip() for c in choices}) != len(choices) or answer not in choices:
        return False
    return isinstance(ids, list) and 1 <= len(ids) <= 2 and all(isinstance(i, int) and i in allowed_ids for i in ids)


def token_counts(result):
    usage = result.usage.total if result.usage else None
    return {"input": usage.input_tokens, "output": usage.output_tokens} if usage else {}


def run_one(codex, image_id, image, objects):
    if STOP.is_set():
        return {"image_id": image_id, "status": "limit_stop", "records": []}
    path = Path(image["path"])
    if not path.is_file():
        return {"image_id": image_id, "status": "missing_image", "records": []}
    selected = objects[:8]
    limit = min(8, 2 * len(selected))
    annotations = {x["annotation_id"]: x for x in selected}
    generation = codex.thread_start(model=MODEL, cwd=str(ROOT), sandbox=Sandbox.read_only,
                                    approval_mode=ApprovalMode.deny_all, ephemeral=True)
    candidate_result = generation.run([
        TextInput(GEN_INSTRUCTIONS.format(limit=limit, objects=json.dumps(selected, separators=(",", ":")))),
        LocalImageInput(str(path)),
    ], output_schema=GEN_SCHEMA, effort="low")
    if not candidate_result.final_response:
        return {"image_id": image_id, "status": "empty_generation", "records": []}
    candidates = json.loads(candidate_result.final_response).get("examples", [])
    candidates = [c for c in candidates if validate_candidate(c, annotations)]
    if not candidates:
        return {"image_id": image_id, "status": "no_valid_candidates", "records": [], "generation_usage": token_counts(candidate_result)}
    verifier = codex.thread_start(model=MODEL, cwd=str(ROOT), sandbox=Sandbox.read_only,
                                  approval_mode=ApprovalMode.deny_all, ephemeral=True)
    verification_result = verifier.run([
        TextInput(VERIFY_INSTRUCTIONS.format(candidates=json.dumps(candidates, separators=(",", ":")))),
        LocalImageInput(str(path)),
    ], output_schema=VERIFY_SCHEMA, effort="low")
    verdicts = json.loads(verification_result.final_response or "{}").get("verdicts", [])
    accepted = {v["index"] for v in verdicts if isinstance(v, dict) and v.get("keep") is True
                and isinstance(v.get("index"), int)}
    records = []
    seen_questions = set()
    for index, candidate in enumerate(candidates):
        if index not in accepted:
            continue
        question = re.sub(r"\s+", " ", candidate["question"]).strip()
        if question.casefold() in seen_questions:
            continue
        seen_questions.add(question.casefold())
        raw_id = f"{image_id}:{candidate['task']}:{question.casefold()}"
        digest = hashlib.sha256(raw_id.encode()).hexdigest()[:16]
        ids = candidate["evidence_annotation_ids"]
        records.append({
            "schema_version": "1.1",
            "id": f"coco2017-train2017-{image_id}-luna-{digest}",
            "split": "train",
            "image": image,
            "task": candidate["task"],
            "question": question,
            "choices": candidate["choices"],
            "answer_index": candidate["choices"].index(candidate["answer"]),
            "candidates": [],
            "evidence": {"annotation_ids": ids, "boxes_xyxy": [annotations[i]["bbox_xyxy"] for i in ids]},
            "provenance": {"method": "model_generated", "source": "coco_instances+image",
                           "generator_model": MODEL, "verifier_model": MODEL,
                           "generation_turn_id": candidate_result.id,
                           "verification_turn_id": verification_result.id,
                           "review_status": "model_verified_pending_audit"},
        })
    return {"image_id": image_id, "status": "ok", "records": records,
            "generation_usage": token_counts(candidate_result),
            "verification_usage": token_counts(verification_result)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-images", type=int, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--cached-only", action="store_true")
    args = parser.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    ledger_path = DATA / "luna_runs.jsonl"
    done = set()
    if ledger_path.exists():
        with ledger_path.open(encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                if item["status"] not in ("error", "missing_image", "limit_stop"):
                    done.add(item["image_id"])
    targets = []
    for image_id, image, objects in grouped_train_images():
        if image_id in done or len(objects) < 2:
            continue
        if args.cached_only and not Path(image["path"]).is_file():
            continue
        targets.append((image_id, image, objects))
        if len(targets) >= args.max_images:
            break
    print(f"Selected {len(targets)} images; {len(done)} already completed", flush=True)
    if not targets:
        return
    with Codex() as codex, ThreadPoolExecutor(max_workers=args.workers) as pool, ledger_path.open("a", encoding="utf-8") as ledger:
        futures = {pool.submit(run_one, codex, *target): target[0] for target in targets}
        for future in as_completed(futures):
            image_id = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                message = str(exc).casefold()
                if any(token in message for token in ("usage limit", "rate limit", "quota", "limit reached")):
                    STOP.set()
                result = {"image_id": image_id, "status": "error", "error": str(exc)[:500], "records": []}
            ledger.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
            ledger.flush()
            print(f"{image_id}: {result['status']} {len(result['records'])} records", flush=True)


if __name__ == "__main__":
    main()
