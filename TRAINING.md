# Training and benchmarking

`full_modal.py` is the full run orchestrator. It pins the [Hugging Face annotation dataset](https://huggingface.co/datasets/harshnandwana/visual-jev-decisions-v1) to commit `687c745c34846d104ee85af802b9fd444a854f5d` and downloads the 100,008 referenced photos from official COCO URLs into a Modal Volume. The public dataset and GitHub repository contain no photo bytes.

## Pipeline

1. `prepare_full_data` fetches all train, validation, test, image manifest, and release manifest files from Hugging Face, then checks their SHA-256 hashes.
2. It downloads and archives photos in groups of 5,000. Each complete tar shard and its checksum are committed to the Modal Volume. A rerun skips shards whose checksums match, so a client cancellation loses at most the current group.
3. After all image shards are present, staging writes a completion report and starts `train_complete_dataset`. This avoids an idle waiting container.
4. `full_worker.py` runs one LoRA epoch on all 694,255 training rows across four A100 80 GB GPUs, accumulating 16 examples per GPU before each optimizer step. It calculates baseline and tuned negative log-likelihood over **every** validation row and generates base and tuned answers for **every** held-out test row.
5. `publish_full_model.py` refuses to release if any split or task count is incomplete. When checks pass, it prepares the adapter, processor, prediction log, full metrics, benchmark image, source scripts, and model card, then uploads the model repository using the Modal `HF_TOKEN` secret.

The current implementation uses Qwen3.5-0.8B-Base, rank 16 LoRA on `q_proj` and `v_proj`, AdamW at `5e-5`, gradient accumulation 8 per GPU, and 512 pixel maximum image dimensions. Grounding uses mean intersection-over-union; the other four tasks use greedy generation exact match. Read `full_worker.py` for the exact prompt and target serialization.

## Run

Create a Modal account and configure its CLI. If you want automatic model publication, create a secret named `visual-jev-hf-publish` with `HF_TOKEN` set to a token with model-write permission. Use your own model repository ID if you fork the project; the default in `publish_full_model.py` targets the author's account.

```bash
modal deploy full_modal.py
python3 - <<'PY'
import modal
call = modal.Function.from_name("visual-jev-full-data", "prepare_full_data").spawn()
print("staging call:", call.object_id)
PY
```

The deployed App has no idle GPU container. Modal starts a CPU container for staging and starts GPU training only after its completion report has been committed. View current logs with `modal app logs visual-jev-full-data --tail 100` and staged files with `modal volume ls visual-jev-full-v1 /full_dataset`.

If staging is interrupted, call `prepare_full_data` again. It verifies and reuses committed shards. If training fails, inspect the Modal app logs and `/full_run` artifacts on the Volume before rerunning; the current worker saves periodic LoRA checkpoints but does not yet restore optimizer state automatically. Do not report or publish a partial run as a full benchmark.

## Existing evidence

The small pilot L4 run completed 60 optimizer steps. Its 20 row held-out loss fell from 1.293 to 0.201. Across all 97 pilot validation records, constrained first-token choice accuracy rose from 46/97 to 71/97; a 20 record real-versus-blank image check gave 18/20 versus 11/20. These are pilot results only. The full dataset benchmark is in progress and must be reported from its own `metrics.json` after completion.
