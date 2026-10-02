# Visual Jev

Open source code for building and training an image grounded decision dataset from COCO 2017, Visual Genome v1.2, and separately reviewed `gpt-6-luna` candidates. The released annotation dataset is [Visual Jev decisions v1](https://huggingface.co/datasets/harshnandwana/visual-jev-decisions-v1): **694,255 train**, **24,137 validation**, and **6,254 test** records across grounding, box choice, spatial, attribute, and relation tasks.

The Hugging Face dataset and this GitHub repository contain **no photo files**. The dataset's `images.jsonl` gives official COCO URLs and per-image license information. The optional Luna candidate shard has 6,823 records and remains outside all training and evaluation splits pending human audit.

## What is here

| File | Purpose |
| --- | --- |
| `dataset.py`, `build_full_dataset.py`, `build_vg.py`, `package_hf_dataset.py` | Validate and build the annotation dataset. |
| `generate_luna.py`, `export_luna.py` | Generate and export candidate questions with the Codex Python SDK. |
| `train_modal.py`, `train_full_modal.py`, `full_modal.py`, `full_worker.py`, `budget_modal.py` | Pilot, sampled, complete, and budget-limited Modal training. |
| `plot_results.py`, `plot_public_pilot.py`, `plot_full_benchmark.py`, `publish_full_model.py` | Plot and publish measured results. |
| `fixtures/`, `test_dataset.py` | Synthetic fixture and local tests. |
| `data/train.jsonl`, `data/validation.jsonl` | Small pilot annotations; photos are separate. |
| `data/hf_release/manifest.json` | Pinned full release counts and checksums used by the Modal code. |

Large JSONL splits are hosted on Hugging Face. COCO and Visual Genome source files, local photos, Modal Volume shards, checkpoints, and model weights are deliberately excluded from GitHub. See [DATA.md](DATA.md) for sources, licenses, and split construction.

## Quick start

Python 3.11 or newer is recommended.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest -q
.venv/bin/python dataset.py build-coco \
  --annotations fixtures/coco_instances.json \
  --images-dir fixtures --source-split train2017 \
  --output /tmp/visual-jev-demo.jsonl
.venv/bin/python dataset.py validate /tmp/visual-jev-demo.jsonl
```

The fixture is synthetic and tests the conversion path; it is not model training data.

## Get the full annotation dataset

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="harshnandwana/visual-jev-decisions-v1",
    repo_type="dataset",
    revision="687c745c34846d104ee85af802b9fd444a854f5d",
    local_dir="data/hf_release",
)
```

The revision above pins the exact release used by `full_modal.py`. Image paths in the JSONL rows are relative to this repository root. To fetch referenced photos directly from the official COCO source and check them locally:

```bash
.venv/bin/python dataset.py download-images data/hf_release/train.jsonl --workers 12
.venv/bin/python dataset.py download-images data/hf_release/validation.jsonl --workers 8
.venv/bin/python dataset.py download-images data/hf_release/test.jsonl --workers 8
.venv/bin/python dataset.py validate data/hf_release/train.jsonl --check-images
```

For rebuilding from raw annotations, use `build_full_dataset.py`, `build_vg.py`, then `package_hf_dataset.py`. These scripts expect the original COCO annotations and Visual Genome source files under `data/`; consult their path constants before running. The public Hugging Face release is the easier reproducible starting point.

## Training and evaluation

The pilot script `train_modal.py` demonstrates a small L4 LoRA run. Its measured results are in `data/train_smoke_metrics.json` and `data/choice_eval.json`; the [public pilot plot](data/pilot_metrics.png) has no photo thumbnails. These small-sample results do not establish full-dataset performance.

The two-L4 budget run has a verified one-step training result in [BUDGET_TRAINING.md](BUDGET_TRAINING.md). The larger 40,000-row run uses the same batch configuration and publishes its full selected-subset benchmark after successful completion.

`full_modal.py` is the complete-dataset training path. It stages photos outside Hugging Face and can train all 694,255 rows on four A100 80 GB GPUs, followed by full validation and test benchmarking. The complete run was stopped to stay within the available Modal credit budget. **No full-dataset metric or model release is claimed.** The bounded release path in [`BUDGET_TRAINING.md`](BUDGET_TRAINING.md) uses a reproducible 40,000-row subset and two L4 GPUs, with an explicit six-hour GPU timeout and sampled evaluation.

Set up [Modal](https://modal.com/docs/guide) and a Hugging Face token with model-write access before running the automatic publication path. The current project configuration uses a Modal secret named `visual-jev-hf-publish` containing `HF_TOKEN`; keep the token out of source control. Edit `REPO_ID`, `DATASET_REVISION`, the model repo ID, and Modal Volume name if you fork the project. Then:

```bash
modal deploy full_modal.py
python3 - <<'PY'
import modal
call = modal.Function.from_name("visual-jev-full-data", "prepare_full_data").spawn()
print(call.object_id)
PY
```

The deployed function uses one staging container and commits progress to the Modal Volume. Its final stage launches training, and successful training launches benchmark validation and model publication. No idle waiting container is needed. GPU time and storage are billed by Modal. See [TRAINING.md](TRAINING.md) for the training configuration and recovery steps.

## Data quality and scope

COCO box tasks are derived from annotated visible instances. An unannotated object is never treated as absent. Visual Genome attribute and relationship labels are noisy and marked as pending audit. Luna generated records were checked by another run of the same model family, so they still require human review. Do not place `answer_index`, `answer_box_xyxy`, `answer_text`, `evidence`, or `provenance` in the model input.

The repository contains research code and examples. Source annotations and photos retain their own terms; this repository does not grant rights to third party photos. See [DATA.md](DATA.md).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The code license is in [LICENSE](LICENSE).
