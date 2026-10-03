# Budget-limited model release

The complete annotation dataset remains available at [Visual Jev decisions v1](https://huggingface.co/datasets/harshnandwana/visual-jev-decisions-v1). The full 694,255-row training epoch and full 24,137-row validation / 6,254-row test benchmark do not fit the roughly $20 Modal credit budget with the current per-image worker. This run produced an honestly labeled [sampled model](https://huggingface.co/harshnandwana/visual-jev-budget20-qwen35-0.8b-lora) instead.

## Fixed scope and ceiling

- Stratified reservoir sample from the pinned Hugging Face revision `687c745c34846d104ee85af802b9fd444a854f5d`: 8,000 training rows and 200 validation and test rows **per task**, totaling 40,000 / 1,000 / 1,000.
- The existing 100,008 photo shards stay on the private Modal Volume. `prepare_budget_dataset` copies only the selected images into a separate archive. Neither the dataset nor the model repository receives photo bytes.
- One LoRA epoch on two L4 GPUs, with an effective global batch of 64. The GPU function is capped at six hours. Selected validation loss and selected test generation are both run for the base model and adapter.
- A short one-L4 microbatch probe tests target alignment, memory, and examples per second before the larger run. Use its results to choose a safe per-GPU microbatch; do not infer throughput from VRAM occupancy alone.
- At [Modal's listed rates](https://modal.com/pricing), two L4 GPUs, eight CPU cores, and 32 GiB requested RAM cost about $2.23/hour, or **$13.39 at the six-hour timeout**. The one-hour CPU staging and 15-minute probe caps add less than $0.55 at listed rates. Publication is CPU-only. Actual charges may vary with usage and platform metering; inspect the live credit balance before starting.
- The training function saves LoRA checkpoints every 100 optimizer steps. The model release is automatic only after all selected counts and predictions pass validation. A timeout or incomplete benchmark is never described as a complete result.

## Run and audit

Run these commands from the repository root after installing `requirements.txt` and configuring Modal. For your own fork of the same five-task schema, set `VISUAL_JEV_DATASET_REPO`, `VISUAL_JEV_DATASET_REVISION`, and `VISUAL_JEV_MODEL_REPO` in the shell before deployment. You can also set `VISUAL_JEV_MODAL_VOLUME` to isolate your staged photos and results.

```bash
modal run scripts/modal/budget_modal.py::probe
modal run scripts/modal/budget_modal.py::stage
modal run scripts/modal/budget_modal.py::smoke
modal deploy scripts/modal/budget_modal.py
modal run scripts/modal/budget_modal.py::train
```

The Modal app is `visual-jev-budget20`; staged metadata and the selected image archive are under `/budget20` in Volume `visual-jev-full-v1`. The publisher uses the Modal secret `visual-jev-hf-publish` for `HF_TOKEN` and targets [the budget model repository](https://huggingface.co/harshnandwana/visual-jev-budget20-qwen35-0.8b-lora). Check the model card's exact split sizes and per-task metrics after upload. The complete-dataset model is a separate future run that would require a larger compute budget.

The `train` entrypoint calls the deployed app. Its returned function-call ID identifies the persistent GPU run; the local command can exit while training continues.

## Completed sampled run

One epoch on 40,000 sampled training records completed in 625 optimizer steps on two L4 GPUs. Baseline and adapter validation covered all 1,000 selected validation records; greedy generation for both covered all 1,000 selected test records, 200 per task. The adapter and model card are on [Hugging Face](https://huggingface.co/harshnandwana/visual-jev-budget20-qwen35-0.8b-lora). The [metrics JSON](../results/budget40k_metrics.json) and [chart](../results/budget40k_benchmark.png) are also in this repository. The Hugging Face dataset and model repositories contain no photo bytes; this GitHub repository includes a composite example figure with COCO photo thumbnails.

| Task | Base test score | Adapter test score | Metric |
| --- | ---: | ---: | --- |
| Ground box | 0.044 | 0.530 | Mean IoU |
| Box choice | 0.000 | 0.940 | Exact match |
| Spatial Boolean | 0.710 | 0.900 | Exact match |
| Attribute text | 0.005 | 0.785 | Exact match |
| Relation text | 0.000 | 0.800 | Exact match |

These scores describe the selected held-out test subset. They do not establish performance on the full published test split or on new domains.

## Completed 120k follow-up

The [120k adapter](https://huggingface.co/harshnandwana/visual-jev-120k-qwen35-0.8b-lora) completed one epoch on 24,000 sampled training rows per task, totaling 120,000 rows and 1,875 optimizer steps on two L4 GPUs. It evaluated all 1,000 selected validation rows and generated predictions for all 1,000 selected test rows. The 40k and 120k prediction logs have identical sets of 1,000 test IDs. Its [metrics JSON](../results/budget120k_metrics.json) and [benchmark chart](../results/budget120k_benchmark.png) are saved in this repository.

| Task | 40k adapter | 120k adapter | Difference | Metric |
| --- | ---: | ---: | ---: | --- |
| Ground box | 0.530 | 0.598 | +0.068 | Mean IoU |
| Box choice | 0.940 | 0.975 | +0.035 | Exact match |
| Spatial Boolean | 0.900 | 0.945 | +0.045 | Exact match |
| Attribute text | 0.785 | 0.815 | +0.030 | Exact match |
| Relation text | 0.800 | 0.805 | +0.005 | Exact match |

Each task has 200 selected test rows. The base model scores are unchanged because the same test IDs were used. The 120k run took 12,981.8 seconds (3.61 hours) from GPU function start through benchmark and used peak allocated GPU memory of 7.27 GB per rank; peak reserved memory was 14.62 GB on a 23.66 GB L4. Better scores on this subset do not establish improvement on the full published test split or unseen image domains. The released adapter and model card contain no photo bytes.

For case-level inspection, see the [comparison graphic](../results/test_case_comparison.png) and [full example data](../results/test_case_examples.json). The figure uses the published prediction log, pinned test annotations, and locally fetched COCO photos. Its grounding success threshold is IoU ≥ 0.50; the other tasks use the published exact-match score. This case classification is separate from the mean scores above. The photos appear only in the GitHub composite, not on Hugging Face.

## Verified training check

The first two-L4 smoke run completed one optimizer step on 64 examples, then evaluated all 64 selected validation and 64 selected test examples. It is a pipeline check, not the 40,000-row model benchmark. The measured validation mean negative log-likelihood fell from 1.033 to 0.983 on box choice and from 1.674 to 1.619 on attribute text; other per-task values are in [`budget_smoke_microbatch4_metrics.json`](../results/budget_smoke_microbatch4_metrics.json). Peak GPU memory allocation was 12.66 GB and peak reservation was 16.70 GB on an L4 with 23.66 GB total. A later full-data batch exceeded the 24 GB L4 limit, so the production configuration uses microbatch 2 with gradient accumulation 16. The small test generation scores vary by task and must not be treated as final quality estimates.

The revised two-L4 check completed **four optimizer steps on 256 examples** with microbatch 2. It evaluated 256 selected validation rows and generated predictions for 256 selected test rows. The measured results are in [`budget_smoke_metrics.json`](../results/budget_smoke_metrics.json):

| Task | Validation NLL, base → adapter | Test score, base → adapter |
| --- | ---: | ---: |
| Ground box | 0.953 → 0.934 | Mean IoU 0.034 → 0.083 |
| Box choice | 1.038 → 0.754 | Exact match 0.000 → 0.078 |
| Spatial Boolean | 0.400 → 0.328 | Exact match 0.679 → 0.660 |
| Attribute text | 1.560 → 1.098 | Exact match 0.020 → 0.060 |
| Relation text | 4.117 → 3.690 | Exact match 0.000 → 0.000 |

This confirms the training and evaluation path runs end to end. Its tiny training set does not support a quality claim. Peak reserved GPU memory was 15.34 GB of 23.66 GB; the production run's first optimizer step reserved 8.79 GB at most across its two L4 ranks.
