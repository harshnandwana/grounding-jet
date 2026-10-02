# Contributing

Issues and pull requests for reproducibility, dataset validation, training reliability, and clearer documentation are welcome.

Before proposing code changes, run `python3 -m unittest -q` and `python3 -m py_compile *.py`. Keep tests small and independent of COCO downloads, Codex usage, and Modal GPU time.

Do not commit photos, upstream raw archives, full JSONL splits, model weights, access tokens, Modal secrets, or private run logs. Link to the Hugging Face dataset and official COCO/Visual Genome sources instead. For label corrections, include the record ID, source annotation IDs, and a short explanation; avoid copying third party photos into issues.

Changes to train/test selection should preserve image disjointness and update the dataset manifest and documentation. New model results should state the exact dataset revision, split counts, metric definition, and hardware.
