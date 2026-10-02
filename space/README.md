---
title: Visual Jev
emoji: 👁️
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
models:
  - harshnandwana/visual-jev-budget20-qwen35-0.8b-lora
---

# Visual Jev demo

Try the published Qwen3.5 0.8B vision LoRA with your own image or a synthetic diagram. The preset images are generated with Pillow; no photographs are included in this Space repository. Presets demonstrate the input format and are not benchmark examples.

This app is configured for CPU Basic to avoid an always-on GPU bill. First load and inference can take time. Creating a Gradio Space on this account requires Hugging Face PRO; see the [repository instructions](https://github.com/harshnandwana/grounding-jet#gradio-demo). For faster local inference, use the [CLI](https://github.com/harshnandwana/grounding-jet/blob/main/scripts/infer.py) or [Colab notebook](https://colab.research.google.com/github/harshnandwana/grounding-jet/blob/main/notebooks/try_visual_jev.ipynb).

The model predicts normalized boxes, box choices, spatial yes/no/unknown, colors, and short relations. See the [model card](https://huggingface.co/harshnandwana/visual-jev-budget20-qwen35-0.8b-lora) for training and evaluation details.
