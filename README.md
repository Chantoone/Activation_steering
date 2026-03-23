<div align="center">

# Exploring the Translation Mechanism of Large Language Models

[![Paper](https://img.shields.io/badge/Paper-NeurIPS_2025-4B8BBE)](https://openreview.net/pdf?id=3QjESmXftM)
[![GitHub](https://img.shields.io/badge/GitHub-azurestarz%2Fexploring_translation_mechanism-24292e?logo=github)](https://github.com/AzureStarz/exploring_translation_mechanism)

</div>

This repository contains the code and experiments for **Exploring the Translation Mechanism of Large Language Models**, focusing on how translation knowledge is represented, routed, and can be surgically intervened in decoder-only LLMs.

---

## ✨ Highlights
- **Subspace Path Patching**: Implements task-steering subspace identification and projection-only path patching to isolate translation-specific directions.
- **Translation Circuit Probing**: Re-uses the EasyTransformer-style hooks for path patching, head/MLP knockout, and activation visualization.
- **Reproducible Notebook**: `interpret_translation.ipynb` walks through dataset creation, baseline translation metrics, standard path patching, and subspace path patching.

---

## 🚀 Quick Start
1) Install dependencies (editable mode):
```bash
pip install -e ./
```

2) Run the notebook:
- Open `interpret_translation.ipynb` and execute cells in order.
- Use `translation_dataset.py` to generate the contrastive translation pairs (`translation_dataset` vs. `flipped_translation_dataset`).
- The “Subspace Path Patching” section mirrors the standard path patching pipeline but patches only task subspace directions.

3) Core scripts:
- `translation_utils.py`: translation metrics, activation utilities, path patching, and subspace patching (Algorithms 1 & 2).
- `translation_dataset.py`: dataset construction, token indices (END/src/tgt), flipped contrastive sets.
- `easy_transformer/`: lightweight transformer implementation and hook utilities used throughout the experiments.

---

## 🔍 Experiment Map
- **Baseline translation quality**: `get_translation_logits`, `get_translation_acc`.
- **Standard path patching**: `batch_path_patching` (Heads/MLPs → final residual).
- **Subspace path patching**: `task_steering_subspace_identification`, `subspace_intervene_path_patching_batch`.
- **Visualization**: `show_attention_patterns`, `show_pp` heatmaps for head/MLP effects.

---

## 📂 Repository Layout
- `interpret_translation.ipynb` — end-to-end walkthrough (baseline → path patching → subspace patching).
- `translation_utils.py` — metrics, caching hooks, path patching, subspace algorithms.
- `translation_dataset.py` — dataset builder with contrastive (flipped) pairs and token index helpers.
- `easy_transformer/` — model, hooks, and caching primitives adapted from EasyTransformer.
- `data/` — translation lexicons and prompt templates (see notebook for loading).

---

## 📜 Citation
If you find this repository useful, please cite:
```
@inproceedings{
zhang2025exploring,
title={Exploring the Translation Mechanism of Large Language Models},
author={Hongbin Zhang and Kehai Chen and Xuefeng Bai and Xiucheng Li and Yang Xiang and Min Zhang},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://openreview.net/forum?id=3QjESmXftM}
}
```

---

## 🙏 Acknowledgements
This project builds on the EasyTransformer codebase and related interpretability work:
- *Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 Small* (Wang et al., 2022). GitHub: https://github.com/redwoodresearch/Easy-Transformer/
- Path patching, hook utilities, and circuit-style analyses adapted for translation-focused experiments.
