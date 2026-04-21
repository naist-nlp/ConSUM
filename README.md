# Enhancing Factuality through Consensus and Consistency in Summarization Using Minimum Bayes Risk Decoding

## TL;DR
---------------------

ConSUM implements Minimum Bayes Risk (MBR) decoding for factuality-aware abstractive summarization. The code samples multiple candidate summaries per source, scores them with a mix of referential and referenceless metrics, and selects MBR-optimal candidates to improve factual consistency in generated summaries.

```bibtex
@inproceedings{TODO2026ConSUM,
   title={Factuality-based Summarization with MBR Decoding},
   author={Author, A. and Author, B.},
   year={2026},
   booktitle={Conference Name},
   url={https://github.com/naist-nlp/ConSUM}
}
```

## Requirements & quick setup
--------------------------

- Python: `>=3.12` (see `pyproject.toml`)
- GPU recommended for sampling/training; CPU is sufficient for small demos.

Minimum install steps:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Quickstart (minimal demo)
-------------------------

This example samples two candidate summaries for each source in the small CNN/DailyMail validation files included under `testing/data/validation` and writes the results to JSON.

```bash
python src/generate.py \
   --dataset_path testing/data/validation/cnn_dm \
   --model bart \
   --candidate_count 2 \
   --decoding_method ancestral \
   --num_beams 1 \
   --output_path testing/original/validation/ancestral/cnn_dm_demo_bart \
   --format json
```

Output (example): `testing/original/validation/ancestral/cnn_dm_demo_bart/candidate_2.json`

Notes
- `--dataset_path` is a prefix: the code reads `${dataset_path}.src` and `${dataset_path}.tgt`.

## Reproduce results
---

Overview of the full pipeline steps to reproduce the main results in the paper:

1. Sample candidate summaries with `testing/script/mbrs_plugin/generate.sh` (wraps `src/generate.py`).
2. Run MBR decoding using `testing/script/mbrs_plugin/decode.sh` (uses templates in `testing/script/mbrs_plugin/configs/`).
3. Compute metric scores with `testing/script/mbrs_plugin/score.sh` (invokes `src/score.hypotheses.py` or `mbrs` score entrypoints).

If you want to pretrain your own simcls models:

1. Preprocess raw datasets → produce `.src` / `.tgt` pairs using scripts in `simcls/preprocessed_dataset/`.
2. Train the models used for generation (see `simcls/train_model/train.sh`).

Example local commands (no SLURM):

```bash
# 1) sample candidates (local)
SLURM_ARRAY_TASK_ID=1 bash testing/script/mbrs_plugin/generate.sh

# 2) run MBR decoding (local)
SLURM_ARRAY_TASK_ID=1 bash testing/script/mbrs_plugin/decode.sh

# 3) compute scores for hypotheses or the MBR outputs
SLURM_ARRAY_TASK_ID=1 bash testing/script/mbrs_plugin/score.sh
```

Scripts write structured outputs under `testing/mbrs_plugin/<subset>/<decoding_method>/`.

Directory map (top-level)
------------------------

- `pyproject.toml`, `requirements.txt` — project metadata and dependencies.
- `README.md`— documentation (this file).
- `simcls/` — pretraining simcls model:
   - `simcls/preprocessed_dataset/` — preprocessing scripts per dataset/split.
   - `simcls/train_model/train.sh` — training wrapper.
- `src/` — mbr code and CLIs:
   - `src/generate.py` — candidate sampling.
   - `src/score.hypotheses.py` — scoring/evaluation CLI.
   - `src/get_dataset.py`, `src/calculate_dataset_length.py` — dataset utilities to support data loading and processing.
   - `src/consum/` — MBR plugin codebase (decoders, metrics, modules, utils).
- `testing/` — example configs, small validation data, and pipeline:
   - `testing/data/validation/` — small `.src`/`.tgt` pairs for demos.
   - `testing/script/mbrs_plugin/` — `generate.sh`, `decode.sh`, `score.sh` and `configs/`.
   - `testing/mbrs_plugin/` —  outputs (candidates, mbr results, scores).

Usage examples (detailed)
------------------------

- Generate 4 candidates for XSum with BART:

```bash
export HF_HUB_CACHE="$HOME/.cache/huggingface"
python src/generate.py \
   --dataset_path testing/data/validation/xsum \
   --model bart \
   --candidate_count 4 \
   --decoding_method ancestral \
   --output_path testing/original/validation/ancestral/xsum_demo_bart \
   --format json
```

- Run MBR decoding (uses the `configs/` templates):

```bash
SLURM_ARRAY_TASK_ID=1 bash testing/script/mbrs_plugin/decode.sh
# outputs: testing/mbrs_plugin/<subset>/<decoding_method>/mbr/
```

- Score hypotheses (per-metric JSON results):

```bash
SLURM_ARRAY_TASK_ID=1 bash testing/script/mbrs_plugin/score.sh
# JSON results: testing/mbrs_plugin/<subset>/<decoding_method>/scores/
```