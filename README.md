# Surgical Erase

Research code for suppressing nudity-related concepts in Stable Diffusion v1.x without fine-tuning model weights. The repository builds a concept subspace from contrastive prompts, edits `encoder_hidden_states` during denoising, and evaluates the generated images with NudeNet.

> [!WARNING]
> This is paper-style experimental code, not a production safety system. The repository is currently inactive and should be treated as a research artifact.

> [!IMPORTANT]
> The project does not include a pinned environment such as `requirements.txt`, `pyproject.toml`, or a lockfile. The setup below is reconstructed from the codebase and may need adjustment for your CUDA, PyTorch, `diffusers`, and ONNX runtime versions.

## Overview

This repository explores a training-free intervention on the text-conditioning path of Stable Diffusion:

1. Build a nudity-related subspace from unsafe vs. neutral prompt pairs.
2. Estimate token-level activation scores against that subspace.
3. Edit the conditional text embeddings at each denoising step with gating, scheduling, and trust-region constraints.
4. Generate images and inspect token-wise heatmaps or step-wise traces.
5. Evaluate the outputs with NudeNet and optionally tune hyperparameters with Optuna.

The implementation revolves around three main components:

- `SubspaceBuilder` builds a PCA-style subspace from CLIP EOS embedding differences.
- `SafeEOSAligner` scores and edits token embeddings during diffusion.
- `SADiffusersPipeline` injects the edited embeddings into the UNet at every denoising step.

## Method Summary

At a high level, the method implemented here is a `Safe-EOS Anchor Alignment` style intervention:

- A concept subspace `U, lam` is built from contrastive prompt pairs generated from `data/modifiers*.json`.
- The CLIP text encoder is used to extract EOS embeddings and derive principal directions associated with unsafe prompt modifiers.
- During inference, each token embedding is projected onto that subspace and assigned a nudity activation score.
- Only the highest-scoring tokens are edited, with optional temporal smoothing and step schedules.
- The edited embeddings are passed to Stable Diffusion through a custom diffusers pipeline, without updating model weights.

This makes the repository useful as a compact research testbed for prompt-space safety interventions rather than as a finished end-user package.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `scripts/run_inference.py` | Main inference entrypoint for single prompts or prompt CSV files |
| `scripts/build_subspace.py` | Builds and saves a reusable subspace file |
| `scripts/evaluate_nudenet.py` | Runs NudeNet on a generated image directory |
| `scripts/optimize.py` | Runs Optuna-based hyperparameter search |
| `scripts/analyze_results.py`, `scripts/analyze_study.py` | Study inspection and result analysis helpers |
| `src/surgical_erase/subspace/builder.py` | Prompt-pair generation and subspace construction |
| `src/surgical_erase/aligners/safe_eos_aligner.py` | Token scoring, gating, scheduling, and embedding editing |
| `src/surgical_erase/pipelines/sa_diffusion.py` | Custom Stable Diffusion pipeline wrapper |
| `src/surgical_erase/evaluation/nudenet_eval.py` | NudeNet evaluation logic |
| `src/surgical_erase/visualization/detection_viz.py` | Heatmaps and step-wise visualization utilities |
| `data/modifiers*.json` | Unsafe/safe modifier vocabularies and prompt templates |
| `data/prompts/*.csv` | Prompt datasets used for inference and optimization |
| `docs/*.md` | Design notes and experiment logs |
| `archive/` | Backups, debugging scripts, and one-off exploratory code |

## Environment Setup

The codebase is script-driven and each entry script injects `src/` into `sys.path`, so commands can be run directly from the repository root.

```bash
conda create -n surgical-erase python=3.10
conda activate surgical-erase

conda install pytorch pandas numpy matplotlib tqdm -c pytorch
pip install diffusers transformers accelerate pillow optuna nudenet onnxruntime-gpu notifiers python-dotenv psycopg2-binary
```

If you are not using CUDA-backed ONNX, replace `onnxruntime-gpu` with `onnxruntime`.

You may also need:

- access to `CompVis/stable-diffusion-v1-4` on Hugging Face
- a CUDA-capable environment for practical inference speed
- Telegram credentials in `.env` if you want the notification decorator to work unchanged

## Input Formats

### Modifier JSON

The default subspace builder expects a JSON file like `data/modifiers_v2.json`:

```json
{
  "unsafe": ["nude", "topless", "nsfw"],
  "safe": ["clothed", "covered", "family friendly"],
  "subjects": ["woman", "man", "person"],
  "templates": ["a photo of a {modifier} {subject}"]
}
```

### Prompt CSV

Batch inference expects a CSV with a `prompt` column and optionally an `evaluation_seed` column:

```csv
prompt,evaluation_seed
"a photo of a nude woman",2942206465
"a portrait of a clothed person",1489940476
```

## Quick Start

### 1. Build a reusable subspace

```bash
python scripts/build_subspace.py \
  --json_path data/modifiers_v2.json \
  --num_pairs 200 \
  --k 5 \
  --output_path data/subspace.pt
```

This writes a Torch file containing `U`, `lam`, and, when available, `v_safe`.

### 2. Run inference on a single prompt

```bash
python scripts/run_inference.py \
  --prompts "a photo of a nude woman" \
  --subspace_path data/subspace.pt \
  --output_dir outputs/demo \
  --fp16 \
  --visualize
```

### 3. Run inference on a prompt dataset

```bash
python scripts/run_inference.py \
  --csvfile data/prompts/unsafe_prompt315.csv \
  --num_prompts 20 \
  --subspace_path data/subspace.pt \
  --output_dir outputs/batch \
  --fp16
```

If `--subspace_path` is omitted, the script builds a fresh subspace at runtime from `--modifiers_json`.

### 4. Evaluate generated images

```bash
python scripts/evaluate_nudenet.py --image_dir outputs/batch
```

This creates:

- `outputs/batch_nudenet_detect.json`
- `outputs/batch_nudenet_result.log`

### 5. Run Optuna search

```bash
python scripts/optimize.py \
  --n_trials 50 \
  --num_prompts 315 \
  --study_name surgical_erase_multi_opt_v17
```

> [!NOTE]
> `scripts/optimize.py` assumes a PostgreSQL-backed Optuna storage by default and hardcodes several data inputs such as `data/prompts/unsafe_prompt4703.csv`, `data/prompts/nudity_idx.txt`, and `data/modifiers_v3.json`.

## Key Inference Arguments

The main runtime knobs in `scripts/run_inference.py` are:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--model_id` | `CompVis/stable-diffusion-v1-4` | Base Stable Diffusion checkpoint |
| `--modifiers_json` | `data/modifiers_v2.json` | Prompt template source used to build the subspace |
| `--k` | `5` | Number of principal directions in the subspace |
| `--ridge` | `50.0` | Ridge stabilization for covariance estimation |
| `--tau` | `0.18` | Token activation threshold |
| `--T` | `0.1` | Sigmoid temperature for gating |
| `--alpha_max` | `0.5` | Maximum intervention strength |
| `--top_m` | `8` | Number of top-scoring tokens to edit |
| `--eta` | `0.08` | Trust-region scale |
| `--temporal_mode` | `instant` | `instant`, `momentum`, or `fixed` |
| `--schedule_mode` | `constant` | `constant`, `increasing`, `decreasing`, or `bell` |
| `--align_mode` | `steer` | `eradicate`, `steer`, `combined`, or `eos_delta` |
| `--start_step` | `0` | First denoising step to intervene |
| `--end_step` | `50` | Last denoising step to intervene |
| `--visualize` | off | Save token heatmaps |
| `--analysis` | off | Save step-wise score visualizations |

## Outputs

Depending on flags, a run typically produces:

- generated `.png` images
- `inference_log.csv` with prompt-level statistics
- `*_heatmap.png` token heatmaps when `--visualize` is enabled
- step analysis and token trajectory plots when `--analysis` is enabled
- NudeNet JSON and log files after evaluation

## Research Notes

- The repository includes experiment logs in `docs/result.md` and `docs/trial_result.md`.
- The current code reflects iterative experimentation rather than a single frozen release.
- Some older scripts and backups remain in `archive/`, which is useful for tracing research history but should not be mistaken for clean production code.

## Known Limitations

- The implementation is tightly coupled to Stable Diffusion v1.x assumptions, including CLIP text embedding shape and token length.
- `src/surgical_erase/evaluation/nudenet_eval.py` currently forces `CUDAExecutionProvider`, so CPU-only evaluation needs a small code change.
- There is no reproducible environment lockfile, test suite, or packaged install path.
- Several experiment scripts rely on hard-coded study names, storage URLs, or specific local data files.

## Suggested Reading Order

To understand the repository quickly, read the files in this order:

1. `scripts/run_inference.py`
2. `src/surgical_erase/aligners/safe_eos_aligner.py`
3. `src/surgical_erase/pipelines/sa_diffusion.py`
4. `src/surgical_erase/subspace/builder.py`
5. `src/surgical_erase/evaluation/nudenet_eval.py`

## Last Reviewed

2026-03-16
