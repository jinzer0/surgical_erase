# Surgical Erase

> [!WARNING]
> This project is no longer updated and should be treated as a research artifact. It contains experimental code, not a production safety system.

## Introduction

This repository contains research code for suppressing nudity-related concepts in Stable Diffusion v1.x without fine-tuning model weights. It explores a training-free intervention on the text-conditioning path of Stable Diffusion. The repository builds a concept subspace from contrastive prompts, edits `encoder_hidden_states` during denoising, and evaluates the generated images with NudeNet.

## Method Summary

At a high level, the method implemented here is a **Safe-EOS Anchor Alignment** style intervention:

1. **Subspace Construction:** Build a nudity-related subspace from unsafe vs. neutral prompt pairs using CLIP EOS embeddings to derive principal directions associated with unsafe concepts.
2. **Token Scoring:** During inference, project each token embedding onto the subspace and estimate a nudity activation score.
3. **Embedding Intervention:** Only the highest-scoring tokens are edited at each denoising step with gating, scheduling, and trust-region constraints.
4. **Generation:** The edited conditional text embeddings are passed to Stable Diffusion through a custom diffusers pipeline, without updating the model weights.

## Installation and Execution

### Environment Setup

The codebase is script-driven. You can run commands directly from the repository root.

```bash
conda create -n surgical-erase python=3.10
conda activate surgical-erase

# Install dependencies
conda install pytorch pandas numpy matplotlib tqdm -c pytorch
pip install diffusers transformers accelerate pillow nudenet onnxruntime-gpu notifiers python-dotenv psycopg2-binary
```

*(Note: If you are not using CUDA-backed ONNX, replace `onnxruntime-gpu` with `onnxruntime`.)*

### Examples

#### 1. Build a reusable subspace

First, build a concept subspace from contrastive prompts.

```bash
python scripts/build_subspace.py \
  --json_path data/modifiers_v2.json \
  --num_pairs 200 \
  --k 5 \
  --output_path data/subspace.pt
```

#### 2. Run inference on a single prompt

Generate an image with the intervention applied using the built subspace.

```bash
python scripts/run_inference.py \
  --prompts "a photo of a nude woman" \
  --subspace_path data/subspace.pt \
  --output_dir outputs/demo \
  --fp16 \
  --visualize
```

#### 3. Run inference on a prompt dataset

Run inference over a CSV dataset of prompts.

```bash
python scripts/run_inference.py \
  --csvfile data/prompts/unsafe_prompt315.csv \
  --num_prompts 20 \
  --subspace_path data/subspace.pt \
  --output_dir outputs/batch \
  --fp16
```

#### 4. Evaluate generated images

Evaluate the generated batch of images using NudeNet.

```bash
python scripts/evaluate_nudenet.py --image_dir outputs/batch
```
