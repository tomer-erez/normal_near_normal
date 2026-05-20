# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Medical image retrieval system using CLIP-based models for image-text alignment on the MIMIC-CXR chest X-ray dataset. The goal is to align chest X-ray images with radiology reports for retrieval (not generation or classification).

## Environment Setup

```bash
source ~/.bashrc
conda activate thesis_clip   # environment defined in environment.yml
```

The environment includes PyTorch ≥2.2 (CUDA 12.1), HuggingFace `transformers`/`peft`, and OpenCLIP (vendored under `open_clip/`).

## Common Commands

### Training (LoRA fine-tuning)

Main script is `train_lora.py` at the repo root. It applies LoRA to an open_clip model and saves merged weights.

```bash
# Vanilla CLIP ViT-B-32
python train_lora.py \
    --base-model ViT-B-32 --pretrained openai \
    --train-csv cxr_data/mimic_cxr_train.csv \
    --image-dir /mnt/walkure_public/users/tomererez/mimic_cxr_jpg_images/mimic_cxr_jpg_images_from_google_cloud/mimic-cxr-jpg-2.1.0.physionet.org/files/ \
    --output-dir ./experiments/lora_vitb32

# BiomedCLIP (fit 11 GB VRAM with grad-accum)
python train_lora.py \
    --base-model hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 \
    --train-csv cxr_data/mimic_cxr_train.csv \
    --image-dir /mnt/walkure_public/users/tomererez/mimic_cxr_jpg_images/mimic_cxr_jpg_images_from_google_cloud/mimic-cxr-jpg-2.1.0.physionet.org/files/ \
    --output-dir ./experiments/lora_biomedclip \
    --batch-size 32 --grad-accum-steps 2 \
    --lora-modules "qkv,proj,fc1,fc2,query,key,value"

# Quick smoke-test (1 epoch, 1000 samples)
python train_lora.py --base-model ViT-B-32 --pretrained openai \
    --train-csv cxr_data/mimic_cxr_train.csv \
    --image-dir /mnt/.../files/ \
    --output-dir ./experiments/lora_smoke --epochs 1 --max-samples 1000

# Multi-GPU via torchrun
torchrun --nproc_per_node=4 train_lora.py ...

# Resume from checkpoint
python train_lora.py ... --resume ./experiments/lora_vitb32/epoch_5_adapter.pt
```

**Training outputs**: `experiments/<name>/epoch_N_adapter.pt` (LoRA adapter ~11 MB) and `experiments/<name>/final_merged.pt` (merged model for eval).

### Evaluation Pipeline

**Step 0** — Build the official test CSV (run once):
```bash
python baseline_eval/build_official_test_set.py
# → cxr_data/mimic_cxr_official_test.csv  (5,159 images from the MIMIC-CXR test split)
```

**Step 1** — Build the image gallery (symlinks, no embedding):
```bash
python baseline_eval/build_baseline.py \
    --csv cxr_data/mimic_cxr_official_test.csv \
    --output_dir ./eval_outputs/exp1 --max_samples 5000
```

**Step 2** — Evaluate a single model:
```bash
# Fine-tuned model
python baseline_eval/eval_model.py \
    --model_type finetuned --name lora_vitb32 \
    --finetuned_base_model ViT-B-32 --finetuned_pretrained openai \
    --finetuned_checkpoint ./experiments/lora_vitb32/final_merged.pt \
    --paired_dir ./eval_outputs/exp1/paired_data \
    --csv cxr_data/mimic_cxr_official_test.csv

# Baselines
python baseline_eval/eval_model.py --model_type vanilla_clip \
    --paired_dir ./eval_outputs/exp1/paired_data --csv cxr_data/mimic_cxr_official_test.csv
python baseline_eval/eval_model.py --model_type biomedclip \
    --paired_dir ./eval_outputs/exp1/paired_data --csv cxr_data/mimic_cxr_official_test.csv
```

**Step 3** — Evaluate all models and compare:
```bash
python baseline_eval/run_all_evals.py \
    --paired_dir ./eval_outputs/exp1/paired_data \
    --csv cxr_data/mimic_cxr_official_test.csv \
    --output_dir ./experiments/my_exp1 --batch_size 64 [--skip_existing]
```

To add a fine-tuned model to the multi-eval run, edit the `MODELS` list in `baseline_eval/run_all_evals.py`.

### Running Tests

```bash
cd open_clip
python -m pytest tests/test_training_simple.py -v
python -m pytest tests/test_inference_simple.py -v
python -m pytest tests/test_training_simple.py::test_training_simple -v   # single test
```

## Architecture

### Repository Structure

- `train_lora.py` — Main LoRA fine-tuning script (argparse, DDP-aware, saves adapter + merged weights)
- `train/`
  - `cxr_label_dataset.py` — `CXRLabelDataset`: loads MIMIC-CXR rows, resolves image paths from `txt_file_path` + `metadata_dicom_id`, builds CheXpert label captions
  - `label_aware_loss.py` — `LabelAwareClipLoss` and `LabelAwareSigLipLoss`: multi-positive and conflict-aware contrastive losses
- `baseline_eval/`
  - `build_official_test_set.py` — Filters the 377k-row CSV to the 5,159 official MIMIC-CXR test images
  - `build_baseline.py` — Creates symlinks for the test image gallery
  - `eval_model.py` — Unified evaluation for vanilla_clip / biomedclip / cxrclip / finetuned backends; reports P@1/5/10 and R@1/5/10
  - `run_all_evals.py` — Loops over a `MODELS` list and calls `eval_model.py` for each
- `core/` — Legacy data loaders (not used by the current LoRA training pipeline)
- `open_clip/` — Vendored OpenCLIP library (`src/open_clip/`, `src/open_clip_train/`, `tests/`)
- `cxr_data/` — CSVs: `all_txt_data_and_labels.csv` (full), `mimic_cxr_train.csv`, `mimic_cxr_official_test.csv`

### Dataset

MIMIC-CXR JPG images: `/mnt/walkure_public/users/tomererez/mimic_cxr_jpg_images/`
- 377,110 chest X-ray JPGs
- Path structure resolved as: `image_dir / p{subject} / p{patient} / s{study} / {dicom_id}.jpg`

### Key Training Knobs

**`--caption-mode`** — text caption paired with each training image:
- `single`: one random positive label (`"atelectasis"`)
- `pair`: two random positive labels (`"fracture and edema"`)
- `both` (default): 75% single, 25% pair — matches eval query distribution

**`--match-mode`** — which batch pairs count as positives:
- `standard`: diagonal only (vanilla CLIP)
- `single_label`: share ≥1 positive label
- `two_label`: share ≥2 positive labels (use with `--caption-mode pair`)
- `negative_aware`: `single_label` + hinge repulsion for conflict pairs (one says "Edema", other says "No edema"); extra flags `--negative-weight λ` (default 0.5) and `--negative-margin τ` (default 0.0)

**`--loss`** — loss function:
- `clip` (default): softmax multi-positive cross-entropy
- `siglip`: sigmoid binary cross-entropy per pair (decoupled from batch size; SigLIP model checkpoints have a learned `logit_bias` that is picked up automatically)

**`--query_mode`** (eval only):
- `single` (13 queries), `pair` (78), `negative` (156, "atelectasis and no edema"), `all` (247, default)

### Label Encoding

| CSV value | Meaning | Tensor value |
|-----------|---------|--------------|
| 1 | Pathology positively mentioned | 1.0 (positive) |
| 0 | Pathology explicitly ruled out | −1.0 (negative) |
| NaN | Label not mentioned | −1.0 (same as 0) |
| −1 | Ambiguous / uncertain | 0.0 (ignored) |

NaN and 0 are both encoded as −1.0 (absence of evidence = evidence of absence). Uncertain labels (−1) are excluded from both positive matching and conflict detection.

### CLIP Model Components

- **Image encoder**: ViT, ConvNext, or modified ResNet (configured via JSON in `open_clip/src/open_clip/model_configs/`)
- **Text encoder**: SimpleTokenizer or HuggingFace tokenizer (77-token context)
- **Loss**: `ClipLoss` / `SigLipLoss` in `open_clip/src/open_clip/loss.py` for standard mode; custom `LabelAwareClipLoss` / `LabelAwareSigLipLoss` in `train/label_aware_loss.py` for label-aware modes
- **Model factory**: `open_clip/src/open_clip/factory.py` — `create_model_and_transforms()` is the entry point; `sys.path` is patched in `train_lora.py` to use the vendored copy
- **LoRA**: applied via `peft` (~1.9% trainable params, ~2.9M / 154M for ViT-B-32); merged into base weights at end of training via `merge_and_unload()`
