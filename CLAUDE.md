# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Medical image retrieval system using CLIP-based models for image-text alignment on the MIMIC-CXR chest X-ray dataset. The goal is to align chest X-ray images with radiology reports for retrieval (not generation or classification).

## Environment Setup

```bash
source ~/.bashrc
conda activate clip_mimic
```

## Common Commands

### Training

Run from the repo root with the `open_clip` module:

```bash
python -m open_clip_train.main \
  --train-data <csv_file> \
  --csv-img-key filepath \
  --csv-caption-key title \
  --dataset-type csv \
  --model ViT-B-32 \
  --batch-size 64 \
  --epochs 32 \
  --lr 1e-3 \
  --workers 4 \
  --logs ./logs/
```

Key training parameters:
- `--model`: Architecture (e.g., `RN50`, `ViT-B-32`, `ViT-L-14`, `coca_ViT-B-32`)
- `--pretrained`: Load from pretrained weights
- `--save-frequency`: Checkpoint save interval (epochs)
- `--zeroshot-frequency`: Zero-shot eval interval

### Running Tests

```bash
cd open_clip
python -m pytest tests/test_training_simple.py -v
python -m pytest tests/test_inference_simple.py -v
```

Run a single test:
```bash
python -m pytest tests/test_training_simple.py::test_training_simple -v
```

## Architecture

### Repository Structure

- `core/` — Custom data loaders for MIMIC-CXR
  - `new_cxr_data_loader.py` — Simple single-image/text pair loader
  - `data_loader_cxr.py` — Advanced loader with `image` and `subject` modes; converts text CSV paths to image paths, handles multi-view patient grouping
- `open_clip/` — OpenCLIP library (included as submodule/vendored)
  - `src/open_clip/` — Model definitions, tokenizers, loss functions
  - `src/open_clip_train/` — Training pipeline (`main.py`, `train.py`, `params.py`)
  - `tests/` — Test suite
- `cxr_data/all_txt_data_and_labels.csv` — MIMIC-CXR text/label data (~335MB)
- `explore/` — EDA scripts and notebooks

### Dataset

MIMIC-CXR JPG images at `/mnt/walkure_public/users/tomererez/mimic_cxr_jpg_images/`
- 377,110 chest X-ray JPGs
- Path structure: `p{subject_id}/{patient_id}/{study_id}/{image}.jpg`

### Data Loading Modes

Two modes in `core/data_loader_cxr.py`:
- **`image` mode**: Single image + text pairs. Simple, but introduces label noise when a study has multiple views.
- **`subject` mode**: All images for a patient grouped with one report. Clinically more accurate; uses padding/masking for variable image counts.

### CLIP Model Components

- **Image encoder**: ViT, ConvNext, or modified ResNet (configured via JSON in `open_clip/src/open_clip/model_configs/`)
- **Text encoder**: SimpleTokenizer or HuggingFace tokenizer (77-token context)
- **Loss**: `ClipLoss` in `open_clip/src/open_clip/loss.py` — contrastive loss with distributed all-gather and Horovod support
- **Model factory**: `open_clip/src/open_clip/factory.py` — `create_model_and_transforms()` is the main entry point for instantiating models
