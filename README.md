# CXR-Retrieve: Compositional Text-to-Image Retrieval in Chest Radiography

Label-aware contrastive fine-tuning of CLIP/CXR-CLIP on MIMIC-CXR for retrieval of chest X-rays by structured text queries — single findings, co-occurring findings, and negation queries.

---

## Installation

```bash
conda env create -f environment.yml
conda activate thesis_clip
```

Requires CUDA 12.1 and ≥11 GB VRAM (single GPU) or two GPUs for the default multi-GPU setup.

---

## Dataset Setup

**Files needed:**
- `cxr_data/all_txt_data_and_labels.csv` — full MIMIC-CXR CSV with CheXpert labels (from PhysioNet)
- `mimic-cxr-2.0.0-split.csv.gz` — official split file (PhysioNet)
- MIMIC-CXR JPG images at any local path (passed as `--image-dir`)

**Step 1 — Create train/test CSVs (once):**

```bash
python baseline_eval/create_train_test_sets.py \
    --csv       cxr_data/all_txt_data_and_labels.csv \
    --split     <PATH/TO/mimic-cxr-2.0.0-split.csv.gz> \
    --out_train cxr_data/mimic_cxr_train.csv \
    --out_test  cxr_data/mimic_cxr_official_test.csv
```

Produces ~227k train rows and 5,159 official test rows.

**Step 2 — Build the eval image gallery (once):**

```bash
python baseline_eval/build_baseline.py \
    --csv        cxr_data/mimic_cxr_official_test.csv \
    --image_dir  <PATH/TO/MIMIC-CXR-JPG/files/> \
    --output_dir ./eval_outputs/baseline_output_official_test
```

Creates symlinks for all 5,159 test images under `eval_outputs/baseline_output_official_test/paired_data/`.

---

## CXR-CLIP Baseline Checkpoints

The evaluation compares against [CXR-CLIP](https://github.com/Soomit/cxr-clip) pretrained checkpoints.
Download the checkpoint archives from the CXR-CLIP GitHub releases and extract them under `valid_pretrained_models_to_try/`:

```
valid_pretrained_models_to_try/
    r50_m/          ← ResNet-50, MIMIC only
    r50_mc/         ← ResNet-50, MIMIC + CheXpert
    swint_m/        ← Swin-T, MIMIC only
    swint_mc/       ← Swin-T, MIMIC + CheXpert
```

Pack each directory into a `.pt` file that `torch.load()` accepts (run once):

```bash
python baseline_eval/build_cxrclip_checkpoints.py
```

Produces `r50_m.pt`, `r50_mc.pt`, `swint_m.pt`, `swint_mc.pt` in the same directory.
Use `--overwrite` to re-pack after adding new checkpoints.

---

## Training

The best-performing configuration is LoRA fine-tuning from the CXR-CLIP Swin-T checkpoint using the `label_dot` match mode with hard negative mining. Run:

```bash
IMAGE_DIR=<PATH/TO/MIMIC-CXR-JPG/files/> bash run_finetune.sh   # CXR-CLIP SwinT fine-tune (paper model)
IMAGE_DIR=<PATH/TO/MIMIC-CXR-JPG/files/> bash run.sh            # ViT-B/32 from OpenAI CLIP
```

Both scripts launch a tmux session. Key flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--base-model` | `ViT-B-32` | OpenCLIP model name (ignored when `--cxrclip-finetune` is set) |
| `--pretrained` | `openai` | OpenCLIP pretrained tag |
| `--match-mode` | `label_dot` | `K(i,j) = l_i · l_j` soft multi-positive target |
| `--caption-mode` | `all` | 50% single / 25% pair / 25% negation captions |
| `--nan-mode` | `ignore` | NaN labels treated as unknown (not as absent) |
| `--hnm-weight` | `0.3` | Hard negative mining repulsion strength |
| `--batch-size` | `70` | Per-GPU; effective batch = `70 × 2 GPUs = 140` |
| `--lr` / `--min-lr` | `1e-4` / `1e-8` | Cosine schedule bounds |
| `--warmup-epochs` | `5` | Linear LR warmup |
| `--patience` | `10` | Early stopping (epochs without improvement) |
| `--lora-r` / `--lora-alpha` | `16` / `16` | LoRA rank and scaling |

Training saves:
- `experiments/<name>/best_adapter.pt` — LoRA adapter at the best epoch (~11 MB)
- `experiments/<name>/final_merged.pt` — merged weights, ready for evaluation

---

## Evaluation

**Run all models and generate comparison plots:**

```bash
bash eval.sh
```

Or directly:

```bash
python baseline_eval/run_all_evals.py \
    --paired_dir ./eval_outputs/baseline_output_official_test/paired_data \
    --csv        cxr_data/mimic_cxr_official_test.csv \
    --output_dir ./eval_outputs/my_results \
    --batch_size 128 \
    --skip_existing
```

Each model is evaluated in **4 passes** automatically:
- **Pass 1:** single + pair + negative queries (standard template `"A and no B"`, lenient)
- **Pass 2:** negative queries only, rephrased template (`"an image with A but without B"`, lenient)
- **Pass 3:** negative queries only, standard template, strict (NaN not counted as absent)
- **Pass 4:** negative queries only, rephrased template, strict

**To add your fine-tuned model**, add an entry to the `MODELS` list in `baseline_eval/run_all_evals.py`:

```python
{
    "name": "my_run",
    "model_type": "cxrclip_finetune",
    "checkpoint": None,
    "cxrclip_finetune_image_checkpoint": "valid_pretrained_models_to_try/swint_mc.pt",
    "cxrclip_finetune_merged_checkpoint": "experiments/my_run/final_merged.pt",
},
```

**Evaluate a single model:**

```bash
python baseline_eval/eval_model.py \
    --model_type cxrclip_finetune --name my_run \
    --cxrclip_finetune_image_checkpoint valid_pretrained_models_to_try/swint_mc.pt \
    --cxrclip_finetune_merged_checkpoint experiments/my_run/final_merged.pt \
    --paired_dir ./eval_outputs/baseline_output_official_test/paired_data \
    --csv cxr_data/mimic_cxr_official_test.csv
```

**Output files** (in `--output_dir`):

| File | Contents |
|------|----------|
| `summary_single.csv` | P@K / R@K / MAP@K, single-label queries |
| `summary_pair.csv` | P@K / R@K / MAP@K, pair queries |
| `summary_negative.csv` | P@K / R@K / MAP@K / HNRR@K, negation queries (lenient) |
| `summary_negative_strict.csv` | Same queries, strict negation (NaN ≠ absent) |
| `summary_negative_robust.csv` | Rephrased template, lenient |
| `summary_negative_robust_strict.csv` | Rephrased template, strict |
| `summary_macro.csv` | Macro-averaged across all query types, one row per model |
| `plots/radar.pdf` | Spider chart: all models × all metrics (single-column paper figure) |
| `plots/parallel_coordinates.png` | Parallel coordinates comparison |
| `plots/parallel_coordinates_vertical.pdf` | Single-column variant for papers |

**Metrics:**

| Metric | Description |
|--------|-------------|
| P@K | Precision at K — fraction of top-K retrieved images satisfying all query constraints, macro-averaged |
| R@K | Recall at K — 1 if any relevant image appears in top-K, else 0, macro-averaged |
| MAP@K | Mean Average Precision at K — macro-averaged AP@K across queries with at least one relevant image |
| HNRR@K | Hard Negative Retrieval Rate — for `"A and no B"` queries: fraction of top-K results where both A and B are confirmed present; **lower is better** |

**Query types** (10 CheXpert labels active by default):

| Type | Count | Example |
|------|-------|---------|
| single | 10 | `"atelectasis"` |
| pair | 45 | `"atelectasis and edema"` |
| negative | 90 | `"atelectasis and no edema"` |
| negative (robust) | 90 | `"an image with atelectasis but without edema"` |

The set of active labels is controlled by `constants.py` — comment out entries there to exclude labels from both training and evaluation.
