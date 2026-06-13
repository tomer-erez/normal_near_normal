# CXR-Retrieve: CLIP-based Chest X-ray Retrieval

CLIP fine-tuning with label-aware contrastive learning on MIMIC-CXR.  
Aligns chest X-ray images with radiology report text for retrieval tasks.

---

## Installation

```bash
conda env create -f environment.yml
conda activate thesis_clip
```

> Requires CUDA 12.1 and a GPU with ≥11 GB VRAM (or two GPUs for the default multi-GPU setup).

---

## Dataset Setup

**Data required:**
- `cxr_data/all_txt_data_and_labels.csv` — full MIMIC-CXR CSV with CheXpert labels
- `mimic-cxr-2.0.0-split.csv.gz` — official split file from PhysioNet
- MIMIC-CXR JPG images (default path: `cxr_data/images/mimic_cxr_jpg_images_from_google_cloud/...`)

**Step 1 — Create train/test CSVs (run once):**

```bash
python baseline_eval/create_train_test_sets.py \
    --csv       cxr_data/all_txt_data_and_labels.csv \
    --split     cxr_data/images/mimic_cxr_jpg_images_from_google_cloud/mimic-cxr-jpg-2.1.0.physionet.org/mimic-cxr-2.0.0-split.csv.gz \
    --out_train cxr_data/mimic_cxr_train.csv \
    --out_test  cxr_data/mimic_cxr_official_test.csv
```

Produces ~227k train rows and 5,159 official test rows.

**Step 2 — Build the eval image gallery (run once):**

```bash
python baseline_eval/build_baseline.py \
    --csv        cxr_data/mimic_cxr_official_test.csv \
    --output_dir ./eval_outputs/baseline_output_official_test
```

Creates symlinks and paired text files for the 5,159 test images under `eval_outputs/baseline_output_official_test/paired_data/`.

---

## Training

The best-performing configuration uses LoRA on ViT-B/32 (OpenAI CLIP) with the `label_dot` match mode:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_lora.py \
    --base-model ViT-B-32 --pretrained openai \
    --precision bf16 \
    --lr 1e-4 --min-lr 1e-8 \
    --batch-size 90 --epochs 100 \
    --warmup-epochs 5 --patience 10 \
    --nan-mode ignore \
    --loss clip \
    --caption-mode all \
    --match-mode label_dot \
    --lora-r 16 --lora-alpha 16 \
    --output-dir ./experiments/my_run
```

Or just run the provided script (launches in a tmux session):

```bash
bash run.sh
```

**Key training flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--match-mode` | `standard` | `label_dot` is the recommended mode |
| `--caption-mode` | `both` | `all` adds negation captions (50% single / 25% pair / 25% negation) |
| `--nan-mode` | `negative` | `ignore` treats unlabelled findings as unknown (recommended) |
| `--lora-r` | 16 | LoRA rank |
| `--lora-alpha` | 16 | LoRA scaling factor |
| `--batch-size` | — | Per-GPU; effective batch = `batch_size × num_gpus` |

Training saves `experiments/my_run/final_merged.pt` (merged weights, ready for eval).

---

## Evaluation

**Evaluate all models and compare:**

```bash
python baseline_eval/run_all_evals.py \
    --paired_dir ./eval_outputs/baseline_output_official_test/paired_data \
    --csv        cxr_data/mimic_cxr_official_test.csv \
    --output_dir ./eval_outputs/my_results \
    --batch_size 64
```

Add `--skip_existing` to skip models already evaluated.

**To include your fine-tuned model**, add an entry to the `MODELS` list in `baseline_eval/run_all_evals.py`:

```python
{
    "name": "my_run",
    "model_type": "finetuned",
    "finetuned_base_model": "ViT-B-32",
    "finetuned_pretrained": "openai",
    "finetuned_checkpoint": "experiments/my_run/final_merged.pt",
},
```

**Evaluate a single model:**

```bash
python baseline_eval/eval_model.py \
    --model_type finetuned --name my_run \
    --finetuned_base_model ViT-B-32 --finetuned_pretrained openai \
    --finetuned_checkpoint ./experiments/my_run/final_merged.pt \
    --paired_dir ./eval_outputs/baseline_output_official_test/paired_data \
    --csv cxr_data/mimic_cxr_official_test.csv
```

**Eval metrics:**

| Metric | Description |
|--------|-------------|
| P@K | Precision at K — fraction of top-K retrieved images that are relevant |
| R@K | Recall at K — 1 if any relevant image appears in top-K, else 0 (binary, averaged over queries) |
| HNRR@K | Hard Negative Retrieval Rate — fraction of top-K results that violate the negation in "A and no B" queries; lower is better |

Three query tiers: **single** (13 queries), **pair** (78 queries), **negative** (156 queries).

---

## Reference

See `updated_guide.md` for detailed documentation on all training knobs (match modes, caption modes, loss variants, nan modes, recipes).
