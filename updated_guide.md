# MIMIC-CXR CLIP Fine-Tuning Guide

## Full Pipeline

### Step 0 — Build train/test CSVs (run once)

```bash
python baseline_eval/create_train_test_sets.py \
    --csv       cxr_data/all_txt_data_and_labels.csv \
    --split     cxr_data/images/mimic_cxr_jpg_images_from_google_cloud/mimic-cxr-jpg-2.1.0.physionet.org/mimic-cxr-2.0.0-split.csv.gz \
    --out_train cxr_data/mimic_cxr_train.csv \
    --out_test  cxr_data/mimic_cxr_official_test.csv
```

Produces ~227k train rows and 5,159 official test rows.

### Step 1 — Build the image gallery (run once)

```bash
python baseline_eval/build_baseline.py \
    --csv        cxr_data/mimic_cxr_official_test.csv \
    --output_dir ./eval_outputs/baseline_output_official_test
```

Creates symlinks + paired `.txt` files for 5,159 test images in `eval_outputs/baseline_output_official_test/paired_data/`.

### Step 2 — Train a model

```bash
python train_lora.py --base-model ViT-B-32 --pretrained openai \
    --output-dir ./experiments/my_experiment \
    [... other flags ...]
```

Produces `experiments/my_experiment/final_merged.pt`.

### Step 3 — Evaluate all models and compare

```bash
python baseline_eval/run_all_evals.py \
    --paired_dir $PAIRED_DIR --csv $TEST_CSV \
    --output_dir ./eval_outputs/all_results --batch_size 64 \
```

Add `--skip_existing` to skip models already evaluated. Produces per-model result CSVs, `summary_macro.csv`, and plots in `--output_dir`.

#### Adding a fine-tuned model to `run_all_evals.py`

Edit the `MODELS` list in `baseline_eval/run_all_evals.py`:

```python
{
    "name": "lora_vitb32",
    "model_type": "finetuned",
    "finetuned_base_model": "ViT-B-32",
    "finetuned_pretrained": "openai",
    "finetuned_checkpoint": "experiments/lora_vitb32/final_merged.pt",
},
```

---

## Label Encoding

| CSV value | Meaning | Tensor value |
|-----------|---------|--------------|
| `1` | Pathology positively mentioned | `1.0` (positive) |
| `0` | Pathology explicitly ruled out | `-1.0` (negative) |
| `NaN` | Label not mentioned in report | `-1.0` (same as 0) |
| `-1` | Ambiguous / uncertain mention | `0.0` (ignored) |

NaN and 0 are both treated as `-1.0`: absence of evidence = evidence of absence.
Only `-1` (uncertain language like "possible" or "cannot exclude") is ignored.

> **Note:** if sample A has Edema=1 and B has Edema=NaN, they are flagged as a conflict in `negative_aware` mode.

---

## Knob 1 — Caption Mode (`--caption-mode`)

Controls what text caption is paired with each training image.

| Mode | Caption format | Example |
|------|---------------|---------|
| `single` | One random positive label | `"atelectasis"` |
| `pair` | Two random positive labels | `"fracture and edema"` |
| `negative` | One positive + one absent label | `"atelectasis and no cardiomegaly"` |
| `both` | 75% single / 25% pair | *(default)* |
| `all` | 50% single / 25% pair / 25% negative | *(best for full eval coverage)* |

Negative labels are drawn from labels with CSV value `0` (explicitly ruled out) or `NaN` (not mentioned) — matching the definition used in `--query_mode negative`. If an image has no absent labels, the caption falls back to single.

Use `--caption-mode negative` (or `all`) with `--match-mode negative_aware` for the strongest training signal on "label A and no label B" queries.

---

## Knob 2 — Match Mode (`--match-mode`)

Controls which (image, text) pairs count as positives in the loss. Standard CLIP uses only the diagonal — harmful in multi-label data where two images with identical findings should attract each other.

| Mode | Positives | Notes |
|------|-----------|-------|
| `standard` | Diagonal only | Uses distributed all-gather; works with any batch size |
| `single_label` | Share ≥1 positive label | Soft target matrix |
| `two_label` | Share ≥2 positive labels | Pair with `--caption-mode pair` |
| `negative_aware` | `single_label` + hinge repulsion for conflict pairs | Use large batch; no grad accum |

**Conflict pair** = one sample has label=1, the other has label=0/NaN for the same pathology.

Extra flags for `negative_aware`:
- `--negative-weight λ` (default `0.5`) — repulsion strength
- `--negative-margin τ` (default `0.0`) — cosine-sim threshold below which repulsion is skipped

**Diagnostics** logged at step 0 of each epoch (label-aware modes):

| Stat | Healthy range | Action if out of range |
|------|--------------|----------------------|
| `avg_positives_per_sample` | > 2.0 | Increase `--batch-size` or switch to `single_label` |
| `diagonal_only` | < 20% | Same as above |
| `conflict_pairs` | 5–15% | Only meaningful for `negative_aware` |

> **Important:** label-aware modes mine positives from the **live batch**. `--grad-accum-steps` does NOT help — batches across accumulation steps never see each other's labels. Increase `--batch-size` directly.

---

## Knob 3 — Loss Variant (`--loss`)

| Loss | Description | Best for |
|------|-------------|---------|
| `clip` | Softmax multi-positive cross-entropy *(default)* | General retrieval (ranking) |
| `siglip` | Sigmoid binary BCE per pair — decoupled from batch size | Negative query retrieval |

**Which to use:**
- `clip` produces better-separated embedding spaces for ranking tasks (P@K metrics) because the softmax forces each sample to rank its positives above *all* in-batch negatives simultaneously.
- `siglip` gives stronger repulsion for conflict pairs — conflicting pairs get target=−1 in the main loss itself, plus an optional hinge on top. Better if negative queries are the primary goal.

`--negative-weight` tuning for `negative_aware`:

| λ | Behaviour |
|---|-----------|
| `0.25` | Conservative — use if training is unstable |
| `0.5` | Default balanced |
| `1.0` | Aggressive repulsion |

With SigLIP, prefer `--negative-weight 0.25` since the main loss already repels conflicts.

---

## Knob 4 — Query Mode (`--query_mode`, eval only)

| Mode | # Queries | Format | Relevant images |
|------|-----------|--------|----------------|
| `single` | 13 | `"atelectasis"` | label_A == 1 |
| `pair` | 78 | `"atelectasis and edema"` | label_A == 1 AND label_B == 1 |
| `negative` | 156 | `"atelectasis and no edema"` | label_A == 1 AND label_B == 0/NaN |
| `all` | 247 | all of the above | *(default)* |

---

## Recipes (Quick Reference)

| Recipe | Flags | Goal |
|--------|-------|------|
| A — Vanilla CLIP baseline | `--match-mode standard --loss clip` | Baseline |
| B — Soft positives | `--match-mode single_label` | Better single/pair retrieval |
| C — Strict positives | `--match-mode two_label --caption-mode pair` | Pair queries |
| D — Negative-aware CLIP | `--match-mode negative_aware --negative-weight 0.5` | Negative queries |
| E — SigLIP label-aware | `--loss siglip --match-mode single_label` | General, small batch |
| F — SigLIP + negative-aware | `--loss siglip --match-mode negative_aware --negative-weight 0.25` | Strongest negative repulsion |

---

## Experiments

### Idea 1 — Baseline CLIP (standard mode)
```bash
python train_lora.py --base-model ViT-B-32 --pretrained openai \
    --match-mode standard --loss clip
```

### Idea 2 — Soft positives (single label match)
```bash
python train_lora.py --base-model ViT-B-32 --pretrained openai \
    --match-mode single_label --loss clip
```

### Idea 3 — Strict positives (two shared labels)
```bash
python train_lora.py --base-model ViT-B-32 --pretrained openai \
    --match-mode two_label --caption-mode pair --loss clip
```

### Idea 4 — Negative-aware CLIP
```bash
python train_lora.py --base-model ViT-B-32 --pretrained openai \
    --match-mode negative_aware --negative-weight 0.5 --negative-margin 0.0 \
    --loss clip --batch-size 128
```

> Use a large `--batch-size` to get enough conflict pairs per batch. Do not use `--grad-accum-steps > 1` or multi-GPU — conflict signal is local to the batch.

### Idea 5 — SigLIP loss
```bash
python train_lora.py --base-model ViT-B-32 --pretrained openai \
    --match-mode single_label --loss siglip
```

### Idea 6 — SigLIP + negative-aware (strongest repulsion)
```bash
python train_lora.py --base-model ViT-B-32 --pretrained openai \
    --loss siglip --match-mode negative_aware \
    --negative-weight 0.25 --negative-margin 0.0 --batch-size 128
```

> SigLIP already repels conflict pairs via target=−1; lower `--negative-weight` accordingly.

### Idea 7 — Frozen CXR-CLIP image encoder + fine-tuned text encoder
```bash
# ResNet50 image encoder
python train_lora.py --base-model ViT-B-32 --pretrained openai \
    --cxrclip-checkpoint valid_pretrained_models_to_try/r50_mc.pt \
    --match-mode single_label

# Swin-T image encoder
python train_lora.py --base-model ViT-B-32 --pretrained openai \
    --cxrclip-checkpoint valid_pretrained_models_to_try/swint_mc.pt \
    --match-mode single_label
```

### Idea 8 — BiomedCLIP base model
```bash
python train_lora.py \
    --base-model hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 \
    --match-mode single_label \
    --lora-modules "qkv,proj,fc1,fc2,query,key,value"
```

---

## Ablation Ideas

| # | What | Key flag |
|---|------|---------|
| 9 | LoRA encoder target (image / text / both) | `--lora-target` |
| 10 | LoRA rank ablation | `--lora-r`, `--lora-alpha` |
| 11 | Repulsion strength (λ and τ) | `--negative-weight`, `--negative-margin` |
| 12 | Batch size effect on label-aware mining | `--batch-size` |
| 13 | Caption mode ablation (single / pair / both / all) | `--caption-mode` |
| 14 | Negative caption training ("label A and no label B") | `--caption-mode negative` |

---

## Quick Diagnostics Checklist

After each training run, check the per-epoch log line:
```
[batch stats] avg_positives_per_sample=X.XX  diagonal_only=Y.Y%  conflict_pairs=Z.Z%
```

| Symptom | Cause | Fix |
|---------|-------|-----|
| `avg_positives_per_sample ~= 1.0` | Batch too small or mode too strict | Increase `--batch-size` or switch to `single_label` |
| `diagonal_only > 20%` | Same as above | Same fix |
| `conflict_pairs < 1%` | Conflicting labels rare in batch | `negative_aware` hinge has little effect |
| `val_loss` rises, `train_loss` falls | Overfitting | Reduce `--lr`, `--lora-r`, or add `--lora-dropout 0.1` |
| `val_loss` plateaus quickly | LR too high or rank too low | Tune `--lr` and `--lora-r` |
