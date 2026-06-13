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

Add `--skip_existing` to skip models already evaluated.

Each model is evaluated in **four passes** automatically:

| Pass | `nan_mode` | `neg_template` | Output type |
|------|-----------|----------------|-------------|
| 1 | `negative` | default (`"A and no B"`) | `single`, `pair`, `negative_nan` |
| 2 | `ignore` | default | `negative_strict` |
| 3 | `negative` | robust (`"an image with A but without B"`) | `negative_robust_nan` |
| 4 | `ignore` | robust | `negative_robust_strict` |

Output files in `--output_dir`:
- `summary_single.csv`, `summary_pair.csv`
- `summary_negative_nan.csv`, `summary_negative_strict.csv`
- `summary_negative_robust_nan.csv`, `summary_negative_robust_strict.csv`
- `summary_macro.csv` — one row per model, all metrics
- `plots/` — bar charts, heatmaps, publication-style table PNGs

#### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| P@K | Precision at K — fraction of top-K retrieved images that satisfy all query constraints, macro-averaged |
| R@K | Recall at K — 1 if any relevant image appears in top-K, else 0 (binary), macro-averaged |
| HNRR@K | Hard Negative Retrieval Rate — for `"A and no B"` queries: fraction of top-K results where both A and B are confirmed present (direct violation of the negation). Lower is better; 0 = no retrieved image violates the negation. Only reported for negative-type queries. |

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

| CSV value | Meaning | `--nan-mode negative` (default) | `--nan-mode ignore` |
|-----------|---------|--------------------------------|---------------------|
| `1` | Pathology positively mentioned | `1.0` (positive) | `1.0` (positive) |
| `0` | Pathology explicitly ruled out | `-1.0` (negative) | `-1.0` (negative) |
| `NaN` | Label not mentioned in report | `-1.0` (negative, same as 0) | `0.0` (ignored) |
| `-1` | Ambiguous / uncertain mention | `0.0` (ignored) | `0.0` (ignored) |

**`--nan-mode negative`** (default): absence of evidence = evidence of absence. NaN and CSV 0 are treated identically as negative.

**`--nan-mode ignore`**: only labels *explicitly ruled out* (CSV 0) count as negative. NaN ("not mentioned") is ignored — useful when you want a stricter definition of a conflict.

> **Note:** in `negative_aware` mode, if sample A has Edema=1 and B has Edema=NaN, they are flagged as a conflict under `--nan-mode negative` but **not** under `--nan-mode ignore`.

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
| `single_label` | Share ≥1 positive label | Soft target matrix; binary (0 or 1/N per pair) |
| `two_label` | Share ≥2 positive labels | Pair with `--caption-mode pair` |
| `negative_aware` | `single_label` + hinge repulsion for conflict pairs | Use large batch; no grad accum |
| `graded` | Soft target proportional to number of shared labels | Distinguishes "shares 1" from "shares 2" labels |
| `label_dot` ⭐ | Soft target T_ij = max(y_i·y_j, 0) / row_sum | **Best-performing mode** — selective attraction, no explicit repulsion |
| `image_pair` | SigLIP diagonal + image-image attraction & repulsion | Explicit image-level attract/repel |

**`label_dot` mode** is the best-performing training objective. For each pair (i, j) it computes a dot product of the label vectors: K(i,j) = y_i · y_j, where positive labels are +1, confirmed-absent labels are −1 (under `--nan-mode ignore` only CSV 0 counts as −1, NaN counts as 0). The soft target is T_ij = max(K, 0) / row_sum. Conflicting pairs (K < 0) get T = 0 — they are neither attracted nor explicitly repelled; they enter the softmax denominator identically to any standard negative pair. Always use with `--nan-mode ignore --caption-mode all --loss clip`.

**`image_pair` mode** combines three loss terms:

1. **Image-text CLIP** (`L_it`): symmetric softmax cross-entropy with diagonal target — each image must rank its own paired caption above all others in the batch, and vice versa.
2. **Image-image attraction** (`L_attr`, weight `--attract-weight λ_a`): any two images sharing the same confirmed label value (both positive *or* both negative for any label) are pulled closer: `(1 − cosine_sim).mean()` over attract pairs.
3. **Image-image repulsion** (`L_rep`, weight `--repel-weight λ_r`): image pairs with a conflicting label (one positive, one negative for the same label) are pushed apart via a hinge on cosine similarity.

Crucially, **attraction and repulsion act only on image embeddings** — text embeddings receive gradient only from the image-text term. This avoids entangling the text encoder with training-set co-occurrence patterns.

**Conflict pair** = one sample has label=1 and the other has label=−1 (explicit CSV 0, not NaN) for the same pathology. If a pair also shares a matching label, attraction takes priority and the pair is excluded from repulsion.

Always pair with `--nan-mode ignore` — this ensures only *explicitly ruled-out* labels (CSV 0) trigger repulsion, not NaN (unmentioned) labels.

Extra flags for `image_pair`:
- `--attract-weight λ_a` (default `0.1`) — weight of the image-image attraction term
- `--repel-weight λ_r` (default `0.1`) — weight of the image-image repulsion hinge
- `--repel-margin τ_r` (default `0.0`) — cosine-sim threshold; pairs already below τ are not penalised

**`graded` mode** weights attraction by the count of shared positive labels: a pair sharing 2 labels attracts twice as strongly as a pair sharing 1. Works with both `--loss clip` and `--loss siglip`. Pair with `--caption-mode all --nan-mode ignore`.

**Conflict pair** (for `negative_aware`/`graded`) = one sample has label=1, the other has label=0/NaN for the same pathology, **and they share no confirmed positive labels**. Attraction wins otherwise.

Extra flags for `negative_aware`:
- `--negative-weight λ` (default `0.5`) — repulsion strength relative to the contrastive loss
- `--negative-margin τ` (default `0.0`) — **logit-space** threshold; pairs with `logit_scale × cosine_sim < τ` are not penalised.

> **Important:** the repulsion hinge in `negative_aware` operates in logit space (`logit_scale × cosine_sim`, range ≈ ±100). The `image_pair` repulsion hinge operates in raw cosine space (range [−1, 1]).

**Diagnostics** logged at step 0 of each epoch (label-aware modes):

| Stat | Healthy range | Action if out of range |
|------|--------------|----------------------|
| `avg_positives_per_sample` | > 2.0 | Increase `--batch-size` or switch to `single_label` |
| `diagonal_only` | < 20% | Same as above |
| `conflict_pairs` | 1–10% with `--nan-mode ignore` | > 10% usually means NaN pollution; switch to `--nan-mode ignore` |
| `attract_pairs` (`image_pair`) | 20–60% | < 20%: batch too small; > 60%: labels too dense, fine |

> **Important:** label-aware modes mine positives from the **live batch**. `--grad-accum-steps` does NOT help — batches across accumulation steps never see each other's labels. Increase `--batch-size` directly.

---

## Knob 3 — Loss Variant (`--loss`)

| Loss | Description | Best for |
|------|-------------|---------|
| `clip` | Softmax multi-positive cross-entropy *(default)* | General retrieval (ranking) |
| `siglip` | Sigmoid binary BCE per pair — decoupled from batch size | Negative query retrieval |

**Which to use:**
- `clip` produces better-separated embedding spaces for ranking tasks (P@K metrics) because the softmax forces each sample to rank its positives above *all* in-batch negatives simultaneously.
- `siglip` gives stronger repulsion for conflict pairs — conflicting pairs get target=−1 in the main loss itself, with no separate hinge needed. Better if negative queries are the primary goal.

**SigLIP + `single_label`**: conflict pairs automatically receive target=−1 and are repelled without any hinge. `--negative-weight` and `--negative-margin` are not needed and can be omitted.

**SigLIP + `negative_aware`**: adds an optional hinge on top of the −1 targets. Since SigLIP already repels, keep λ small.

`--negative-weight` tuning for `negative_aware`:

| λ | Behaviour |
|---|-----------|
| `0.1` | Very conservative |
| `0.25` | Conservative — use with SigLIP (main loss already repels) |
| `0.5` | Default balanced — use with CLIP |
| `1.0` | Aggressive repulsion |

---

## Knob 4 — NaN Mode (`--nan-mode`)

Controls how labels that are *not mentioned* (NaN in the CSV) are encoded during training and how they affect relevance during evaluation of negative queries.

| Mode | NaN encoding (train) | NaN in negative query relevance (eval) |
|------|---------------------|----------------------------------------|
| `negative` (default) | `-1.0` — treated as absent | image is relevant if neg_label == 0 **or** NaN |
| `ignore` | `0.0` — treated as unknown | image is **not** considered relevant for that query |

Use `--nan-mode ignore` when:
- You want only *explicitly ruled-out* labels to define negatives (stricter standard).
- You are training `negative_aware` and want to suppress spurious conflicts from NaN-labelled images.

> **For `negative_aware`:** always pair with `--nan-mode ignore`. Under `--nan-mode negative`, ~78% of off-diagonal pairs are flagged as conflicts (mostly NaN-derived, not genuine contradictions), drowning out the meaningful repulsion signal. The same applies to `label_dot`.

---

## Knob 5 — Query Mode (`--query_mode`, eval only)

10 CheXpert labels are used (4 excluded: No Finding, Enlarged Cardiomediastinum, Pleural Other, Support Devices).

| Mode | # Queries | Format | Relevant images |
|------|-----------|--------|----------------|
| `single` | 10 | `"atelectasis"` | label_A == 1 |
| `pair` | 45 | `"atelectasis and edema"` | label_A == 1 AND label_B == 1 |
| `negative` | 90 | `"atelectasis and no edema"` | label_A == 1 AND label_B == 0/NaN |
| `all` | 145 | all of the above | *(default)* |

The text format for negative queries is controlled by `--neg-template`:
- Default: `"{pos} and no {neg}"` → e.g. `"atelectasis and no edema"`
- Robust: `"an image with {pos} but without {neg}"` — used by `run_all_evals.py` for robustness passes 3 & 4

Relevance for negative queries depends on `--nan-mode`:
- **lenient** (`--nan-mode negative`): NaN and CSV 0 both count as absent (image is relevant if neg label == 0 OR NaN)
- **strict** (`--nan-mode ignore`): only CSV 0 counts as absent (NaN images are excluded from relevant set)

---

## Recipes (Quick Reference)

| Recipe | Flags | Goal |
|--------|-------|------|
| A — Vanilla CLIP baseline | `--match-mode standard --loss clip` | Baseline |
| B — Soft positives | `--match-mode single_label` | Better single/pair retrieval |
| C — Strict positives | `--match-mode two_label --caption-mode pair` | Pair queries |
| D — Negative-aware CLIP | `--match-mode negative_aware --negative-weight 0.5 --nan-mode ignore` | Negative queries |
| E — SigLIP label-aware | `--loss siglip --match-mode single_label --nan-mode ignore` | General, implicit repulsion |
| F — SigLIP + negative-aware | `--loss siglip --match-mode negative_aware --negative-weight 0.25 --nan-mode ignore` | Strongest negative repulsion |
| G — Graded CLIP | `--loss clip --match-mode graded --caption-mode all --nan-mode ignore --batch-size 128` | Graded attraction across all query types |
| H — Image-pair | `--loss clip --match-mode image_pair --caption-mode all --nan-mode ignore --attract-weight 0.1 --repel-weight 0.1` | Image-level attraction + repulsion |
| I — Label-Dot CLIP ⭐ | `--loss clip --match-mode label_dot --caption-mode all --nan-mode ignore --lora-r 16 --lora-alpha 16` | **Best overall: selective attraction via label dot product** |

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
    --loss clip --batch-size 128 --nan-mode ignore
```

> Use `--nan-mode ignore` to limit conflict detection to explicit label contradictions (CSV 1 vs CSV 0), not NaN-derived ones. Use a large `--batch-size` to get enough conflict pairs per batch. Do not use `--grad-accum-steps > 1` — conflict signal is local to the batch.

### Idea 5 — Label-Dot CLIP (best overall) ⭐
```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_lora.py \
    --base-model ViT-B-32 --pretrained openai \
    --precision bf16 \
    --lr 1e-4 --min-lr 1e-8 \
    --batch-size 64 --epochs 100 \
    --warmup-epochs 5 --patience 10 \
    --nan-mode ignore \
    --loss clip \
    --caption-mode all \
    --match-mode label_dot \
    --lora-r 16 --lora-alpha 16 \
    --output-dir ./experiments/label_dot
```

> Soft target T_ij = max(y_i·y_j, 0) / row_sum. Conflicting pairs (K < 0) get T = 0 and are treated as ordinary negatives — no explicit repulsion, no hinge. `--caption-mode all` (50% single / 25% pair / 25% negation captions) ensures full coverage of all query tiers. `--nan-mode ignore` means only explicitly ruled-out labels (CSV 0) count as −1; NaN ("not mentioned") is 0. Best results with 2 GPUs × 64 per-GPU batch = 128 effective batch size.

### Idea 5b — Graded SigLIP
```bash
python train_lora.py --base-model ViT-B-32 --pretrained openai \
    --match-mode graded --loss siglip \
    --caption-mode all --nan-mode ignore --batch-size 128
```

> Graded soft targets weight attraction proportionally to the number of shared positive labels. SigLIP's soft BCE handles both attraction and repulsion cleanly. Use a large batch (≥128) so graded weights have enough diversity to be meaningful.

### Idea 6 — SigLIP + single_label (simpler baseline)
```bash
python train_lora.py --base-model ViT-B-32 --pretrained openai \
    --match-mode single_label --loss siglip --nan-mode ignore --batch-size 128
```

> Conflict pairs automatically receive target=−1 in the SigLIP loss, providing repulsion with no hinge term needed. Simpler than graded but doesn't distinguish "shares 1 label" from "shares 2 labels" — pair query performance will be weaker.

### Idea 6 — SigLIP + negative-aware (extra emphasis on conflicts)
```bash
python train_lora.py --base-model ViT-B-32 --pretrained openai \
    --loss siglip --match-mode negative_aware \
    --negative-weight 0.25 --negative-margin 0.0 --batch-size 128 --nan-mode ignore
```

> SigLIP already repels conflict pairs via target=−1; the hinge adds extra emphasis. Keep `--negative-weight` low (0.1–0.25).

### Idea 7 — Image-pair loss
```bash
python train_lora.py --base-model ViT-B-32 --pretrained openai \
    --loss clip --match-mode image_pair \
    --caption-mode all --nan-mode ignore \
    --attract-weight 0.1 --repel-weight 0.1 \
    --batch-size 64
```

> Three-component loss: (1) SigLIP diagonal image-text pairing, (2) image-image attraction for same-label pairs, (3) image-image repulsion hinge for conflicting-label pairs. Attraction and repulsion act only on image embeddings. Use `--nan-mode ignore` so only explicitly ruled-out labels (CSV 0) trigger repulsion.

---

### Idea 9 — Frozen CXR-CLIP image encoder + fine-tuned text encoder
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

### Idea 10 — BiomedCLIP base model
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
| 10 | LoRA encoder target (image / text / both) | `--lora-target` |
| 11 | LoRA rank ablation | `--lora-r`, `--lora-alpha` |
| 12 | Repulsion strength (λ and τ in logit space) | `--negative-weight`, `--negative-margin` |
| 13 | Batch size effect on label-aware mining | `--batch-size` |
| 14 | Caption mode ablation (single / pair / both / all) | `--caption-mode` |
| 15 | Graded vs single_label vs two_label — pair query ablation | `--match-mode` |

---

## Quick Diagnostics Checklist

After each training run, check the per-epoch log line:
```
# negative_aware mode:
[batch stats] avg_positives_per_sample=X.XX  diagonal_only=Y.Y%  conflict_pairs=Z.Z%
Epoch N/M — train=X.XXXX  [clip=X.XXXX  rep=X.XXXX]  val=X.XXXX  lr=X.XXe-XX

# image_pair mode:
[batch stats] avg_positives_per_sample=1.00  diagonal_only=0.0%  conflict_pairs=Z.Z%  attract_pairs=A.A%
Epoch N/M — train=X.XXXX  [clip=X.XXXX  attract=X.XXXX  repel=X.XXXX]  val=X.XXXX  lr=X.XXe-XX
```

| Symptom | Cause | Fix |
|---------|-------|-----|
| `avg_positives_per_sample ~= 1.0` | Batch too small or mode too strict | Increase `--batch-size` or switch to `single_label` |
| `diagonal_only > 20%` | Same as above | Same fix |
| `conflict_pairs > 20%` | NaN labels polluting conflict detection | Switch to `--nan-mode ignore` |
| `conflict_pairs < 1%` | Conflicting labels rare in batch | `negative_aware` hinge has little effect |
| `rep ~= 0` with `negative_aware` | All conflict pairs already separated below margin | Lower `--negative-margin` (e.g. `-20`) |
| `val_loss` rises, `train_loss` falls | Overfitting | Reduce `--lr`, `--lora-r`, or add `--lora-dropout 0.1` |
| `val_loss` plateaus quickly | LR too high or rank too low | Tune `--lr` and `--lora-r` |
