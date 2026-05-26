================================================================================
 MIMIC-CXR CLIP FINE-TUNING — FULL GUIDE
================================================================================
This guide provides a comprehensive overview of the training and evaluation process for fine-tuning CLIP on the MIMIC-CXR dataset.
It covers running the training in the background, tracking with Weights & Biases, and the full pipeline from data preparation to evaluation.

================================================================================
1. FULL PIPELINE

--- Step 0 — Build train/test CSVs (run once) ---

python baseline_eval/create_train_test_sets.py \
    --csv       cxr_data/all_txt_data_and_labels.csv \
    --split     cxr_data/images/mimic_cxr_jpg_images_from_google_cloud/mimic-cxr-jpg-2.1.0.physionet.org/mimic-cxr-2.0.0-split.csv.gz \
    --out_train cxr_data/mimic_cxr_train.csv \
    --out_test  cxr_data/mimic_cxr_official_test.csv

Produces ~227k train rows and 5,159 official test rows.

--- Step 1 — Build the image gallery (run once) ---

python baseline_eval/build_baseline.py \
    --csv        cxr_data/mimic_cxr_official_test.csv \
    --output_dir ./eval_outputs/baseline_output_official_test

Creates symlinks + paired .txt files for 5,159 test images in
eval_outputs/baseline_output_official_test/paired_data/.


--- Step 2 — Evaluate all models and compare ---

python baseline_eval/run_all_evals.py \
    --paired_dir $PAIRED_DIR --csv $TEST_CSV \
    --output_dir ./eval_outputs/all_results --batch_size 64 \
    --wandb-project mimic-cxr-clip --wandb-run-name all_results

Add --skip_existing to skip models already evaluated. Produces per-model
result CSVs, summary_macro.csv, and plots/ in --output_dir.

you can train a model with train_lora.py, which produces a final_merged.pt checkpoint and add to the MODELS list to evaluate it with run_all_evals.py.

ADDING A FINE-TUNED MODEL TO run_all_evals.py:

Edit the MODELS list in baseline_eval/run_all_evals.py:

  {
      "name": "lora_vitb32",
      "model_type": "finetuned",
      "finetuned_base_model": "ViT-B-32",
      "finetuned_pretrained": "openai",
      "finetuned_checkpoint": "experiments/lora_vitb32/final_merged.pt",
  },
================================================================================


================================================================================
 LABEL ENCODING


  CSV value  Meaning                          Tensor value
  ---------  -------------------------------- ------------
      1      Pathology positively mentioned   1.0  (positive)
      0      Pathology explicitly ruled out   -1.0 (negative)
     NaN     Label not mentioned in report    -1.0 (negative — same as 0)
     -1      Ambiguous / uncertain mention    0.0  (ignored)

NaN and 0 are both treated as -1.0: absence of evidence = evidence of absence.
Only -1 (uncertain language like "possible" or "cannot exclude") is ignored.
Consequence: if sample A has Edema=1 and B has Edema=NaN, they are flagged as
a conflict in negative_aware mode.
================================================================================

================================================================================
 KNOB 1 — CAPTION MODE  (--caption-mode): responsible for generating the text queries used in training and eval to match the image labels.

  single    One random positive label                       "atelectasis"
  pair      Two random positive labels                      "fracture and edema"
  negative  One positive + one absent label                 "atelectasis and no cardiomegaly"
  both      Randomly single or pair (75%/25%)               (DEFAULT)
  all       Mix: 50% single / 25% pair / 25% negative       (best for full eval coverage)

Negative labels are drawn from labels with CSV value 0 (explicitly ruled out)
or NaN (not mentioned) — matching the definition used in eval query_mode=negative.
If an image has no absent labels (all 13 are positive or uncertain), the caption
falls back to a single-label caption.

Use --caption-mode negative (or all) with --match-mode negative_aware for the
strongest training signal on "label A and no label B" queries.
================================================================================


================================================================================
 KNOB 2 — MATCH MODE  (--match-mode)

Controls which (image, text) pairs count as positives. Standard CLIP uses only
the diagonal — harmful in multi-label data where two images with identical
findings should attract each other.

  standard       Diagonal-only. Uses ClipLoss with distributed all-gather.
  single_label   Positive if >= 1 shared label. Soft target matrix.
  two_label      Positive if >= 2 shared labels. Pair with --caption-mode pair.
  negative_aware single_label + hinge repulsion for conflict pairs
                 (one sample label=1, other label=0/NaN).
                 Extra flags: --negative-weight lambda (default 0.5)
                              --negative-margin tau   (default 0.0)

Diagnostics logged at step 0 of each epoch (label-aware modes):
  avg_positives_per_sample  should be >1.0; if ~1.0 -> increase --batch-size
  diagonal_only (%)         keep <20%
  conflict_pairs (%)        5-15% is healthy for negative_aware
================================================================================


================================================================================
 KNOB 3 — LOSS VARIANT  (--loss)

  clip    (DEFAULT) Softmax multi-positive cross-entropy per row.
  siglip  Sigmoid binary BCE per pair — loss decoupled from batch size.

Both work with any --match-mode. Tuning lambda/tau for negative_aware:
  lambda = 0.25  conservative (use if training is unstable)
  lambda = 0.5   default balanced
  lambda = 1.0   aggressive repulsion
  tau = 0.0      penalise any positive conflict similarity (default)
  tau = 0.2      more aggressive
================================================================================


================================================================================
 KNOB 4 — QUERY MODE  (--query_mode, eval only)
  single    13 queries   "atelectasis"               label_A == 1
  pair      78 queries   "atelectasis and edema"     label_A == 1 AND label_B == 1
  negative  156 queries  "atelectasis and no edema"  label_A == 1 AND label_B == 0/NaN
  all       247 queries  all of the above            (DEFAULT)
================================================================================


================================================================================
 RECIPES (quick-reference)


  Recipe A — Baseline (vanilla CLIP, diagonal pairs)
    --match-mode standard --loss clip

  Recipe B — Soft positives (single label match, CLIP loss)
    --match-mode single_label

  Recipe C — Strict positives (two shared labels)
    --match-mode two_label --caption-mode pair

  Recipe D — Negative-aware CLIP
    --match-mode negative_aware --negative-weight 0.5 --negative-margin 0.0

  Recipe E — SigLIP loss, label-aware
    --loss siglip --match-mode single_label

  Recipe F — SigLIP + negative-aware (strongest repulsion)
    --loss siglip --match-mode negative_aware --negative-weight 0.5 --negative-margin 0.0
================================================================================


================================================================================
 DESIGN NOTES

Label-aware modes operate on the LOCAL batch only. For multi-GPU, positive
mining is per-GPU; standard mode retains full all-gather. For label-aware +
multi-GPU, increase --batch-size rather than using torchrun.

Conflict detection covers CSV 0 AND NaN (both -> -1.0). Uncertain labels
(CSV -1 -> 0.0) are fully ignored in both positive matching and conflict
detection.
================================================================================




================================================================================
================================================================================
================================================================================
================================================================================


================================================================================
EXPERIMENTS TO RUN
================================================================================

IDEA 1 — Baseline CLIP (standard mode)
python train_lora.py --base-model ViT-B-32 --pretrained openai --match-mode standard --loss clip

IDEA 2 — Soft positives (single label match)
python train_lora.py --base-model ViT-B-32 --pretrained openai --match-mode single_label --loss clip

IDEA 3 — Strict positives (two shared labels)
python train_lora.py --base-model ViT-B-32 --pretrained openai --match-mode two_label --caption-mode pair --loss clip

IDEA 4 — Negative-aware CLIP
python train_lora.py --base-model ViT-B-32 --pretrained openai --match-mode negative_aware --negative-weight 0.5 --negative-margin 0.0 --loss clip
# here it is important to increase --batch-size to get enough conflict pairs in each batch for the hinge loss to work well. also dont use gradient accumulation with negative_aware, as the conflict signal is diluted across steps. also dont use multi-GPU with negative_aware, as the conflict signal is local to the batch.

IDEA 5 — SigLIP loss
python train_lora.py --base-model ViT-B-32 --pretrained openai --match-mode single_label --loss siglip

IDEA 6 — SigLIP + negative-aware (strongest repulsion)
python train_lora.py --base-model ViT-B-32 --pretrained openai --loss siglip --match-mode negative_aware --negative-weight 0.5 --negative-margin 0.0
# this is the most aggressive repulsion for conflict pairs, as the sigmoid loss decouples the positive and negative terms, allowing the conflict pairs to be pushed to negative similarity without affecting the positive pairs. again, use a large batch size and no grad accumelation step and avoid multi-GPU for the same reasons as IDEA 4.


IDEA 7 use frozen CXR-CLIP models
# What: Keep a CXR-CLIP image encoder frozen (pre-trained on biomedical data)
      and fine-tune a ViT-B-32 text encoder to align with it. Both project
      to 512-dim — no architecture changes needed. LoRA applies to text only.
      Checkpoints in valid_pretrained_models_to_try/:
        r50_mc.pt   — ResNet50 image encoder
        swint_mc.pt — Swin-T image encoder
python train_lora.py --base-model ViT-B-32 --pretrained valid_pretrained_models_to_try/r50_mc.pt --match-mode single_label
# or
python train_lora.py --base-model ViT-B-32 --pretrained valid_pretrained_models_to_try/swint_mc.pt --match-mode single_label

IDEA 8 BioMedClip base model
python train_lora.py --base-model biomedclip --pretrained biomedclip --lora-modules "qkv,proj,fc1,fc2,query,key,value"

================================================================================
ABLATION IDEAS
================================================================================

IDEA 9
LoRA encoder target ablation (image / text / both)
What: Does fine-tuning the image encoder, text encoder, or both matter most?
      Hypothesis: text encoder dominates as it shapes the query space.

IDEA 10
LORA rank ablaion (rank, alpha)

IDEA 11
Repulsion strength ablation (lambda and tau tuning)

IDEA 12
Batch size effect on label-aware mining
What: Label-aware modes need in-batch positive partners. If
      avg_positives_per_sample ~= 1.0 the batch is too small.

IDEA 13 Caption mode ablation (single / pair / both / all)
What: Does training with only single-label captions hurt pair query retrieval?
"all" (50%/25%/25%) should generalise best across all three eval query types.

IDEA 14 Negative caption training (optimise "label A and no label B" queries)


================================================================================
 QUICK DIAGNOSTICS CHECKLIST
================================================================================

After each training run, check the per-epoch log line:
  [batch stats] avg_positives_per_sample=X.XX  diagonal_only=Y.Y%  conflict_pairs=Z.Z%

  avg_positives_per_sample
    ~= 1.0  -> batch too small / mode too strict -> increase --batch-size or
               switch from two_label to single_label
    > 2.0   -> healthy label-aware signal

  diagonal_only > 20%  -> same remedies as above

  conflict_pairs
    < 1%    -> --negative-weight has little effect
    5-15%   -> healthy for negative_aware mode

Training/val loss:
  val_loss rises, train_loss falls  -> overfitting; reduce --lr, --lora-r,
                                        or add --lora-dropout 0.1
  val_loss plateaus quickly         -> LR too high or rank too low
