"""
Smoke-test runner for all research ideas in guide.txt.
Runs every train + eval pair sequentially, one at a time.

Parameters are intentionally tiny (epochs=1, batch=4, max_samples=10) to
verify each code path works before committing to full training runs.

Output dirs: ./experiments/smoke/<idea_name>/
Results CSVs: written next to each experiment dir.

Usage:
    python all_scripts_to_execute.py              # run all steps
    python all_scripts_to_execute.py --start 5   # resume from step 5
    python all_scripts_to_execute.py --only 3    # run only step 3
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

PYTHON = sys.executable

TRAIN_CSV  = "cxr_data/mimic_cxr_train.csv"
IMAGE_DIR  = "cxr_data/images/mimic_cxr_jpg_images_from_google_cloud/mimic-cxr-jpg-2.1.0.physionet.org/files"
PAIRED_DIR = "./eval_outputs/baseline_output_official_test/paired_data"
TEST_CSV   = "cxr_data/mimic_cxr_official_test.csv"
SMOKE_DIR  = "./experiments/smoke"

EPOCHS     = 1
BATCH      = 4
SAMPLES    = 10
EVAL_BATCH = 32


# ── helpers ───────────────────────────────────────────────────────────────────

def _train(name, extra):
    return [
        PYTHON, "train_lora.py",
        "--base-model",  "ViT-B-32",
        "--pretrained",  "openai",
        "--train-csv",   TRAIN_CSV,
        "--image-dir",   IMAGE_DIR,
        "--output-dir",  f"{SMOKE_DIR}/{name}",
        "--batch-size",  str(BATCH),
        "--epochs",      str(EPOCHS),
        "--max-samples", str(SAMPLES),
    ] + extra


def _eval(name, extra=None):
    cmd = [
        PYTHON, "baseline_eval/eval_model.py",
        "--model_type",           "finetuned",
        "--name",                 name,
        "--finetuned_base_model", "ViT-B-32",
        "--finetuned_pretrained", "openai",
        "--finetuned_checkpoint", f"{SMOKE_DIR}/{name}/final_merged.pt",
        "--paired_dir",           PAIRED_DIR,
        "--csv",                  TEST_CSV,
        "--batch_size",           str(EVAL_BATCH),
    ]
    return cmd + (extra or [])


def _biomed_train(name, extra):
    return [
        PYTHON, "train_lora.py",
        "--base-model",   "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        "--train-csv",    TRAIN_CSV,
        "--image-dir",    IMAGE_DIR,
        "--output-dir",   f"{SMOKE_DIR}/{name}",
        "--batch-size",   str(BATCH),
        "--epochs",       str(EPOCHS),
        "--max-samples",  str(SAMPLES),
        "--lora-modules", "qkv,proj,fc1,fc2,query,key,value",
    ] + extra


def _biomed_eval(name):
    return [
        PYTHON, "baseline_eval/eval_model.py",
        "--model_type",           "finetuned",
        "--name",                 name,
        "--finetuned_base_model", "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        "--finetuned_pretrained", "",
        "--finetuned_checkpoint", f"{SMOKE_DIR}/{name}/final_merged.pt",
        "--paired_dir",           PAIRED_DIR,
        "--csv",                  TEST_CSV,
        "--batch_size",           str(EVAL_BATCH),
    ]


# ── step list ─────────────────────────────────────────────────────────────────

STEPS = [
    # ── IDEA 1a — vanilla CLIP baseline ──────────────────────────────────────
    ("TRAIN 1a | vanilla CLIP (diagonal pairs, softmax loss)",
     _train("1a_standard_clip", ["--match-mode", "standard", "--loss", "clip"])),

    ("EVAL  1a | vanilla CLIP baseline",
     _eval("1a_standard_clip")),

    # ── IDEA 1b — single-label soft positives ────────────────────────────────
    ("TRAIN 1b | single-label soft positives (CLIP loss)",
     _train("1b_single_clip", ["--match-mode", "single_label", "--loss", "clip"])),

    ("EVAL  1b | single-label soft positives",
     _eval("1b_single_clip")),

    # ── IDEA 1c — two-label strict positives + pair captions ─────────────────
    ("TRAIN 1c | two-label strict positives + pair captions",
     _train("1c_two_clip", ["--match-mode", "two_label", "--caption-mode", "pair", "--loss", "clip"])),

    ("EVAL  1c | two-label strict positives",
     _eval("1c_two_clip")),

    # ── IDEA 1d — negative-aware CLIP ────────────────────────────────────────
    ("TRAIN 1d | negative-aware CLIP (attract+repel, λ=0.5)",
     _train("1d_negaware_clip", ["--match-mode", "negative_aware", "--loss", "clip",
                                  "--negative-weight", "0.5", "--negative-margin", "0.0"])),

    ("EVAL  1d | negative-aware CLIP (all query modes)",
     _eval("1d_negaware_clip")),

    # ── IDEA 1e — single-label SigLIP ────────────────────────────────────────
    ("TRAIN 1e | single-label SigLIP loss",
     _train("1e_single_siglip", ["--match-mode", "single_label", "--loss", "siglip"])),

    ("EVAL  1e | single-label SigLIP",
     _eval("1e_single_siglip")),

    # ── IDEA 1f — negative-aware SigLIP ──────────────────────────────────────
    ("TRAIN 1f | negative-aware SigLIP (strongest repulsion)",
     _train("1f_negaware_siglip", ["--match-mode", "negative_aware", "--loss", "siglip",
                                    "--negative-weight", "0.5", "--negative-margin", "0.0"])),

    ("EVAL  1f | negative-aware SigLIP",
     _eval("1f_negaware_siglip")),

    # ── IDEA 2 — batch size effect ────────────────────────────────────────────
    # (Both use batch=4 in smoke mode — verifies code paths, not the effect itself)
    ("TRAIN 2a | batch-size 32 (smoke: 4)",
     _train("2a_bs32", ["--match-mode", "single_label"])),

    ("EVAL  2a | batch-size 32",
     _eval("2a_bs32")),

    ("TRAIN 2b | batch-size 128 (smoke: 4)",
     _train("2b_bs128", ["--match-mode", "single_label"])),

    ("EVAL  2b | batch-size 128",
     _eval("2b_bs128")),

    # ── IDEA 3 — LoRA encoder target ──────────────────────────────────────────
    ("TRAIN 3a | LoRA on image encoder only",
     _train("3a_image_only", ["--match-mode", "single_label", "--lora-target", "image"])),

    ("EVAL  3a | image encoder only",
     _eval("3a_image_only")),

    ("TRAIN 3b | LoRA on text encoder only",
     _train("3b_text_only", ["--match-mode", "single_label", "--lora-target", "text"])),

    ("EVAL  3b | text encoder only",
     _eval("3b_text_only")),

    ("TRAIN 3c | LoRA on both encoders",
     _train("3c_both_encoders", ["--match-mode", "single_label", "--lora-target", "both"])),

    ("EVAL  3c | both encoders",
     _eval("3c_both_encoders")),

    # ── IDEA 4 — LoRA rank ablation ───────────────────────────────────────────
    ("TRAIN 4a | LoRA rank r=4  (α=8)",
     _train("4a_r4", ["--match-mode", "single_label", "--lora-r", "4", "--lora-alpha", "8"])),

    ("EVAL  4a | LoRA rank r=4",
     _eval("4a_r4")),

    ("TRAIN 4b | LoRA rank r=16 (α=32, default)",
     _train("4b_r16", ["--match-mode", "single_label", "--lora-r", "16", "--lora-alpha", "32"])),

    ("EVAL  4b | LoRA rank r=16",
     _eval("4b_r16")),

    ("TRAIN 4c | LoRA rank r=64 (α=128)",
     _train("4c_r64", ["--match-mode", "single_label", "--lora-r", "64", "--lora-alpha", "128"])),

    ("EVAL  4c | LoRA rank r=64",
     _eval("4c_r64")),

    # ── IDEA 5 — repulsion strength ablation ──────────────────────────────────
    ("TRAIN 5a | repulsion λ=0.25 (conservative)",
     _train("5a_negaware_w025", ["--match-mode", "negative_aware", "--loss", "clip",
                                  "--negative-weight", "0.25", "--negative-margin", "0.0"])),

    ("EVAL  5a | repulsion λ=0.25 (negative queries)",
     _eval("5a_negaware_w025", ["--query_mode", "negative"])),

    ("TRAIN 5b | repulsion λ=0.5  (balanced, default)",
     _train("5b_negaware_w05", ["--match-mode", "negative_aware", "--loss", "clip",
                                 "--negative-weight", "0.5", "--negative-margin", "0.0"])),

    ("EVAL  5b | repulsion λ=0.5 (negative queries)",
     _eval("5b_negaware_w05", ["--query_mode", "negative"])),

    ("TRAIN 5c | repulsion λ=1.0  (aggressive)",
     _train("5c_negaware_w10", ["--match-mode", "negative_aware", "--loss", "clip",
                                 "--negative-weight", "1.0", "--negative-margin", "0.0"])),

    ("EVAL  5c | repulsion λ=1.0 (negative queries)",
     _eval("5c_negaware_w10", ["--query_mode", "negative"])),

    ("TRAIN 5d | repulsion τ=0.2  (lenient margin)",
     _train("5d_negaware_margin02", ["--match-mode", "negative_aware", "--loss", "clip",
                                      "--negative-weight", "0.5", "--negative-margin", "0.2"])),

    ("EVAL  5d | repulsion τ=0.2 (negative queries)",
     _eval("5d_negaware_margin02", ["--query_mode", "negative"])),

    # ── IDEA 6 — BiomedCLIP base model ────────────────────────────────────────
    ("TRAIN 6a | BiomedCLIP + standard loss",
     _biomed_train("6a_biomedclip_standard", ["--match-mode", "standard", "--loss", "clip"])),

    ("EVAL  6a | BiomedCLIP + standard",
     _biomed_eval("6a_biomedclip_standard")),

    ("TRAIN 6b | BiomedCLIP + negative-aware",
     _biomed_train("6b_biomedclip_negaware", ["--match-mode", "negative_aware", "--loss", "clip",
                                               "--negative-weight", "0.5", "--negative-margin", "0.0"])),

    ("EVAL  6b | BiomedCLIP + negative-aware",
     _biomed_eval("6b_biomedclip_negaware")),

    # ── IDEA 7 — caption mode ablation ────────────────────────────────────────
    ("TRAIN 7a | single captions only",
     _train("7a_caption_single", ["--match-mode", "single_label", "--caption-mode", "single"])),

    ("EVAL  7a | single captions",
     _eval("7a_caption_single")),

    ("TRAIN 7b | pair captions only (two_label match)",
     _train("7b_caption_pair", ["--match-mode", "two_label", "--caption-mode", "pair"])),

    ("EVAL  7b | pair captions",
     _eval("7b_caption_pair")),

    ("TRAIN 7c | both captions — 75% single, 25% pair (default)",
     _train("7c_caption_both", ["--match-mode", "single_label", "--caption-mode", "both"])),

    ("EVAL  7c | both captions",
     _eval("7c_caption_both")),
]


# ── runner ────────────────────────────────────────────────────────────────────

def run_step(idx, label, cmd):
    bar = "─" * 70
    print(f"\n{bar}")
    print(f"[{idx:>2}/{len(STEPS)}] {label}")
    print(f"CMD: {' '.join(cmd)}")
    print(bar)
    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0
    status = "✓ OK" if result.returncode == 0 else "✗ FAILED"
    print(f"{status}  ({elapsed:.0f}s)")
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=1,
                        help="Resume from this step number (1-indexed)")
    parser.add_argument("--only", type=int, default=None,
                        help="Run only this single step number")
    args = parser.parse_args()

    Path(SMOKE_DIR).mkdir(parents=True, exist_ok=True)

    if args.only is not None:
        idx = args.only
        label, cmd = STEPS[idx - 1]
        run_step(idx, label, cmd)
        return

    failed = []
    for i, (label, cmd) in enumerate(STEPS, start=1):
        if i < args.start:
            print(f"[{i:>2}/{len(STEPS)}] skipping: {label}")
            continue
        rc = run_step(i, label, cmd)
        if rc != 0:
            failed.append((i, label))

    print(f"\n{'═' * 70}")
    passed = len(STEPS) - len(failed) - (args.start - 1)
    total  = len(STEPS) - (args.start - 1)
    print(f"DONE  {passed}/{total} steps passed")
    if failed:
        print("FAILED:")
        for idx, label in failed:
            print(f"  [{idx:>2}] {label}")
    print('═' * 70)


if __name__ == "__main__":
    main()
