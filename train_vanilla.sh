#!/bin/bash
cd "$(dirname "$0")"
source ~/.bashrc && conda activate clip_mimic

# ── Single-GPU (original) ──────────────────────────────────────────────────
# CUDA_VISIBLE_DEVICES=2 python train_lora.py \
#     --base-model ViT-B-32 \
#     --pretrained openai \
#     --train-csv cxr_data/mimic_cxr_train.csv \
#     --image-dir /mnt/walkure_public/users/tomererez/mimic_cxr_jpg_images/mimic_cxr_jpg_images_from_google_cloud/mimic-cxr-jpg-2.1.0.physionet.org/files/ \
#     --output-dir ./experiments/lora_vitb32_with_5_ep \
#     --batch-size 64 \
#     --grad-accum-steps 2 \
#     --epochs 5 \
#     --report-interval-steps 50 \
#     2>&1 | tee logs/vanilla_5_ep.log

# ── Multi-GPU via torchrun (4× RTX 2080 Ti) ───────────────────────────────
# Effective batch = batch-size(64) × grad-accum(2) × gpus(4) = 512
CUDA_VISIBLE_DEVICES=1,2,3,4 torchrun --nproc_per_node=4 train_lora.py \
    --base-model ViT-B-32 \
    --pretrained openai \
    --train-csv cxr_data/mimic_cxr_train.csv \
    --image-dir /cxr_data/images/mimic_cxr_jpg_images_from_google_cloud/mimic-cxr-jpg-2.1.0.physionet.org/files/ \
    --output-dir ./experiments/full_vanilla_run \
    --batch-size 64 \
    --grad-accum-steps 2 \
    --epochs 100 \
    --report-interval-steps 50 \
    2>&1 | tee logs/vanilla_full_run.log
