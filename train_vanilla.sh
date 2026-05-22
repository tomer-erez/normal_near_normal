#!/bin/bash
cd "$(dirname "$0")"
source ~/.bashrc && conda activate thesis_clip

# ── Multi-GPU via torchrun (4× RTX 2080 Ti) ───────────────────────────────
# Effective batch = batch-size(16) × grad-accum(2) × gpus(4) = 128
CUDA_VISIBLE_DEVICES=1,2,3,4 torchrun --nproc_per_node=4 --master_port=29504 train_lora.py \
    --base-model ViT-B-32 \
    --pretrained openai \
    --output-dir experiments/_vanilla_bsz16x8_with_siglip \
    --loss siglip \
    --batch-size 16 \
    --grad-accum-steps 2 \
    --epochs 100 \
    --report-interval-steps 500 \
    2>&1 | tee logs/_vanilla_bsz16x8_with_siglip.log
