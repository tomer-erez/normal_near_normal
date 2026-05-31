#!/bin/bash

name="negaware_w08"
set -e
cd "$(dirname "$0")"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis_clip
mkdir -p logs experiments/$name
tmux new-session -d -s $name "
    CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 --master_port=29500 train_lora.py \
        --base-model ViT-B-32 \
        --pretrained openai \
        --precision bf16 \
        --lr 1e-4 \
        --min-lr 1e-8 \
        --loss clip \
        --batch-size 50 \
        --match-mode negative_aware \
        --negative-weight 0.80 \
        --negative-margin 0.0 \
        --lora-r 16 \
        --lora-alpha 16 \
        --output-dir ./experiments/$name \
        --wandb-project mimic-cxr-clip \
        --wandb-run-name $name \
        --wandb-entity tomererez1998-technion-israel-institute-of-technology \
        2>&1 | tee logs/${name}.log
"

tmux attach -t $name