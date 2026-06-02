#!/bin/bash

name="caption_mode_all_clip"
set -e
cd "$(dirname "$0")"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis_clip
mkdir -p logs experiments/$name
tmux new-session -d -s $name "
    CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29507 train_lora.py \
        --base-model ViT-B-32 \
        --pretrained openai \
        --precision bf16 \
        --lr 1e-4 \
        --min-lr 1e-8 \
        --batch-size 128 \
        --nan-mode ignore \
        --loss clip \
        --caption-mode all \
        --match-mode single_label \
        --lora-r 16 \
        --lora-alpha 16 \
        --output-dir ./experiments/$name \
        --wandb-project mimic-cxr-clip \
        --wandb-run-name $name \
        --wandb-entity tomererez1998-technion-israel-institute-of-technology \
        2>&1 | tee logs/${name}.log
"

tmux attach -t $name