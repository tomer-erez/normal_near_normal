#!/bin/bash

name="claude_hyperparms_label_dot_clip_r32_a32"
set -e
cd "$(dirname "$0")"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis_clip
mkdir -p logs experiments/$name
tmux new-session -d -s $name "
    CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 --master_port=29510 train_lora.py \
        --base-model ViT-B-32 \
        --pretrained openai \
        --precision bf16 \
        --lr 1e-4 \
        --min-lr 1e-8 \
        --batch-size 90 \
        --epochs 100 \
        --warmup-epochs 5 \
        --patience 10 \
        --nan-mode ignore \
        --loss clip \
        --caption-mode all \
        --match-mode label_dot \
        --lora-r 32 \
        --lora-alpha 32 \
        --output-dir ./experiments/$name \
        --wandb-project mimic-cxr-clip \
        --wandb-run-name $name \
        --wandb-entity tomererez1998-technion-israel-institute-of-technology \
        2>&1 | tee logs/${name}.log
"

tmux attach -t $name