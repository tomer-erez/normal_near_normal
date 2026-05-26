#!/bin/bash

name="vitb32_cliploss_negaware_bs32_no_conflict_zeriong_version"
set -e
cd "$(dirname "$0")"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis_clip
mkdir -p logs experiments/$name
export CUDA_VISIBLE_DEVICES=1,2
tmux new-session -d -s $name "
    torchrun --nproc_per_node=2 --master_port=29500 train_lora.py \
        --base-model ViT-B-32 \
        --pretrained openai \
        --precision bf16 \
        --lr 5e-5 \
        --min-lr 1e-8 \
        --loss clip \
        --match-mode negative_aware \
        --negative-weight 0.1 \
        --negative-margin 0.0 \
        --batch-size 64 \
        --lora-r 32 \
        --lora-alpha 32 \
        --output-dir ./experiments/$name \
        2>&1 | tee logs/${name}.log
"

tmux attach -t $name