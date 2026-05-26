#!/bin/bash

name="vitl14_singlelabel_bf16"
set -e
cd "$(dirname "$0")"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis_clip
mkdir -p logs experiments/$name
export CUDA_VISIBLE_DEVICES=1,2,3,4
tmux new-session -d -s $name "
    torchrun --nproc_per_node=4 --master_port=29500 train_lora.py \
        --base-model ViT-L-14 \
        --pretrained openai \
        --precision bf16 \
        --lr 3e-5 \
        --min-lr 1e-8 \
        --loss clip \
        --match-mode single_label \
        --caption-mode single \
        --output-dir ./experiments/$name \
        --lora-r 64 \
        --lora-alpha 64 \
        --batch-size 16 \
        --grad-accum-steps 2 \
        2>&1 | tee logs/${name}.log
"

tmux attach -t $name