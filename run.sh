#!/bin/bash

name="vit_l_14_r16a32"

set -e
cd "$(dirname "$0")"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis_clip

mkdir -p logs experiments/$name

export CUDA_VISIBLE_DEVICES=1,2,3,4

tmux new-session -d -s $name "
    torchrun --nproc_per_node=4 --master_port=29503 train_lora.py \
        --base-model ViT-L-14 \
        --pretrained openai \
        --loss clip \
        --output-dir ./experiments/$name \
        --lora-r 16 \
        --lora-alpha 32 \
        --batch-size 16 \
        --grad-accum-steps 2 \
        2>&1 | tee logs/${name}.log
"

tmux attach -t $name