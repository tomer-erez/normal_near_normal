#!/bin/bash

name="biomedclip_matchmode_graded_captionmode_all_10labels"
set -e
cd "$(dirname "$0")"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis_clip
mkdir -p logs experiments/$name
tmux new-session -d -s $name "
    CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 --master_port=29503 train_lora.py \
        --base-model hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 \
        --precision bf16 \
        --lr 1e-4 \
        --min-lr 1e-8 \
        --batch-size 38 \
        --grad-accum-steps 4 \
        --lora-modules \"qkv,proj,fc1,fc2,query,key,value\" \
        --nan-mode ignore \
        --loss clip \
        --caption-mode all \
        --match-mode graded \
        --lora-r 16 \
        --lora-alpha 16 \
        --output-dir ./experiments/$name \
        --wandb-project mimic-cxr-clip \
        --wandb-run-name $name \
        --wandb-entity tomererez1998-technion-israel-institute-of-technology \
        2>&1 | tee logs/${name}.log
"

tmux attach -t $name
