#!/bin/bash
# Fine-tune the CXR-CLIP swint_mc checkpoint.
#
# Architecture: frozen SwinT image encoder + trainable Bio_ClinicalBERT text encoder.
# Both encoders are transformers so LoRA applies to both via the same module names
# (query, key, value, dense) — image and text adapt at ~2% trainable params each.
# This gives a fair parameter-budget comparison with the ViT-B-32 LoRA models.
#
# Loss: label_dot clip — soft multi-positive labels, nan=ignore.

name="labeldot_hnm_swint_M_hnm01"
set -e
cd "$(dirname "$0")"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis_clip
mkdir -p logs experiments/$name
tmux new-session -d -s $name "
    CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 --master_port=29504 train_lora.py \
        --cxrclip-finetune ./valid_pretrained_models_to_try/swint_m.pt \
        --precision bf16 \
        --lr 1e-4 \
        --min-lr 1e-8 \
        --batch-size 70 \
        --epochs 100 \
        --warmup-epochs 5 \
        --patience 10 \
        --nan-mode ignore \
        --loss clip \
        --caption-mode all \
        --match-mode label_dot \
        --hnm-weight 0.1 \
        --lora-r 12 \
        --lora-alpha 12 \
        --lora-dropout 0.05 \
        --lora-target both \
        --lora-modules \"query,key,value,dense\" \
        --output-dir ./experiments/$name \
        --wandb-project mimic-cxr-clip \
        --wandb-run-name $name \
        --wandb-entity tomererez1998-technion-israel-institute-of-technology \
        2>&1 | tee logs/${name}.log
"

tmux attach -t $name
