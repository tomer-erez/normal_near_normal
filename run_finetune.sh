#!/bin/bash
# Fine-tune the CXR-CLIP r50_mc checkpoint.
#
# Architecture: frozen ResNet50 image encoder (strong medical features)
#               + trainable Bio_ClinicalBERT text encoder via LoRA.
#
# LoRA targets: ClinicalBERT Q/K/V attention layers + all dense (FFN) layers.
# Loss:         label_dot clip — soft multi-positive labels, nan=ignore.
# LR:           5e-5 (lower than ViT-B-32 because ClinicalBERT is BERT-scale).

name="test_cxrclip_finetune_r50_labeldot"
set -e
cd "$(dirname "$0")"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis_clip
mkdir -p logs experiments/$name
tmux new-session -d -s $name "
    CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29511 train_lora.py \
        --cxrclip-finetune ./valid_pretrained_models_to_try/r50_mc.pt \
        --precision bf16 \
        --lr 5e-5 \
        --min-lr 1e-8 \
        --batch-size 6 \
        --max-samples 36 \
        --epochs 10 \
        --warmup-epochs 5 \
        --patience 10 \
        --nan-mode ignore \
        --loss clip \
        --caption-mode all \
        --match-mode label_dot \
        --lora-r 16 \
        --lora-alpha 32 \
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
