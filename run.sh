#!/bin/bash
# LoRA fine-tuning from OpenAI ViT-B/32 CLIP.
#
# Loss: label_dot clip — soft multi-positive labels, nan=ignore.
#
# Before running, set:
#   IMAGE_DIR  — root directory of MIMIC-CXR-JPG images
#   GPUS       — comma-separated list of GPU IDs to use
#   NPROC      — number of GPUs (must match len(GPUS))

IMAGE_DIR="${IMAGE_DIR:-<PATH/TO/MIMIC-CXR-JPG/files/>}"
GPUS="${GPUS:-0,1}"
NPROC="${NPROC:-2}"

name="lora_vitb32_labeldot"
set -e
cd "$(dirname "$0")"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis_clip
mkdir -p logs experiments/$name
tmux new-session -d -s $name "
    CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node=$NPROC --master_port=29502 train_lora.py \
        --base-model ViT-B-32 \
        --pretrained openai \
        --train-csv cxr_data/mimic_cxr_train.csv \
        --image-dir $IMAGE_DIR \
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
        --hnm-weight 0.3 \
        --lora-r 16 \
        --lora-alpha 16 \
        --output-dir ./experiments/$name \
        2>&1 | tee logs/${name}.log
"

tmux attach -t $name
