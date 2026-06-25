#!/bin/bash
# Fine-tune the CXR-CLIP swint_mc checkpoint.
#
# Architecture: frozen SwinT image encoder + trainable Bio_ClinicalBERT text encoder.
# LoRA targets both encoder types via module names (query, key, value, dense).
#
# Loss: label_dot clip — soft multi-positive labels, nan=ignore.
#
# Before running, set environment variables (or edit the defaults below):
#   IMAGE_DIR  — root directory of MIMIC-CXR-JPG images
#   CKPT       — path to CXR-CLIP swint_mc.pt checkpoint
#   GPUS       — comma-separated GPU IDs (e.g. "0,1")
#   NPROC      — number of GPUs (must equal the count in GPUS)

IMAGE_DIR="${IMAGE_DIR:-/mnt/walkure_public/users/tomererez/mimic_cxr_jpg_images/mimic_cxr_jpg_images_from_google_cloud/mimic-cxr-jpg-2.1.0.physionet.org/files/}"
CKPT="${CKPT:-valid_pretrained_models_to_try/swint_mc.pt}"
GPUS="${GPUS:-1,2}"
NPROC="${NPROC:-2}"

name="current_method_from_cxrclip"
set -e
cd "$(dirname "$0")"
mkdir -p logs experiments/$name

tmux new-session -d -s "$name" "
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate thesis_clip
    CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node=$NPROC --master_port=29500 train_lora.py \
        --cxrclip-finetune $CKPT \
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
        --caption-weights 0.50 0.25 0.25 \
        --match-mode label_dot \
        --hnm-weight 0.3 \
        --lora-r 12 \
        --lora-alpha 12 \
        --lora-dropout 0.05 \
        --lora-target both \
        --lora-modules 'query,key,value,dense' \
        --output-dir ./experiments/$name \
        2>&1 | tee logs/${name}.log
"

echo "Training started in tmux session: $name"
echo "Run 'tmux attach -t $name' to watch logs, or: tail -f logs/${name}.log"
tmux attach -t "$name"
