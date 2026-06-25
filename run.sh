#!/bin/bash
# LoRA fine-tuning from OpenAI ViT-B/32 CLIP.
#
# Loss: label_dot clip — soft multi-positive labels, nan=ignore.
#
# Before running, set environment variables (or edit the defaults below):
#   IMAGE_DIR  — root directory of MIMIC-CXR-JPG images
#   GPUS       — comma-separated GPU IDs (e.g. "0,1")
#   NPROC      — number of GPUs (must equal the count in GPUS)

IMAGE_DIR="${IMAGE_DIR:-/mnt/walkure_public/users/tomererez/mimic_cxr_jpg_images/mimic_cxr_jpg_images_from_google_cloud/mimic-cxr-jpg-2.1.0.physionet.org/files/}"
GPUS="${GPUS:-3,4}"
NPROC="${NPROC:-2}"

name="currend_method_from_vanilla"
set -e
cd "$(dirname "$0")"
mkdir -p logs experiments/$name

tmux new-session -d -s "$name" "
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate thesis_clip
    CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node=$NPROC --master_port=29501 train_lora.py \
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

echo "Training started in tmux session: $name"
echo "Run 'tmux attach -t $name' to watch logs, or: tail -f logs/${name}.log"
tmux attach -t "$name"
