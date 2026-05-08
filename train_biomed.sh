#!/bin/bash
cd "$(dirname "$0")"
source ~/.bashrc && conda activate clip_mimic

CUDA_VISIBLE_DEVICES=1 python train_lora.py \
    --base-model hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 \
    --train-csv cxr_data/mimic_cxr_train.csv \
    --image-dir /mnt/walkure_public/users/tomererez/mimic_cxr_jpg_images/mimic_cxr_jpg_images_from_google_cloud/mimic-cxr-jpg-2.1.0.physionet.org/files/ \
    --output-dir ./experiments/lora_biomedclip_with_5_ep \
    --batch-size 32 \
    --grad-accum-steps 2 \
    --epochs 5 \
    --lora-modules "qkv,proj,fc1,fc2,query,key,value" \
    2>&1 | tee logs/biomed_5_ep.log
