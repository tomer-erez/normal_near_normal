#!/bin/bash

name="clip_cxrclip_us"
set -e
cd "$(dirname "$0")"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis_clip
mkdir -p logs eval_outputs/$name

tmux new-session -d -s $name "
    CUDA_VISIBLE_DEVICES=0 python baseline_eval/run_all_evals.py \
        --paired_dir ./eval_outputs/baseline_output_official_test/paired_data \
        --csv cxr_data/mimic_cxr_official_test.csv \
        --output_dir ./eval_outputs/$name \
        --batch_size 512 \
        --skip_existing \
        --wandb-project mimic-cxr-clip \
        --wandb-run-name $name \
        --wandb-entity tomererez1998-technion-israel-institute-of-technology \
        2>&1 | tee logs/${name}.log
"

tmux attach -t $name
