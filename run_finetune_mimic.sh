#!/usr/bin/env bash
# Train a LoRA-fine-tuned CLIP on MIMIC-CXR, then evaluate it separately on
# the CheXpert validation set and the MIMIC-CXR official test set.
#
# GPUs: physical devices 4,5 (2 GPUs via torchrun DDP).
# Runs inside a detached tmux session (survives shell/SSH disconnects).
# Re-run this script to attach-or-relaunch; logs go to ./logs/run_finetune_mimic.log
set -eo pipefail  # no -u: conda's own activate.d hooks reference unset vars

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "${SCRIPT_PATH}")"
cd "${SCRIPT_DIR}"

SESSION="run_finetune_mimic"

if [ -z "${TMUX:-}" ]; then
    if tmux has-session -t "${SESSION}" 2>/dev/null; then
        echo "tmux session '${SESSION}' is already running."
        echo "Attach with: tmux attach -t ${SESSION}"
        exit 0
    fi
    tmux new-session -d -s "${SESSION}" -c "${SCRIPT_DIR}" "${SCRIPT_PATH}"
    echo "Launched in detached tmux session '${SESSION}'."
    echo "Attach with: tmux attach -t ${SESSION}"
    echo "Logs: ${SCRIPT_DIR}/logs/${SESSION}.log"
    exit 0
fi

# --- Everything below runs inside the tmux session ---
mkdir -p logs
exec > >(tee "logs/${SESSION}.log") 2>&1

source /home/tomererez/miniconda3/etc/profile.d/conda.sh
conda activate thesis_clip

export CUDA_VISIBLE_DEVICES=4,5

EXP_NAME="lora_vitb32_mimic"
OUTPUT_DIR="./experiments/${EXP_NAME}"

MIMIC_EVAL_DIR="./eval_outputs/exp1"
CHEXPERT_EVAL_DIR="./eval_outputs/chexpert_valid"

echo "=== [1/3] Train: MIMIC-CXR ==="
torchrun --nproc_per_node=2 --master_port=29502 train_lora.py \
    --dataset cxr \
    --base-model ViT-B-32 --pretrained openai \
    --output-dir "${OUTPUT_DIR}"

echo "=== [2/3] Build eval galleries (skipped if already built) ==="
if [ ! -d "${CHEXPERT_EVAL_DIR}/paired_data" ]; then
    python baseline_eval/build_chexpert_baseline.py \
        --chexpert_dir chexpert/chexpert --split valid \
        --output_dir "${CHEXPERT_EVAL_DIR}"
fi
if [ ! -d "${MIMIC_EVAL_DIR}/paired_data" ]; then
    python baseline_eval/build_baseline.py \
        --csv cxr_data/mimic_cxr_official_test.csv \
        --output_dir "${MIMIC_EVAL_DIR}"
fi

echo "=== [3/3] Eval: CheXpert, then MIMIC-CXR (separately) ==="
python baseline_eval/eval_model.py \
    --model_type finetuned --dataset cxr \
    --finetuned_base_model ViT-B-32 --finetuned_pretrained openai \
    --finetuned_checkpoint "${OUTPUT_DIR}/final_merged.pt" \
    --paired_dir "${CHEXPERT_EVAL_DIR}/paired_data" \
    --csv "${CHEXPERT_EVAL_DIR}/chexpert_valid_labels.csv" \
    --query_mode all \
    --name "${EXP_NAME}_on_chexpert"

python baseline_eval/eval_model.py \
    --model_type finetuned --dataset cxr \
    --finetuned_base_model ViT-B-32 --finetuned_pretrained openai \
    --finetuned_checkpoint "${OUTPUT_DIR}/final_merged.pt" \
    --paired_dir "${MIMIC_EVAL_DIR}/paired_data" \
    --csv cxr_data/mimic_cxr_official_test.csv \
    --query_mode all \
    --name "${EXP_NAME}_on_mimic"

echo "Done. Results -> results_${EXP_NAME}_on_chexpert.csv, results_${EXP_NAME}_on_mimic.csv"
