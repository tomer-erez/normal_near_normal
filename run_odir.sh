#!/usr/bin/env bash
# Train a LoRA-fine-tuned CLIP on ODIR-5K, then evaluate it on the ODIR-5K
# test split.
#
# GPUs: physical devices 0,1 (2 GPUs via torchrun DDP).
# Runs inside a detached tmux session (survives shell/SSH disconnects).
# Re-run this script to attach-or-relaunch; logs go to ./logs/run_odir.log
set -eo pipefail  # no -u: conda's own activate.d hooks reference unset vars

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "${SCRIPT_PATH}")"
cd "${SCRIPT_DIR}"

SESSION="run_odir"

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

export CUDA_VISIBLE_DEVICES=0,1

EXP_NAME="lora_vitb32_odir"
OUTPUT_DIR="./experiments/${EXP_NAME}"
EVAL_DIR="./eval_outputs/odir_test"

echo "=== [1/3] Train: ODIR ==="
torchrun --nproc_per_node=2 --master_port=29500 train_lora.py \
    --dataset odir \
    --base-model ViT-B-32 --pretrained openai \
    --output-dir "${OUTPUT_DIR}"

echo "=== [2/3] Build ODIR test gallery (skipped if already built) ==="
if [ ! -d "${EVAL_DIR}/paired_data" ]; then
    python baseline_eval/build_odir_gallery.py \
        --csv odir_data/odir_test.csv \
        --image_dir odir/odir/preprocessed_images \
        --output_dir "${EVAL_DIR}"
fi

echo "=== [3/3] Eval: ODIR ==="
python baseline_eval/eval_model.py \
    --model_type finetuned --dataset odir \
    --finetuned_base_model ViT-B-32 --finetuned_pretrained openai \
    --finetuned_checkpoint "${OUTPUT_DIR}/final_merged.pt" \
    --paired_dir "${EVAL_DIR}/paired_data" \
    --csv odir_data/odir_test.csv \
    --query_mode all \
    --ks 1 3 5 \
    --name "${EXP_NAME}_on_odir"

echo "Done. Results -> results_${EXP_NAME}_on_odir.csv"
