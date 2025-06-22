#!/bin/bash

# This script runs all four training sessions in parallel, assigning each
# to a different GPU. It waits for all jobs to complete.

echo "Starting parallel training for all ViT models..."

# Set PYTHONPATH to include the project's root directory
export PYTHONPATH=.

# --- Configuration ---
# Set the GPU IDs you want to use for the 4 training runs.
GPUS=(4 5 6 7)
EPOCHS=50

# Common training parameters
COMMON_PARAMS="--epochs $EPOCHS --lr 1e-4 --batch_size 512"

# Define model-specific flags
declare -A MODELS
MODELS=(
    ["Baseline"]="--save_path ./models/vit_baseline.pth"
    ["Approach A"]="--use_klein_features --save_path ./models/vit_approach_a.pth"
    ["Approach B"]="--use_topological_attention --save_path ./models/vit_approach_b.pth"
    ["Approach C"]="--use_gated_attention --save_path ./models/vit_approach_c.pth"
)

# --- Launch Training Jobs ---
PIDS=()
i=0
for name in "${!MODELS[@]}"; do
    GPU_ID=${GPUS[$i]}
    FLAGS=${MODELS[$name]}

    echo "==> Launching '$name' on GPU $GPU_ID..."

    # Launch the training in the background
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 src/training/trainer.py \
        $COMMON_PARAMS \
        $FLAGS &

    PIDS+=($!)
    i=$((i+1))
done

# --- Wait for all jobs to complete ---
echo -e "\nAll training jobs launched. Waiting for completion..."
echo "Process IDs: ${PIDS[*]}"

for pid in "${PIDS[@]}"; do
    wait $pid
    if [ $? -eq 0 ]; then
        echo "Process $pid finished successfully."
    else
        echo "Process $pid failed."
    fi
done

echo -e "\nAll training runs have completed."