#!/bin/bash

# Script for running the training process for the KleinViT.

echo "Running training with Klein Features (Topological Attention)..."

# Set PYTHONPATH to include the project's root directory
export PYTHONPATH=.

# Run the training script with specified hyperparameters
# and the flag to use topological attention (Approach B).
CUDA_VISIBLE_DEVICES=4 python3 src/training/trainer.py \
    --epochs 50 \
    --lr 1e-4 \
    --batch_size 256 \
    --patch_size 3 \
    --dim 256 \
    --depth 6 \
    --heads 8 \
    --mlp_dim 512 \
    --use_topological_attention

echo "Training script finished."