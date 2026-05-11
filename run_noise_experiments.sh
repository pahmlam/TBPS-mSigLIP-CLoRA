#!/bin/bash
# Run Circle Loss experiments across noise rates 0.0-0.8
# Adapted from RDE's train_multiple_noise.sh

set -e

NOISY_RATES=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8)
DATASET_NAME="VN3K_VI"

# Ensure noiseindex directory exists
mkdir -p ./noiseindex

for noisy_rate in "${NOISY_RATES[@]}"; do
    echo "=========================================="
    echo "Training with noisy_rate = $noisy_rate"
    echo "=========================================="

    uv run trainer.py -cn cir_msiglip \
        dataset.noisy_rate=$noisy_rate \
        dataset.noisy_file="./noiseindex/${DATASET_NAME}_${noisy_rate}.npy" \
        trainer.max_epochs=60 \
        trainer.accumulate_grad_batches=3 \
        ++trainer.precision=16-mixed \
        optimizer=cir_test \
        optimizer.param_groups.default.lr=1e-4 \
        +lora=default

    echo "Completed noisy_rate = $noisy_rate"
done

echo "All noise rate experiments completed."
