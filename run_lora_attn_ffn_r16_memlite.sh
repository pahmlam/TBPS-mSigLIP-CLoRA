#!/bin/bash

source "$(dirname "${BASH_SOURCE[0]}")/scripts/training_paths.sh"

uv run trainer.py -cn cir_msiglip \
    trainer.max_epochs=60 \
    trainer.accumulate_grad_batches=6 \
    ++trainer.precision=16-mixed \
    \
    dataset.batch_size=12 \
    dataset.test_batch_size=32 \
    dataset.num_workers=2 \
    \
    optimizer=cir_test \
    optimizer.param_groups.default.lr=1e-4 \
    loss.NACIR=false \
    \
    +lora=attn_ffn_r16 \
    "$@"
