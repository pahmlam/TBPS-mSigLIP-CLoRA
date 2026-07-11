#!/bin/bash

source "$(dirname "${BASH_SOURCE[0]}")/scripts/training_paths.sh"

uv run trainer.py -cn cir_msiglip \
    trainer.max_epochs=60 \
    trainer.accumulate_grad_batches=3 \
    ++trainer.precision=16-mixed \
    \
    optimizer=cir_test \
    optimizer.param_groups.default.lr=1e-4 \
    loss.nitc_mvs_mode=direct \
    loss.NACIR=false \
    loss.MNEB=false \
    loss.PART_ALIGN=false \
    \
    +lora=default \
    "$@"
