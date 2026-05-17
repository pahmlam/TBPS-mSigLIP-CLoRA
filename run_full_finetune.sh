#!/bin/bash

source "$(dirname "${BASH_SOURCE[0]}")/scripts/training_paths.sh"

uv run trainer.py -cn cir_msiglip \
    trainer.max_epochs=60 \
    trainer.accumulate_grad_batches=3 \
    ++trainer.precision=16-mixed \
    trainer.gradient_clip_val=1.0 \
    trainer.gradient_clip_algorithm=norm \
    \
    dataset.batch_size=24 \
    optimizer=cir_test \
    optimizer.param_groups.default.lr=1e-5 \
    optimizer.param_groups.backbone.lr=1e-5
