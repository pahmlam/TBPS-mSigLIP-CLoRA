#!/bin/bash

source "$(dirname "${BASH_SOURCE[0]}")/scripts/training_paths.sh"

uv run trainer.py -cn cir_msiglip \
    trainer.max_epochs=60 \
    trainer.accumulate_grad_batches=3 \
    ++trainer.precision=16-mixed \
    \
    optimizer=cir_test \
    optimizer.param_groups.default.lr=1e-4 \
    \
    +lora=default \
    \
    loss.NACIR=true \
    loss.nacir_config.fn_prior=0.010 \
    loss.nacir_config.epsilon_n=0.60 \
    loss.nacir_config.fn_enable_epoch=999 \
    loss.nacir_config.fp_enable_epoch=999 \
    "$@"
