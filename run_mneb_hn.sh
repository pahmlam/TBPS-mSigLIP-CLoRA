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
    loss.MNEB=true \
    loss.NACIR=false \
    loss.mneb_config.evidence_bank.enabled=true \
    loss.mneb_config.fnm_aux.enabled=true \
    loss.mneb_config.rde_aux.enabled=true \
    "$@"
