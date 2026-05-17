#!/usr/bin/env bash

# Resolve training paths for both the standardized layout and existing server
# workspaces. Explicit environment variables always win.
MSIGLIP_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$MSIGLIP_PROJECT_ROOT"

if [ -z "${MSIGLIP_DATA_ROOT:-}" ]; then
    if [ -d "$MSIGLIP_PROJECT_ROOT/data/raw/VN3K" ]; then
        export MSIGLIP_DATA_ROOT="$MSIGLIP_PROJECT_ROOT/data/raw"
    elif [ -d "$MSIGLIP_PROJECT_ROOT/VN3K" ]; then
        export MSIGLIP_DATA_ROOT="$MSIGLIP_PROJECT_ROOT"
    fi
fi

if [ -z "${MSIGLIP_PRETRAINED_ROOT:-}" ]; then
    if [ -f "$MSIGLIP_PROJECT_ROOT/artifacts/models/pretrained/m_siglip_checkpoints/model.safetensors" ]; then
        export MSIGLIP_PRETRAINED_ROOT="$MSIGLIP_PROJECT_ROOT/artifacts/models/pretrained"
    elif [ -f "$MSIGLIP_PROJECT_ROOT/m_siglip_checkpoints/model.safetensors" ]; then
        export MSIGLIP_PRETRAINED_ROOT="$MSIGLIP_PROJECT_ROOT"
    fi
fi
