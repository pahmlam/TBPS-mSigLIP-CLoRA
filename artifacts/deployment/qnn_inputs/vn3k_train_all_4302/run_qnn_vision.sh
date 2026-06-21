#!/usr/bin/env bash
set -euo pipefail

QAIRT="${QAIRT:-/opt/qcom/qairt/2.45.40.260406}"
QNN_BIN="${QNN_BIN:-$QAIRT/bin/aarch64-ubuntu-gcc9.4}"
QNN_LIB="${QNN_LIB:-$QAIRT/lib/aarch64-ubuntu-gcc9.4}"
export LD_LIBRARY_PATH="$QNN_LIB:${LD_LIBRARY_PATH:-}"

# Update this if your skel files live in another 2.45 directory.
if [ -z "${ADSP_LIBRARY_PATH:-}" ]; then
  for candidate in "$QAIRT"/lib/hexagon-v68/* "$QAIRT"/lib/hexagon-v68; do
    if ls "$candidate"/*Skel*.so >/dev/null 2>&1; then
      export ADSP_LIBRARY_PATH="$candidate"
      break
    fi
  done
fi

cd "$(dirname "$0")"

"$QNN_BIN/qnn-net-run" \
  --backend "$QNN_LIB/libQnnHtp.so" \
  --retrieve_context "${VISION_BIN:-../../qnn_inputs/vision_encoder.bin}" \
  --config_file "${HTP_CONFIG:-../../../../deployment/config/qnn/htp_config_245.json}" \
  --input_list "input_list.txt" \
  --output_dir "${OUTPUT_DIR:-../../qnn_runs/vision_results}" \
  --profiling_level basic \
  --perf_profile high_performance \
  --log_level info
