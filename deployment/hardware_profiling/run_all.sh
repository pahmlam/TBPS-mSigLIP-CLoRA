#!/bin/bash
# Hardware profiling benchmarks (proxy models: MobileNetV2, ResNet18, EfficientNet)
# NOT mSigLIP — this profiles RB3 Gen2 hardware capabilities
# All terminal output is automatically logged with timestamp
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/../logs"
ARTIFACT_DIR="${MSIGLIP_DEPLOYMENT_ARTIFACTS:-artifacts/deployment/hardware_profiling}"

cd ~/sigm
source venv/bin/activate

# Setup logging — tee all stdout+stderr to timestamped log file
mkdir -p "$ARTIFACT_DIR/logs"
LOG_FILE="$ARTIFACT_DIR/logs/hardware_profiling_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=========================================="
echo "Qualcomm RB3 Gen2 Hardware Profiling"
echo "(Proxy models — NOT mSigLIP)"
echo "=========================================="
echo "Started: $(date)"
echo "Log file: $LOG_FILE"
echo ""

# Step 1: Collect system info
echo "[1/3] Collecting system information..."
./collect_sysinfo.sh

# Step 2: Run benchmarks
echo "[2/3] Running proxy model benchmarks..."
python benchmark.py

# Step 3: Summary
echo "[3/3] Generating summary..."
echo ""
echo "=========================================="
echo "Files generated:"
ls -la *.md *.json *.txt 2>/dev/null
echo "=========================================="
echo "Completed: $(date)"
echo "Log saved: $LOG_FILE"
