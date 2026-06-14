#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR="${1:-/tmp}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
PKG_NAME="msiglip_colab_training_code_${TIMESTAMP}"
STAGE="${OUT_DIR%/}/${PKG_NAME}"
OUT="${OUT_DIR%/}/${PKG_NAME}.tar.gz"

require_file() {
    local path="$1"
    if [[ ! -e "$path" ]]; then
        echo "Missing required file or directory: $path" >&2
        exit 1
    fi
}

assert_archive_absent() {
    local pattern="$1"
    local label="$2"
    if tar -tzf "$OUT" | grep -E "$pattern" >/dev/null; then
        echo "BAD: ${label} included" >&2
        tar -tzf "$OUT" | grep -E "$pattern" | head -20 >&2
        exit 1
    fi
    echo "OK: no ${label}"
}

assert_archive_present() {
    local needle="$1"
    if ! tar -tzf "$OUT" | grep -F "$needle" >/dev/null; then
        echo "BAD: required archive path missing: $needle" >&2
        exit 1
    fi
    tar -tzf "$OUT" | grep -F "$needle" | head -1
}

require_file src
require_file configs
require_file scripts/training_paths.sh
require_file tests/test_lora_configs.py
require_file tests/test_part_alignment_loss.py
require_file tests/test_evidence_bank.py
require_file tests/test_mneb_objectives.py
require_file tests/test_mneb_integration.py
require_file notebooks/colab_training_experiments.ipynb
require_file trainer.py
require_file test.py
require_file pyproject.toml
require_file README.md
require_file run_mneb_hn.sh
require_file src/msiglip/model/evidence_bank.py
require_file src/msiglip/data/prw_tps_cn.py
require_file configs/loss/cir_msiglip.yaml
require_file configs/dataset/prw_tps_cn.yaml
require_file configs/lora/attn_ffn_r16.yaml
require_file configs/lora/attn_ffn_r32.yaml
require_file configs/lora/attn_ffn_r32_pissa.yaml
require_file configs/lora/attn_ffn_r32_dora.yaml
require_file configs/lora/attn_ffn_r32_rslora.yaml

rm -rf "$STAGE"
mkdir -p "$STAGE"

RSYNC_EXCLUDES=(
    --exclude="__pycache__/"
    --exclude="*.pyc"
    --exclude="*.pyo"
    --exclude=".DS_Store"
    --exclude="*.egg-info/"
)

rsync -a "${RSYNC_EXCLUDES[@]}" src configs scripts tests "$STAGE"/
rsync -a trainer.py test.py pyproject.toml README.md "$STAGE"/

mkdir -p "$STAGE/notebooks"
rsync -a notebooks/colab_training_experiments.ipynb "$STAGE/notebooks/"
rsync -a run_*.sh "$STAGE"/

cat >"$STAGE/COLAB_PACKAGE_MANIFEST.txt" <<EOF
Package: ${PKG_NAME}
Created from: ${REPO_ROOT}
Created at: $(date)

Included:
- trainer.py, test.py, pyproject.toml, README.md
- src/
- configs/
- scripts/
- tests/
- run_*.sh
- notebooks/colab_training_experiments.ipynb

Excluded by whitelist:
- deployment/
- artifacts/
- data/raw/
- VN3K/
- m_siglip_checkpoints/
- knowledge/
- ref/
- experiments/
- venv/, .venv/, caches

Expected Colab-side external assets:
- VN3K/
- CUHK-PEDES/ (for natural-noise English experiments)
- PRW-TPS-CN/ (for Chinese experiments)
- m_siglip_checkpoints/model.safetensors
EOF

tar -C "$OUT_DIR" -czf "$OUT" "$PKG_NAME"

echo "Archive created:"
echo "$OUT"
ls -lh "$OUT"

echo
echo "Archive preview:"
tar -tzf "$OUT" | head -50

echo
echo "Archive validation:"
assert_archive_absent '(^|/)deployment/' "deployment"
assert_archive_absent '(^|/)artifacts/' "artifacts"
assert_archive_absent '(^|/)data/raw/' "data/raw"
assert_archive_absent '(^|/)VN3K/' "VN3K"
assert_archive_absent '(^|/)m_siglip_checkpoints/' "pretrained checkpoint"
assert_archive_absent '(^|/)knowledge/' "knowledge"
assert_archive_absent '(^|/)ref/' "ref"
assert_archive_absent '(^|/)venv/' "venv"
assert_archive_absent '(^|/)\\.venv/' ".venv"
assert_archive_absent '(^|/)__pycache__/' "__pycache__"
assert_archive_absent '\\.pyc$' "pyc files"

assert_archive_present "configs/loss/cir_msiglip.yaml"
assert_archive_present "configs/dataset/prw_tps_cn.yaml"
assert_archive_present "configs/lora/attn_ffn_r16.yaml"
assert_archive_present "configs/lora/attn_ffn_r32.yaml"
assert_archive_present "configs/lora/attn_ffn_r32_pissa.yaml"
assert_archive_present "configs/lora/attn_ffn_r32_dora.yaml"
assert_archive_present "configs/lora/attn_ffn_r32_rslora.yaml"
assert_archive_present "tests/test_lora_configs.py"
assert_archive_present "tests/test_part_alignment_loss.py"
assert_archive_present "tests/test_evidence_bank.py"
assert_archive_present "tests/test_mneb_objectives.py"
assert_archive_present "tests/test_mneb_integration.py"
assert_archive_present "notebooks/colab_training_experiments.ipynb"
assert_archive_present "trainer.py"
assert_archive_present "src/msiglip/train.py"
assert_archive_present "src/msiglip/model/evidence_bank.py"
assert_archive_present "src/msiglip/data/prw_tps_cn.py"
assert_archive_present "run_mneb_hn.sh"

echo
echo "Colab extract command template:"
cat <<EOF
mkdir -p /content/drive/MyDrive/mSigLIP/code
tar -xzf /content/drive/MyDrive/mSigLIP/${PKG_NAME}.tar.gz \\
  -C /content/drive/MyDrive/mSigLIP/code \\
  --strip-components=1
EOF
