import os
from pathlib import Path

from huggingface_hub import snapshot_download


pretrained_root = Path(
    os.environ.get("MSIGLIP_PRETRAINED_ROOT", "artifacts/models/pretrained")
)

snapshot_download(
    repo_id="google/siglip-base-patch16-256-multilingual",
    local_dir=pretrained_root / "m_siglip_checkpoints",
    max_workers=4,
)
