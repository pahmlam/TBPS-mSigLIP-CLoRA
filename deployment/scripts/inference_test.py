"""
Step 3: Run inference test on target device (Qualcomm RB3 Gen2 or any CPU machine).
Tests model loading, single-sample inference, latency, and memory usage.

Usage (PyTorch FP16):
    python deployment/scripts/inference_test.py \
        --model-dir artifacts/deployment/exports/msiglip_lora \
        --dtype fp16 \
        --dataset-root /path/to/VN3K

Usage (ONNX):
    python deployment/scripts/inference_test.py \
        --model-dir artifacts/deployment/exports/msiglip_lora \
        --backend onnx \
        --dataset-root /path/to/VN3K

If --dataset-root is not provided, uses random dummy data for a quick smoke test.
"""

import argparse
import os
import sys
import time
import gc

import numpy as np

# Add deployment root to path
_deployment_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_project_root = os.path.dirname(_deployment_root)
sys.path.insert(0, _deployment_root)
from deploy_utils import TeeLogger

sys.path.insert(0, os.path.join(_project_root, "src"))

DEFAULT_MODEL_DIR = os.path.join(
    _project_root, "artifacts", "deployment", "exports", "msiglip_lora"
)
DEFAULT_LOG_DIR = os.path.join(_project_root, "artifacts", "deployment", "logs")


def get_memory_mb():
    """Get current process memory usage in MB."""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Linux: KB -> MB
    except Exception:
        return -1


def test_pytorch(model_dir: str, dtype: str, dataset_root: str = None, num_samples: int = 5):
    """Test inference with PyTorch."""
    import torch
    import yaml

    print(f"\n{'='*60}")
    print(f"PYTORCH INFERENCE TEST (dtype={dtype})")
    print(f"{'='*60}")

    mem_before = get_memory_mb()

    # Load config
    config_path = os.path.join(model_dir, "config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    img_size = config["backbone"]["vision_config"]["image_size"]
    if isinstance(img_size, (list, tuple)):
        h, w = img_size
    else:
        h = w = img_size
    max_text_len = config["tokenizer"]["model_max_length"]

    # Load model
    model_file = f"model_{dtype}.pt"
    model_path = os.path.join(model_dir, model_file)
    print(f"Loading {model_path}...")

    t0 = time.time()
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    load_time = time.time() - t0
    print(f"State dict loaded in {load_time:.2f}s")

    # Reconstruct model
    from omegaconf import OmegaConf

    def resolve_tuple(*args):
        return tuple(args)

    OmegaConf.register_new_resolver("tuple", resolve_tuple, replace=True)
    OmegaConf.register_new_resolver("eval", eval, replace=True)

    from msiglip.model.build import build_backbone_with_proper_layer_resize
    from msiglip.model.tbps import TBPS

    omegaconf = OmegaConf.create(config)
    backbone = build_backbone_with_proper_layer_resize(omegaconf.backbone)
    model = TBPS(config=omegaconf, backbone=backbone)

    # Load weights
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    if dtype == "fp16":
        model = model.half()

    mem_after_load = get_memory_mb()
    print(f"Memory: before={mem_before:.0f}MB, after load={mem_after_load:.0f}MB, delta={mem_after_load-mem_before:.0f}MB")

    # Prepare input
    if dataset_root:
        print(f"\nLoading real data from {dataset_root}...")
        test_pytorch_with_dataset(model, omegaconf, dataset_root, dtype, num_samples)
    else:
        print("\nNo dataset provided — using random dummy data for smoke test.")
        test_pytorch_with_dummy(model, h, w, max_text_len, dtype, num_samples)


def test_pytorch_with_dummy(model, h, w, max_text_len, dtype, num_samples):
    """Smoke test with random data."""
    import torch

    torch_dtype = torch.float16 if dtype == "fp16" else torch.float32

    # --- Image encoding ---
    print("\n--- Image Encoder ---")
    dummy_image = torch.randn(1, 3, h, w, dtype=torch_dtype)

    latencies = []
    for i in range(num_samples + 1):  # +1 for warmup
        t0 = time.time()
        with torch.no_grad():
            img_feat = model.encode_image(dummy_image)
        elapsed = time.time() - t0
        if i > 0:  # skip warmup
            latencies.append(elapsed)
        if i == 0:
            print(f"  Warmup: {elapsed:.3f}s")

    print(f"  Output shape: {img_feat.shape}")
    print(f"  Latency: {np.mean(latencies):.3f}s +/- {np.std(latencies):.3f}s (n={len(latencies)})")
    print(f"  Throughput: {1/np.mean(latencies):.1f} images/sec")

    # --- Text encoding ---
    print("\n--- Text Encoder ---")
    dummy_input_ids = torch.zeros(1, max_text_len, dtype=torch.long)
    dummy_attention_mask = torch.ones(1, max_text_len, dtype=torch.long)
    caption_input = {"input_ids": dummy_input_ids, "attention_mask": dummy_attention_mask}

    latencies = []
    for i in range(num_samples + 1):
        t0 = time.time()
        with torch.no_grad():
            txt_feat = model.encode_text(caption_input)
        elapsed = time.time() - t0
        if i > 0:
            latencies.append(elapsed)
        if i == 0:
            print(f"  Warmup: {elapsed:.3f}s")

    print(f"  Output shape: {txt_feat.shape}")
    print(f"  Latency: {np.mean(latencies):.3f}s +/- {np.std(latencies):.3f}s (n={len(latencies)})")
    print(f"  Throughput: {1/np.mean(latencies):.1f} texts/sec")

    # --- Similarity ---
    import torch.nn.functional as F
    img_norm = F.normalize(img_feat, dim=-1)
    txt_norm = F.normalize(txt_feat, dim=-1)
    similarity = (img_norm @ txt_norm.t()).item()
    print(f"\n--- Similarity ---")
    print(f"  Image-Text similarity (random): {similarity:.4f}")

    mem_final = get_memory_mb()
    print(f"\n--- Final Memory: {mem_final:.0f}MB ---")
    print("\nSMOKE TEST PASSED - model runs on this device.")


def test_pytorch_with_dataset(model, config, dataset_root, dtype, num_samples):
    """Test with real VN3K dataset."""
    import torch
    import torch.nn.functional as F

    from msiglip.lightning_data import TBPSDataModule

    torch_dtype = torch.float16 if dtype == "fp16" else torch.float32

    config.dataset_root_dir = dataset_root
    dm = TBPSDataModule(config)
    dm.setup()

    test_loader = dm.test_dataloader()
    img_loader = test_loader.iterables["img"]
    txt_loader = test_loader.iterables["txt"]

    # Test a few image batches
    print("\n--- Image Encoder (real data) ---")
    img_latencies = []
    for i, batch in enumerate(img_loader):
        if i >= num_samples + 1:
            break
        images = batch["images"]
        if dtype == "fp16":
            images = images.half()
        t0 = time.time()
        with torch.no_grad():
            img_feat = model.encode_image(images)
        elapsed = time.time() - t0
        if i > 0:
            img_latencies.append(elapsed)
            print(f"  Batch {i}: {images.shape[0]} images in {elapsed:.3f}s ({images.shape[0]/elapsed:.1f} img/s)")
        else:
            print(f"  Warmup: {elapsed:.3f}s")

    # Test a few text batches
    print("\n--- Text Encoder (real data) ---")
    txt_latencies = []
    for i, batch in enumerate(txt_loader):
        if i >= num_samples + 1:
            break
        caption_input = {
            "input_ids": batch["caption_input_ids"],
            "attention_mask": batch["caption_attention_mask"],
        }
        t0 = time.time()
        with torch.no_grad():
            txt_feat = model.encode_text(caption_input)
        elapsed = time.time() - t0
        if i > 0:
            txt_latencies.append(elapsed)
            print(f"  Batch {i}: {batch['caption_input_ids'].shape[0]} texts in {elapsed:.3f}s")
        else:
            print(f"  Warmup: {elapsed:.3f}s")

    mem_final = get_memory_mb()
    print(f"\n--- Final Memory: {mem_final:.0f}MB ---")
    print("\nREAL DATA TEST PASSED.")


def test_onnx(model_dir: str, dataset_root: str = None, num_samples: int = 5):
    """Test inference with ONNX Runtime."""
    import onnxruntime as ort
    import yaml

    print(f"\n{'='*60}")
    print("ONNX RUNTIME INFERENCE TEST")
    print(f"{'='*60}")

    mem_before = get_memory_mb()

    # Load config
    with open(os.path.join(model_dir, "config.yaml")) as f:
        config = yaml.safe_load(f)

    img_size = config["backbone"]["vision_config"]["image_size"]
    if isinstance(img_size, (list, tuple)):
        h, w = img_size
    else:
        h = w = img_size
    max_text_len = config["tokenizer"]["model_max_length"]

    # Available providers
    providers = ort.get_available_providers()
    print(f"ONNX Runtime version: {ort.__version__}")
    print(f"Available providers: {providers}")

    # Prefer CPU for ARM
    session_opts = ort.SessionOptions()
    session_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_opts.intra_op_num_threads = 4  # Use performance cores

    # --- Vision encoder ---
    vision_path = os.path.join(model_dir, "vision_onnx", "vision_encoder.onnx")
    print(f"\nLoading vision encoder: {vision_path}")
    vision_session = ort.InferenceSession(vision_path, session_opts, providers=["CPUExecutionProvider"])

    mem_after_vision = get_memory_mb()
    print(f"Memory after vision load: {mem_after_vision:.0f}MB (delta: {mem_after_vision-mem_before:.0f}MB)")

    dummy_image = np.random.randn(1, 3, h, w).astype(np.float32)
    latencies = []
    for i in range(num_samples + 1):
        t0 = time.time()
        result = vision_session.run(None, {"image": dummy_image})
        elapsed = time.time() - t0
        if i > 0:
            latencies.append(elapsed)
        if i == 0:
            print(f"  Warmup: {elapsed:.3f}s")

    img_feat = result[0]
    print(f"  Output shape: {img_feat.shape}")
    print(f"  Latency: {np.mean(latencies):.3f}s +/- {np.std(latencies):.3f}s")

    # --- Text encoder ---
    text_path = os.path.join(model_dir, "text_onnx", "text_encoder.onnx")
    print(f"\nLoading text encoder: {text_path}")
    text_session = ort.InferenceSession(text_path, session_opts, providers=["CPUExecutionProvider"])

    mem_after_text = get_memory_mb()
    print(f"Memory after text load: {mem_after_text:.0f}MB (delta: {mem_after_text-mem_after_vision:.0f}MB)")

    dummy_ids = np.zeros((1, max_text_len), dtype=np.int64)
    dummy_mask = np.ones((1, max_text_len), dtype=np.int64)
    latencies = []
    for i in range(num_samples + 1):
        t0 = time.time()
        result = text_session.run(None, {"input_ids": dummy_ids, "attention_mask": dummy_mask})
        elapsed = time.time() - t0
        if i > 0:
            latencies.append(elapsed)
        if i == 0:
            print(f"  Warmup: {elapsed:.3f}s")

    txt_feat = result[0]
    print(f"  Output shape: {txt_feat.shape}")
    print(f"  Latency: {np.mean(latencies):.3f}s +/- {np.std(latencies):.3f}s")

    # Similarity
    img_norm = img_feat / np.linalg.norm(img_feat, axis=-1, keepdims=True)
    txt_norm = txt_feat / np.linalg.norm(txt_feat, axis=-1, keepdims=True)
    sim = float(img_norm @ txt_norm.T)
    print(f"\n  Image-Text similarity (random): {sim:.4f}")

    mem_final = get_memory_mb()
    print(f"\n--- Final Memory: {mem_final:.0f}MB ---")
    print("\nONNX SMOKE TEST PASSED.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR,
        help="Directory with exported model files",
    )
    parser.add_argument("--backend", choices=["pytorch", "onnx"], default="pytorch")
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--dataset-root", default=None, help="Path to VN3K dataset root (optional)")
    parser.add_argument("--num-samples", type=int, default=5)
    args = parser.parse_args()

    print(f"{'='*60}")
    print("mSigLIP INFERENCE TEST")
    print(f"{'='*60}")
    print(f"Model dir:    {args.model_dir}")
    print(f"Backend:      {args.backend}")
    print(f"Device:       CPU (ARM64)")
    print(f"Dataset root: {args.dataset_root or 'NONE (dummy data)'}")

    # System info
    import platform
    print(f"\nPlatform:     {platform.machine()}")
    print(f"Python:       {platform.python_version()}")

    try:
        import torch
        print(f"PyTorch:      {torch.__version__}")
    except ImportError:
        if args.backend == "pytorch":
            print("ERROR: PyTorch not installed. Use --backend onnx or install PyTorch.")
            return

    try:
        import onnxruntime as ort
        print(f"ONNX Runtime: {ort.__version__}")
    except ImportError:
        if args.backend == "onnx":
            print("ERROR: onnxruntime not installed. pip install onnxruntime")
            return

    if args.backend == "pytorch":
        test_pytorch(args.model_dir, args.dtype, args.dataset_root, args.num_samples)
    else:
        test_onnx(args.model_dir, args.dataset_root, args.num_samples)


if __name__ == "__main__":
    logger = TeeLogger(DEFAULT_LOG_DIR, "inference")
    main()
    logger.close()
