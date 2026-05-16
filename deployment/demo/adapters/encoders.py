"""Image and text encoder adapters for the modular demo."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from ..core.utils import (
    deterministic_embedding,
    ensure_dir,
    l2_normalize,
    read_float32_raw,
    write_msiglip_image_raw,
)


class FakeVisionEncoder:
    runtime_name = "fake-vision"

    def encode(self, crop: Any) -> list[float]:
        image = crop.convert("RGB").resize((32, 32))
        return deterministic_embedding(image.tobytes())


class FakeTextEncoder:
    runtime_name = "fake-text"

    def encode(self, text: str) -> list[float]:
        return deterministic_embedding(text.encode("utf-8"))


class OnnxVisionEncoder:
    runtime_name = "onnx-vision-cpu"

    def __init__(self, model_path: Path, image_size: int = 256):
        self.model_path = model_path.expanduser().resolve()
        self.image_size = image_size
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX vision model not found: {self.model_path}")
        try:
            import numpy as np
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("OnnxVisionEncoder requires numpy and onnxruntime") from exc
        self._np = np
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            str(self.model_path),
            options,
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name

    def _preprocess(self, crop: Any):
        img = crop.convert("RGB").resize((self.image_size, self.image_size))
        arr = self._np.asarray(img).astype("float32") / 255.0
        arr = (arr - 0.5) / 0.5
        arr = arr.transpose(2, 0, 1)[None, ...]
        return arr

    def encode(self, crop: Any) -> list[float]:
        output = self._session.run(None, {self._input_name: self._preprocess(crop)})[0]
        values = output.reshape(-1).astype("float32").tolist()
        return l2_normalize(values)


class QnnVisionEncoder:
    """Run a QNN context binary with qnn-net-run on RB3.

    This adapter is the real deployment path for the current `vision_encoder.bin`
    artifact. It writes qnn-net-run inputs to a small runtime directory, invokes
    QAIRT, reads Result_0/output_0.raw, validates the vector, and L2-normalizes it.
    """

    runtime_name = "qnn-vision-htp"

    def __init__(
        self,
        vision_bin: Path,
        htp_config: Path,
        qairt: Path = Path("/opt/qcom/qairt/2.45.40.260406"),
        qnn_bin: Path | None = None,
        qnn_lib: Path | None = None,
        runtime_dir: Path = Path("deployment/demo_runtime/qnn"),
        image_size: int = 256,
        output_name: str = "output_0.raw",
        keep_artifacts: bool = False,
        adsp_library_path: Path | None = None,
    ):
        self.vision_bin = vision_bin.expanduser().resolve()
        self.htp_config = htp_config.expanduser().resolve()
        self.qairt = qairt.expanduser().resolve()
        self.qnn_bin = (qnn_bin or self.qairt / "bin/aarch64-ubuntu-gcc9.4").expanduser().resolve()
        self.qnn_lib = (qnn_lib or self.qairt / "lib/aarch64-ubuntu-gcc9.4").expanduser().resolve()
        self.runtime_dir = runtime_dir.expanduser().resolve()
        self.image_size = image_size
        self.output_name = output_name
        self.keep_artifacts = keep_artifacts
        self.adsp_library_path = adsp_library_path.expanduser().resolve() if adsp_library_path else None
        self._validate_paths()

    def _validate_paths(self) -> None:
        qnn_net_run = self.qnn_bin / "qnn-net-run"
        backend = self.qnn_lib / "libQnnHtp.so"
        for path in [self.vision_bin, self.htp_config, qnn_net_run, backend]:
            if not path.exists():
                raise FileNotFoundError(f"QNN runtime path not found: {path}")

    def _run_dir(self) -> Path:
        ensure_dir(self.runtime_dir)
        if self.keep_artifacts:
            return Path(tempfile.mkdtemp(prefix="qnn_", dir=self.runtime_dir))
        return Path(tempfile.mkdtemp(prefix="qnn_", dir=self.runtime_dir))

    def _env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = f"{self.qnn_lib}:{env.get('LD_LIBRARY_PATH', '')}"
        if self.adsp_library_path:
            env["ADSP_LIBRARY_PATH"] = str(self.adsp_library_path)
        return env

    def encode(self, crop: Any) -> list[float]:
        run_dir = self._run_dir()
        raw_dir = ensure_dir(run_dir / "raw")
        output_dir = run_dir / "qnn_results"
        raw_path = raw_dir / "image.raw"
        input_list = run_dir / "input_list.txt"
        write_msiglip_image_raw(crop, raw_path, image_size=self.image_size)
        input_list.write_text(f"image:={raw_path}\n", encoding="utf-8")

        command = [
            str(self.qnn_bin / "qnn-net-run"),
            "--backend",
            str(self.qnn_lib / "libQnnHtp.so"),
            "--retrieve_context",
            str(self.vision_bin),
            "--config_file",
            str(self.htp_config),
            "--input_list",
            str(input_list),
            "--output_dir",
            str(output_dir),
            "--profiling_level",
            "basic",
            "--perf_profile",
            "high_performance",
            "--log_level",
            "info",
        ]
        try:
            proc = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                env=self._env(),
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    "qnn-net-run failed with code "
                    f"{proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
                )
            output_path = output_dir / "Result_0" / self.output_name
            values = read_float32_raw(output_path)
            return l2_normalize(values)
        finally:
            if not self.keep_artifacts:
                shutil.rmtree(run_dir, ignore_errors=True)
