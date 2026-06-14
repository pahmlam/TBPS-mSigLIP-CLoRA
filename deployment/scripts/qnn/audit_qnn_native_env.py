#!/usr/bin/env python3
"""Audit local/server QNN or QAIRT native toolchain availability."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


TOOL_NAMES = (
    "qairt-converter",
    "qairt-quantizer",
    "qairt-model-lib-generator",
    "qairt-context-binary-generator",
    "qairt-net-run",
    "qnn-onnx-converter",
    "qnn-model-lib-generator",
    "qnn-context-binary-generator",
    "qnn-net-run",
    "qnn-throughput-net-run",
    "qnn-profile-viewer",
    "qnn-platform-validator",
)

ENV_KEYS = (
    "QNN_SDK_ROOT",
    "QAIRT_ROOT",
    "QNN_ROOT",
    "SNPE_ROOT",
    "PATH",
    "LD_LIBRARY_PATH",
    "DYLD_LIBRARY_PATH",
    "PYTHONPATH",
)

COMMON_ROOTS = (
    "/opt/qcom/qairt",
    "/opt/qcom/aistack",
    "/opt/qcom/qnn",
    "/opt/qcom/snpe",
    "/opt/qcom",
    "/usr/local/qcom",
    "/home/ubuntu/qairt",
    "/home/ubuntu/qnn",
)


def _existing_dirs(values: list[str]) -> list[Path]:
    dirs: list[Path] = []
    for value in values:
        if not value:
            continue
        path = Path(value).expanduser()
        if path.is_dir():
            dirs.append(path.resolve())
    return dirs


def _candidate_bin_dirs(extra_roots: list[Path]) -> list[Path]:
    roots = []
    for key in ("QNN_SDK_ROOT", "QAIRT_ROOT", "QNN_ROOT", "SNPE_ROOT"):
        value = os.environ.get(key)
        if value:
            roots.append(Path(value).expanduser())
    roots.extend(extra_roots)
    roots.extend(Path(root) for root in COMMON_ROOTS)

    bin_dirs: list[Path] = []
    suffixes = (
        "bin",
        "bin/x86_64-linux-clang",
        "bin/aarch64-ubuntu-gcc9.4",
        "lib/x86_64-linux-clang",
    )
    for root in roots:
        if not root.exists():
            continue
        candidates = [root / suffix for suffix in suffixes]
        if root.name in {"bin", "x86_64-linux-clang"}:
            candidates.append(root)
        for child in root.glob("*/bin"):
            candidates.append(child)
        for child in root.glob("*/bin/x86_64-linux-clang"):
            candidates.append(child)
        for directory in candidates:
            if directory.is_dir():
                resolved = directory.resolve()
                if resolved not in bin_dirs:
                    bin_dirs.append(resolved)
    return bin_dirs


def _find_tool(name: str, bin_dirs: list[Path]) -> Path | None:
    found = shutil.which(name)
    if found:
        return Path(found).resolve()
    for directory in bin_dirs:
        candidate = directory / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    return None


def _run_probe(path: Path, args: tuple[str, ...], timeout: float) -> dict[str, object]:
    try:
        proc = subprocess.run(
            [str(path), *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except Exception as exc:  # pragma: no cover - environment-specific
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout_first_lines": proc.stdout.splitlines()[:12],
        "stderr_first_lines": proc.stderr.splitlines()[:12],
    }


def _tool_report(name: str, bin_dirs: list[Path], timeout: float) -> dict[str, object]:
    path = _find_tool(name, bin_dirs)
    if path is None:
        return {"found": False}

    probes: dict[str, object] = {}
    for label, args in (
        ("version", ("--version",)),
        ("help", ("--help",)),
    ):
        probes[label] = _run_probe(path, args, timeout)
    return {
        "found": True,
        "path": str(path),
        "probes": probes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit QNN/QAIRT native toolchain availability."
    )
    parser.add_argument(
        "--root",
        action="append",
        default=[],
        help="Additional QNN/QAIRT root directory to search. Repeatable.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=Path("artifacts/deployment/runtime/qnn_native/env_audit.json"),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Seconds per tool probe.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extra_roots = _existing_dirs(args.root)
    bin_dirs = _candidate_bin_dirs(extra_roots)

    result = {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "python": sys.version,
        },
        "environment": {key: os.environ.get(key, "") for key in ENV_KEYS},
        "searched_bin_dirs": [str(path) for path in bin_dirs],
        "tools": {
            name: _tool_report(name, bin_dirs, args.timeout) for name in TOOL_NAMES
        },
    }

    args.json.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    found = [name for name, info in result["tools"].items() if info["found"]]
    missing = [name for name in TOOL_NAMES if name not in found]
    print(json.dumps({
        "output_json": str(args.json),
        "found_tools": found,
        "missing_tools": missing,
    }, indent=2))


if __name__ == "__main__":
    main()
