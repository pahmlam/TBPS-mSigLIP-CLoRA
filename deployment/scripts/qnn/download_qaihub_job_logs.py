#!/usr/bin/env python3
"""Download curated Qualcomm AI Hub job logs into semantic evidence paths."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


EVIDENCE_PATTERNS = (
    "floating-point type",
    "incorrect Value 68",
    "expected >= 73",
    "QnnBackend_validateOpConfig",
    "Failed to validate op",
    "Conversion to context binary failed",
    "JobStatus",
    "FAILED",
    "SUCCESS",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_jobs(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        data = data.get("jobs", [])
    if not isinstance(data, list):
        raise ValueError(f"Expected a list or {{'jobs': [...]}} in {path}")
    jobs: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict) or not item.get("job_id"):
            raise ValueError(f"Invalid job entry in {path}: {item!r}")
        if not item.get("target_relpath"):
            raise ValueError(f"Missing target_relpath for job {item['job_id']}")
        jobs.append(item)
    return jobs


def _extract_evidence(log_path: Path) -> list[dict[str, Any]]:
    if not log_path.exists():
        return []
    evidence = []
    for line_no, line in enumerate(log_path.read_text(errors="replace").splitlines(), 1):
        if any(pattern in line for pattern in EVIDENCE_PATTERNS):
            evidence.append({"line": line_no, "text": line})
    return evidence[:20]


def _job_url(job: Any, job_id: str) -> str:
    url = getattr(job, "url", None)
    return str(url) if url else f"https://workbench.aihub.qualcomm.com/jobs/{job_id}/"


def _download_one(job_spec: dict[str, Any], output_dir: Path, overwrite: bool) -> dict[str, Any]:
    job_id = str(job_spec["job_id"])
    target = output_dir / str(job_spec["target_relpath"])
    target.parent.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        **job_spec,
        "target_path": str(target.relative_to(output_dir)),
        "ok": False,
    }

    if target.exists() and not overwrite:
        record.update(
            {
                "ok": True,
                "download_status": "already_present",
                "downloaded_logs": [str(target.relative_to(output_dir))],
                "evidence_lines": _extract_evidence(target),
            }
        )
        return record

    try:
        import qai_hub as hub

        job = hub.get_job(job_id)
        record["url"] = _job_url(job, job_id)
        try:
            record["status"] = str(job.get_status())
        except Exception as exc:  # noqa: BLE001 - preserve AI Hub client errors in manifest.
            record["status_error"] = repr(exc)

        tmp_dir = output_dir / ".download_tmp" / job_id
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        downloaded = job.download_job_logs(str(tmp_dir))
        record["download_result"] = str(downloaded) if downloaded else ""

        logs = sorted(tmp_dir.rglob("*.log"))
        if not logs:
            files = [p for p in sorted(tmp_dir.rglob("*")) if p.is_file()]
            if files:
                logs = files
        if not logs:
            raise RuntimeError("AI Hub did not produce a log file")

        shutil.move(str(logs[0]), target)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        record.update(
            {
                "ok": True,
                "download_status": "downloaded",
                "downloaded_logs": [str(target.relative_to(output_dir))],
                "evidence_lines": _extract_evidence(target),
            }
        )
    except Exception as exc:  # noqa: BLE001 - manifest must record unavailable jobs.
        record.update({"download_status": "unavailable", "error": repr(exc)})
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs-json", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_repo_root() / "artifacts/deployment/logs",
        help="Root of the canonical deployment logs archive.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    jobs = _load_jobs(args.jobs_json)
    output_dir = args.output_dir.expanduser().resolve()

    if args.dry_run:
        for job in jobs:
            print(f"{job['job_id']} -> {output_dir / job['target_relpath']}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for job in jobs:
        record = _download_one(job, output_dir, args.overwrite)
        records.append(record)
        print(f"{record['job_id']}: {record.get('download_status')} -> {record.get('target_path')}")

    manifest_path = output_dir / "aihub/curated_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
