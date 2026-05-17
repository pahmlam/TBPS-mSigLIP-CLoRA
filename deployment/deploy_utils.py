"""Shared utilities for deployment scripts."""

import os
import sys
from datetime import datetime


class TeeLogger:
    """Duplicates all stdout/stderr to a timestamped log file.

    Usage:
        logger = TeeLogger("artifacts/deployment/logs", "export")
        # ... all print() output is now also written to the log file
        logger.close()  # restore original stdout/stderr
    """

    def __init__(self, log_dir: str, prefix: str):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(log_dir, f"{prefix}_{timestamp}.log")
        self.file = open(self.log_path, "w")
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = self
        sys.stderr = _TeeStream(self._stderr, self.file)
        print(f"Logging to: {self.log_path}")

    def write(self, data):
        self._stdout.write(data)
        self.file.write(data)
        self.file.flush()

    def flush(self):
        self._stdout.flush()
        self.file.flush()

    def close(self):
        print(f"\nLog saved: {self.log_path}")
        sys.stdout = self._stdout
        sys.stderr = self._stderr
        self.file.close()


class _TeeStream:
    """Tee for stderr."""

    def __init__(self, original, log_file):
        self.original = original
        self.file = log_file

    def write(self, data):
        self.original.write(data)
        self.file.write(data)
        self.file.flush()

    def flush(self):
        self.original.flush()
        self.file.flush()
