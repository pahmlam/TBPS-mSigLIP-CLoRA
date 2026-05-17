from pathlib import Path
import sys

import fire


SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from msiglip.evaluate import run_test


if __name__ == "__main__":
    fire.Fire(run_test)
