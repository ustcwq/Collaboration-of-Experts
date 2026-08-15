from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from .artifacts import innovation_code_manifest, manifest_sha256, sha256_file, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run innovation tests and bind the result to code/config hashes")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, action="append", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    command = [sys.executable, "-m", "unittest", "discover", "-s", "tests/innovation", "-v"]
    started = time.time()
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    combined_output = completed.stdout + completed.stderr
    receipt = {
        "command": command,
        "started_unix": started,
        "finished_unix": time.time(),
        "exit_code": completed.returncode,
        "test_output": combined_output,
        "test_count": combined_output.count(" ... ok"),
        "code_manifest_sha256": manifest_sha256(innovation_code_manifest()),
        "config_hashes": {str(path): sha256_file(path) for path in args.config},
    }
    write_json(args.output, receipt)
    print(combined_output, end="")
    if completed.returncode:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
