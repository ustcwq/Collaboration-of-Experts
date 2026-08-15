from __future__ import annotations

import argparse
from pathlib import Path

from .artifacts import validate_test_receipt
from .locked_protocol import load_protocol
from .locked_protocol import prepare_locked_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze and materialize label-free MuSR observables")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_protocol(args.config)
    validate_test_receipt(Path(str(config["test_receipt"])), args.config)
    preregistration = prepare_locked_run(args.config)
    print(f"Frozen preregistration: {preregistration}")


if __name__ == "__main__":
    main()
