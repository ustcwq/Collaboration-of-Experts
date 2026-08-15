from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path


ROOT = Path("/home/sm5/ys/Project/ITNorm/O/O")
CHECKPOINT_RE = re.compile(r"^(?:f?ckpt_epoch_|epoch_)(\d+)\.(?:pth|pt|ckpt)$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("/home/sm5/ys/FCS/benchcoe_assets/manifests/itnorm_checkpoint_cleanup_20260731.json"),
    )
    return parser.parse_args()


def collect() -> list[dict[str, object]]:
    groups: dict[Path, list[tuple[int, Path]]] = {}
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        match = CHECKPOINT_RE.match(path.name)
        if match:
            groups.setdefault(path.parent, []).append((int(match.group(1)), path))

    selected: list[dict[str, object]] = []
    for parent, checkpoints in sorted(groups.items()):
        checkpoints.sort(key=lambda item: (item[0], item[1].stat().st_mtime, item[1].name))
        delete_count = int(len(checkpoints) * 0.30)
        for epoch, path in checkpoints[:delete_count]:
            stat = path.stat()
            selected.append(
                {
                    "path": str(path),
                    "parent": str(parent),
                    "epoch": epoch,
                    "size": stat.st_size,
                    "mtime": stat.st_mtime,
                    "group_checkpoint_count": len(checkpoints),
                    "group_delete_count": delete_count,
                }
            )
    return selected


def main() -> int:
    args = parse_args()
    selected = collect()
    payload = {
        "root": str(ROOT),
        "policy": "delete earliest 30 percent of epoch checkpoints in each training-run directory",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "execute": args.execute,
        "file_count": len(selected),
        "bytes": sum(int(item["size"]) for item in selected),
        "files": selected,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.execute:
        for item in selected:
            path = Path(str(item["path"]))
            path.relative_to(ROOT)
            path.unlink(missing_ok=True)
    print(json.dumps({key: value for key, value in payload.items() if key != "files"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
