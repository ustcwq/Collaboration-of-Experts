#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench_coe.assets.paths import AssetPaths
from bench_coe.assets.validation import verify_lock


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Verify immutable Bench-CoE locks and optional local model loading")
    value.add_argument("--asset-root")
    value.add_argument("--modelscope-cache")
    value.add_argument("--smoke-load-model", type=Path, help="Local model path for offline tokenizer/processor load check")
    value.add_argument("--trust-remote-code", action="store_true")
    return value


def offline_load(path: Path, trust_remote_code: bool) -> dict:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoProcessor, AutoTokenizer

    started = time.monotonic()
    loaded = []
    AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=trust_remote_code)
    loaded.append("config")
    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=trust_remote_code)
        loaded.append("tokenizer")
    except Exception:
        AutoProcessor.from_pretrained(path, local_files_only=True, trust_remote_code=trust_remote_code)
        loaded.append("processor")
    generation = []
    try:
        model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, trust_remote_code=trust_remote_code)
        loaded.append("causal_lm_weights")
        if tokenizer is not None:
            for prompt in ("Return the word ready.", "What is two plus three?"):
                inputs = tokenizer(prompt, return_tensors="pt")
                with torch.inference_mode():
                    output = model.generate(**inputs, max_new_tokens=8, do_sample=False)
                generation.append({"prompt": prompt, "output": tokenizer.decode(output[0], skip_special_tokens=True)})
    except Exception:
        AutoModel.from_pretrained(path, local_files_only=True, trust_remote_code=trust_remote_code)
        loaded.append("model_weights")
    peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    return {"status": "loaded_offline", "path": str(path.resolve()), "components": loaded, "elapsed_seconds": time.monotonic() - started, "peak_cuda_bytes": peak, "generation": generation, "enable_thinking": False}


def main() -> int:
    args = parser().parse_args()
    paths = AssetPaths.from_env(args.asset_root, args.modelscope_cache)
    manifest = paths.directories()["manifests"]
    results = {
        "asset_lock": verify_lock(manifest / "asset_lock.json", manifest / "asset_lock.sha256"),
        "protocol_lock": verify_lock(manifest / "protocol_lock.yaml", manifest / "protocol_lock.sha256"),
    }
    if args.smoke_load_model:
        results["offline_load"] = offline_load(args.smoke_load_model, args.trust_remote_code)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0 if all(item.get("status") in {"valid", "loaded_offline"} for item in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
