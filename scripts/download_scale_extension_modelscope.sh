#!/usr/bin/env bash
set -euo pipefail

cd /home/sm5/ys/FCS

export MODELSCOPE_DOMAIN=modelscope.cn
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

CACHE_DIR=/home/sm5/ys/FCS/modelscope_cache
LOG_DIR=/home/sm5/ys/FCS/benchcoe_assets/logs/scale_extension_20260731
MAX_WORKERS=${MAX_WORKERS:-16}
mkdir -p "$CACHE_DIR" "$LOG_DIR"

download_model() {
  local model_id="$1"
  local destination="$2"
  local marker="$destination/.benchcoe_modelscope_complete.json"
  if [[ -f "$marker" ]]; then
    printf 'skip completed %s -> %s\n' "$model_id" "$destination"
    return
  fi
  mkdir -p "$destination"
  printf 'download %s -> %s\n' "$model_id" "$destination"
  modelscope download \
    --model "$model_id" \
    --revision master \
    --cache_dir "$CACHE_DIR" \
    --local_dir "$destination" \
    --max-workers "$MAX_WORKERS" \
    2>&1 | tee "$LOG_DIR/$(basename "$destination").log"
  printf '{"modelscope_id":"%s","revision":"master","completed_at":"2026-07-31","local_dir":"%s"}\n' \
    "$model_id" "$destination" > "$marker"
}

# Text-only LLMs -> models/
download_model Qwen/Qwen3-1.7B /home/sm5/ys/FCS/models/Qwen3-1.7B
download_model Qwen/Qwen2.5-3B-Instruct /home/sm5/ys/FCS/models/Qwen2.5-3B-Instruct
download_model Qwen/Qwen3-14B /home/sm5/ys/FCS/models/Qwen3-14B
download_model Qwen/Qwen2.5-14B-Instruct /home/sm5/ys/FCS/models/Qwen2.5-14B-Instruct

# Vision-language models -> models_v/
download_model OpenGVLab/InternVL3_5-8B /home/sm5/ys/FCS/models_v/InternVL3_5-8B
download_model Qwen/Qwen2.5-VL-7B-Instruct /home/sm5/ys/FCS/models_v/Qwen2.5-VL-7B-Instruct
download_model OpenGVLab/InternVL3_5-14B /home/sm5/ys/FCS/models_v/InternVL3_5-14B
