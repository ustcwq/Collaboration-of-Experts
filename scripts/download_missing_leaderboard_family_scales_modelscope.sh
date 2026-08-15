#!/usr/bin/env bash
set -uo pipefail

cd /home/sm5/ys/FCS

export MODELSCOPE_DOMAIN=modelscope.cn
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

CACHE_DIR=/home/sm5/ys/FCS/modelscope_cache
LOG_DIR=/home/sm5/ys/FCS/benchcoe_assets/logs/missing_leaderboard_family_scales_20260801
MAX_WORKERS=${MAX_WORKERS:-16}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-3}
mkdir -p "$CACHE_DIR" "$LOG_DIR"

download_model() {
  local model_id="$1"
  local destination="$2"
  local marker="$destination/.benchcoe_modelscope_complete.json"
  local failure_marker="$destination/.benchcoe_modelscope_failed.json"
  local model_log="$LOG_DIR/$(basename "$destination").log"

  if [[ -f "$marker" ]]; then
    printf 'skip completed %s -> %s\n' "$model_id" "$destination"
    return 0
  fi

  mkdir -p "$destination"
  local attempt
  for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    printf 'download attempt %s/%s %s -> %s\n' \
      "$attempt" "$MAX_ATTEMPTS" "$model_id" "$destination"
    if modelscope download \
      --model "$model_id" \
      --revision master \
      --cache_dir "$CACHE_DIR" \
      --local_dir "$destination" \
      --max-workers "$MAX_WORKERS" \
      2>&1 | tee -a "$model_log"; then
      rm -f "$failure_marker"
      printf '{"modelscope_id":"%s","revision":"master","completed_at":"%s","local_dir":"%s"}\n' \
        "$model_id" "$(date '+%F %T')" "$destination" > "$marker"
      return 0
    fi
    sleep 20
  done

  printf '{"modelscope_id":"%s","failed_at":"%s","attempts":%s,"local_dir":"%s"}\n' \
    "$model_id" "$(date '+%F %T')" "$MAX_ATTEMPTS" "$destination" > "$failure_marker"
  printf 'failed after %s attempts: %s\n' "$MAX_ATTEMPTS" "$model_id" >&2
  return 0
}

# Language-model families missing strict or useful scale comparisons.
download_model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  /home/sm5/ys/FCS/models/DeepSeek-R1-Distill-Qwen-1.5B
download_model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
  /home/sm5/ys/FCS/models/DeepSeek-R1-Distill-Qwen-14B
download_model 01ai/Yi-1.5-6B-Chat \
  /home/sm5/ys/FCS/models/Yi-1.5-6B-Chat

# Matching larger Thinking variant for the tested Qwen3-VL-2B-Thinking model.
download_model Qwen/Qwen3-VL-8B-Thinking \
  /home/sm5/ys/FCS/models_v/Qwen3-VL-8B-Thinking
