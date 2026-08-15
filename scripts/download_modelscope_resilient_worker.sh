#!/usr/bin/env bash
set -uo pipefail

if [[ "$#" -ne 3 ]]; then
  echo "usage: $0 MODEL_ID LOCAL_DIR LOG_FILE" >&2
  exit 2
fi

MODEL_ID="$1"
LOCAL_DIR="$2"
LOG_FILE="$3"
MARKER="$LOCAL_DIR/.benchcoe_modelscope_complete.json"
MAX_WORKERS=${MAX_WORKERS:-4}
RETRY_DELAY=${RETRY_DELAY:-60}
CACHE_DIR=${CACHE_DIR:-/home/sm5/ys/FCS/modelscope_cache}

cd /home/sm5/ys/FCS
export MODELSCOPE_DOMAIN=modelscope.cn
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
mkdir -p "$LOCAL_DIR" "$(dirname "$LOG_FILE")" "$CACHE_DIR"

log() {
  printf '%s %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG_FILE"
}

while [[ ! -f "$MARKER" ]]; do
  if pgrep -af "[m]odelscope download --model $MODEL_ID" >/dev/null; then
    log "$MODEL_ID already downloading in another queue; waiting"
    sleep "$RETRY_DELAY"
    continue
  fi

  log "starting or resuming $MODEL_ID"
  if modelscope download \
    --model "$MODEL_ID" \
    --revision master \
    --cache_dir "$CACHE_DIR" \
    --local_dir "$LOCAL_DIR" \
    --max-workers "$MAX_WORKERS" >> "$LOG_FILE" 2>&1; then
    printf '{"modelscope_id":"%s","revision":"master","completed_at":"%s","local_dir":"%s"}\n' \
      "$MODEL_ID" "$(date '+%F %T')" "$LOCAL_DIR" > "$MARKER"
    log "$MODEL_ID completed"
    exit 0
  fi

  log "$MODEL_ID failed or disconnected; retrying after ${RETRY_DELAY}s"
  sleep "$RETRY_DELAY"
done

log "$MODEL_ID already completed"
