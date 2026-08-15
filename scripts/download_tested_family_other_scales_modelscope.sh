#!/usr/bin/env bash
set -euo pipefail

cd /home/sm5/ys/FCS

export MODELSCOPE_DOMAIN=modelscope.cn
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

CACHE_DIR=/home/sm5/ys/FCS/modelscope_cache
LOG_DIR=/home/sm5/ys/FCS/benchcoe_assets/logs/family_scale_expansion_20260731
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
  printf '{"modelscope_id":"%s","revision":"master","completed_at":"%s","local_dir":"%s"}\n' \
    "$model_id" "$(date '+%F %T')" "$destination" > "$marker"
}

# Other parameter scales from language-model families already represented in the report.
download_model ibm-granite/granite-3.3-2b-instruct /home/sm5/ys/FCS/models/granite-3.3-2b-instruct
download_model Shanghai_AI_Laboratory/internlm2_5-1_8b-chat /home/sm5/ys/FCS/models/internlm2_5-1_8b-chat
download_model LLM-Research/gemma-2-2b-it /home/sm5/ys/FCS/models/gemma-2-2b-it
download_model mistralai/Ministral-3-3B-Instruct-2512 /home/sm5/ys/FCS/models/Ministral-3-3B-Instruct-2512
download_model LLM-Research/Llama-3.2-3B-Instruct /home/sm5/ys/FCS/models/Llama-3.2-3B-Instruct
download_model baichuan-inc/Baichuan2-13B-Chat /home/sm5/ys/FCS/models/Baichuan2-13B-Chat
download_model LLM-Research/Mistral-Nemo-Instruct-2407 /home/sm5/ys/FCS/models/Mistral-Nemo-Instruct-2407

# Other parameter scales from vision-language families represented in the report.
download_model ZhipuAI/GLM-4.1V-9B-Thinking /home/sm5/ys/FCS/models_v/GLM-4.1V-9B-Thinking
download_model LLM-Research/gemma-3-12b-it /home/sm5/ys/FCS/models_v/gemma-3-12b-it
