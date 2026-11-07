#!/usr/bin/env bash
# Open-weight half of the main study, run from your laptop because the cloud workspace cannot reach Modal.
#
#     bash scripts/run_openweight.sh
#
# It pulls the repo, deploys Qwen3-32B and gpt-oss-20b on Modal, has both models write board posts, has both audit
# every board, and stops the servers. It is safe to rerun: finished requests are skipped. Expect about an hour and
# roughly $5 to $10 of Modal credit. Keep the laptop plugged in; caffeinate keeps it awake while the script runs.
set -euo pipefail
cd "$(dirname "$0")/.."
caffeinate -dims -w $$ >/dev/null 2>&1 &

git pull --ff-only
uv sync --extra modal

# 1. A server key, kept in the Modal secret "vllm-key" and in compute.env.
if ! grep -q '^VLLM_API_KEY=' compute.env; then
  KEY=$(openssl rand -hex 24)
  uv run modal secret create vllm-key VLLM_API_KEY="$KEY" --force
  printf '\nVLLM_API_KEY=%s\n' "$KEY" >> compute.env
fi

# 2. Deploy both servers and record their URLs in compute.env.
for app in qwen gptoss; do
  url=$(uv run modal deploy "modal_jobs/serve_${app}.py" 2>&1 | tee /dev/stderr | grep -Eo 'https://[A-Za-z0-9.-]+\.modal\.run' | tail -1)
  var="VLLM_BASE_URL_$(echo "$app" | tr '[:lower:]' '[:upper:]')"
  if [ -z "$url" ]; then echo "Could not read the $app URL from modal deploy. Add $var to compute.env by hand." >&2; exit 1; fi
  grep -v "^${var}=" compute.env > compute.env.tmp && mv compute.env.tmp compute.env
  printf '%s=%s/v1\n' "$var" "$url" >> compute.env
done

# 3. Warm up. The first call downloads the weights and can take 10 minutes, so keep trying for 20.
KEY=$(grep '^VLLM_API_KEY=' compute.env | tail -1 | cut -d= -f2-)
for var in VLLM_BASE_URL_QWEN VLLM_BASE_URL_GPTOSS; do
  base=$(grep "^${var}=" compute.env | tail -1 | cut -d= -f2-)
  for i in $(seq 1 40); do
    if curl -sf -m 60 "$base/models" -H "Authorization: Bearer $KEY" >/dev/null; then echo "$var is up"; break; fi
    echo "waiting for $var ($i/40)"; sleep 30
  done
done

# 4. Both open models write the routine posts of all 40 boards.
uv run python scripts/author_boards.py --data data/main --authors vllm/qwen,vllm/gptoss --workers 16

# 5. Both open models audit every board from all four authors, in parallel, and then run the two exploratory
#    follow-ups (E1 alibis and E2 logged effects) on the Haiku and Gemini boards.
uv run python scripts/study.py requests --investigators vllm/qwen,vllm/gptoss
uv run python scripts/study.py --config configs/explore_alibi.json requests --investigators vllm/qwen,vllm/gptoss
uv run python scripts/study.py --config configs/explore_logging.json requests --investigators vllm/qwen,vllm/gptoss
pids=()
for m in qwen gptoss; do
  ( uv run python -m swarm.batch "data/main/req/vllm_${m}.jsonl" "data/main/out/vllm_${m}.jsonl" --concurrency vllm=48 &&
    uv run python -m swarm.batch "data/main/req/vllm_${m}_e1.jsonl" "data/main/out/vllm_${m}_e1.jsonl" --concurrency vllm=48 &&
    uv run python -m swarm.batch "data/main/req/vllm_${m}_e2.jsonl" "data/main/out/vllm_${m}_e2.jsonl" --concurrency vllm=48 ) &
  pids+=($!)
done
wait "${pids[@]}"

# 6. Stop the servers so they bill nothing more.
uv run modal app stop -y swarm-inv-qwen || true
uv run modal app stop -y swarm-inv-gptoss || true
echo "Done. Tell Claude the open-weight run finished."
