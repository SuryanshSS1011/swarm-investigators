#!/usr/bin/env bash
# E3 extension (docs/explore_reasoning.md): Qwen3-32B with thinking on and gpt-oss-20b at reasoning effort "high"
# audit the 160 E3 boards. Run from your laptop, because the cloud workspace cannot reach Modal:
#
#     bash scripts/run_openweight_e3.sh
#
# It pulls the repo, redeploys both servers with a 32,768-token context, runs the audits and stops the servers. It is
# safe to rerun, since finished requests are skipped. Expect 30 to 45 minutes and a few dollars of Modal credit.
set -euo pipefail
cd "$(dirname "$0")/.."
caffeinate -dims -w $$ >/dev/null 2>&1 &

git pull --ff-only
uv sync --extra modal

if ! grep -q '^VLLM_API_KEY=' compute.env; then
  KEY=$(openssl rand -hex 24)
  uv run modal secret create vllm-key VLLM_API_KEY="$KEY" --force
  printf '\nVLLM_API_KEY=%s\n' "$KEY" >> compute.env
fi

# Redeploy both servers so they pick up the longer context, and record their URLs in compute.env.
for app in qwen gptoss; do
  url=$(uv run modal deploy "modal_jobs/serve_${app}.py" 2>&1 | tee /dev/stderr | grep -Eo 'https://[A-Za-z0-9.-]+\.modal\.run' | tail -1)
  var="VLLM_BASE_URL_$(echo "$app" | tr '[:lower:]' '[:upper:]')"
  if [ -z "$url" ]; then echo "Could not read the $app URL from modal deploy. Add $var to compute.env by hand." >&2; exit 1; fi
  grep -v "^${var}=" compute.env > compute.env.tmp && mv compute.env.tmp compute.env
  printf '%s=%s/v1\n' "$var" "$url" >> compute.env
done

KEY=$(grep '^VLLM_API_KEY=' compute.env | tail -1 | cut -d= -f2-)
for var in VLLM_BASE_URL_QWEN VLLM_BASE_URL_GPTOSS; do
  base=$(grep "^${var}=" compute.env | tail -1 | cut -d= -f2-)
  for i in $(seq 1 40); do
    if curl -sf -m 60 "$base/models" -H "Authorization: Bearer $KEY" >/dev/null; then echo "$var is up"; break; fi
    echo "waiting for $var ($i/40)"; sleep 30
  done
done

uv run python scripts/study.py --config configs/explore_reasoning.json requests --investigators vllm/qwen@high,vllm/gptoss@high
pids=()
for m in qwen gptoss; do
  uv run python -m swarm.batch "data/main/req/vllm_${m}@high_e3.jsonl" "data/main/out/vllm_${m}@high_e3.jsonl" --concurrency vllm=32 &
  pids+=($!)
done
wait "${pids[@]}"

uv run modal app stop -y swarm-inv-qwen || true
uv run modal app stop -y swarm-inv-gptoss || true
echo "Done. Tell Claude the open-weight E3 run finished."
