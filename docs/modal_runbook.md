# Running the open-weight models on Modal (laptop runbook)

The open-weight models run on vLLM servers on Modal, and their requests run from your laptop, because the cloud
workspace cannot reach Modal, Hugging Face, OpenRouter or OpenAI. The laptop and the workspace run the same code in
`swarm/batch.py`, so outputs have the same format and resume the same way. Two models are set up, and both have
Apache-2.0 weights.

1. **Qwen3-32B-FP8** is `vllm/qwen`. `modal_jobs/serve_qwen.py` serves it as the Modal app `swarm-inv-qwen`, and its
   thinking mode is off.
2. **gpt-oss-20b** is `vllm/gptoss`. `modal_jobs/serve_gptoss.py` serves it as the Modal app `swarm-inv-gptoss`, and
   its reasoning effort is low.

Each server is one H100 billed per GPU-second, and it scales to zero after 10 idle minutes. Run the steps in order.

## 0. One-time setup

1. Install `uv` from https://docs.astral.sh/uv/.
2. Clone the repo and install it with the Modal extra.

   ```bash
   git clone https://github.com/SuryanshSS1011/swarm-investigators.git
   cd swarm-investigators
   uv sync --extra modal
   uv run pytest -q          # offline checks, no API calls
   ```

3. Log in to Modal with `uv run modal setup`, which opens a browser.
4. Copy `compute.env.example` to `compute.env`. It is gitignored, so it is never committed.

## 1. Start the servers

Both servers read their API key from a Modal secret named `vllm-key`, the one collusion-sting used. If it exists and
you still have its value, reuse it. Otherwise make a new key, and `--force` replaces the old secret.

```bash
export VLLM_KEY=$(openssl rand -hex 24)
uv run modal secret create vllm-key VLLM_API_KEY=$VLLM_KEY --force
uv run modal deploy modal_jobs/serve_qwen.py
uv run modal deploy modal_jobs/serve_gptoss.py
```

Modal prints one URL per app, such as `https://<you>--swarm-inv-qwen-serve.modal.run` and
`https://<you>--swarm-inv-gptoss-serve.modal.run`. Fill in these lines of `compute.env`, adding `/v1` to each URL.

```bash
VLLM_API_KEY=<the value of $VLLM_KEY>
VLLM_BASE_URL_QWEN=https://<you>--swarm-inv-qwen-serve.modal.run/v1
VLLM_BASE_URL_GPTOSS=https://<you>--swarm-inv-gptoss-serve.modal.run/v1
```

Warm both servers up and check them. The first call downloads the weights, about 33 GB for Qwen and 14 GB for
gpt-oss, and it can take 5 to 10 minutes. If a call times out, run it again. The weights are cached in the Modal
volume `huggingface-cache`, which collusion-sting also used, so they may already be there.

```bash
curl -s https://<you>--swarm-inv-qwen-serve.modal.run/v1/models -H "Authorization: Bearer $VLLM_KEY"
curl -s https://<you>--swarm-inv-gptoss-serve.modal.run/v1/models -H "Authorization: Bearer $VLLM_KEY"
```

Both server scripts pin transformers, because an unpinned install crashed vLLM 0.9.2 on import with "'aimv2' is
already used by a Transformers config". If a deploy still fails, send the last lines of
`uv run modal app logs swarm-inv-qwen` or `uv run modal app logs swarm-inv-gptoss` to Claude rather than changing
versions locally.

## 2. Smoke test

This writes a two-line request file and runs it through the laptop runner, which takes a minute or two.

```bash
mkdir -p data/raw
uv run python - <<'EOF'
import json
with open("data/raw/smoke.jsonl", "w") as f:
    for model in ("vllm/qwen", "vllm/gptoss"):
        f.write(json.dumps({"id": "smoke-" + model, "model": model, "system": "",
                            "user": 'Reply with the JSON object {"ok": true} and nothing else.',
                            "max_tokens": 50}) + "\n")
EOF
uv run python modal_jobs/run_requests_laptop.py data/raw/smoke.jsonl data/raw/smoke_out.jsonl
```

Both lines of `data/raw/smoke_out.jsonl` should have `"error": null` and text that contains `{"ok": true}`. Failed
calls go to `data/raw/smoke_out.errors.jsonl`, and rerunning the command retries only those.

Then check gpt-oss's output tokens per call. Its reasoning counts as output, so every gpt-oss call is given at least
4,000 output tokens whatever the request says. At low reasoning effort a call should use a few hundred. If calls use
thousands, stop and tell Claude. Another sprint team saw reasoning models produce 15,000 to 20,000 output tokens per
call, which would make a run slow and leave its answers truncated.

## 3. Real runs

Run a request file from the experiment workstream with the same command. It resumes, so rerun it after any failure.

```bash
uv run python modal_jobs/run_requests_laptop.py data/raw/<name>.jsonl data/raw/<name>_out.jsonl \
    --concurrency vllm/qwen=32,vllm/gptoss=32
```

1. **Concurrency:** A full model id gets a worker pool of its own, so the two servers do not share one. Lower the
   numbers to 16 if you see many timeouts or 5xx errors.
2. **Context length:** Each server accepts at most 16,384 tokens of prompt plus output, set by `MAX_MODEL_LEN` in its
   script. A longer request fails with HTTP 400 and lands in the errors file. Raise `MAX_MODEL_LEN` and redeploy if
   prompts outgrow it. Qwen3 goes up to 32,768 without YaRN, and gpt-oss goes up to 131,072.
3. **Other routes:** The same command runs `openrouter/*` and `openai/*` requests when `OPENROUTER_API_KEY` or
   `OPENAI_API_KEY` is set, and it runs `claude/*` and `gemini/*` requests as well. Paid routes need
   `SWARM_BUDGET_USD`, and an `openai/*` model needs an entry in `OPENAI_PRICES` in `swarm/llm.py` before its first
   call.

## 4. Hand the outputs back

`data/raw/` is gitignored, so copy the finished outputs to a folder that is not, and push them.

```bash
mkdir -p data/laptop
cp data/raw/<name>_out.jsonl data/laptop/
git add data/laptop/<name>_out.jsonl
git commit -m "Laptop outputs: <name>"
git push
```

## 5. Stop billing

```bash
uv run modal app stop swarm-inv-qwen
uv run modal app stop swarm-inv-gptoss
```

Each server also scales to zero after 10 idle minutes.

## Adding a third model

1. Copy `modal_jobs/serve_gptoss.py` to `modal_jobs/serve_<name>.py`. Change `MODEL`, set the app name to
   `swarm-inv-<name>`, and bump the vLLM pin if the model needs a newer release.
2. Add `"<name>": "<Hugging Face id>"` to `VLLM_SERVED` in `swarm/llm.py`, or set `VLLM_MODEL_<NAME>` in
   `compute.env`.
3. Set `VLLM_BASE_URL_<NAME>` in `compute.env`, and call the model as `vllm/<name>`.
4. Check its reasoning switch. Every vLLM model except gpt-oss is sent `chat_template_kwargs` with
   `enable_thinking: false`, which Qwen3 templates read and other templates ignore. A model with a different switch
   needs its own case in `vllm_reasoning_args` in `swarm/llm.py`.

## Notes for the report

1. Qwen's thinking mode is off, and gpt-oss runs at low reasoning effort with at least 4,000 output tokens, as in
   collusion-sting.
2. Temperature is whatever each request sets. A request without one gets the server's default sampling settings.
3. Open-weight calls have `cost_usd` 0, because Modal bills GPU time rather than tokens. Record the Modal bill
   separately.
4. collusion-sting never sent a system prompt to the open-weight models, so the system-role path for them is new.
   Check one transcript per model before a full run.
