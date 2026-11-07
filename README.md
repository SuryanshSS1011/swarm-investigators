# The Last Word Wins

**Retractions and peer alibis in swarm logs mislead LLM incident investigators.** This repository holds the
benchmark, pre-registrations, raw outputs and analysis for an entry in Apart Research's
[Swarm Alignment Hackathon](https://apartresearch.com/sprints/swarm-alignment-2026) (November 2026).

## What the study does

After the July 2026 Hugging Face incident, METR and Redwood delegated much of their transcript review to LLM
agents. We built 40 synthetic swarm logs with planted ground-truth violations, each with a benign twin that matches
it line for line, had several model families write the agents' posts, and measured how text that agents write into
their own logs changes LLM investigators' verdicts.

1. `docs/prereg_pilot.md` and `results/pilot/` hold the pilot and its decision rules.
2. `docs/prereg_main.md` is the main pre-registration, committed before any main-study call.
3. `docs/explore_alibi.md` (E1), `docs/explore_logging.md` (E2) and `docs/explore_reasoning.md` (E3) are
   exploratory follow-ups, each committed before its first call.
4. `docs/prereg_replication.md` (R) is a pre-registered replication with model-written retractions and GPT-5.6 Sol
   as an investigator, committed before any of its calls.
5. `results/main/summary.json` holds every pre-registered test of the main study, `results/main/secondary.json` the
   secondary and exploratory quantities (scored leniently, split by investigator group, and the E1, E2 and probe
   results), `results/main/replication.json` the replication, `results/main/reasoning.json` E3, and
   `results/main/fig*.png` the figures.
6. `results/main/raw/` holds the gzipped raw model outputs, scored audits and probe answers behind every number.

## Reproducing

```bash
uv sync && cp compute.env.example compute.env      # add ANTHROPIC_API_KEY and GEMINI_API_KEY
python3 scripts/study.py skeletons                 # seeded boards (already in data/main/)
python3 scripts/author_boards.py --data data/main --authors claude/claude-haiku-5-5,gemini/gemini-3.8-flash
python3 scripts/study.py requests --investigators claude/claude-haiku-5-5      # one file per investigator
python3 -m swarm.anthropic_batches data/main/req/claude_claude-haiku-5-5.jsonl data/main/out/claude_claude-haiku-5-5.jsonl
bash scripts/run_openweight.sh                     # Qwen3-32B and gpt-oss-20b on Modal, from a laptop
bash scripts/build_all.sh                          # score, analyze, draw figures
```

To rebuild the results without any model calls, unzip the committed outputs first. The audit files go to
`data/main/out/`, and `scored`, `probes` and `author_log` go to `data/main/`. The replication also reads
`data/main/gen_retractions.jsonl`, which is committed.

```bash
mkdir -p data/main/out && for f in results/main/raw/{claude,gemini,vllm}_*.jsonl.gz; do gunzip -c "$f" > data/main/out/$(basename "${f%.gz}"); done
for f in scored probes author_log; do gunzip -c results/main/raw/$f.jsonl.gz > data/main/$f.jsonl; done
python3 scripts/analyze.py && python3 scripts/analyze_secondary.py && python3 scripts/figures.py
```

The board generator is in `swarm/world.py`, `swarm/skeleton.py`, `swarm/author.py` and `swarm/assemble.py`. The
investigator prompt, parser and scorer are in `swarm/investigate.py`, and the rule-based baseline is in
`swarm/baseline.py`.

## Infrastructure

### Setup

```bash
git clone https://github.com/SuryanshSS1011/swarm-investigators.git
cd swarm-investigators
uv sync                              # on the laptop, use: uv sync --extra modal
cp compute.env.example compute.env   # then fill in only the keys you need
uv run pytest -q                     # offline checks, no API calls
```

Run every command from the repo root. `compute.env` is gitignored. The open-weight models also need the Modal servers described in [`docs/modal_runbook.md`](docs/modal_runbook.md).

### Calling a model

`swarm.llm.complete(model, system, user, max_tokens, temperature=None, replicate=0)` returns an `LLMResult` with `text`, `input_tokens`, `output_tokens`, `cost_usd`, `latency_s` and `error`, plus `model`, `stop_reason` and `cached`. The prefix of the model id picks the route.

| Model id | Endpoint | Keys in compute.env | Runs from |
|---|---|---|---|
| `claude/<id>` | api.anthropic.com | `ANTHROPIC_API_KEY`, optionally `ANTHROPIC_WORKSPACE_ID` | workspace or laptop |
| `gemini/<id>` | Google's OpenAI-compatible endpoint | `GEMINI_API_KEY` | workspace or laptop |
| `vllm/qwen`, `vllm/gptoss` | vLLM on Modal | `VLLM_API_KEY`, `VLLM_BASE_URL_<NAME>` | laptop |
| `openrouter/<id>` | openrouter.ai | `OPENROUTER_API_KEY` | laptop |
| `openai/<id>` | api.openai.com | `OPENAI_API_KEY` | laptop |
| `mock/<name>` | none | none | tests |

Each route keeps its models at their lowest reasoning setting.

1. **Claude:** Claude 5.5 models never get temperature or top_p. Haiku runs with thinking disabled, Sonnet with `between_tools`, and Opus with no thinking field because it always thinks. Every call asks for low effort. The base URL is pinned to api.anthropic.com, so an `ANTHROPIC_BASE_URL` exported by other tooling is ignored.
2. **vLLM:** Qwen3's thinking is switched off. gpt-oss runs at low reasoning effort and gets at least 4,000 output tokens, because its reasoning counts against `max_tokens`.
3. **Gemini:** The lowest `reasoning_effort` each model accepts is found on its first call by climbing from `none`, and thinking tokens are counted as output tokens.
4. **OpenRouter:** The cost of each call is read from the response.
5. **OpenAI:** Requests send `max_completion_tokens`, and `OPENAI_REASONING` sets each model's lowest accepted reasoning effort. Add a model's prices to `OPENAI_PRICES` in `swarm/llm.py` before its first call.

Successful answers are cached in `data/cache/llm.sqlite`, keyed by sha256 over model, system, user, max_tokens, temperature and replicate. A cache hit makes no call and costs nothing, and `replicate` draws independent samples of one prompt. Calls are retried with exponential backoff and jitter on HTTP 408, 429 and 5xx, which includes Anthropic's 529 overloaded error, and on connection errors. An empty, refused or truncated answer is a result rather than an error, so it is never retried, because retrying until an answer parses would bias the sample. Check `stop_reason` instead.

### Running request files

A request file is JSONL with one object per line. Each object holds `id`, `model`, `system`, `user` and `max_tokens`, and it may add `temperature`, `replicate` and a `meta` dict that is copied to the output.

1. `uv run python -m swarm.batch requests.jsonl out.jsonl --concurrency claude=8,vllm=32` runs any route. It resumes by request id, prints progress, and writes failed calls to `out.errors.jsonl` so the next run retries them. It stops starting calls once the output file's spend reaches `SWARM_BUDGET_USD`.
2. `uv run python -m swarm.anthropic_batches requests.jsonl out.jsonl` sends the `claude/*` requests through the Message Batches API at half price and writes the same output format. Add `--no-wait` to submit and exit, and rerun the same command to collect.
3. `uv run python modal_jobs/run_requests_laptop.py requests.jsonl out.jsonl` is the laptop entry point for the routes this workspace cannot reach. It runs `swarm.batch` unchanged.

Keep raw outputs in `data/raw/`, which is gitignored along with `data/cache/`.

### Parsing answers

`swarm.parse.parse_json_block(text)` returns the answer object from model text, or None. It skips reasoning and code fences and repairs common JSON mistakes. It returns None when the last JSON value is cut off or broken, rather than falling back to an earlier draft.

## Preexisting code

The sprint rules require declaring prior work. The files below were copied or adapted from the author's earlier project [collusion-sting](https://github.com/SuryanshSS1011/collusion-sting) at commit `4be9164` (`4be91648cbc501db8c2896fb5983db8cfe6347c1`), an entry in the Apart Research AI Collusion Research Sprint of October 2026. Everything else in this repository was written for this sprint.

| File here | Original path at 4be9164 | What was kept and what changed |
|---|---|---|
| `swarm/llm.py` | `sting/clients.py` | The Claude request handling and price table are copied, as are the vLLM reasoning switches. One `complete()` function replaces the client classes, retries gained jitter, Retry-After and a stop on spend-limit errors, and the client timeout rose from 120 s to 600 s. vLLM URLs now come from `VLLM_BASE_URL_<NAME>`. The Gemini, OpenAI, OpenRouter and mock routes are new. collusion-sting sent OpenRouter calls through its upstream harness, which is not copied. |
| `modal_jobs/serve_qwen.py` | `modal_jobs/serve_qwen.py` | The image, volumes, secret and server flags are unchanged. The app is renamed `swarm-inv-qwen`, the context length became the constant `MAX_MODEL_LEN` with the same value, later raised from 16,384 to 32,768 for E3, and the docstring is new. |
| `modal_jobs/serve_gptoss.py` | `modal_jobs/serve_gptoss.py` | The same changes, with the app renamed `swarm-inv-gptoss`. The docstring's deploy command, which named `serve_qwen.py`, is fixed. |
| `modal_jobs/run_requests_laptop.py` | `modal_jobs/run_prompts.py` | Replaced by a thin command-line wrapper over `swarm/batch.py`, so the laptop runs the workspace's code. |
| `swarm/batch.py` | `modal_jobs/run_prompts.py` and `analysis/monitor_test.py` | New code that follows their pattern of a thread pool that resumes by request id and stops at a spend cap. Unlike `run_prompts.py`, failed calls are not marked done. |
| `docs/modal_runbook.md` | `docs/modal_runbook.md` | The setup, secret, warm-up, transformers pin and gpt-oss token check are kept. The rest is rewritten for the new apps, variables and request runner. |
| `compute.env.example` | `compute.env.example` | The Anthropic key and workspace entries and their comments are kept, and the other variables are new. |
| `.gitignore` | `.gitignore` | The secrets and environment entries are kept, and the result folders are replaced by `data/cache/` and `data/raw/`. |
| `pyproject.toml` | `pyproject.toml` | The uv non-package layout and pytest settings are kept. The dependencies and the Python floor are new. |
| `tests/test_llm.py` | `tests/test_offline.py` | `test_claude_request_shape` is adapted from `test_claude_client_request_shape`. The other tests in the file are new. |
| mock route in `swarm/llm.py` | `sting/mock.py` | Only the idea of a deterministic offline model for tests is reused, and no code is copied. |

`swarm/cache.py`, `swarm/anthropic_batches.py`, `swarm/parse.py` and the other test files are new. collusion-sting parsed model JSON with the upstream harness's `extract_json`, and `swarm/parse.py` neither uses nor copies it.
