"""Run a request file from your laptop through swarm/batch.py, the same code this workspace uses.

    uv run python modal_jobs/run_requests_laptop.py data/raw/requests.jsonl data/raw/results.jsonl --concurrency vllm=32

Use it for the routes this workspace cannot reach, which are vllm/* on Modal, openrouter/* and openai/*. Claude and
Gemini requests in the same file run too. It takes the arguments of `python -m swarm.batch`, reads compute.env from
the repo root, and resumes the same way, so rerunning a command only does what is missing. See
docs/modal_runbook.md.

Adapted from collusion-sting modal_jobs/run_prompts.py at 4be9164, which it replaces.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from swarm import batch  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(batch.main(sys.argv[1:]))
