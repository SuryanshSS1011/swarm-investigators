"""Have author models write the routine posts of every skeleton, retrying a failed board with new replicates.

    python scripts/author_boards.py --data data/kt --authors claude/claude-haiku-5-5,claude/claude-sonnet-5-5

Writes <data>/bases.jsonl (one row per author and seed that passed validation) and <data>/author_log.jsonl
(one row per call, with its stop reason and validation errors). Calls go through swarm.llm, so they are cached.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from swarm import author as AU  # noqa: E402
from swarm import llm  # noqa: E402
from swarm import skeleton as SK  # noqa: E402
from swarm.parse import parse_json_block  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/kt")
    p.add_argument("--authors", required=True)
    p.add_argument("--max-replicates", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=6000)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--budget", type=float, default=5.0)
    a = p.parse_args()
    data = ROOT / a.data
    sks = [SK.Skeleton.from_dict(json.loads(l)) for l in (data / "skeletons.jsonl").read_text().splitlines()]
    spent = [0.0]
    log_rows = []

    def one(job):
        model, sk = job
        for rep in range(a.max_replicates):
            if spent[0] > a.budget:
                return model, sk.seed, None, "budget"
            r = llm.complete(model, AU.AUTHOR_SYSTEM, AU.author_prompt(sk), max_tokens=a.max_tokens, replicate=rep)
            spent[0] += r.cost_usd or 0.0
            texts, errs = AU.parse_author(sk, parse_json_block(r.text or "")) if r.text else ({}, ["empty"])
            log_rows.append({"author": model, "seed": sk.seed, "replicate": rep, "stop_reason": r.stop_reason,
                             "output_tokens": r.output_tokens, "cost_usd": r.cost_usd, "errors": errs[:5],
                             "error": r.error})
            if not errs:
                return model, sk.seed, {"author": model, "seed": sk.seed, "replicate": rep,
                                        "texts": {str(k): v for k, v in texts.items()}}, "ok"
        return model, sk.seed, None, "failed"

    jobs = [(m, sk) for m in a.authors.split(",") for sk in sks]
    bases_path = data / "bases.jsonl"
    keep = {}
    if bases_path.exists():
        for l in bases_path.read_text().splitlines():
            row = json.loads(l)
            keep[(row["author"], row["seed"])] = row
    jobs = [j for j in jobs if (j[0], j[1].seed) not in keep]
    status = collections.Counter()
    with ThreadPoolExecutor(a.workers) as ex:
        for model, seed, row, st in ex.map(one, jobs):
            status[(model, st)] += 1
            if row:
                keep[(model, seed)] = row
    bases_path.write_text("".join(json.dumps(r) + "\n" for r in sorted(keep.values(), key=lambda r: (r["author"], r["seed"]))))
    with (data / "author_log.jsonl").open("a") as f:
        for r in log_rows:
            f.write(json.dumps(r) + "\n")
    stops = collections.Counter((r["author"], r["stop_reason"]) for r in log_rows)
    print("status:", dict(status))
    print("stop reasons over all calls:", dict(stops))
    print("bases per author:", dict(collections.Counter(r["author"] for r in keep.values())))
    print(f"spent ${spent[0]:.3f}")


if __name__ == "__main__":
    main()
