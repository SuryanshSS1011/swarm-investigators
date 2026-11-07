"""Run the claude/* requests of a request file through the Message Batches API, at half the list price.

    uv run python -m swarm.anthropic_batches data/raw/requests.jsonl data/raw/results.jsonl            # submit, wait, collect
    uv run python -m swarm.anthropic_batches data/raw/requests.jsonl data/raw/results.jsonl --no-wait  # submit, then exit

It reads and writes the formats of swarm.batch (see that module), so either tool can finish a file the other
started, though not both at once. Each submitted batch is recorded in <out stem>.batches.json as soon as the API
returns its id. Rerunning the same command collects every batch that has ended and submits whatever is still
missing, including requests whose batch result was an error. Requests already in the disk cache are written
without a call, and collected answers go into the cache, so swarm.llm.complete() reuses them. An answer whose
request changed, or was finished another way, while its batch ran is kept in the cache and logged to the errors
file, so its cost still counts. Batch results have latency_s null, because the API reports no per-request timing.

Cost is half the list price in swarm/llm.py. A submitted batch cannot be stopped, so each submission is sized to
fit what is left of SWARM_BUDGET_USD (or --budget) after the spend so far and the worst case of batches still
running. The worst case assumes 3 characters per input token and the full max_tokens of output. Requests that do
not fit wait for a later run, by which time collected costs show what was really spent.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from swarm import llm
from swarm.batch import (Appender, Budget, Request, RequestFileError, _usd, batches_path, done_ids, errors_path,
                         load_requests, read_rows, resolve_budget, result_row, spend_text, spent_usd)
from swarm.llm import ConfigError, LLMResult

log = logging.getLogger(__name__)

MAX_PER_BATCH = 10_000
MAX_BATCH_BYTES = 100_000_000  # well under the API's per-batch size limit
CUSTOM_ID_LEN = 40  # hex characters of the request's cache key; custom ids allow letters, digits, - and _
CHARS_PER_TOKEN_BOUND = 3.0
POLL_S = 60.0
# A submission refused with one of these was not created, so sending it again cannot pay for it twice.
SUBMIT_RETRY_STATUSES = frozenset({429, 503, 529})


class BatchSubmitError(RuntimeError):
    """A submission that may or may not have created a batch."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_state(path) -> dict:
    path = Path(path)
    if not path.exists():
        return {"batches": []}
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(path, state: dict) -> None:
    """Write atomically, so a crash never leaves a half-written state file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(state, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def worst_case_usd(params: dict, price: tuple[float, float]) -> float:
    chars = len(params.get("system", "")) + sum(len(m["content"]) for m in params["messages"])
    input_tokens = math.ceil(chars / CHARS_PER_TOKEN_BOUND) + 50
    return llm.cost_usd(price, input_tokens, params["max_tokens"], llm.BATCH_DISCOUNT)


def _submit(client, requests: list[dict]):
    """Create one batch. Only refusals are retried. A lost connection or an ambiguous server error stops the run,
    because the batch may exist, and resubmitting it would pay for the same requests twice."""

    def should_retry(e: BaseException) -> bool:
        return llm.status_of(e) in SUBMIT_RETRY_STATUSES and llm.is_retryable(e)

    try:
        return llm.with_retries(lambda: client.messages.batches.create(requests=requests),
                                what=f"batch submission ({len(requests)} requests)", should_retry=should_retry)
    except Exception as e:  # noqa: BLE001
        status = llm.status_of(e)
        if llm.is_retryable(e) and status not in SUBMIT_RETRY_STATUSES:
            raise BatchSubmitError(
                f"submitting a batch of {len(requests)} requests failed with {llm.describe(e)}, so the batch may"
                " still have been created. Check the Batches page of the Claude Console before rerunning, or the"
                " same requests may be paid for twice."
            ) from e
        raise


def _row_for(rid: str, req: Optional[Request], model: str, res: LLMResult, batch_id: str) -> dict:
    if req is None:  # the request is no longer in the file
        req = Request(rid, model, "", "", 1)
    return result_row(req, res, batch_id)


def _error_text(result) -> str:
    inner = getattr(getattr(result, "error", None), "error", None)
    kind, message = getattr(inner, "type", None), getattr(inner, "message", None)
    return f"{kind}: {message}" if kind or message else str(result.type)


def _collect(client, entry: dict, by_id: dict, out: Path, ok_out: Appender, err_out: Appender, store, budget: Budget,
             ) -> Counter:
    """Write the results of one ended batch. Safe to repeat if an earlier collection was interrupted."""
    bid = entry["id"]
    done_rows = {r["id"]: r for r in read_rows(out) if "id" in r and r.get("error") is None}
    err_seen = {(r.get("id"), r.get("batch_id")) for r in read_rows(errors_path(out))}
    items = llm.with_retries(lambda: list(client.messages.batches.results(bid)), what=f"results of {bid}")
    counts: Counter = Counter()
    for item in items:
        info = entry["items"].get(item.custom_id)
        if info is None:
            log.warning("batch %s returned an unknown custom_id %s", bid, item.custom_id)
            continue
        key, model, ids = info["key"], info["model"], info["ids"]
        result = item.result
        if result.type != "succeeded":
            res = LLMResult(model=model, error=f"batch {result.type}: {_error_text(result)}")
            for rid in ids:
                if rid in done_rows or (rid, bid) in err_seen:
                    continue
                err_out.write(_row_for(rid, by_id.get(rid), model, res, bid))
                counts["error"] += 1
            continue
        price = llm.price_for("claude", model.removeprefix("claude/"))
        res = llm.claude_result(result.message, price, llm.BATCH_DISCOUNT)
        res.model = model
        if store is not None:
            store.put(key, model, res.to_dict(), res.cost_usd)
        # An interrupted earlier collection may have written this answer, and its cost, already.
        cost_left = 0.0 if any(done_rows.get(rid, {}).get("batch_id") == bid for rid in ids) else res.cost_usd
        if cost_left and any((rid, bid) in err_seen for rid in ids):
            cost_left = 0.0
        paid = cost_left
        for rid in ids:
            req = by_id.get(rid)
            if rid in done_rows or req is None or req.key != key:
                continue
            ok_out.write(result_row(req, replace(res, cost_usd=cost_left, cached=cost_left == 0.0), bid))
            done_rows[rid] = {"batch_id": bid}
            cost_left = 0.0
            counts["ok"] += 1
        if cost_left:
            unused = replace(res, error="answer not used: its request changed or was finished another way while the"
                                        " batch ran; it is kept in the cache")
            err_out.write(_row_for(ids[0], by_id.get(ids[0]), model, unused, bid))
            counts["unused"] += 1
        budget.add(paid)
    return counts


def _collect_ended(client, state: dict, state_path: Path, by_id: dict, out: Path, ok_out: Appender,
                   err_out: Appender, store, budget: Budget) -> Counter:
    total: Counter = Counter()
    for entry in state["batches"]:
        if entry.get("collected"):
            continue
        bid = entry["id"]
        info = llm.with_retries(lambda b=bid: client.messages.batches.retrieve(b), what=f"status of {bid}")
        c = info.request_counts
        if info.processing_status != "ended":
            print(f"  batch {bid}: {info.processing_status}, {c.processing} processing, {c.succeeded} succeeded,"
                  f" {c.errored} errored", flush=True)
            continue
        got = _collect(client, entry, by_id, out, ok_out, err_out, store, budget)
        entry["collected"] = True
        entry["collected_at"] = _now()
        save_state(state_path, state)
        print(f"  batch {bid}: collected {got['ok']} answers and {got['error']} errors", flush=True)
        total.update(got)
    return total


def _submit_within_budget(client, groups: dict, state: dict, state_path: Path, budget: Budget,
                          max_per_batch: int) -> Counter:
    counts: Counter = Counter()
    running = sum(b.get("worst_case_usd", 0.0) for b in state["batches"] if not b.get("collected"))
    room = budget.limit - budget.spent - running
    chunk: list = []
    size = [0, 0.0]  # bytes and worst-case USD of the chunk being built

    def flush() -> None:
        if not chunk:
            return
        requests = [{"custom_id": key[:CUSTOM_ID_LEN], "params": params} for key, _, params in chunk]
        if len({r["custom_id"] for r in requests}) != len(requests):
            raise RuntimeError("two requests share a custom_id prefix; raise CUSTOM_ID_LEN")
        batch = _submit(client, requests)
        n_ids = sum(len(group) for _, group, _ in chunk)
        state["batches"].append({
            "id": batch.id,
            "submitted_at": _now(),
            "collected": False,
            "worst_case_usd": round(size[1], 8),
            "items": {key[:CUSTOM_ID_LEN]: {"key": key, "model": group[0].model, "ids": [r.id for r in group]}
                      for key, group, _ in chunk},
        })
        save_state(state_path, state)
        print(f"  submitted batch {batch.id}: {len(requests)} calls for {n_ids} requests, worst case"
              f" ${size[1]:.4f}", flush=True)
        counts["submitted"] += n_ids
        chunk.clear()
        size[0], size[1] = 0, 0.0

    for key, group in groups.items():
        head = group[0]
        model_id = head.model.removeprefix("claude/")
        params = llm.claude_params(model_id, head.system, head.user, head.max_tokens, head.temperature)
        worst = worst_case_usd(params, llm.price_for("claude", model_id))
        if worst > room:
            counts["held"] += len(group)
            continue
        nbytes = len(json.dumps(params, ensure_ascii=False).encode("utf-8")) + 100
        if chunk and (len(chunk) >= max_per_batch or size[0] + nbytes > MAX_BATCH_BYTES):
            flush()
        chunk.append((key, group, params))
        size[0] += nbytes
        size[1] += worst
        room -= worst
    flush()
    if counts["held"]:
        print(f"  {counts['held']} requests held back, because their worst case does not fit in what is left of the"
              f" {_usd(budget.limit)} budget; rerun after the running batches are collected", flush=True)
    return counts


def run_batches(requests_jsonl, out_jsonl, *, wait: bool = True, poll_s: float = POLL_S,
                max_per_batch: int = MAX_PER_BATCH, budget_usd: Optional[float] = None, cache=None,
                limit: Optional[int] = None, client=None) -> dict:
    """Collect ended batches, write cached answers, submit what is missing, and optionally wait for the rest."""
    llm.load_env()
    requests = load_requests(requests_jsonl)
    if limit:
        requests = requests[:limit]
    claude = [r for r in requests if llm.provider_of(r.model) == "claude"]
    others = len(requests) - len(claude)
    if others:
        print(f"  {others} requests are not claude/* and are skipped; run them with python -m swarm.batch", flush=True)
    out = Path(out_jsonl)
    err_path, state_path = errors_path(out), batches_path(out)
    state = load_state(state_path)
    by_id = {r.id: r for r in claude}
    models = sorted({r.model for r in claude})
    for model in models:
        llm.check_route(model)
    budget = Budget(resolve_budget(budget_usd, models), spent_usd(read_rows(out), read_rows(err_path)))
    store = llm.resolve_cache(cache)
    if client is None and (claude or any(not b.get("collected") for b in state["batches"])):
        client = llm.anthropic_client()
    ok_out, err_out = Appender(out), Appender(err_path)
    counts: Counter = Counter()
    try:
        counts.update(_collect_ended(client, state, state_path, by_id, out, ok_out, err_out, store, budget))
        done = done_ids(read_rows(out))
        pending = {it["key"] for b in state["batches"] if not b.get("collected") for it in b["items"].values()}
        groups: dict[str, list[Request]] = {}
        for r in claude:
            if r.id not in done and r.key not in pending:
                groups.setdefault(r.key, []).append(r)
        for key in list(groups):
            hit = store.get(key) if store is not None else None
            if hit is None:
                continue
            res = replace(LLMResult.from_dict(hit), cost_usd=0.0, cached=True, error=None)
            for r in groups.pop(key):
                ok_out.write(result_row(r, res))
                counts["cached"] += 1
        counts.update(_submit_within_budget(client, groups, state, state_path, budget, max_per_batch))
        while wait and any(not b.get("collected") for b in state["batches"]):
            llm._sleep(poll_s)
            counts.update(_collect_ended(client, state, state_path, by_id, out, ok_out, err_out, store, budget))
    finally:
        ok_out.close()
        err_out.close()

    done = done_ids(read_rows(out)) & set(by_id)
    running = sum(1 for b in state["batches"] if not b.get("collected"))
    summary = {
        "requests": len(requests),
        "claude_requests": len(claude),
        "skipped_not_claude": others,
        "done": len(done),
        "cached": counts["cached"],
        "submitted": counts["submitted"],
        "held_for_budget": counts["held"],
        "collected_ok": counts["ok"],
        "collected_errors": counts["error"],
        "batches_running": running,
        "spent_usd_total": round(budget.spent, 8),
        "budget_usd": None if math.isinf(budget.limit) else budget.limit,
        "out": str(out),
        "state_file": str(state_path),
    }
    print(f"{out.name}: {len(done)} of {len(claude)} claude requests done, {running} batches still running,"
          f" {spend_text(budget)}", flush=True)
    return summary


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m swarm.anthropic_batches", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("requests", help="request JSONL (only claude/* lines are sent)")
    p.add_argument("out", help="output JSONL, in the swarm.batch format")
    p.add_argument("--no-wait", action="store_true", help="submit and exit; rerun the same command to collect")
    p.add_argument("--poll", type=float, default=POLL_S, help="seconds between status checks while waiting")
    p.add_argument("--max-per-batch", type=int, default=MAX_PER_BATCH, help="requests per submitted batch")
    p.add_argument("--budget", type=float, default=None, help="USD cap for this output file (default SWARM_BUDGET_USD)")
    p.add_argument("--limit", type=int, default=0, help="use only the first N requests, for a smoke test")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    try:
        s = run_batches(a.requests, a.out, wait=not a.no_wait, poll_s=a.poll, max_per_batch=a.max_per_batch,
                        budget_usd=a.budget, limit=a.limit or None)
    except (ConfigError, RequestFileError, BatchSubmitError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 3
    if s["held_for_budget"]:
        return 2
    return 0 if s["done"] == s["claude_requests"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
