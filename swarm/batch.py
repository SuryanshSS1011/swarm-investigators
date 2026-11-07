"""Run a JSONL file of model requests through swarm.llm. Runs are resumable, concurrent per provider and capped in spend.

    uv run python -m swarm.batch data/raw/requests.jsonl data/raw/results.jsonl
    uv run python -m swarm.batch data/raw/requests.jsonl data/raw/results.jsonl --concurrency claude=8,vllm=32 --budget 20

Each request line is a JSON object with "id" (a unique string), "model", "system", "user" and "max_tokens", and
optionally "temperature" (not sent when absent), "replicate" (default 0) and "meta" (any object, copied to the
output untouched).

Each finished request gets one output line with its "id", "model", "replicate" and "meta" and the result fields
"text", "input_tokens", "output_tokens", "cost_usd", "latency_s", "stop_reason", "cached", "error" and "batch_id".
A failed call goes to <out stem>.errors.jsonl in the same format and is tried again on the next run, so rerunning
a command only does what is missing. Requests that differ only in id share one call, and only the first of them
carries its cost. swarm.anthropic_batches writes the same format with "batch_id" set.

The spend of an output file is the sum of cost_usd over it and its errors file, so it carries across reruns. Once
it reaches SWARM_BUDGET_USD (or --budget), no new call starts. Calls already running finish and are written, so
the overshoot is at most one call per worker. Cached answers are still written, because they cost nothing. vLLM
calls count as $0, because Modal bills GPU time rather than tokens.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional

from swarm import llm
from swarm.cache import cache_key
from swarm.llm import ConfigError, LLMResult

log = logging.getLogger(__name__)

REQUIRED_FIELDS = ("id", "model", "system", "user", "max_tokens")
OPTIONAL_FIELDS = ("temperature", "replicate", "meta")
DEFAULT_CONCURRENCY = {"claude": 8, "vllm": 32, "openrouter": 8, "gemini": 8, "openai": 8, "mock": 8}


class RequestFileError(ValueError):
    """A request file line that cannot be run."""


@dataclass
class Request:
    id: str
    model: str
    system: str
    user: str
    max_tokens: int
    temperature: Optional[float] = None
    replicate: int = 0
    meta: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return cache_key(self.model, self.system, self.user, self.max_tokens, self.temperature, self.replicate)


def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _parse_request(d, where: str) -> Request:
    if not isinstance(d, dict):
        raise RequestFileError(f"{where}: each line must be a JSON object")
    missing = [k for k in REQUIRED_FIELDS if k not in d]
    if missing:
        raise RequestFileError(f"{where}: missing {', '.join(missing)}")
    unknown = sorted(set(d) - set(REQUIRED_FIELDS) - set(OPTIONAL_FIELDS))
    if unknown:
        raise RequestFileError(f"{where}: unknown field {', '.join(unknown)}; put extra data in meta")
    rid, model, system, user = d["id"], d["model"], d["system"] or "", d["user"]
    if not isinstance(rid, str) or not rid:
        raise RequestFileError(f"{where}: id must be a non-empty string")
    if not isinstance(model, str):
        raise RequestFileError(f"{where}: model must be a string")
    try:
        llm.split_model(model)
    except ConfigError as e:
        raise RequestFileError(f"{where}: {e}") from None
    if not isinstance(system, str) or not isinstance(user, str):
        raise RequestFileError(f"{where}: system and user must be strings")
    max_tokens = d["max_tokens"]
    if not _is_int(max_tokens) or max_tokens < 1:
        raise RequestFileError(f"{where}: max_tokens must be a positive integer")
    temperature = d.get("temperature")
    if temperature is not None and (isinstance(temperature, bool) or not isinstance(temperature, (int, float))):
        raise RequestFileError(f"{where}: temperature must be a number or null")
    replicate = 0 if d.get("replicate") is None else d["replicate"]
    if not _is_int(replicate) or replicate < 0:
        raise RequestFileError(f"{where}: replicate must be a non-negative integer")
    meta = {} if d.get("meta") is None else d["meta"]
    if not isinstance(meta, dict):
        raise RequestFileError(f"{where}: meta must be an object")
    return Request(rid, model, system, user, max_tokens, temperature, replicate, meta)


def load_requests(path) -> list[Request]:
    """Read and check a request file. Every problem names its line."""
    requests, seen = [], set()
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            where = f"{path}:{n}"
            try:
                d = json.loads(line)
            except json.JSONDecodeError as e:
                raise RequestFileError(f"{where}: not valid JSON ({e.msg})") from None
            r = _parse_request(d, where)
            if r.id in seen:
                raise RequestFileError(f"{where}: duplicate id {r.id!r}")
            seen.add(r.id)
            requests.append(r)
    return requests


# ---------------------------------------------------------------- output files


def errors_path(out) -> Path:
    out = Path(out)
    return out.with_name(f"{out.stem}.errors{out.suffix}")


def batches_path(out) -> Path:
    out = Path(out)
    return out.with_name(f"{out.stem}.batches.json")


def read_rows(path) -> list[dict]:
    """Rows of a JSONL file. A line cut short by a crash is skipped with a warning."""
    path = Path(path)
    if not path.exists():
        return []
    rows, bad = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if isinstance(row, dict):
                rows.append(row)
    if bad:
        log.warning("%s: skipped %d unreadable line(s), probably cut short by a crash", path, bad)
    return rows


def spent_usd(*row_lists) -> float:
    return sum(float(r.get("cost_usd") or 0.0) for rows in row_lists for r in rows)


def done_ids(rows) -> set:
    return {r["id"] for r in rows if "id" in r and r.get("error") is None}


class Appender:
    """Appends JSON lines and flushes each one, so a crash loses at most the line being written. The file is
    opened on the first write, and a last line left without its newline by a crash is ended first."""

    def __init__(self, path):
        self.path = Path(path)
        self._f = None
        self._lock = threading.Lock()

    def _open(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and self.path.stat().st_size:
            with open(self.path, "rb") as f:
                f.seek(-1, os.SEEK_END)
                ended = f.read(1) == b"\n"
            if not ended:
                with open(self.path, "ab") as f:
                    f.write(b"\n")
        self._f = open(self.path, "a", encoding="utf-8")

    def write(self, row: dict) -> None:
        line = json.dumps(row, ensure_ascii=False) + "\n"
        with self._lock:
            if self._f is None:
                self._open()
            self._f.write(line)
            self._f.flush()

    def close(self) -> None:
        with self._lock:
            if self._f is not None:
                self._f.close()
                self._f = None


def result_row(req: Request, res: LLMResult, batch_id: Optional[str] = None) -> dict:
    """One output line. swarm.anthropic_batches writes the same fields."""
    return {
        "id": req.id,
        "model": req.model,
        "replicate": req.replicate,
        "text": res.text,
        "input_tokens": res.input_tokens,
        "output_tokens": res.output_tokens,
        "cost_usd": res.cost_usd,
        "latency_s": res.latency_s,
        "stop_reason": res.stop_reason,
        "cached": res.cached,
        "error": res.error,
        "batch_id": batch_id,
        "meta": req.meta,
    }


# ---------------------------------------------------------------- budget and concurrency


class Budget:
    """Spend so far against a cap, shared by worker threads."""

    def __init__(self, limit_usd: float, spent: float = 0.0):
        self.limit = float(limit_usd)
        self.spent = float(spent)
        self._lock = threading.Lock()

    def add(self, usd: float) -> None:
        with self._lock:
            self.spent += usd

    @property
    def exhausted(self) -> bool:
        with self._lock:
            return self.spent >= self.limit


def _usd(x: float) -> str:
    return "no cap" if math.isinf(x) else f"${x:.2f}"


def spend_text(budget: "Budget") -> str:
    """For progress lines, such as "$1.2345 of $20.00 spent" or "$1.2345 spent (no cap)"."""
    if math.isinf(budget.limit):
        return f"${budget.spent:.4f} spent (no cap)"
    return f"${budget.spent:.4f} of ${budget.limit:.2f} spent"


def resolve_budget(budget_usd, models) -> float:
    """--budget, else SWARM_BUDGET_USD, else no cap, which is allowed only when no model costs money."""
    if budget_usd is not None:
        return float(budget_usd)
    raw = os.environ.get("SWARM_BUDGET_USD", "").strip()
    if raw:
        try:
            return float(raw)
        except ValueError:
            raise ConfigError(f"SWARM_BUDGET_USD must be a number of US dollars, not {raw!r}") from None
    paid = sorted({m for m in models if llm.provider_of(m) in llm.PAID_PROVIDERS})
    if paid:
        raise ConfigError("SWARM_BUDGET_USD is not set, and these models cost money: " + ", ".join(paid)
                          + ". Set it in compute.env or pass --budget.")
    return math.inf


def _positive(value, name: str) -> int:
    value = int(value)
    if value < 1:
        raise ValueError(f"concurrency for {name} must be at least 1")
    return value


def parse_concurrency(spec=None) -> dict:
    """Workers per provider, or per full model id, which then gets a pool of its own. Takes a dict, one number for
    every provider, or a string such as "claude=8,vllm=32,vllm/gptoss=16"."""
    conc = dict(DEFAULT_CONCURRENCY)
    if spec is None or spec == "":
        return conc
    if isinstance(spec, int):
        return {k: _positive(spec, k) for k in conc}
    if isinstance(spec, str):
        if spec.strip().isdigit():
            return {k: _positive(int(spec), k) for k in conc}
        parsed = {}
        for part in spec.split(","):
            name, sep, value = part.partition("=")
            if not sep or not name.strip():
                raise ValueError(f"bad concurrency entry {part!r}; write name=count, for example claude=8")
            parsed[name.strip()] = value
        spec = parsed
    for name, value in spec.items():
        if name not in conc and "/" not in name:
            raise ValueError(f"unknown concurrency key {name!r}; use a provider ({', '.join(DEFAULT_CONCURRENCY)})"
                             " or a full model id")
        conc[name] = _positive(value, name)
    return conc


def lane_of(model: str, conc: dict) -> str:
    return model if model in conc else llm.provider_of(model)


# ---------------------------------------------------------------- the runner


def run_requests(requests_jsonl, out_jsonl, concurrency=None, *, budget_usd: Optional[float] = None, cache=None,
                 limit: Optional[int] = None, progress_every_s: float = 15.0) -> dict:
    """Run every request in requests_jsonl that out_jsonl does not hold yet, and return a summary."""
    llm.load_env()
    requests = load_requests(requests_jsonl)
    if limit:
        requests = requests[:limit]
    out = Path(out_jsonl)
    err_path = errors_path(out)
    prior, prior_err = read_rows(out), read_rows(err_path)
    done = done_ids(prior)
    todo = [r for r in requests if r.id not in done]
    models = sorted({r.model for r in todo})
    budget = Budget(resolve_budget(budget_usd, models), spent_usd(prior, prior_err))
    for model in models:
        llm.check_route(model)  # fail before anything is spent
    conc = parse_concurrency(concurrency)
    store = llm.resolve_cache(cache)
    cache_arg = store if store is not None else False

    groups: dict[str, list[Request]] = {}
    for r in todo:
        groups.setdefault(r.key, []).append(r)
    calls = f"{len(groups)} distinct call" + ("" if len(groups) == 1 else "s")
    print(f"{out.name}: {len(requests)} requests, {len(requests) - len(todo)} already done, {len(todo)} to run"
          f" ({calls}); {spend_text(budget)}", flush=True)

    ok_out, err_out = Appender(out), Appender(err_path)
    stop = threading.Event()
    run_spend = [0.0]
    spend_lock = threading.Lock()

    def work(group: list[Request]) -> tuple[str, int]:
        head = group[0]
        if stop.is_set():
            return "skipped", len(group)
        res = llm.cached_result(head.model, head.system, head.user, head.max_tokens, head.temperature,
                                head.replicate, cache=cache_arg)
        if res is None:
            if budget.exhausted:
                return "skipped", len(group)
            res = llm.complete(head.model, head.system, head.user, head.max_tokens, head.temperature,
                               head.replicate, cache=cache_arg)
            budget.add(res.cost_usd)
            with spend_lock:
                run_spend[0] += res.cost_usd
        writer = ok_out if res.error is None else err_out
        for j, r in enumerate(group):
            shared = res if j == 0 else replace(res, cost_usd=0.0, cached=res.error is None)
            writer.write(result_row(r, shared))
        return ("error" if res.error else "cached" if res.cached else "ok"), len(group)

    lanes: dict[str, list[list[Request]]] = {}
    for group in groups.values():
        lanes.setdefault(lane_of(group[0].model, conc), []).append(group)
    pools = {name: ThreadPoolExecutor(max_workers=conc[name], thread_name_prefix="swarm-" + name.replace("/", "-"))
             for name in lanes}
    counts: Counter = Counter()
    t0 = last = time.monotonic()
    announced = False
    try:
        futures = [pools[name].submit(work, g) for name, gs in lanes.items() for g in gs]
        finished = 0
        for fut in as_completed(futures):
            status, n = fut.result()
            counts[status] += n
            finished += n
            if budget.exhausted and not announced and not math.isinf(budget.limit):
                announced = True
                print(f"  budget reached: {spend_text(budget)}, so no new call will start", flush=True)
            now = time.monotonic()
            if now - last >= progress_every_s or finished == len(todo):
                last = now
                print(f"  [{finished}/{len(todo)}] {counts['ok']} ok, {counts['cached']} cached, {counts['error']}"
                      f" errors, {counts['skipped']} not started; ${run_spend[0]:.4f} this run, {spend_text(budget)}"
                      f" in all; {now - t0:.0f}s", flush=True)
    except BaseException:
        stop.set()
        for pool in pools.values():
            pool.shutdown(wait=True, cancel_futures=True)
        raise
    finally:
        for pool in pools.values():
            pool.shutdown(wait=True)
        ok_out.close()
        err_out.close()

    summary = {
        "requests": len(requests),
        "already_done": len(requests) - len(todo),
        "ok": counts["ok"],
        "cached": counts["cached"],
        "errors": counts["error"],
        "skipped_for_budget": counts["skipped"],
        "stopped_on_budget": counts["skipped"] > 0,
        "spent_usd_this_run": round(run_spend[0], 8),
        "spent_usd_total": round(budget.spent, 8),
        "budget_usd": None if math.isinf(budget.limit) else budget.limit,
        "out": str(out),
        "errors_file": str(err_path) if counts["error"] else None,
    }
    tail = f"; {counts['error']} failed (see {err_path.name}, rerun to retry)" if counts["error"] else ""
    if counts["skipped"]:
        tail += f"; {counts['skipped']} not started because the budget is spent"
    print(f"{out.name}: {counts['ok']} new, {counts['cached']} from cache{tail}. ${run_spend[0]:.4f} this run,"
          f" {spend_text(budget)} in all.", flush=True)
    return summary


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m swarm.batch", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("requests", help="request JSONL")
    p.add_argument("out", help="output JSONL, appended to; failed calls go to <stem>.errors.jsonl")
    p.add_argument("--concurrency", default=None,
                   help='workers per provider or model, e.g. "claude=8,vllm=32,vllm/gptoss=16", or one number for all')
    p.add_argument("--budget", type=float, default=None, help="USD cap for this output file (default SWARM_BUDGET_USD)")
    p.add_argument("--limit", type=int, default=0, help="run only the first N requests, for a smoke test")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    try:
        conc = parse_concurrency(a.concurrency)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 3
    try:
        s = run_requests(a.requests, a.out, conc, budget_usd=a.budget, limit=a.limit or None)
    except (ConfigError, RequestFileError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 3
    if s["stopped_on_budget"]:
        return 2
    return 1 if s["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
