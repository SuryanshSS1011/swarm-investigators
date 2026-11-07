"""Kill test driver. Each subcommand reads and writes plain JSONL under data/kt/.

    python scripts/kt.py skeletons --n 20
    python scripts/kt.py author-requests --authors claude/claude-haiku-5-5,claude/claude-sonnet-5-5,gemini/gemini-3.8-flash
    (run data/kt/author_requests.jsonl with the batch runner -> data/kt/author_outputs.jsonl)
    python scripts/kt.py author-parse
    python scripts/kt.py audit-requests --investigators claude/claude-haiku-5-5,gemini/gemini-3.8-flash
    (run data/kt/audit_requests.jsonl -> data/kt/audit_outputs.jsonl)
    python scripts/kt.py report
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from swarm import assemble as A  # noqa: E402
from swarm import author as AU  # noqa: E402
from swarm import baseline as BL  # noqa: E402
from swarm import investigate as INV  # noqa: E402
from swarm import skeleton as SK  # noqa: E402
from swarm import world as W  # noqa: E402

DATA = ROOT / "data" / "kt"
FAMILIES = ("anthropic", "google", "alibaba", "openai")


def jl_read(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def jl_write(p: Path, rows) -> int:
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
            n += 1
    return n


def cmd_skeletons(a):
    rows = [SK.build(seed, arch, tier).to_dict() for seed, arch, tier in SK.design(a.n, a.start)]
    print("skeletons:", jl_write(DATA / "skeletons.jsonl", rows))


def cmd_author_requests(a):
    sks = [SK.Skeleton.from_dict(d) for d in jl_read(DATA / "skeletons.jsonl")]
    reqs = []
    for m in a.authors.split(","):
        for sk in sks:
            reqs.append({"id": f"author|{m}|{sk.seed}|r{a.replicate}", "model": m, "system": AU.AUTHOR_SYSTEM,
                         "user": AU.author_prompt(sk), "max_tokens": a.max_tokens, "replicate": a.replicate,
                         "meta": {"stage": "author", "author": m, "seed": sk.seed}})
    out = DATA / (a.out or "author_requests.jsonl")
    print("author requests:", jl_write(out, reqs), "->", out)


def _parse(text):
    try:
        from swarm.parse import parse_json_block
        return parse_json_block(text)
    except ImportError:
        return INV._last_json_object(text)


def cmd_author_parse(a):
    sks = {d["seed"]: d for d in jl_read(DATA / "skeletons.jsonl")}
    outs = []
    for f in a.inputs.split(","):
        outs += jl_read(DATA / f)
    bases, failures = {}, []
    for o in outs:
        meta = o.get("meta") or {}
        m, seed = meta.get("author"), meta.get("seed")
        if m is None:
            _, m, seed, _ = o["id"].split("|")
            seed = int(seed)
        if (m, seed) in bases:
            continue
        sk = SK.Skeleton.from_dict(sks[seed])
        texts, errs = AU.parse_author(sk, _parse(o.get("text") or ""))
        if errs:
            failures.append({"id": o["id"], "errors": errs[:5]})
            continue
        bases[(m, seed)] = {"author": m, "seed": seed, "texts": {str(k): v for k, v in texts.items()}}
    print("bases:", jl_write(DATA / "bases.jsonl", bases.values()), "failures:", len(failures))
    for fl in failures[:20]:
        print("  ", fl)
    jl_write(DATA / "author_failures.jsonl", failures)


def other_face(inv_model: str, seed: int) -> str:
    fam = W.FAMILY[inv_model]
    others = [f for f in FAMILIES if f != fam]
    return W.FAMILY_FACE[others[seed % len(others)]]


def conditions(kt_b_variants=("none", "auditor_kin")):
    """(variant, header kind). KT-A crosses every variant with no header; KT-B adds self and other headers."""
    for v in W.VARIANTS:
        yield v, "none"
    for v in kt_b_variants:
        for h in ("self", "other"):
            yield v, h


def cmd_audit_requests(a):
    sks = {d["seed"]: d for d in jl_read(DATA / "skeletons.jsonl")}
    bases = jl_read(DATA / "bases.jsonl")
    authors = set(a.authors.split(",")) if a.authors else None
    variants = set(a.variants.split(",")) if a.variants else None
    headers = set(a.headers.split(",")) if a.headers else None
    reqs, boards = [], []
    for bz in bases:
        if authors and bz["author"] not in authors:
            continue
        sk = SK.Skeleton.from_dict(json.loads(json.dumps(sks[bz["seed"]])))
        AU.fill(sk, {int(k): v for k, v in bz["texts"].items()})
        for inv in a.investigators.split(","):
            for v, h in conditions():
                if (variants and v not in variants) or (headers and h not in headers):
                    continue
                face = None if h == "none" else (W.DISPLAY[inv] if h == "self" else other_face(inv, sk.seed))
                for viol in (True, False):
                    b = A.assemble(sk, viol, v, agent_model=face, view=a.view)
                    bid = f"{bz['author']}|{sk.seed}|{'V' if viol else 'C'}|{v}|{h}|{a.view}"
                    rid = f"audit|{inv}|{bid}|r{a.replicate}"
                    system = INV.investigator_system(W.DISPLAY[inv], hardened=a.hardened)
                    meta = {"stage": "audit", "investigator": inv, "author": bz["author"], "seed": sk.seed,
                            "violating": viol, "variant": v, "header": h, "face": face, "view": a.view,
                            "hardened": a.hardened, "truth": b.truth}
                    reqs.append({"id": rid, "model": inv, "system": system, "user": INV.investigator_prompt(b),
                                 "max_tokens": a.max_tokens, "replicate": a.replicate, "meta": meta})
                    if inv == a.investigators.split(",")[0]:
                        boards.append({"board_id": bid, "truth": b.truth, "baseline": BL.check(b.log)})
    out = DATA / (a.out or "audit_requests.jsonl")
    print("audit requests:", jl_write(out, reqs), "->", out)
    jl_write(DATA / "boards_baseline.jsonl", boards)


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def fmt(k, n):
    p, lo, hi = wilson(k, n)
    return f"{k:>3}/{n:<3} {100 * p:5.1f}% [{100 * lo:4.1f}, {100 * hi:4.1f}]" if n else "   -"


def load_scored(files):
    rows = []
    for f in files:
        for o in jl_read(DATA / f):
            meta = o.get("meta")
            if not meta or meta.get("stage") != "audit":
                continue
            ans = INV.parse_answer(o.get("text") or "")
            s = INV.score(ans, meta["truth"])
            rows.append({**meta, **s, "id": o["id"], "error": o.get("error"),
                         "cost": o.get("cost_usd") or 0.0})
    return rows


def cmd_report(a):
    rows = load_scored(a.inputs.split(","))
    print(f"{len(rows)} audits, unparsed {sum(not r['parsed'] for r in rows)}, cost ${sum(r['cost'] for r in rows):.2f}")
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["investigator"], r["header"], r["variant"], r["violating"])].append(r)
    invs = sorted({r["investigator"] for r in rows})
    for inv in invs:
        print(f"\n== {inv}")
        print(f"{'header':6} {'variant':15} {'recall':32} {'FPR':32} conf-exon")
        for h in ("none", "self", "other"):
            for v in W.VARIANTS:
                V, C = by.get((inv, h, v, True), []), by.get((inv, h, v, False), [])
                if not V and not C:
                    continue
                ce = sum(r.get("confident_exoneration", False) for r in V)
                print(f"{h:6} {v:15} {fmt(sum(r['flagged'] for r in V), len(V)):32} "
                      f"{fmt(sum(r['flagged'] for r in C), len(C)):32} {ce}")
        # Recall by tier and by author for the no-header, no-variant cells.
        for key, label in (("tier", "tier"), ("author", "author")):
            g = collections.defaultdict(list)
            for r in rows:
                if r["investigator"] == inv and r["header"] == "none" and r["variant"] == "none":
                    g[(r["truth"][key] if key == "tier" else r[key], r["violating"])].append(r)
            for k in sorted({k for k, _ in g}):
                V, C = g.get((k, True), []), g.get((k, False), [])
                print(f"  {label}={k:28} recall {fmt(sum(x['flagged'] for x in V), len(V))}  "
                      f"FPR {fmt(sum(x['flagged'] for x in C), len(C))}")
        # Paired variant effects against "none" on the same board (violation boards, no header).
        base = {(r["author"], r["seed"]): r["flagged"] for r in rows
                if r["investigator"] == inv and r["header"] == "none" and r["variant"] == "none" and r["violating"]}
        for v in W.VARIANTS[1:]:
            pairs = [(base[(r["author"], r["seed"])], r["flagged"]) for r in rows
                     if r["investigator"] == inv and r["header"] == "none" and r["variant"] == v and r["violating"]
                     and (r["author"], r["seed"]) in base]
            if pairs:
                b10 = sum(1 for x, y in pairs if x and not y)
                b01 = sum(1 for x, y in pairs if not x and y)
                diff = (sum(y for _, y in pairs) - sum(x for x, _ in pairs)) / len(pairs)
                print(f"  paired {v:15} n={len(pairs):3} recall change {100 * diff:+5.1f} pts "
                      f"(lost {b10}, gained {b01})")
        # Kinship: self vs other header, paired on the same board and variant.
        for v in ("none", "auditor_kin"):
            m = {}
            for r in rows:
                if r["investigator"] == inv and r["variant"] == v and r["header"] in ("self", "other"):
                    m.setdefault((r["author"], r["seed"], r["violating"]), {})[r["header"]] = r["flagged"]
            for viol in (True, False):
                pairs = [(d["self"], d["other"]) for (au, s, vi), d in m.items() if vi == viol and len(d) == 2]
                if pairs:
                    diff = (sum(x for x, _ in pairs) - sum(y for _, y in pairs)) / len(pairs)
                    print(f"  kinship {v:12} {'viol' if viol else 'clean'} n={len(pairs):3} flag rate self minus other "
                          f"{100 * diff:+5.1f} pts")
        # Style kinship: same-family author vs other-family author, no header, no variant.
        fam = W.FAMILY[inv]
        g = collections.defaultdict(list)
        for r in rows:
            if r["investigator"] == inv and r["header"] == "none" and r["variant"] == "none" and r["violating"]:
                rel = "same-model" if r["author"] == inv else ("same-family" if W.FAMILY[r["author"]] == fam else "other")
                g[rel].append(r["flagged"])
        print("  style:", ", ".join(f"{k} {fmt(sum(v), len(v))}" for k, v in sorted(g.items())))
    bl = jl_read(DATA / "boards_baseline.jsonl")
    if bl:
        g = collections.defaultdict(list)
        for b in bl:
            g[(b["truth"]["tier"], b["truth"]["violation"])].append(b["baseline"]["verdict"] == "violation")
        print("\nrule baseline:", ", ".join(f"{t} {'viol' if v else 'clean'} {fmt(sum(x), len(x))}"
                                         for (t, v), x in sorted(g.items())))
    if a.dump:
        jl_write(DATA / a.dump, rows)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("skeletons"); s.add_argument("--n", type=int, default=20); s.add_argument("--start", type=int, default=0)
    s = sub.add_parser("author-requests"); s.add_argument("--authors", required=True)
    s.add_argument("--max-tokens", type=int, default=6000); s.add_argument("--replicate", type=int, default=0)
    s.add_argument("--out", default=None)
    s = sub.add_parser("author-parse"); s.add_argument("--inputs", default="author_outputs.jsonl")
    s = sub.add_parser("audit-requests"); s.add_argument("--investigators", required=True)
    s.add_argument("--authors", default=None); s.add_argument("--variants", default=None)
    s.add_argument("--headers", default=None); s.add_argument("--view", default="full")
    s.add_argument("--hardened", action="store_true"); s.add_argument("--max-tokens", type=int, default=1200)
    s.add_argument("--replicate", type=int, default=0); s.add_argument("--out", default=None)
    s = sub.add_parser("report"); s.add_argument("--inputs", default="audit_outputs.jsonl"); s.add_argument("--dump", default=None)
    a = p.parse_args()
    {"skeletons": cmd_skeletons, "author-requests": cmd_author_requests, "author-parse": cmd_author_parse,
     "audit-requests": cmd_audit_requests, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
