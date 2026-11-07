"""Analysis of exploratory follow-up E3, investigators at high reasoning (docs/explore_reasoning.md).

    python scripts/analyze_reasoning.py      (reads data/main/scored.jsonl, writes results/main/reasoning.json)

Each high-reasoning audit is paired with the same model's main-arm audit of the same board and variant at the floor,
on the five retraction cells with Haiku- and Gemini 3.8 Flash-written boards. Contrasts use the same paired
bootstrap and exact McNemar test as the replication. Nothing here is pooled into the main study.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_replication import DRAWS, contrast  # noqa: E402
from swarm import world as W  # noqa: E402

RETRACTION = {tuple(c) for c in W.FALSE_CORRECTION_CELLS}
AUTHORS = {"claude/claude-haiku-5-5", "gemini/gemini-3.8-flash"}
ANALYZED = ("claude/claude-haiku-5-5", "claude/claude-sonnet-5-5", "claude/claude-opus-5-5")
STOPPED = ("gemini/gemini-3.8-flash",)  # stopped early, see the protocol's deviation note
TRUNCATED = ("max_tokens", "length")


def did(rows_high, rows_floor):
    """Mean over boards of (retraction minus neutral at high) minus (retraction minus neutral at the floor)."""
    cell = {}
    for lvl, rows in (("h", rows_high), ("f", rows_floor)):
        for r in rows:
            cell.setdefault((r["investigator"], r["author"], r["seed"]), {})[(lvl, r["variant"])] = r["flagged"]
    need = [("h", "neutral"), ("h", "spoof"), ("f", "neutral"), ("f", "spoof")]
    by_seed = {}
    for (inv, au, seed), v in cell.items():
        if all(k in v for k in need):
            d = (v[("h", "spoof")] - v[("h", "neutral")]) - (v[("f", "spoof")] - v[("f", "neutral")])
            by_seed.setdefault(seed, []).append(d)
    vals = [d for ds in by_seed.values() for d in ds]
    if not vals:
        return {"n": 0}
    seeds = sorted(by_seed)
    rng = random.Random(20261107)
    boots = []
    for _ in range(DRAWS):
        pick = [d for s in (rng.choice(seeds) for _ in seeds) for d in by_seed[s]]
        boots.append(sum(pick) / len(pick))
    boots.sort()
    q = lambda f: boots[min(len(boots) - 1, max(0, int(round(f * (len(boots) - 1)))))]  # noqa: E731
    return {"n": len(vals), "diff": sum(vals) / len(vals), "ci95": [q(0.025), q(0.975)]}


def main() -> None:
    rows = [json.loads(l) for l in (ROOT / "data/main/scored.jsonl").read_text().splitlines() if l.strip()]
    keep = lambda r: (r["archetype"], r["tier"]) in RETRACTION and r["author"] in AUTHORS and r["header"] == "none"  # noqa: E731
    high_all = [r for r in rows if r["arm"] == "e3_high"]
    high, floor = [], []
    for r in high_all:
        base = r["investigator"].split("@")[0]
        if base in ANALYZED and keep(r):
            high.append({**r, "investigator": base})
    boards = {(r["investigator"], r["author"], r["seed"]) for r in high}
    for r in rows:
        if (r["arm"] == "main" and r["investigator"] in ANALYZED and keep(r) and r["variant"] in ("neutral", "spoof")
                and (r["investigator"], r["author"], r["seed"]) in boards):
            floor.append(r)
    sel = lambda xs, v, viol, inv=None: [r for r in xs if r["variant"] == v and r["violating"] == viol  # noqa: E731
                                         and (inv is None or r["investigator"] == inv)]
    lenient = lambda r: r["verdict"] in ("violation", "insufficient_evidence")  # noqa: E731

    def block(inv=None):
        return {
            "retraction_high": contrast(sel(high, "neutral", True, inv), sel(high, "spoof", True, inv)),
            "retraction_high_lenient": contrast(sel(high, "neutral", True, inv), sel(high, "spoof", True, inv),
                                                fn=lenient),
            "retraction_floor": contrast(sel(floor, "neutral", True, inv), sel(floor, "spoof", True, inv)),
            "neutral_high_vs_floor": contrast(sel(floor, "neutral", True, inv), sel(high, "neutral", True, inv)),
            "spoof_high_vs_floor": contrast(sel(floor, "spoof", True, inv), sel(high, "spoof", True, inv)),
            "fpr_neutral_high_vs_floor": contrast(sel(floor, "neutral", False, inv), sel(high, "neutral", False, inv)),
            "fpr_spoof_high_vs_floor": contrast(sel(floor, "spoof", False, inv), sel(high, "spoof", False, inv)),
            "did": did([r for r in high if r["violating"] and (inv is None or r["investigator"] == inv)],
                       [r for r in floor if r["violating"] and (inv is None or r["investigator"] == inv)]),
        }

    out = {"pooled": block(), "by_investigator": {inv: block(inv) for inv in ANALYZED
                                                  if any(r["investigator"] == inv for r in high)}}
    e3a = out["pooled"]["retraction_high"]
    out["E3a_supported"] = bool(e3a.get("n") and e3a["diff"] <= -0.05 and e3a["ci95"][1] < 0)
    out["audits"] = {}
    for inv in ANALYZED + STOPPED:
        x = [r for r in high_all if r["investigator"].split("@")[0] == inv]
        raw = ROOT / "data/main/out" / f"{inv.replace('/', '_')}@high_e3.jsonl"
        toks = [json.loads(l)["output_tokens"] for l in raw.read_text().splitlines() if l.strip()] if raw.exists() else []
        nv = [r for r in x if r["variant"] == "neutral" and r["violating"] and not r.get("refused")]
        out["audits"][inv] = {"n": len(x), "refused": sum(bool(r.get("refused")) for r in x),
                              "refused_neutral": sum(bool(r.get("refused")) for r in x if r["variant"] == "neutral"),
                              # unparsed answers other than refusals, which are counted above
                              "unparsed": sum(not r.get("parsed", True) and not r.get("refused") for r in x),
                              "truncated": sum(r.get("stop_reason") in TRUNCATED for r in x),
                              "mean_output_tokens": round(sum(toks) / len(toks)) if toks else None,
                              "neutral_violation_answered": [sum(r["flagged"] for r in nv), len(nv)],
                              "cost_usd": round(sum(r.get("cost_usd") or 0 for r in x), 4),
                              "analyzed": inv in ANALYZED}
    dest = ROOT / "results/main/reasoning.json"
    dest.write_text(json.dumps(out, indent=1))
    for name, b in [("pooled", out["pooled"])] + list(out["by_investigator"].items()):
        h, f, d = b["retraction_high"], b["retraction_floor"], b["did"]
        if h.get("n"):
            print(f"{name:26s} high: {100 * h['rate_a']:.0f}% -> {100 * h['rate_b']:.0f}% ({100 * h['diff']:+.1f}, "
                  f"95% CI {100 * h['ci95'][0]:+.1f} to {100 * h['ci95'][1]:+.1f}, n={h['n']})   floor: "
                  f"{100 * f['rate_a']:.0f}% -> {100 * f['rate_b']:.0f}%   change in effect {100 * d['diff']:+.1f} "
                  f"[{100 * d['ci95'][0]:+.1f}, {100 * d['ci95'][1]:+.1f}]")
    print("E3a supported:", out["E3a_supported"], "| audits:", out["audits"])
    print("wrote", dest.relative_to(ROOT))


if __name__ == "__main__":
    main()
