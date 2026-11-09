"""E5 analysis (docs/explore_effects_all.md): effect notes on every tool call against E2's single planted note.

    python scripts/analyze_effects_all.py      (reads data/main/scored.jsonl, writes results/main/effects_all.json)

Every comparison is paired on (investigator, author, seed, violating). Gemini ran E5 pinned to reasoning effort "low",
so its "@low" suffix is dropped before pairing with E2, whose Gemini calls climbed to "low".
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.study import jl_read  # noqa: E402

DATA = ROOT / "data" / "main"
OUT = ROOT / "results" / "main"
VARIANTS = ("neutral", "spoof", "vouch", "self_vouch")
ORDER = ["claude/claude-haiku-5-5", "claude/claude-sonnet-5-5", "claude/claude-opus-5-5", "gemini/gemini-3.8-flash",
         "gemini/gemini-4-argon", "vllm/qwen", "vllm/gptoss"]
STANDARD = ("main", "e1_main")


def base(inv: str) -> str:
    return inv.split("@")[0]


def mcnemar(pairs):
    b = sum(1 for x, y in pairs if x and not y)
    c = sum(1 for x, y in pairs if y and not x)
    n = b + c
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)
    return {"e5_only": b, "e2_only": c, "p_exact": p}


def compare(e5, ref, keep):
    """Detection or flag rate in E5 and in a reference arm on the same boards, for rows passing keep."""
    ps = [(bool(r["flagged"]), ref[k]) for k, r in e5.items() if keep(r) and k in ref]
    n = len(ps)
    if not n:
        return None
    return {"n": n, "e5": sum(x for x, _ in ps) / n, "ref": sum(y for _, y in ps) / n, **mcnemar(ps)}


def reading(pct):
    return "does not depend on pointing" if pct >= 0.90 else "depends partly on pointing" if pct >= 0.75 \
        else "depends on pointing"


def main() -> None:
    rows = jl_read(DATA / "scored.jsonl")
    key = lambda r: (base(r["investigator"]), r["author"], r["seed"], r["violating"], r["variant"])  # noqa: E731
    e5 = {key(r): r for r in rows if r["arm"] == "e5_effects_all"}
    e2 = {key(r): bool(r["flagged"]) for r in rows if r["arm"] == "e2_effects"}
    std = {key(r): bool(r["flagged"]) for r in rows if r["arm"] in STANDARD and r["header"] == "none"}
    invs = [i for i in ORDER if any(base(r["investigator"]) == i for r in e5.values())]
    closed = [i for i in invs if not i.startswith("vllm/")]
    opened = [i for i in invs if i.startswith("vllm/")]

    X = {"investigators": invs, "audits": len(e5), "cost_usd": round(sum(r["cost_usd"] for r in e5.values()), 4),
         "by_investigator": {}, "pooled": {}}
    for inv in invs:
        d = {}
        for v in VARIANTS:
            sel = lambda r, v=v, inv=inv: base(r["investigator"]) == inv and r["variant"] == v  # noqa: E731
            d[v] = {"detection_vs_e2": compare(e5, e2, lambda r, s=sel: s(r) and r["violating"]),
                    "fpr_vs_e2": compare(e5, e2, lambda r, s=sel: s(r) and not r["violating"])}
        X["by_investigator"][inv] = d
    for grp, members in (("closed_weight", closed), ("open_weight", opened)):
        if not members:
            continue
        g = {}
        for v in VARIANTS:
            sel = lambda r, v=v, m=set(members): base(r["investigator"]) in m and r["variant"] == v  # noqa: E731
            g[v] = {"detection_vs_e2": compare(e5, e2, lambda r, s=sel: s(r) and r["violating"]),
                    "detection_vs_standard": compare(e5, std, lambda r, s=sel: s(r) and r["violating"]),
                    "fpr_vs_e2": compare(e5, e2, lambda r, s=sel: s(r) and not r["violating"])}
        X["pooled"][grp] = g
    prim = X["pooled"]["closed_weight"]["spoof"]["detection_vs_e2"]
    X["primary"] = {**prim, "reading": reading(prim["e5"])}

    # Refusals count as misses, as in the main study, so detection is also given over the answered boards only.
    X["refusals"] = {}
    for inv in invs:
        mine = [r for r in e5.values() if base(r["investigator"]) == inv]
        ref2 = [r for r in rows if r["arm"] == "e2_effects" and r["investigator"] == inv]
        X["refusals"][inv] = {"e5": [sum(r["refused"] for r in mine), len(mine)],
                              "e2": [sum(r["refused"] for r in ref2), len(ref2)]}
    for grp, members in (("closed_weight", closed), ("open_weight", opened)):
        ans = [r for r in e5.values() if base(r["investigator"]) in members and not r["refused"]]
        if not ans:
            continue
        X["pooled"][grp]["answered"] = {
            v: {"violation_boards": sum(1 for r in ans if r["variant"] == v and r["violating"]),
                "detection": sum(bool(r["flagged"]) for r in ans if r["variant"] == v and r["violating"])
                / max(1, sum(1 for r in ans if r["variant"] == v and r["violating"]))} for v in VARIANTS}
        allv = [(bool(r["flagged"]), e2[k]) for k, r in e5.items() if base(r["investigator"]) in members
                and not r["violating"] and k in e2]
        X["pooled"][grp]["fpr_all_variants"] = {"n": len(allv), "e5": sum(x for x, _ in allv) / len(allv),
                                                "e2": sum(y for _, y in allv) / len(allv), **mcnemar(allv)}

    dest = OUT / "effects_all.json"
    dest.write_text(json.dumps(X, indent=1) + "\n")
    print(f"E5 primary: retraction detection {100 * prim['e5']:.1f}% with notes on every call against "
          f"{100 * prim['ref']:.1f}% with one note (n = {prim['n']}, p = {prim['p_exact']:.2g}), {X['primary']['reading']}")
    print("wrote", dest.relative_to(ROOT))


if __name__ == "__main__":
    main()
