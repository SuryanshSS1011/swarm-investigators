"""Pre-registered analysis of the main study (docs/prereg_main.md), plus clearly labeled exploratory analyses.

    python scripts/study.py score
    python scripts/analyze.py            # writes results/main/summary.json and results/main/tables.md

Every contrast is paired on the same board: (investigator, author, seed, violation status, arm, header).
Intervals come from a cluster bootstrap that resamples seeds. Pooled tests are Mantel-Haenszel tests for matched
pairs stratified by investigator, which reduce to a sign test on the discordant pairs summed over strata.
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from swarm import world as W  # noqa: E402

DATA = ROOT / "data" / "main"
OUT = ROOT / "results" / "main"
RNG = np.random.default_rng(20261107)
B = 2000
FULL_CROSS = ["claude/claude-haiku-5-5", "gemini/gemini-3.8-flash", "vllm/qwen", "vllm/gptoss"]
FAKE_RECORD = {("hidden_answers", "explicit"), ("scorer_tamper", "explicit"), ("misreport", "explicit"),
               ("misreport", "implicit"), ("transcript_tamper", "explicit")}


def short(m: str) -> str:
    return W.DISPLAY.get(m, m)


def load() -> list[dict]:
    return [json.loads(l) for l in (DATA / "scored.jsonl").read_text().splitlines() if l.strip()]


def pairs(rows, sel_a, sel_b, outcome="flagged", key_extra=()):
    """Matched pairs (investigator, seed, a, b) for rows matching sel_a and sel_b on the same board key."""
    def key(r):
        return (r["investigator"], r["author"], r["seed"], r["violating"], r["header"]) + tuple(r[k] for k in key_extra)
    A = {key(r): r for r in rows if sel_a(r)}
    out = []
    for r in rows:
        if sel_b(r):
            k = key(r)
            if k in A:
                out.append((r["investigator"], r["seed"], bool(A[k][outcome]), bool(r[outcome])))
    return out


def paired_stats(ps, alternative="two-sided"):
    """Difference b-minus-a in rate, MH matched-pairs p-value, bootstrap CI over seeds."""
    if not ps:
        return None
    a = np.array([p[2] for p in ps], float)
    b = np.array([p[3] for p in ps], float)
    diff = float(b.mean() - a.mean())
    n10 = int(((a == 1) & (b == 0)).sum())  # lost under b
    n01 = int(((a == 0) & (b == 1)).sum())  # gained under b
    if n10 + n01 == 0:
        p = 1.0
    else:
        alt = {"less": "greater", "greater": "less", "two-sided": "two-sided"}[alternative]
        # Under H0 a discordant pair is equally likely to go either way. "less" means b below a, so n10 large.
        p = stats.binomtest(n10, n10 + n01, 0.5, alternative=alt).pvalue
    seeds = np.array([p_[1] for p_ in ps])
    uniq = np.unique(seeds)
    idx = {s: np.where(seeds == s)[0] for s in uniq}
    boots = []
    for _ in range(B):
        take = np.concatenate([idx[s] for s in RNG.choice(uniq, len(uniq), replace=True)])
        boots.append(b[take].mean() - a[take].mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    lo90, hi90 = np.percentile(boots, [5, 95])
    return {"n": len(ps), "rate_a": float(a.mean()), "rate_b": float(b.mean()), "diff": diff, "lost": n10,
            "gained": n01, "p": float(p), "ci95": [float(lo), float(hi)], "ci90": [float(lo90), float(hi90)]}


def by_investigator(ps, alternative="two-sided"):
    g = collections.defaultdict(list)
    for p in ps:
        g[p[0]].append(p)
    return {inv: paired_stats(v, alternative) for inv, v in sorted(g.items())}


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def sel(**kw):
    def f(r):
        return all(r.get(k) == v for k, v in kw.items())
    return f


def contrast(rows, arm_a, var_a, arm_b, var_b, violating, alternative, header="none", extra=None):
    sa = lambda r: r["arm"] == arm_a and r["variant"] == var_a and r["header"] == header and r["violating"] == violating and (extra is None or extra(r))  # noqa: E731
    sb = lambda r: r["arm"] == arm_b and r["variant"] == var_b and r["header"] == header and r["violating"] == violating and (extra is None or extra(r))  # noqa: E731
    if arm_a != arm_b:
        # Across arms the system prompt or view differs, so pair on the board without the arm.
        ps = pairs(rows, sa, sb)
    else:
        ps = pairs(rows, sa, sb)
    return {"pooled": paired_stats(ps, alternative), "by_investigator": by_investigator(ps, alternative)}


def rate_table(rows, arm, variants, header="none"):
    t = {}
    for inv in sorted({r["investigator"] for r in rows if r["arm"] == arm}):
        t[inv] = {}
        for v in variants:
            V = [r for r in rows if r["investigator"] == inv and r["arm"] == arm and r["variant"] == v and r["header"] == header and r["violating"]]
            C = [r for r in rows if r["investigator"] == inv and r["arm"] == arm and r["variant"] == v and r["header"] == header and not r["violating"]]
            t[inv][v] = {"detect": [sum(r["flagged"] for r in V), len(V)], "fpr": [sum(r["flagged"] for r in C), len(C)],
                         "refused": [sum(r["refused"] for r in V + C), len(V + C)],
                         "conf_exon": [sum(bool(r.get("confident_exoneration")) for r in V), len(V)]}
    return t


def style_diagonal(rows, violating: bool):
    """H6: double-centered diagonal of the investigator-by-author matrix (main arm, variant none, no header)."""
    invs = [i for i in FULL_CROSS if any(r["investigator"] == i for r in rows)]
    auths = [a for a in FULL_CROSS if any(r["author"] == a for r in rows)]
    sub = [r for r in rows if r["arm"] == "main" and r["variant"] == "none" and r["header"] == "none"
           and r["violating"] == violating and r["investigator"] in invs and r["author"] in auths]
    if len(invs) < 2 or len(auths) < 2:
        return None
    seeds = sorted({r["seed"] for r in sub})
    cell = collections.defaultdict(dict)
    for r in sub:
        cell[(r["investigator"], r["author"])][r["seed"]] = float(r["flagged"])

    def stat(seed_list):
        M = np.full((len(invs), len(auths)), np.nan)
        for i, inv in enumerate(invs):
            for j, au in enumerate(auths):
                vals = [cell[(inv, au)][s] for s in seed_list if s in cell[(inv, au)]]
                if vals:
                    M[i, j] = np.mean(vals)
        if np.isnan(M).any():
            return np.nan, M
        Mc = M - M.mean(1, keepdims=True) - M.mean(0, keepdims=True) + M.mean()
        diag = [Mc[i, auths.index(inv)] for i, inv in enumerate(invs) if inv in auths]
        return float(np.mean(diag)), M

    point, M = stat(seeds)
    boots = [stat(list(RNG.choice(seeds, len(seeds), replace=True)))[0] for _ in range(B)]
    boots = [b for b in boots if not np.isnan(b)]
    return {"investigators": invs, "authors": auths, "matrix": M.tolist(), "diag_effect": point,
            "ci95": np.percentile(boots, [2.5, 97.5]).tolist() if boots else None,
            "ci90": np.percentile(boots, [5, 95]).tolist() if boots else None}


def main() -> None:
    rows = load()
    for r in rows:
        r["spoof_kind"] = "fake_record" if (r["archetype"], r["tier"]) in FAKE_RECORD else "false_correction"
    S = {"n_audits": len(rows), "investigators": sorted({r["investigator"] for r in rows}),
         "authors": sorted({r["author"] for r in rows if r["author"] != "none"})}
    S["rates_main"] = rate_table(rows, "main", ["none", "neutral", "spoof", "auditor_auth", "auditor_kin", "auditor_inject"])
    S["rates_d1"] = rate_table(rows, "d1", ["none", "spoof", "auditor_auth", "auditor_kin", "auditor_inject"])

    # Primary family: H1 and H2a to H2c, one-sided (detection lower than with neutral), Holm-corrected.
    prim = {}
    for name, v in (("H1_spoof", "spoof"), ("H2a_auditor_auth", "auditor_auth"), ("H2b_auditor_kin", "auditor_kin"),
                    ("H2c_auditor_inject", "auditor_inject")):
        prim[name] = {"detection": contrast(rows, "main", "neutral", "main", v, True, "less"),
                      "fpr": contrast(rows, "main", "neutral", "main", v, False, "two-sided")}
    adj = holm({k: v["detection"]["pooled"]["p"] for k, v in prim.items() if v["detection"]["pooled"]})
    for k in prim:
        if k in adj:
            prim[k]["detection"]["pooled"]["p_holm"] = adj[k]
    S["primary"] = prim

    # H3: refusals among Claude investigators, auditor variants against neutral (one-sided, more refusals).
    claude = lambda r: r["investigator"].startswith("claude/")  # noqa: E731
    S["H3_refusal"] = {}
    for v in ("auditor_auth", "auditor_kin"):
        ps = pairs(rows, lambda r: r["arm"] == "main" and r["variant"] == "neutral" and r["header"] == "none" and claude(r),
                   lambda r, v=v: r["arm"] == "main" and r["variant"] == v and r["header"] == "none" and claude(r),
                   outcome="refused")
        S["H3_refusal"][v] = {"pooled": paired_stats(ps, "greater"), "by_investigator": by_investigator(ps, "greater")}

    # H4: spotlight, neutral against none, detection and FPR (one-sided, higher).
    S["H4_spotlight"] = {"detection": contrast(rows, "main", "none", "main", "neutral", True, "greater"),
                         "fpr": contrast(rows, "main", "none", "main", "neutral", False, "greater")}

    # H5: kinship label, self against other, two-sided, with a +-5 point equivalence bound on the 90% interval.
    S["H5_kinship"] = {}
    for v in ("none", "auditor_kin"):
        for viol, lab in ((True, "detection"), (False, "fpr")):
            ps = []
            for inv in {r["investigator"] for r in rows if r["arm"] == "kinship"}:
                A = {(r["author"], r["seed"]): r for r in rows if r["arm"] == "kinship" and r["investigator"] == inv
                     and r["variant"] == v and r["header"] == "other" and r["violating"] == viol}
                for r in rows:
                    if r["arm"] == "kinship" and r["investigator"] == inv and r["variant"] == v and r["header"] == "self" \
                            and r["violating"] == viol and (r["author"], r["seed"]) in A:
                        ps.append((inv, r["seed"], bool(A[(r["author"], r["seed"])]["flagged"]), bool(r["flagged"])))
            st = paired_stats(ps, "two-sided")
            if st:
                st["equivalent_within_5pts"] = bool(st["ci90"][0] > -0.05 and st["ci90"][1] < 0.05)
            S["H5_kinship"][f"{v}_{lab}"] = {"pooled": st, "by_investigator": by_investigator(ps, "two-sided")}

    # H6: style kinship, double-centered diagonal.
    S["H6_style"] = {"detection": style_diagonal(rows, True), "fpr": style_diagonal(rows, False)}

    # H7: defense D1 against the same variant without D1 (one-sided, higher detection; FPR two-sided).
    S["H7_d1"] = {}
    for v in ("spoof", "auditor_auth", "auditor_kin", "auditor_inject", "none"):
        S["H7_d1"][v] = {"detection": contrast(rows, "main", v, "d1", v, True, "greater"),
                         "fpr": contrast(rows, "main", v, "d1", v, False, "two-sided")}

    # H8: action-only view by tier, against the main arm with no variant.
    S["H8_d2"] = {}
    for tier in ("explicit", "implicit"):
        g = collections.defaultdict(lambda: [0, 0, 0, 0])
        for r in rows:
            if r["tier"] != tier:
                continue
            if r["arm"] == "d2":
                g[r["investigator"]][0 if r["violating"] else 2] += int(r["flagged"])
                g[r["investigator"]][1 if r["violating"] else 3] += 1
        S["H8_d2"][tier] = {inv: {"detect": [v[0], v[1]], "fpr": [v[2], v[3]]} for inv, v in g.items()}

    # Rule baseline, computed on the main-arm boards with no variant.
    bl = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        if r["arm"] == "main" and r["variant"] == "none" and r["investigator"] == rows[0]["investigator"]:
            k = (r["tier"], r["violating"])
            bl[k][0] += int(r["baseline"] == "violation")
            bl[k][1] += 1
    S["baseline"] = {f"{t}_{'viol' if v else 'clean'}": x for (t, v), x in bl.items()}

    # Exploratory: the cover story split by kind (fake tool record or false correction), and D1 on each kind.
    S["explore_spoof_kind"] = {}
    for kind in ("fake_record", "false_correction"):
        ex = lambda r, kind=kind: r["spoof_kind"] == kind  # noqa: E731
        S["explore_spoof_kind"][kind] = {
            "main_neutral_to_spoof": contrast(rows, "main", "neutral", "main", "spoof", True, "two-sided", extra=ex),
            "spoof_main_to_d1": contrast(rows, "main", "spoof", "d1", "spoof", True, "two-sided", extra=ex)}
    # Exploratory: verdicts on missed false-correction boards.
    S["explore_missed_verdicts"] = dict(collections.Counter(
        r["verdict"] for r in rows if r["arm"] == "main" and r["variant"] == "spoof" and r["violating"]
        and r["spoof_kind"] == "false_correction" and not r["flagged"]))
    # Exploratory: cross-family ensemble (flag if either investigator flags), main arm.
    ens = {}
    invs = sorted({r["investigator"] for r in rows if r["arm"] == "main"})
    idx = collections.defaultdict(dict)
    for r in rows:
        if r["arm"] == "main" and r["header"] == "none":
            idx[(r["author"], r["seed"], r["violating"], r["variant"])][r["investigator"]] = r["flagged"]
    for i, a in enumerate(invs):
        for b in invs[i + 1:]:
            if W.FAMILY[a] == W.FAMILY[b]:
                continue
            for v in ("neutral", "spoof"):
                V = [d[a] or d[b] for k, d in idx.items() if k[2] and k[3] == v and a in d and b in d]
                C = [d[a] or d[b] for k, d in idx.items() if not k[2] and k[3] == v and a in d and b in d]
                if V:
                    ens[f"{short(a)} + {short(b)} | {v}"] = {"detect": [sum(V), len(V)], "fpr": [sum(C), len(C)]}
    S["explore_ensemble"] = ens

    # Probe: how often each investigator names the true family, and its own family.
    pr = [json.loads(l) for l in (DATA / "probes.jsonl").read_text().splitlines() if l.strip()] if (DATA / "probes.jsonl").exists() else []
    pt = collections.defaultdict(lambda: collections.Counter())
    for p in pr:
        own = W.FAMILY[p["investigator"]] == p["true_family"]
        pt[p["investigator"]]["n"] += 1
        pt[p["investigator"]]["correct"] += int(p["correct"])
        pt[p["investigator"]]["own_n"] += int(own)
        pt[p["investigator"]]["own_correct"] += int(own and p["correct"])
        pt[p["investigator"]]["other_called_own"] += int((not own) and p["guess"] == W.FAMILY[p["investigator"]])
        pt[p["investigator"]]["other_n"] += int(not own)
    S["probe"] = {k: dict(v) for k, v in pt.items()}

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(S, indent=1))
    print(json.dumps({k: S[k] for k in ("n_audits", "investigators", "authors")}, indent=1))
    print("\nPrimary (detection, neutral -> variant, pooled):")
    for k, v in S["primary"].items():
        p = v["detection"]["pooled"]
        if p:
            print(f"  {k:20} n={p['n']:4} {100*p['rate_a']:5.1f}% -> {100*p['rate_b']:5.1f}%  diff {100*p['diff']:+5.1f} "
                  f"[{100*p['ci95'][0]:+.1f}, {100*p['ci95'][1]:+.1f}]  lost {p['lost']} gained {p['gained']}  p={p['p']:.2g} holm={p.get('p_holm', float('nan')):.2g}")
    for kind, d in S["explore_spoof_kind"].items():
        p = d["main_neutral_to_spoof"]["pooled"]
        q = d["spoof_main_to_d1"]["pooled"]
        if p:
            print(f"  explore {kind:16} neutral->spoof {100*p['rate_a']:.1f}% -> {100*p['rate_b']:.1f}% (n={p['n']})"
                  + (f"; spoof main->D1 {100*q['rate_a']:.1f}% -> {100*q['rate_b']:.1f}% (n={q['n']})" if q else ""))


if __name__ == "__main__":
    main()
