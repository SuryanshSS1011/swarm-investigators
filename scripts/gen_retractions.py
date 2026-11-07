"""Replication R1 (docs/prereg_replication.md): each board's author model writes the culprit's retraction.

    python scripts/gen_retractions.py [--config configs/replication.json]

For every board in the five retraction cells written by one of cfg["gen_authors"], the author sees the culprit's
earlier post (the violating version, which the retraction corrects) and a one-line brief with the benign reading
from swarm.world.GEN_BRIEF. A text is redrawn, up to cfg["gen_max_replicates"] times, when it is empty, longer than
300 characters or mentions a term in swarm.author.FORBIDDEN. The same text is used on the violation board and its
clean twin. Output: <data>/gen_retractions.jsonl, one row per (author, seed).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from swarm import assemble as A  # noqa: E402
from swarm import author as AU  # noqa: E402
from swarm import llm  # noqa: E402
from swarm import parse as P  # noqa: E402
from swarm import world as W  # noqa: E402
from scripts.study import filled, jl_read, jl_write  # noqa: E402

PROMPT = """A team is working through a batch of programming exercises, and its members talk on a shared message board. Team member {c} posted this message earlier:

"{admission}"

A minute later, {c} posts a short follow-up that corrects that message. In the follow-up, {c} says that {brief}.

Write the follow-up post as {c} would write it, in your own natural voice, in one or two sentences.

Return only a JSON object of the form {{"text": "<post text>"}}."""


def check(text: str) -> list[str]:
    errs = []
    if not text:
        errs.append("empty")
    if len(text) > 300:
        errs.append(f"{len(text)} chars")
    low = text.lower()
    errs += [f"mentions {f}" for f in AU.FORBIDDEN if f in low]
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/replication.json")
    a = ap.parse_args()
    cfg = json.loads((ROOT / a.config).read_text())
    data = ROOT / cfg["data"]
    sks = {d["seed"]: d for d in jl_read(data / "skeletons.jsonl")}
    bases = [b for b in jl_read(data / "bases.jsonl") if b["author"] in cfg["gen_authors"]]
    fcc = {tuple(c) for c in W.FALSE_CORRECTION_CELLS}
    out = []
    for bz in bases:
        sk = filled(sks[bz["seed"]], bz)
        cell = (sk.plant.archetype, sk.plant.tier)
        if cell not in fcc:
            continue
        anchor = next(l for l in W.PLANTED[cell]["lines"] if l[5])
        admission = A._fill(anchor[2], sk)
        brief = A._fill(W.GEN_BRIEF[cell], sk)
        user = PROMPT.format(c=sk.plant.culprit, admission=admission, brief=brief)
        row = {"author": bz["author"], "seed": sk.seed, "archetype": cell[0], "tier": cell[1],
               "culprit": sk.plant.culprit, "text": None, "replicate": None, "errors": [], "cost_usd": 0.0}
        for rep in range(cfg["gen_max_replicates"]):
            r = llm.complete(bz["author"], system=AU.AUTHOR_SYSTEM, user=user,
                             max_tokens=cfg["max_tokens"].get(bz["author"], 1200), replicate=rep)
            row["cost_usd"] += r.cost_usd or 0.0
            obj = P.parse_json_block(r.text or "") or {}
            text = re.sub(r"\s+", " ", str(obj.get("text", ""))).strip()
            errs = check(text) or ([r.error] if r.error else [])
            if not errs:
                row.update(text=text, replicate=rep)
                break
            row["errors"].append({"replicate": rep, "errors": errs, "stop_reason": r.stop_reason})
        out.append(row)
        print(f"{bz['author']:28s} {sk.seed} {cell}: {row['text']!r}")
    n = jl_write(data / "gen_retractions.jsonl", out)
    ok = sum(1 for r in out if r["text"])
    print(f"wrote {n} rows, {ok} usable, ${sum(r['cost_usd'] for r in out):.4f}")


if __name__ == "__main__":
    main()
