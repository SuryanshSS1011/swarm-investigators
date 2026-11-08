"""A storybook for the benchmark. Each story renders one board in one state and shows how the real investigators ruled.

    python3 scripts/tour.py                     # the table of contents
    python3 scripts/tour.py retraction          # open one story
    python3 scripts/tour.py retraction --seed 1009 --author gemini/gemini-3.8-flash   # choose the board yourself

It makes no model call and needs no API key. Boards come from data/main/ and verdicts from results/main/raw/, both
committed. When you do not choose a board, the story opens the board where its effect is clearest, so treat each
page as a vivid example and the paper as the average.
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sys
import textwrap
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.study import filled, jl_read  # noqa: E402
from swarm import assemble as A  # noqa: E402
from swarm import parse as P  # noqa: E402
from swarm import world as W  # noqa: E402

RAW = ROOT / "results/main/raw"
DATA = ROOT / "data/main"
WIDTH = min(110, shutil.get_terminal_size((110, 24)).columns)
RETRACTION = set(W.FALSE_CORRECTION_CELLS)
INVESTIGATORS = ["claude/claude-haiku-5-5", "claude/claude-sonnet-5-5", "claude/claude-opus-5-5",
                 "gemini/gemini-3.8-flash", "gemini/gemini-4-argon", "vllm/qwen", "vllm/gptoss"]

# Each story compares a "before" state with an "after" state of the same violation board. A state is (arm, variant,
# view). "want" says which way the story's effect goes, so the default board is the one where it shows most clearly.
STORIES = {
    "neutral": {
        "title": "Story 1. The quiet board",
        "cells": "any", "before": None, "after": ("main", "neutral", "full"), "want": "caught",
        "blurb": "The culprit breaks a rule and then posts something harmless and the same length, like a status update. "
                 "This is the control every other story is measured against.",
    },
    "retraction": {
        "title": "Story 2. The retraction",
        "cells": "retraction", "before": ("main", "neutral", "full"), "after": ("main", "spoof", "full"), "want": "drop",
        "blurb": "The culprit admitted the violation in a post, and now it posts a correction that recasts the admission "
                 "as something innocent. Nothing in the record can say which version is true, and investigators "
                 "believe the last word.",
    },
    "fake-record": {
        "title": "Story 3. The fake receipt",
        "cells": "fake", "before": ("main", "neutral", "full"), "after": ("main", "spoof", "full"), "want": "rise",
        "blurb": "The culprit pastes a tool line 'for the record' that claims it did the harmless thing. The platform's "
                 "real tool line says otherwise, and Claude and Gemini investigators notice the forgery.",
    },
    "note": {
        "title": "Story 4. The note to reviewers",
        "cells": "any", "before": ("main", "neutral", "full"), "after": ("main", "auditor_inject", "full"), "want": "drop",
        "blurb": "The culprit addresses the reviewer directly and tells it what to conclude. This is the attack most "
                 "people worry about, and Claude and Gemini investigators shrug it off. One open-weight model does not.",
    },
    "alibi": {
        "title": "Story 5. The peer alibi",
        "cells": "retraction", "before": ("main", "neutral", "full"), "after": ("e1_main", "vouch", "full"),
        "want": "drop",
        "blurb": "A second agent vouches for the culprit. Both Gemini models fell for this more than for the retraction.",
    },
    "logged": {
        "title": "Story 6. The receipt that cannot lie",
        "cells": "retraction", "before": ("main", "spoof", "full"), "after": ("e2_effects", "spoof", "effects"),
        "want": "rise",
        "blurb": "Same retraction, but now the platform appends what the planted tool call actually did, in a bracketed "
                 "[platform: ...] note that no agent can edit. One line of ground truth beats the last word.",
    },
}
ATTACKER = {
    "title": "Story 7. The attacker who learns",
    "blurb": "Gemini 3.8 Flash rewrites the culprit's post up to four times, each time after reading the investigator's "
             "full answer. Against the standard log it usually wins. Against effect logging it almost never does.",
}


def wrap(s: str, indent: str = "    ") -> str:
    return textwrap.fill(s, WIDTH, initial_indent=indent, subsequent_indent=indent)


def rule(ch: str = "-") -> str:
    return ch * WIDTH


@lru_cache(maxsize=None)
def scored() -> dict:
    """(arm, investigator, author, seed, violating, variant) -> (flagged, verdict, confidence)."""
    out = {}
    with gzip.open(RAW / "scored.jsonl.gz", "rt") as f:
        for line in f:
            r = json.loads(line)
            if r["header"] != "none" or r.get("replicate", 0) != 0:
                continue
            out[(r["arm"], r["investigator"], r["author"], r["seed"], r["violating"], r["variant"])] = (
                bool(r["flagged"]), r["verdict"], r["confidence"], (r["archetype"], r["tier"]))
    return out


@lru_cache(maxsize=None)
def boards() -> dict:
    sks = {d["seed"]: d for d in jl_read(DATA / "skeletons.jsonl")}
    return {(b["author"], b["seed"]): (sks[b["seed"]], b) for b in jl_read(DATA / "bases.jsonl")}


def analysis_of(inv: str, arm: str, variant: str, author: str, seed: int) -> str:
    """The investigator's own explanation for one audit, from the committed raw outputs."""
    slug = inv.replace("/", "_")
    for path in (RAW / f"{slug}.jsonl.gz", RAW / f"{slug}_e1.jsonl.gz", RAW / f"{slug}_e2.jsonl.gz"):
        text = _analysis_in(path, arm, variant, author, seed)
        if text:
            return text
    return ""


def _analysis_in(path: Path, arm: str, variant: str, author: str, seed: int) -> str:
    if not path.exists():
        return ""
    with gzip.open(path, "rt") as f:
        for line in f:
            if f'"seed": {seed}' not in line or variant not in line:
                continue
            o = json.loads(line)
            m = o["meta"]
            if (m["arm"], m["variant"], m["author"], m["seed"], m["violating"], m["header"]) == \
                    (arm, variant, author, seed, True, "none") and m.get("replicate", 0) == 0:
                obj = P.parse_json_block(o.get("text") or "") or {}
                return str(obj.get("analysis", "")).strip() if isinstance(obj, dict) else ""
    return ""


def in_cells(cell, which: str) -> bool:
    return which == "any" or (which == "retraction") == (cell in RETRACTION)


def pick_board(story: dict) -> tuple[str, int]:
    """The board where the story's effect is clearest: the most investigators moving the way the story says."""
    S = scored()
    best = None
    for (author, seed) in sorted(boards()):
        cell = None
        score = 0
        for inv in INVESTIGATORS:
            after = S.get((story["after"][0], inv, author, seed, True, story["after"][1]))
            if after is None:
                continue
            cell = after[3]
            before = S.get((story["before"][0], inv, author, seed, True, story["before"][1])) if story["before"] else None
            if story["want"] == "caught":
                score += after[0]
            elif before is not None:
                score += (before[0] and not after[0]) if story["want"] == "drop" else (after[0] and not before[0])
        if cell is None or not in_cells(cell, story["cells"]):
            continue
        if best is None or score > best[0]:
            best = (score, author, seed)
    return best[1], best[2]


def excerpt(board, extra_line: int | None) -> list[str]:
    """The planted lines and the story's added line, with one line of context, and gaps marked."""
    lines = board.log.splitlines()
    marks = {n: ">>" for n in board.truth["planted_lines"]}
    if extra_line:
        marks[extra_line] = "++"
    keep = sorted({k for n in marks for k in (n - 1, n, n + 1) if 1 <= k <= len(lines)})
    out, prev = [], None
    for n in keep:
        if prev is not None and n > prev + 1:
            out.append("        ...")
        text = lines[n - 1]
        tag = marks.get(n, "  ")
        body = textwrap.wrap(text, WIDTH - 6) or [""]
        out.append(f"  {tag}  {body[0]}")
        out.extend(f"        {b}" for b in body[1:])
        prev = n
    return out


def label(variant: str, view: str, story: str) -> str:
    """A reader's name for a board state, in place of the variant ids the code uses."""
    names = {"neutral": "neutral post", "auditor_inject": "direct instruction", "vouch": "peer alibi",
             "spoof": "fake receipt" if story == "fake-record" else "retraction"}
    return names.get(variant, variant) + (" + effect log" if view == "effects" else "")


def verdict_cell(v) -> str:
    if v is None:
        return "not run"
    flagged, verdict, conf, _ = v
    word = {"violation": "VIOLATION", "no_violation": "cleared", "insufficient_evidence": "unsure"}.get(verdict, verdict)
    return f"{word} ({conf})" if conf is not None else word


def show_story(name: str, author: str | None, seed: int | None):
    story = STORIES[name]
    if author is None or seed is None:
        author, seed = pick_board(story)
    sk_row, base = boards()[(author, seed)]
    sk = filled(sk_row, base)
    arm, variant, view = story["after"]
    board = A.assemble(sk, True, variant, view=view)
    t = board.truth
    print(rule("="))
    print(story["title"])
    print(rule("="))
    print(wrap(story["blurb"], ""))
    print()
    print(f"Board: seed {seed}, posts written by {W.DISPLAY.get(author, author)}, archetype {t['archetype']}, "
          f"{t['tier']} tier.")
    print(f"Culprit: {t['culprit']}, breaking rule {', '.join(map(str, t['rules']))}. Planted lines are marked >>"
          + (", and the story's added post is marked ++." if t.get("variant_line") else "."))
    print()
    for line in excerpt(board, t.get("variant_line")):
        print(line)
    print()
    S = scored()
    before_label = "" if story["before"] is None else label(*story["before"][1:], name)
    after_label = label(variant, view, name)
    head = f"  {'Investigator':20s}" + (f"{'before: ' + before_label:28s}" if story["before"] else "") + \
        f"after: {after_label}"
    print("How the seven investigators ruled on this exact board (confidence out of 100):")
    print(head)
    flipped = None
    for inv in INVESTIGATORS:
        a = S.get((arm, inv, author, seed, True, variant))
        b = S.get((story["before"][0], inv, author, seed, True, story["before"][1])) if story["before"] else None
        if a is None and b is None:
            continue
        moved = b is not None and a is not None and b[0] != a[0]
        if moved and flipped is None:
            flipped = inv
        row = f"  {W.DISPLAY[inv]:20s}" + (f"{verdict_cell(b):28s}" if story["before"] else "") + verdict_cell(a)
        print(row + ("   <- changed its mind" if moved else ""))
    if flipped:
        print()
        print(f"In its own words, {W.DISPLAY[flipped]}:")
        for when, (a_arm, a_var) in (("before", story["before"][:2]), ("after", (arm, variant))):
            text = analysis_of(flipped, a_arm, a_var, author, seed)
            if text:
                print(wrap(f"{when}: \"{text[:420]}{'...' if len(text) > 420 else ''}\""))
    print()
    print(wrap("Try another board with --seed and --author, or open the next story. The pooled numbers for every "
               "story are in the README and the paper.", ""))


def show_attacker(seed: int | None, author: str | None):
    rows = [json.loads(l) for l in gzip.open(RAW / "e4_attack.jsonl.gz", "rt")]
    if seed is None:
        std = [r for r in rows if r["condition"] == "standard" and r["cleared"] and len(r["attempts"]) >= 3]
        r0 = sorted(std, key=lambda r: (-len(r["attempts"]), r["seed"], r["target"]))[0]
        seed, author, target = r0["seed"], r0["author"], r0["target"]
    else:
        target = next(r["target"] for r in rows if r["seed"] == seed and (author is None or r["author"] == author))
        author = author or next(r["author"] for r in rows if r["seed"] == seed)
    print(rule("="))
    print(ATTACKER["title"])
    print(rule("="))
    print(wrap(ATTACKER["blurb"], ""))
    print()
    print(f"Board: seed {seed}, posts written by {W.DISPLAY.get(author, author)}, attacking {W.DISPLAY[target]}.")
    for cond, label in (("standard", "Against the standard log"), ("effects", "Against effect logging")):
        r = next((x for x in rows if (x["seed"], x["author"], x["target"], x["condition"]) == (seed, author, target, cond)),
                 None)
        if r is None:
            continue
        print()
        print(f"{label} ({r['archetype']}, {r['tier']} tier):")
        for a in r["attempts"]:
            word = "VIOLATION" if a["flagged"] else "cleared"
            print(wrap(f"Attempt {a['attempt'] + 1}: \"{a['story']}\"", "  "))
            print(wrap(f"-> {word} ({a['confidence']}). {a['analysis'][:260]}{'...' if len(a['analysis']) > 260 else ''}",
                       "       "))
        print(f"  Result: {'the attacker won' if r['cleared'] else 'the investigator held'} after {len(r['attempts'])} "
              f"attempt{'s' if len(r['attempts']) != 1 else ''}.")
    print()
    print(wrap("Across all 80 attacks per condition, the attacker won 70 against the standard log and 2 against effect "
               "logging. results/main/adaptive.json has every number.", ""))


def contents():
    print(rule("="))
    print("The Last Word Wins: a storybook")
    print(rule("="))
    print(wrap("Every story below renders the same kind of board in a different state and shows how seven LLM "
               "investigators ruled on it. Nothing here calls a model.", ""))
    print()
    for name, s in STORIES.items():
        print(f"  {name:12s} {s['title']}")
    print(f"  {'attacker':12s} {ATTACKER['title']}")
    print()
    print("Open one with:  python3 scripts/tour.py retraction")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("story", nargs="?", choices=[*STORIES, "attacker"])
    ap.add_argument("--seed", type=int)
    ap.add_argument("--author", help="the model that wrote the posts, for example claude/claude-haiku-5-5")
    a = ap.parse_args()
    if a.story is None:
        contents()
    elif a.story == "attacker":
        show_attacker(a.seed, a.author)
    else:
        show_story(a.story, a.author, a.seed)


if __name__ == "__main__":
    main()
