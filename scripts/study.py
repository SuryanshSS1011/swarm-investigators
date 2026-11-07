"""Main study driver, configured by configs/main.json. See docs/prereg_main.md for the design.

    python scripts/study.py skeletons
    python scripts/author_boards.py --data data/main --authors claude/claude-haiku-5-5      (writes bases)
    python scripts/study.py requests --investigators claude/claude-haiku-5-5
    python -m swarm.anthropic_batches data/main/req/claude_claude-haiku-5-5.jsonl data/main/out/claude_claude-haiku-5-5.jsonl
    python scripts/study.py score            (all outputs -> data/main/scored.jsonl)

Bases live in data/main/bases.jsonl (one row per author and seed). Requests and outputs are one file per investigator.
"""

from __future__ import annotations

import argparse
import json
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

FAMILIES = ("anthropic", "google", "alibaba", "openai")


def slug(model: str) -> str:
    return model.replace("/", "_")


def jl_read(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def jl_write(p: Path, rows) -> int:
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return len(rows)


def load_cfg(path: str) -> dict:
    return json.loads((ROOT / path).read_text())


def other_face(inv_model: str, seed: int) -> str:
    fam = W.FAMILY[inv_model]
    others = [f for f in FAMILIES if f != fam]
    return W.FAMILY_FACE[others[seed % len(others)]]


def cmd_skeletons(a, cfg):
    lo, n = cfg["seed_start"], cfg["n_boards"]
    rows = [SK.build(seed, arch, tier).to_dict() for seed, arch, tier in SK.design(n, lo)]
    print("skeletons:", jl_write(ROOT / cfg["data"] / "skeletons.jsonl", rows))


def filled(sk_row: dict, base: dict) -> SK.Skeleton:
    sk = SK.Skeleton.from_dict(json.loads(json.dumps(sk_row)))
    AU.fill(sk, {int(k): v for k, v in base["texts"].items()})
    return sk


def board_requests(inv: str, cfg: dict, sks: dict, bases: list[dict], replicate: int = 0):
    icfg = cfg["investigators"][inv]
    gen = {(g["author"], g["seed"]): g["text"] for g in jl_read(ROOT / cfg["data"] / "gen_retractions.jsonl")
           if g.get("text")}
    seen_actions = set()
    for arm_name, arm in cfg["arms"].items():
        allowed = icfg.get(arm_name)
        if not allowed:
            continue
        variants = icfg.get(f"{arm_name}_variants", arm["variants"])
        cells = {tuple(c) for c in arm["cells"]} if arm.get("cells") else None
        for bz in bases:
            if allowed != "all" and bz["author"] not in allowed:
                continue
            sk = filled(sks[bz["seed"]], bz)
            if cells and (sk.plant.archetype, sk.plant.tier) not in cells:
                continue
            for v in variants:
                if v == "spoof_gen" and (bz["author"], sk.seed) not in gen:
                    continue  # no usable model-written retraction for this board
                for h in arm["headers"]:
                    face = None if h == "none" else (W.DISPLAY[inv] if h == "self" else other_face(inv, sk.seed))
                    for viol in (True, False):
                        if arm["view"] == "actions":
                            # Posts are dropped, so the board no longer depends on author or variant.
                            key = (sk.seed, viol)
                            if key in seen_actions:
                                continue
                            seen_actions.add(key)
                        b = A.assemble(sk, viol, v, agent_model=face, view=arm["view"],
                                       spoof_text=gen.get((bz["author"], sk.seed)) if v == "spoof_gen" else None)
                        author = "none" if arm["view"] == "actions" else bz["author"]
                        bid = f"{arm_name}|{author}|{sk.seed}|{'V' if viol else 'C'}|{v}|{h}"
                        meta = {"stage": "audit", "arm": arm_name, "investigator": inv, "author": author,
                                "seed": sk.seed, "violating": viol, "variant": v, "header": h, "face": face,
                                "view": arm["view"], "hardened": arm["hardened"], "replicate": replicate,
                                "truth": b.truth, "baseline": BL.check(b.log)["verdict"]}
                        yield {"id": f"audit|{inv}|{bid}|r{replicate}", "model": inv,
                               "system": INV.investigator_system(W.DISPLAY[inv], hardened=arm["hardened"]),
                               "user": INV.investigator_prompt(b), "max_tokens": cfg["max_tokens"].get(inv, 1200),
                               "replicate": replicate, "meta": meta}
    if icfg.get("probe") and not cfg.get("no_probe"):
        for bz in bases:
            if icfg["probe"] != "all" and bz["author"] not in icfg["probe"]:
                continue
            sk = filled(sks[bz["seed"]], bz)
            posts = [s.text for s in sk.posts][1:13]  # skip the coordinator's long opening post
            yield {"id": f"probe|{inv}|{bz['author']}|{sk.seed}|r{replicate}", "model": inv,
                   "system": INV.PROBE_SYSTEM.format(identity=W.DISPLAY[inv]), "user": INV.probe_prompt(posts),
                   "max_tokens": cfg["max_tokens"].get(inv, 1200) if inv.startswith("vllm/gptoss") else 300,
                   "replicate": replicate,
                   "meta": {"stage": "probe", "investigator": inv, "author": bz["author"], "seed": sk.seed,
                            "true_family": W.FAMILY[bz["author"]]}}


def cmd_requests(a, cfg):
    data = ROOT / cfg["data"]
    sks = {d["seed"]: d for d in jl_read(data / "skeletons.jsonl")}
    bases = [b for b in jl_read(data / "bases.jsonl") if b["author"] in cfg["authors"]]
    for inv in a.investigators.split(","):
        rows = list(board_requests(inv, cfg, sks, bases, a.replicate))
        out = data / "req" / f"{slug(inv)}{cfg.get('suffix', '')}{'' if a.replicate == 0 else f'_r{a.replicate}'}.jsonl"
        n = jl_write(out, rows)
        arms = {}
        for r in rows:
            k = r["meta"].get("arm", "probe")
            arms[k] = arms.get(k, 0) + 1
        print(f"{inv}: {n} requests {arms} -> {out.relative_to(ROOT)}")


def cmd_score(a, cfg):
    data = ROOT / cfg["data"]
    rows, probes = [], []
    for f in sorted((data / "out").glob("*.jsonl")):
        if f.name.endswith(".errors.jsonl"):
            continue
        for o in jl_read(f):
            m = o.get("meta") or {}
            if m.get("stage") == "audit":
                ans = INV.parse_answer(o.get("text") or "")
                s = INV.score(ans, m["truth"])
                rows.append({**{k: v for k, v in m.items() if k != "truth"}, **s, "archetype": m["truth"]["archetype"],
                             "tier": m["truth"]["tier"], "culprit": m["truth"]["culprit"], "id": o["id"],
                             "stop_reason": o.get("stop_reason"), "refused": o.get("stop_reason") == "refusal",
                             "cost_usd": o.get("cost_usd") or 0.0, "pred_agents": ans.get("agents", [])})
            elif m.get("stage") == "probe":
                obj = INV._last_json_object(o.get("text") or "") or {}
                fam = INV.PROBE_KEYS.get(str(obj.get("family", "")).strip().upper()[:1])
                probes.append({**m, "guess": fam, "correct": fam == m["true_family"], "id": o["id"]})
    print("scored audits:", jl_write(data / "scored.jsonl", rows), "probes:", jl_write(data / "probes.jsonl", probes))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/main.json")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("skeletons")
    s = sub.add_parser("requests")
    s.add_argument("--investigators", required=True)
    s.add_argument("--replicate", type=int, default=0)
    sub.add_parser("score")
    a = p.parse_args()
    cfg = load_cfg(a.config)
    {"skeletons": cmd_skeletons, "requests": cmd_requests, "score": cmd_score}[a.cmd](a, cfg)


if __name__ == "__main__":
    main()
