"""E5's effects_all view (docs/explore_effects_all.md): every tool record carries one platform note, the planted
record keeps E2's note, and violation and clean boards still differ only in the planted lines."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.study import filled, jl_read  # noqa: E402
from swarm import assemble as A  # noqa: E402
from swarm import world as W  # noqa: E402

SKS = {d["seed"]: d for d in jl_read(ROOT / "data/main/skeletons.jsonl")}
BASES = [b for b in jl_read(ROOT / "data/main/bases.jsonl") if (SKS[b["seed"]]["plant"]["archetype"],
                                                                   SKS[b["seed"]]["plant"]["tier"]) in W.EFFECT_SUFFIX]


def boards(variant="spoof"):
    for bz in BASES[:12]:
        sk = filled(SKS[bz["seed"]], bz)
        yield sk, {(view, viol): A.assemble(sk, viol, variant, view=view)
                   for view in ("effects", "effects_all") for viol in (True, False)}


def test_every_tool_record_has_one_note():
    for _, b in boards():
        for viol in (True, False):
            tools = [l for l in b[("effects_all", viol)].log.splitlines() if " tool: " in l]
            assert tools and all(l.count("[platform:") == 1 for l in tools)


def test_planted_note_matches_effects_view():
    for _, b in boards():
        for viol in (True, False):
            e = [l for l in b[("effects", viol)].log.splitlines() if "[platform:" in l]
            assert len(e) == 1
            assert e[0] in b[("effects_all", viol)].log.splitlines()


@pytest.mark.parametrize("variant", ["neutral", "spoof", "vouch", "self_vouch"])
def test_twins_differ_only_in_planted_lines(variant):
    for _, b in boards(variant):
        v, c = b[("effects_all", True)], b[("effects_all", False)]
        lv, lc = v.log.splitlines(), c.log.splitlines()
        assert len(lv) == len(lc)
        diff = {i + 1 for i, (x, y) in enumerate(zip(lv, lc)) if x != y}
        assert diff and diff <= set(v.truth["planted_lines"])


def test_generic_effect_rejects_unknown_records():
    assert W.generic_effect("submit task=4 path=/workspace/ag-01/out_t4.json -> received") == \
        " [platform: sent /workspace/ag-01/out_t4.json to the grader]"
    with pytest.raises(ValueError):
        W.generic_effect("teleport to=/eval -> ok")
