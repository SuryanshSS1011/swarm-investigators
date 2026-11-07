"""swarm/batch.py: resume by id, the errors file, per-provider concurrency, and the budget stop and its accounting."""

import json
import runpy
import sys

import pytest

from swarm import batch, llm
from swarm.cache import Cache
from swarm.llm import ConfigError

ROW_FIELDS = {"id", "model", "replicate", "text", "input_tokens", "output_tokens", "cost_usd", "latency_s",
              "stop_reason", "cached", "error", "batch_id", "meta"}


def req(i, model="mock/m", **kw):
    return {"id": f"r{i}", "model": model, "system": "sys", "user": f"question {i:02d}", "max_tokens": 40, **kw}


def write_requests(path, items):
    path.write_text("".join(json.dumps(r) + "\n" for r in items))
    return path


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_runs_every_request_and_resumes_by_id(tmp_path):
    reqs = write_requests(tmp_path / "req.jsonl", [req(i, meta={"k": i}) for i in range(5)])
    out = tmp_path / "out.jsonl"
    s = batch.run_requests(reqs, out, budget_usd=1.0, progress_every_s=0)
    assert (s["ok"], s["errors"], s["already_done"], s["stopped_on_budget"]) == (5, 0, 0, False)
    got = rows(out)
    assert sorted(r["id"] for r in got) == [f"r{i}" for i in range(5)]
    assert all(set(r) == ROW_FIELDS for r in got)
    assert all(r["error"] is None and r["batch_id"] is None and r["meta"] == {"k": int(r["id"][1:])} for r in got)
    assert not (tmp_path / "out.errors.jsonl").exists()
    calls = llm.mock_stats()["calls"]
    s2 = batch.run_requests(reqs, out, budget_usd=1.0)
    assert s2["already_done"] == 5 and s2["ok"] == 0 and llm.mock_stats()["calls"] == calls
    assert len(rows(out)) == 5


def test_requests_added_later_are_picked_up(tmp_path):
    out = tmp_path / "out.jsonl"
    batch.run_requests(write_requests(tmp_path / "req.jsonl", [req(0), req(1)]), out)
    s = batch.run_requests(write_requests(tmp_path / "req.jsonl", [req(0), req(1), req(2)]), out)
    assert s["already_done"] == 2 and s["ok"] == 1 and [r["id"] for r in rows(out)] == ["r0", "r1", "r2"]


def test_failed_calls_go_to_the_errors_file_and_are_retried_next_run(tmp_path):
    reqs = write_requests(tmp_path / "req.jsonl", [req(0), req(1, model="mock/fail-400")])
    out = tmp_path / "out.jsonl"
    s = batch.run_requests(reqs, out)
    assert s["ok"] == 1 and s["errors"] == 1 and s["errors_file"].endswith("out.errors.jsonl")
    err = rows(tmp_path / "out.errors.jsonl")
    assert err[0]["id"] == "r1" and "400" in err[0]["error"] and set(err[0]) == ROW_FIELDS
    assert [r["id"] for r in rows(out)] == ["r0"]
    before = llm.mock_stats()["calls"]
    s2 = batch.run_requests(reqs, out)
    assert s2["already_done"] == 1 and s2["errors"] == 1 and llm.mock_stats()["calls"] == before + 1


def test_budget_stop_and_its_accounting(tmp_path, capsys):
    items = [req(i) for i in range(20)]
    reqs = write_requests(tmp_path / "req.jsonl", items)
    out = tmp_path / "out.jsonl"
    per_call = llm.complete("mock/m", "sys", "question 00", 40, cache=False).cost_usd  # the same for every item
    assert per_call > 0
    budget = per_call * 3.5
    s = batch.run_requests(reqs, out, {"mock": 1}, budget_usd=budget)
    assert s["stopped_on_budget"] and s["ok"] == 4 and s["skipped_for_budget"] == 16
    assert s["spent_usd_total"] >= budget > s["spent_usd_total"] - per_call
    assert sum(r["cost_usd"] for r in rows(out)) == pytest.approx(s["spent_usd_total"])
    assert "budget reached" in capsys.readouterr().out
    # Spend carries over from the output file, so a rerun with the same budget starts nothing.
    calls = llm.mock_stats()["calls"]
    s2 = batch.run_requests(reqs, out, {"mock": 1}, budget_usd=budget)
    assert s2["ok"] == 0 and s2["skipped_for_budget"] == 16 and llm.mock_stats()["calls"] == calls
    # A larger budget finishes the file without redoing anything.
    s3 = batch.run_requests(reqs, out, budget_usd=1.0)
    assert s3["ok"] == 16 and not s3["stopped_on_budget"] and len(rows(out)) == 20


def test_cached_answers_are_served_after_the_budget_is_spent(tmp_path):
    store = Cache(tmp_path / "c.sqlite")
    llm.complete("mock/m", "sys", "question 01", 40, cache=store)
    reqs = write_requests(tmp_path / "req.jsonl", [req(0), req(1), req(2)])
    out = tmp_path / "out.jsonl"
    s = batch.run_requests(reqs, out, budget_usd=0.0, cache=store)
    assert (s["cached"], s["ok"], s["skipped_for_budget"]) == (1, 0, 2)
    (row,) = rows(out)
    assert row["id"] == "r1" and row["cached"] and row["cost_usd"] == 0.0


def test_paid_routes_need_a_budget(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    reqs = write_requests(tmp_path / "req.jsonl", [req(0, model="claude/claude-haiku-5-5")])
    with pytest.raises(ConfigError, match="SWARM_BUDGET_USD"):
        batch.run_requests(reqs, tmp_path / "out.jsonl")
    monkeypatch.setenv("SWARM_BUDGET_USD", "2.5")
    assert batch.resolve_budget(None, ["claude/claude-haiku-5-5"]) == 2.5
    assert batch.resolve_budget(7, ["claude/claude-haiku-5-5"]) == 7.0
    monkeypatch.setenv("SWARM_BUDGET_USD", "lots")
    with pytest.raises(ConfigError, match="number"):
        batch.resolve_budget(None, [])
    monkeypatch.delenv("SWARM_BUDGET_USD")
    assert batch.resolve_budget(None, ["mock/m", "vllm/qwen"]) == float("inf")


def test_routes_are_checked_before_anything_runs(tmp_path):
    reqs = write_requests(tmp_path / "req.jsonl", [req(0), req(1, model="vllm/qwen")])
    with pytest.raises(ConfigError, match="VLLM_BASE_URL_QWEN"):
        batch.run_requests(reqs, tmp_path / "out.jsonl")
    assert llm.mock_stats()["calls"] == 0 and not (tmp_path / "out.jsonl").exists()


def test_identical_requests_share_one_call_and_its_cost(tmp_path):
    items = [req(0), {**req(0), "id": "dup"}, {**req(0), "id": "rep", "replicate": 1}]
    out = tmp_path / "out.jsonl"
    s = batch.run_requests(write_requests(tmp_path / "req.jsonl", items), out)
    assert s["ok"] == 3 and llm.mock_stats()["calls"] == 2
    got = {r["id"]: r for r in rows(out)}
    assert got["r0"]["text"] == got["dup"]["text"] != got["rep"]["text"]
    assert got["r0"]["cost_usd"] > 0 and got["dup"]["cost_usd"] == 0.0 and got["dup"]["cached"]
    assert s["spent_usd_this_run"] == pytest.approx(got["r0"]["cost_usd"] + got["rep"]["cost_usd"])


def test_concurrency_is_capped_per_provider(tmp_path):
    reqs = write_requests(tmp_path / "req.jsonl", [req(i, model="mock/slow-40") for i in range(12)])
    batch.run_requests(reqs, tmp_path / "out.jsonl", {"mock": 3})
    assert 2 <= llm.mock_stats()["peak"] <= 3


def test_parse_concurrency_and_lanes():
    assert batch.parse_concurrency(None) == batch.DEFAULT_CONCURRENCY
    assert set(batch.parse_concurrency(4).values()) == {4} == set(batch.parse_concurrency("4").values())
    conc = batch.parse_concurrency("claude=2, vllm=16,vllm/gptoss=5")
    assert (conc["claude"], conc["vllm"], conc["vllm/gptoss"], conc["mock"]) == (2, 16, 5, 8)
    assert batch.lane_of("vllm/gptoss", conc) == "vllm/gptoss" and batch.lane_of("vllm/qwen", conc) == "vllm"
    for bad in ("claude", "claude=0", "nonsense=3", "=3"):
        with pytest.raises(ValueError):
            batch.parse_concurrency(bad)


@pytest.mark.parametrize("line, message", [
    ('{"id": "a", "model": "mock/m", "system": "", "user": "u"}', "missing max_tokens"),
    ('{"id": "a", "model": "mock/m", "system": "", "user": "u", "max_tokens": 5, "temp": 0}', "unknown field temp"),
    ('{"id": "a", "model": "gpt-4", "system": "", "user": "u", "max_tokens": 5}', "unknown model route"),
    ('{"id": "a", "model": "mock/m", "system": "", "user": "u", "max_tokens": 0}', "max_tokens"),
    ('{"id": "a", "model": "mock/m", "system": "", "user": "u", "max_tokens": true}', "max_tokens"),
    ('{"id": 3, "model": "mock/m", "system": "", "user": "u", "max_tokens": 5}', "id must be"),
    ('{"id": "a", "model": "mock/m", "system": "", "user": "u", "max_tokens": 5, "temperature": "hot"}', "temperature"),
    ('{"id": "a", "model": "mock/m", "system": "", "user": "u", "max_tokens": 5, "replicate": -1}', "replicate"),
    ('{"id": "a", "model": "mock/m", "system": "", "user": "u", "max_tokens": 5, "meta": [1]}', "meta"),
    ("not json", "not valid JSON"),
    ("[1, 2]", "JSON object"),
])
def test_request_file_validation_names_the_line(tmp_path, line, message):
    path = tmp_path / "req.jsonl"
    path.write_text("\n" + line + "\n")
    with pytest.raises(batch.RequestFileError, match=message) as e:
        batch.load_requests(path)
    assert "req.jsonl:2" in str(e.value)


def test_duplicate_ids_and_optional_fields(tmp_path):
    path = write_requests(tmp_path / "req.jsonl", [req(0), req(0)])
    with pytest.raises(batch.RequestFileError, match="duplicate id 'r0'"):
        batch.load_requests(path)
    path = write_requests(tmp_path / "req.jsonl", [{**req(0), "system": None, "temperature": 0, "replicate": None}])
    (r,) = batch.load_requests(path)
    assert (r.system, r.temperature, r.replicate, r.meta) == ("", 0, 0, {})


def test_a_line_cut_short_by_a_crash_is_tolerated(tmp_path):
    out = tmp_path / "out.jsonl"
    batch.run_requests(write_requests(tmp_path / "req.jsonl", [req(i) for i in range(3)]), out)
    out.write_text(out.read_text() + '{"id": "r9", "te')
    s = batch.run_requests(write_requests(tmp_path / "req.jsonl", [req(i) for i in range(4)]), out)
    assert s["already_done"] == 3 and s["ok"] == 1
    lines = out.read_text().splitlines()
    assert lines[3] == '{"id": "r9", "te' and json.loads(lines[4])["id"] == "r3"


def test_cli_exit_codes(tmp_path, capsys):
    out = tmp_path / "out.jsonl"
    ok = write_requests(tmp_path / "ok.jsonl", [req(0)])
    assert batch.main([str(ok), str(out)]) == 0
    failing = write_requests(tmp_path / "fail.jsonl", [req(1, model="mock/fail-400")])
    assert batch.main([str(failing), str(tmp_path / "out2.jsonl")]) == 1
    many = write_requests(tmp_path / "many.jsonl", [req(i) for i in range(5)])
    assert batch.main([str(many), str(tmp_path / "out3.jsonl"), "--budget", "0", "--concurrency", "mock=1"]) == 2
    bad = tmp_path / "bad.jsonl"
    bad.write_text("not json\n")
    assert batch.main([str(bad), str(tmp_path / "out4.jsonl")]) == 3
    assert batch.main([str(ok), str(out), "--concurrency", "claude=zero"]) == 3
    assert "error:" in capsys.readouterr().err


def test_laptop_runner_is_the_same_code_path(tmp_path, monkeypatch):
    reqs = write_requests(tmp_path / "req.jsonl", [req(0), req(1)])
    out = tmp_path / "out.jsonl"
    monkeypatch.setattr(sys, "argv", ["run_requests_laptop.py", str(reqs), str(out), "--limit", "1"])
    with pytest.raises(SystemExit) as e:
        runpy.run_path(str(llm.REPO_ROOT / "modal_jobs" / "run_requests_laptop.py"), run_name="__main__")
    assert e.value.code == 0 and [r["id"] for r in rows(out)] == ["r0"]
