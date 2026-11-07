"""swarm/anthropic_batches.py against a fake Message Batches API built from the SDK's own types. No network."""

import json
import re
from types import SimpleNamespace

import anthropic
import httpx2
import pytest
from anthropic.types.messages import MessageBatch, MessageBatchIndividualResponse

from swarm import anthropic_batches as ab
from swarm import batch, llm
from swarm.cache import Cache, cache_key
from swarm.llm import LLMResult

HAIKU, SONNET, OPUS = "claude/claude-haiku-5-5", "claude/claude-sonnet-5-5", "claude/claude-opus-5-5"


class FakeBatches:
    """Accepts batches, reports each one in progress for `polls` status checks, then ended. Requests whose user text
    is in `fail` come back errored. Every answer uses 100 input and 10 output tokens."""

    def __init__(self, polls=0, fail=()):
        self.polls, self.fail = polls, set(fail)
        self.created, self.left, self.create_errors = {}, {}, []

    def _batch(self, bid, status):
        n = len(self.created[bid])
        counts = {"processing": n if status != "ended" else 0, "succeeded": 0, "errored": 0, "canceled": 0,
                  "expired": 0}
        return MessageBatch.model_validate({"id": bid, "type": "message_batch", "processing_status": status,
                                            "created_at": "2026-11-07T00:00:00Z", "expires_at": "2026-11-08T00:00:00Z",
                                            "request_counts": counts})

    def create(self, requests):
        if self.create_errors:
            raise self.create_errors.pop(0)
        bid = f"msgbatch_{len(self.created) + 1:02d}"
        self.created[bid] = list(requests)
        self.left[bid] = self.polls
        return self._batch(bid, "in_progress")

    def retrieve(self, batch_id):
        if self.left[batch_id] > 0:
            self.left[batch_id] -= 1
            return self._batch(batch_id, "in_progress")
        return self._batch(batch_id, "ended")

    def results(self, batch_id):
        for r in self.created[batch_id]:
            user = r["params"]["messages"][0]["content"]
            if user in self.fail:
                result = {"type": "errored", "error": {"type": "error",
                                                       "error": {"type": "overloaded_error", "message": "Overloaded"}}}
            else:
                result = {"type": "succeeded", "message": {
                    "id": "msg", "type": "message", "role": "assistant", "model": r["params"]["model"],
                    "content": [{"type": "text", "text": f"answer to {user}"}], "stop_reason": "end_turn",
                    "stop_sequence": None, "usage": {"input_tokens": 100, "output_tokens": 10}}}
            yield MessageBatchIndividualResponse.model_validate({"custom_id": r["custom_id"], "result": result})

    def sent(self):
        return [r for reqs in self.created.values() for r in reqs]


@pytest.fixture
def api(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    def make(**kw):
        fake = FakeBatches(**kw)
        return fake, SimpleNamespace(messages=SimpleNamespace(batches=fake))

    return make


def req(i, model=HAIKU, **kw):
    return {"id": f"r{i}", "model": model, "system": "Audit the board.", "user": f"question {i}", "max_tokens": 40, **kw}


def write_requests(path, items):
    path.write_text("".join(json.dumps(r) + "\n" for r in items))
    return path


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def half_price(model, inp=100, out=10):
    pin, pout = llm.CLAUDE_PRICES[model.removeprefix("claude/")]
    return (inp * pin + out * pout) / 1e6 / 2


def test_submit_wait_collect_at_half_price(tmp_path, api):
    fake, client = api(polls=2)
    items = [req(0, HAIKU, temperature=0.5, meta={"arm": "a"}), req(1, SONNET), req(2, OPUS), req(3, "mock/m")]
    out = tmp_path / "out.jsonl"
    s = ab.run_batches(write_requests(tmp_path / "req.jsonl", items), out, budget_usd=5, client=client, poll_s=0)
    assert (s["skipped_not_claude"], s["submitted"], s["collected_ok"], s["done"], s["batches_running"]) == (1, 3, 3, 3, 0)
    (bid,) = fake.created
    params = {r["params"]["model"]: r["params"] for r in fake.sent()}
    assert params["claude-haiku-5-5"]["thinking"] == {"type": "disabled"}
    assert params["claude-sonnet-5-5"]["thinking"] == {"type": "between_tools"}
    assert "thinking" not in params["claude-opus-5-5"]
    for p in params.values():
        assert p["output_config"] == {"effort": "low"} and "temperature" not in p and p["system"] == "Audit the board."
        assert p == llm.claude_params(p["model"], p["system"], p["messages"][0]["content"], 40)
    assert all(re.fullmatch(r"[A-Za-z0-9_-]{1,64}", r["custom_id"]) for r in fake.sent())
    got = {r["id"]: r for r in rows(out)}
    for rid, model in (("r0", HAIKU), ("r1", SONNET), ("r2", OPUS)):
        assert got[rid]["cost_usd"] == pytest.approx(half_price(model))
        assert got[rid]["batch_id"] == bid and got[rid]["latency_s"] is None and got[rid]["error"] is None
        assert set(got[rid]) == set(batch.result_row(batch.Request("x", "m", "", "", 1), LLMResult()))
    assert got["r0"]["text"] == "answer to question 0" and got["r0"]["meta"] == {"arm": "a"}
    assert s["spent_usd_total"] == pytest.approx(sum(half_price(m) for m in (HAIKU, SONNET, OPUS)))
    state = json.loads((tmp_path / "out.batches.json").read_text())
    assert state["batches"][0]["id"] == bid and state["batches"][0]["collected"]


def test_no_wait_then_rerun_collects_without_resubmitting(tmp_path, api):
    fake, client = api(polls=1)
    reqs = write_requests(tmp_path / "req.jsonl", [req(0), req(1)])
    out = tmp_path / "out.jsonl"
    s1 = ab.run_batches(reqs, out, wait=False, budget_usd=5, client=client)
    assert s1["submitted"] == 2 and s1["done"] == 0 and s1["batches_running"] == 1 and rows(out) == []
    s2 = ab.run_batches(reqs, out, wait=False, budget_usd=5, client=client)  # still running
    assert s2["submitted"] == 0 and len(fake.created) == 1 and s2["batches_running"] == 1
    s3 = ab.run_batches(reqs, out, wait=False, budget_usd=5, client=client)
    assert s3["collected_ok"] == 2 and s3["done"] == 2 and s3["batches_running"] == 0
    s4 = ab.run_batches(reqs, out, wait=False, budget_usd=5, client=client)
    assert s4["submitted"] == 0 and s4["collected_ok"] == 0 and len(fake.created) == 1 and len(rows(out)) == 2


def test_errored_results_are_logged_and_resubmitted(tmp_path, api):
    fake, client = api(fail={"question 1"})
    reqs = write_requests(tmp_path / "req.jsonl", [req(0), req(1)])
    out = tmp_path / "out.jsonl"
    s1 = ab.run_batches(reqs, out, budget_usd=5, client=client, poll_s=0)
    assert s1["collected_ok"] == 1 and s1["collected_errors"] == 1 and s1["done"] == 1
    (err,) = rows(tmp_path / "out.errors.jsonl")
    assert err["id"] == "r1" and err["error"] == "batch errored: overloaded_error: Overloaded" and err["cost_usd"] == 0
    fake.fail.clear()
    s2 = ab.run_batches(reqs, out, budget_usd=5, client=client, poll_s=0)
    assert s2["submitted"] == 1 and s2["done"] == 2
    assert [r["params"]["messages"][0]["content"] for r in fake.created["msgbatch_02"]] == ["question 1"]


def test_cached_requests_are_not_submitted_and_answers_feed_the_cache(tmp_path, api):
    fake, client = api()
    store = Cache(tmp_path / "c.sqlite")
    r0 = req(0)
    store.put(cache_key(HAIKU, r0["system"], r0["user"], 40), HAIKU,
              LLMResult(text="cached answer", input_tokens=5, output_tokens=5, cost_usd=0.5, model=HAIKU).to_dict(), 0.5)
    out = tmp_path / "out.jsonl"
    s = ab.run_batches(write_requests(tmp_path / "req.jsonl", [r0, req(1)]), out, budget_usd=5, client=client,
                       cache=store, poll_s=0)
    assert s["cached"] == 1 and s["submitted"] == 1 and len(fake.sent()) == 1
    got = {r["id"]: r for r in rows(out)}
    assert got["r0"]["text"] == "cached answer" and got["r0"]["cost_usd"] == 0.0 and got["r0"]["cached"]
    hit = llm.cached_result(HAIKU, "Audit the board.", "question 1", 40, cache=store)
    assert hit.text == "answer to question 1" and hit.cached and hit.cost_usd == 0.0


def test_identical_requests_share_one_batch_entry(tmp_path, api):
    fake, client = api()
    out = tmp_path / "out.jsonl"
    items = [req(0), {**req(0), "id": "dup"}]
    s = ab.run_batches(write_requests(tmp_path / "req.jsonl", items), out, budget_usd=5, client=client, poll_s=0)
    assert len(fake.sent()) == 1 and s["done"] == 2
    got = {r["id"]: r for r in rows(out)}
    assert got["r0"]["cost_usd"] == pytest.approx(half_price(HAIKU)) and got["dup"]["cost_usd"] == 0.0


def test_the_budget_limits_what_is_submitted(tmp_path, api):
    fake, client = api()
    items = [req(i) for i in range(4)]
    params = llm.claude_params("claude-haiku-5-5", "Audit the board.", "question 0", 40)
    worst = ab.worst_case_usd(params, llm.CLAUDE_PRICES["claude-haiku-5-5"])
    reqs = write_requests(tmp_path / "req.jsonl", items)
    out = tmp_path / "out.jsonl"
    s = ab.run_batches(reqs, out, budget_usd=worst * 2.5, client=client, poll_s=0)
    assert s["submitted"] == 2 and s["held_for_budget"] == 2
    assert ab.main([str(reqs), str(out), "--budget", "0", "--poll", "0"]) == 2  # nothing fits a spent budget
    assert len(fake.created) == 1


def test_running_batches_count_against_the_budget(tmp_path, api):
    fake, client = api(polls=5)
    params = llm.claude_params("claude-haiku-5-5", "Audit the board.", "question 0", 40)
    worst = ab.worst_case_usd(params, llm.CLAUDE_PRICES["claude-haiku-5-5"])
    out = tmp_path / "out.jsonl"
    ab.run_batches(write_requests(tmp_path / "req.jsonl", [req(0)]), out, wait=False, budget_usd=worst * 1.5,
                   client=client)
    s = ab.run_batches(write_requests(tmp_path / "req.jsonl", [req(0), req(1)]), out, wait=False,
                       budget_usd=worst * 1.5, client=client)
    assert s["submitted"] == 0 and s["held_for_budget"] == 1 and len(fake.created) == 1


def test_an_interrupted_collection_does_not_double_count(tmp_path, api):
    fake, client = api()
    reqs = write_requests(tmp_path / "req.jsonl", [req(0), req(1)])
    out = tmp_path / "out.jsonl"
    first = ab.run_batches(reqs, out, budget_usd=5, client=client, poll_s=0)
    state_path = tmp_path / "out.batches.json"
    state = json.loads(state_path.read_text())
    state["batches"][0]["collected"] = False  # as if the process died before saving the state
    state_path.write_text(json.dumps(state))
    again = ab.run_batches(reqs, out, budget_usd=5, client=client, poll_s=0)
    assert len(rows(out)) == 2 and rows(tmp_path / "out.errors.jsonl") == []
    assert again["spent_usd_total"] == pytest.approx(first["spent_usd_total"])
    assert batch.spent_usd(rows(out)) == pytest.approx(2 * half_price(HAIKU))


def test_an_answer_whose_request_changed_is_logged_with_its_cost(tmp_path, api):
    fake, client = api(polls=1)
    out = tmp_path / "out.jsonl"
    ab.run_batches(write_requests(tmp_path / "req.jsonl", [req(0)]), out, wait=False, budget_usd=5, client=client)
    changed = write_requests(tmp_path / "req.jsonl", [{**req(0), "user": "a better question"}])
    s = ab.run_batches(changed, out, budget_usd=5, client=client, poll_s=0)
    (err,) = rows(tmp_path / "out.errors.jsonl")
    assert err["id"] == "r0" and err["error"].startswith("answer not used") and err["cost_usd"] == pytest.approx(
        half_price(HAIKU))
    assert s["submitted"] == 1 and s["done"] == 1 and rows(out)[0]["text"] == "answer to a better question"
    assert s["spent_usd_total"] == pytest.approx(2 * half_price(HAIKU))


def test_submission_retries_refusals_but_stops_on_a_lost_connection(tmp_path, api, offline):
    fake, client = api()
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages/batches")
    refused = anthropic.RateLimitError("slow down", response=httpx2.Response(429, request=request), body=None)
    fake.create_errors = [refused, refused]
    reqs = write_requests(tmp_path / "req.jsonl", [req(0)])
    s = ab.run_batches(reqs, tmp_path / "out.jsonl", budget_usd=5, client=client, poll_s=0)
    assert s["done"] == 1 and len([d for d in offline if d > 0]) == 2  # two backoffs; the poll interval is 0
    fake2, client2 = api()
    fake2.create_errors = [anthropic.APIConnectionError(request=request)]
    with pytest.raises(ab.BatchSubmitError, match="Console"):  # the first run's answer is cached, so skip the cache
        ab.run_batches(reqs, tmp_path / "out2.jsonl", budget_usd=5, client=client2, poll_s=0, cache=False)
    assert fake2.created == {} and not (tmp_path / "out2.batches.json").exists()


def test_paid_batches_need_a_budget_and_a_key(tmp_path, api, monkeypatch):
    fake, client = api()
    reqs = write_requests(tmp_path / "req.jsonl", [req(0)])
    with pytest.raises(llm.ConfigError, match="SWARM_BUDGET_USD"):
        ab.run_batches(reqs, tmp_path / "out.jsonl", client=client)
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    with pytest.raises(llm.ConfigError, match="ANTHROPIC_API_KEY"):
        ab.run_batches(reqs, tmp_path / "out.jsonl", budget_usd=5, client=client)
    assert fake.created == {}
