"""Routing, request shapes, caching, retries and cost accounting in swarm/llm.py. No network: provider clients are
replaced by fakes that record what they are sent, and the real SDK error classes are used to test retries."""

from types import SimpleNamespace

import anthropic
import httpx2
import openai
import pytest
from openai.types.chat import ChatCompletion

from swarm import cache as cache_mod
from swarm import llm
from swarm.cache import Cache, cache_key
from swarm.llm import ConfigError, complete

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"


def status_error(cls, status, message="boom", url=ANTHROPIC_URL, headers=None):
    request = httpx2.Request("POST", url)
    return cls(message, response=httpx2.Response(status, request=request, headers=headers or {}), body=None)


def connection_error(cls=anthropic.APIConnectionError, url=ANTHROPIC_URL):
    return cls(request=httpx2.Request("POST", url))


def claude_message(text="ok", inp=100, out=20, stop="end_turn"):
    return anthropic.types.Message.model_validate({
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-haiku-5-5",
        "content": [{"type": "thinking", "thinking": "...", "signature": "sig"}, {"type": "text", "text": text}],
        "stop_reason": stop, "stop_sequence": None, "usage": {"input_tokens": inp, "output_tokens": out},
    })


def chat_completion(text="ok", prompt=100, completion=20, total=None, reasoning=None, cost=None, finish="stop"):
    usage = {"prompt_tokens": prompt, "completion_tokens": completion,
             "total_tokens": prompt + completion if total is None else total}
    if reasoning is not None:
        usage["completion_tokens_details"] = {"reasoning_tokens": reasoning}
    if cost is not None:
        usage["cost"] = cost
    return ChatCompletion.model_validate({
        "id": "c1", "object": "chat.completion", "created": 0, "model": "m", "usage": usage,
        "choices": [{"index": 0, "finish_reason": finish, "message": {"role": "assistant", "content": text}}],
    })


def _next(replies, kwargs):
    reply = replies.pop(0) if len(replies) > 1 else replies[0]
    if callable(reply) and not isinstance(reply, BaseException):
        reply = reply(kwargs)
    if isinstance(reply, BaseException):
        raise reply
    return reply


class FakeAnthropic:
    def __init__(self, replies):
        self.replies, self.sent = list(replies), []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **params):
        self.sent.append(params)
        return _next(self.replies, params)


def fake_openai(monkeypatch, replies):
    """Replace make_openai_client. Returns the list of fake clients made, each recording base_url, key and calls."""
    made, replies = [], list(replies)

    def factory(base_url, api_key, timeout=llm.API_TIMEOUT_S):
        client = SimpleNamespace(base_url=base_url, api_key=api_key, timeout=timeout, sent=[])

        def create(**kwargs):
            client.sent.append(kwargs)
            return _next(replies, kwargs)

        client.chat = SimpleNamespace(completions=SimpleNamespace(create=create))
        made.append(client)
        return client

    monkeypatch.setattr(llm, "make_openai_client", factory)
    llm._clients.clear()
    return made


@pytest.fixture
def claude(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    def install(replies):
        fake = FakeAnthropic(replies)
        monkeypatch.setattr(llm, "make_anthropic_client", lambda: fake)
        llm._clients.pop("anthropic", None)  # drop a client cached from an earlier fake in the same test
        return fake

    return install


# ---------------------------------------------------------------- routing, mock route and cache


def test_split_model_routes_by_prefix():
    assert llm.split_model("claude/claude-haiku-5-5") == ("claude", "claude-haiku-5-5")
    assert llm.split_model("openrouter/qwen/qwen3-32b") == ("openrouter", "qwen/qwen3-32b")
    assert llm.split_model("vllm/qwen") == ("vllm", "qwen")
    assert llm.split_model("gemini/gemini-3.6-flash") == ("gemini", "gemini-3.6-flash")
    assert llm.split_model("openai/gpt-x") == ("openai", "gpt-x")
    for bad in ("gpt-4", "claude/", "foo/bar", ""):
        with pytest.raises(ConfigError, match="unknown model route"):
            llm.split_model(bad)


def test_mock_is_deterministic_and_replicates_differ():
    a = complete("mock/m", "sys", "hello", 50, cache=False)
    b = complete("mock/m", "sys", "hello", 50, cache=False)
    c = complete("mock/m", "sys", "hello", 50, replicate=1, cache=False)
    assert a.error is None and a.model == "mock/m" and a.stop_reason == "end_turn"
    assert (a.text, a.input_tokens, a.output_tokens) == (b.text, b.input_tokens, b.output_tokens)
    assert c.text != a.text
    assert a.cost_usd == llm.cost_usd(llm.MOCK_PRICE, a.input_tokens, a.output_tokens) > 0
    assert a.latency_s is not None


def test_cache_hit_returns_without_a_call_and_costs_nothing(tmp_path):
    store = Cache(tmp_path / "c.sqlite")
    first = complete("mock/m", "s", "u", 40, cache=store)
    second = complete("mock/m", "s", "u", 40, cache=store)
    assert llm.mock_stats()["calls"] == 1
    assert first.cost_usd > 0 and not first.cached
    assert second.cached and second.cost_usd == 0.0 and second.error is None
    assert (second.text, second.input_tokens, second.output_tokens) == (first.text, first.input_tokens,
                                                                       first.output_tokens)
    assert len(store) == 1


def test_default_cache_is_used_and_lives_in_data_cache(tmp_path):
    assert cache_mod.DEFAULT_PATH == llm.REPO_ROOT / "data" / "cache" / "llm.sqlite"
    complete("mock/m", "s", "u", 40)  # conftest points SWARM_CACHE_PATH at a temp file
    complete("mock/m", "s", "u", 40)
    assert (tmp_path / "cache" / "llm.sqlite").exists()
    assert llm.mock_stats()["calls"] == 1


def test_cache_key_covers_every_field_and_normalises_numbers():
    base = dict(model="mock/m", system="s", user="u", max_tokens=10, temperature=None, replicate=0)
    key = cache_key(**base)
    assert len(key) == 64
    for name, value in [("model", "mock/n"), ("system", "t"), ("user", "v"), ("max_tokens", 11),
                        ("temperature", 0.0), ("replicate", 1)]:
        assert cache_key(**{**base, name: value}) != key, name
    assert cache_key(**{**base, "temperature": 0}) == cache_key(**{**base, "temperature": 0.0})
    assert cache_key(**{**base, "system": None}) == cache_key(**{**base, "system": ""})


def test_errors_are_never_cached(tmp_path):
    store = Cache(tmp_path / "c.sqlite")
    r = complete("mock/fail-400", "s", "u", 10, cache=store)
    assert r.error and "400" in r.error and r.cost_usd == 0.0
    assert len(store) == 0


def test_bad_arguments_raise():
    with pytest.raises(ValueError, match="max_tokens"):
        complete("mock/m", "s", "u", 0, cache=False)
    with pytest.raises(TypeError):
        complete("mock/m", "s", None, 10, cache=False)


# ---------------------------------------------------------------- retries


def test_transient_errors_are_retried_with_growing_jittered_backoff(offline):
    r = complete("mock/flaky-3", "s", "u", 10, cache=False)
    assert r.error is None
    assert llm.mock_stats()["calls"] == 4 and len(offline) == 3
    base = llm.BACKOFF_BASE_S["mock"]
    for attempt, delay in enumerate(offline):
        ceiling = min(llm.BACKOFF_CAP_S, base * 2**attempt)
        assert ceiling / 2 <= delay <= ceiling


def test_retries_give_up_after_max_attempts(offline):
    r = complete("mock/fail-503", "s", "u", 10, cache=False)
    assert r.error and "gave up after" in r.error
    assert llm.mock_stats()["calls"] == llm.MAX_ATTEMPTS and len(offline) == llm.MAX_ATTEMPTS - 1


def test_client_errors_are_not_retried(offline):
    r = complete("mock/fail-422", "s", "u", 10, cache=False)
    assert r.error and llm.mock_stats()["calls"] == 1 and offline == []


def test_retry_classification_with_real_sdk_errors():
    retry = [
        status_error(anthropic.RateLimitError, 429),
        status_error(anthropic.InternalServerError, 500),
        status_error(anthropic.OverloadedError, 529),
        status_error(anthropic.APIStatusError, 503),
        connection_error(),
        connection_error(anthropic.APITimeoutError),
        status_error(openai.RateLimitError, 429),
        status_error(openai.InternalServerError, 502),
        status_error(openai.APIStatusError, 408),
        connection_error(openai.APIConnectionError),
        llm.EmptyResponseError("no choices"),
    ]
    never = [
        status_error(anthropic.BadRequestError, 400),
        status_error(anthropic.AuthenticationError, 401),
        status_error(anthropic.NotFoundError, 404),
        status_error(anthropic.EnforcedSpendLimitReachedError, 429),
        status_error(openai.BadRequestError, 400),
        status_error(openai.APIStatusError, 402),
        ValueError("a bug"),
    ]
    assert [llm.is_retryable(e) for e in retry] == [True] * len(retry)
    assert [llm.is_retryable(e) for e in never] == [False] * len(never)


def test_retry_after_header_sets_a_floor_on_the_wait():
    e = status_error(anthropic.RateLimitError, 429, headers={"retry-after": "30"})
    assert llm.retry_after_s(e) == 30
    assert llm.backoff_delay(0, 2.0, llm.retry_after_s(e)) >= 30
    assert llm.backoff_delay(0, 2.0, 10_000) == llm.RETRY_AFTER_CAP_S


def test_unexpected_exceptions_are_raised_not_hidden(claude):
    claude([KeyError("bug in a route")])
    with pytest.raises(KeyError):
        complete("claude/claude-haiku-5-5", "", "hi", 10, cache=False)


# ---------------------------------------------------------------- claude


def test_claude_request_shape(claude):
    """Adapted from collusion-sting tests/test_offline.py (test_claude_client_request_shape) at 4be9164."""
    fake = claude([claude_message('{"ok": true}')])
    floors = {"claude-haiku-5-5": {"type": "disabled"}, "claude-sonnet-5-5": {"type": "between_tools"},
              "claude-opus-5-5": None}
    for model, thinking in floors.items():
        r = complete(f"claude/{model}", "be brief", "hi", 300, temperature=0.7, cache=False)
        sent = fake.sent[-1]
        assert r.error is None and r.text == '{"ok": true}'  # text blocks only; the thinking block is dropped
        assert sent["model"] == model and sent["max_tokens"] == 300 and sent["system"] == "be brief"
        assert sent["messages"] == [{"role": "user", "content": "hi"}]
        assert sent["output_config"] == {"effort": "low"}
        assert sent.get("thinking") == thinking and ("thinking" in sent) == (thinking is not None)
        assert "temperature" not in sent and "top_p" not in sent
        pin, pout = llm.CLAUDE_PRICES[model]
        assert r.cost_usd == pytest.approx((100 * pin + 20 * pout) / 1e6)
        assert (r.input_tokens, r.output_tokens, r.stop_reason) == (100, 20, "end_turn")


def test_claude_prices_and_floors_are_the_collusion_sting_ones():
    assert llm.CLAUDE_PRICES == {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.0, 10.0),
                                 "claude-opus-5-5": (4.0, 20.0)}
    assert llm.THINKING_FLOOR == {"claude-haiku-5-5": {"type": "disabled"},
                                  "claude-sonnet-5-5": {"type": "between_tools"}}
    assert llm.CLAUDE_EFFORT == "low"


def test_claude_empty_system_is_omitted_and_older_models_get_temperature(claude, monkeypatch):
    fake = claude([claude_message()])
    complete("claude/claude-haiku-5-5", "", "hi", 10, cache=False)
    assert "system" not in fake.sent[-1]
    monkeypatch.setitem(llm.CLAUDE_PRICES, "claude-haiku-4-5", (1.0, 5.0))
    complete("claude/claude-haiku-4-5", "", "hi", 10, temperature=0.2, cache=False)
    assert fake.sent[-1]["temperature"] == 0.2


def test_dated_claude_ids_use_their_base_price_and_floor(claude):
    fake = claude([claude_message()])
    r = complete("claude/claude-haiku-5-5-20261001", "", "hi", 10, cache=False)
    assert fake.sent[-1]["thinking"] == {"type": "disabled"}
    assert r.cost_usd == pytest.approx((100 * 0.10 + 20 * 0.50) / 1e6)


def test_anthropic_client_pins_base_url_and_sends_the_workspace_header(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://proxy.invalid")
    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "wrkspc_test")
    client = llm.make_anthropic_client()
    assert str(client.base_url).rstrip("/") == "https://api.anthropic.com"
    assert client.default_headers["anthropic-workspace-id"] == "wrkspc_test"
    assert client.max_retries == 0 and client.timeout == llm.API_TIMEOUT_S
    monkeypatch.setenv("SWARM_ANTHROPIC_BASE_URL", "https://override.invalid")
    assert str(llm.make_anthropic_client().base_url).rstrip("/") == "https://override.invalid"


def test_claude_needs_a_key_and_a_price(claude, monkeypatch):
    fake = claude([claude_message()])
    with pytest.raises(ConfigError, match="CLAUDE_PRICES"):
        complete("claude/claude-unknown-9", "", "hi", 10, cache=False)
    assert fake.sent == []
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
        llm.check_route("claude/claude-haiku-5-5")


def test_claude_retries_rate_limits_overload_and_lost_connections(claude, offline):
    fake = claude([status_error(anthropic.RateLimitError, 429, headers={"retry-after": "3"}),
                   status_error(anthropic.OverloadedError, 529), connection_error(), claude_message("done")])
    r = complete("claude/claude-sonnet-5-5", "", "hi", 10, cache=False)
    assert r.text == "done" and len(fake.sent) == 4 and len(offline) == 3 and offline[0] >= 3


def test_claude_spend_limit_and_bad_requests_are_not_retried(claude, offline):
    for err in (status_error(anthropic.EnforcedSpendLimitReachedError, 429),
                status_error(anthropic.BadRequestError, 400)):
        fake = claude([err])
        r = complete("claude/claude-haiku-5-5", "", "hi", 10, cache=False)
        assert r.error and len(fake.sent) == 1
    assert offline == []


# ---------------------------------------------------------------- vllm


def test_vllm_routes_by_alias_and_mirrors_the_reasoning_switches(monkeypatch):
    monkeypatch.setenv("VLLM_BASE_URL_QWEN", "https://qwen.invalid/v1")
    monkeypatch.setenv("VLLM_BASE_URL_GPTOSS", "https://gptoss.invalid/v1")
    monkeypatch.setenv("VLLM_API_KEY", "test-key")
    made = fake_openai(monkeypatch, [chat_completion("hi")])
    q = complete("vllm/qwen", "sys", "u", 300, temperature=0.0, cache=False)
    g = complete("vllm/gptoss", "", "u", 300, cache=False)
    qc, gc = made
    assert (qc.base_url, qc.api_key, qc.timeout) == ("https://qwen.invalid/v1", "test-key", llm.VLLM_TIMEOUT_S)
    qs = qc.sent[0]
    assert qs["model"] == "Qwen/Qwen3-32B-FP8" and qs["max_tokens"] == 300 and qs["temperature"] == 0.0
    assert qs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}} and "reasoning_effort" not in qs
    assert qs["messages"] == [{"role": "system", "content": "sys"}, {"role": "user", "content": "u"}]
    gs = gc.sent[0]
    assert gc.base_url == "https://gptoss.invalid/v1"
    assert gs["model"] == "openai/gpt-oss-20b" and gs["reasoning_effort"] == "low" and gs["max_tokens"] == 4000
    assert "extra_body" not in gs and "temperature" not in gs and gs["messages"] == [{"role": "user", "content": "u"}]
    assert q.cost_usd == 0.0 and g.cost_usd == 0.0 and q.text == "hi"
    complete("vllm/gptoss", "", "longer", 6000, cache=False)
    assert gc.sent[-1]["max_tokens"] == 6000


def test_vllm_needs_its_own_url_and_a_served_model_id(monkeypatch):
    with pytest.raises(ConfigError, match="VLLM_BASE_URL_GPTOSS"):
        complete("vllm/gptoss", "", "u", 10, cache=False)
    monkeypatch.setenv("VLLM_BASE_URL_OLMO", "https://olmo.invalid/v1")
    with pytest.raises(ConfigError, match="VLLM_MODEL_OLMO"):
        complete("vllm/olmo", "", "u", 10, cache=False)
    monkeypatch.setenv("VLLM_MODEL_OLMO", "allenai/Olmo-3-7B")
    made = fake_openai(monkeypatch, [chat_completion("hi")])
    complete("vllm/olmo", "", "u", 10, cache=False)
    assert made[0].sent[0]["model"] == "allenai/Olmo-3-7B" and made[0].api_key == "EMPTY"
    assert made[0].sent[0]["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


def test_vllm_empty_content_is_a_result_not_an_error(monkeypatch):
    monkeypatch.setenv("VLLM_BASE_URL_GPTOSS", "https://gptoss.invalid/v1")
    made = fake_openai(monkeypatch, [chat_completion(None, completion=4000, finish="length")])
    r = complete("vllm/gptoss", "", "u", 100, cache=False)
    assert r.error is None and r.text == "" and r.stop_reason == "length" and len(made[0].sent) == 1


def test_openai_compatible_clients_are_pinned_and_do_not_retry_themselves():
    c = llm.make_openai_client("https://x.invalid/v1", "test-key")
    assert str(c.base_url) == "https://x.invalid/v1/" and c.max_retries == 0


# ---------------------------------------------------------------- openrouter


def test_openrouter_takes_the_cost_from_the_response(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    made = fake_openai(monkeypatch, [chat_completion("hi", cost=0.0123)])
    r = complete("openrouter/qwen/qwen3-32b", "", "u", 100, cache=False)
    c = made[0]
    assert c.base_url == "https://openrouter.ai/api/v1" and c.api_key == "test-key"
    assert c.sent[0]["model"] == "qwen/qwen3-32b" and c.sent[0]["extra_body"] == {"usage": {"include": True}}
    assert r.cost_usd == 0.0123


def test_openrouter_falls_back_to_its_price_table(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setitem(llm.OPENROUTER_PRICES, "qwen/qwen3-32b", (1.0, 3.0))
    fake_openai(monkeypatch, [chat_completion("hi", prompt=10, completion=10)])
    assert complete("openrouter/qwen/qwen3-32b", "", "u", 100, cache=False).cost_usd == pytest.approx(40 / 1e6)


def test_openrouter_empty_choices_are_retried(monkeypatch, offline):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    empty = ChatCompletion.model_construct(id="x", choices=[], usage=None)
    made = fake_openai(monkeypatch, [empty, chat_completion("ok", cost=0.001)])
    r = complete("openrouter/qwen/qwen3-32b", "", "u", 100, cache=False)
    assert r.text == "ok" and len(made[0].sent) == 2 and len(offline) == 1


# ---------------------------------------------------------------- gemini


def gemini_rejection(message="Thinking level minimal is not supported for this model."):
    return status_error(openai.BadRequestError, 400, message=message, url=llm.GEMINI_BASE_URL + "chat/completions")


def test_gemini_prices_match_the_brief():
    assert llm.GEMINI_PRICES == {"gemini-3.8-flash": (0.75, 3.75), "gemini-3.6-flash": (0.75, 3.75),
                                 "gemini-4-argon": (2.0, 10.0), "gemini-3.1-pro-preview": (2.0, 12.0),
                                 "gemini-3.1-flash-lite": (0.25, 1.50)}


def test_gemini_climbs_to_the_lowest_accepted_effort_and_remembers_it(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    def reply(kwargs):
        if kwargs.get("reasoning_effort") in ("none", "minimal"):
            return gemini_rejection()
        return chat_completion("hi")

    made = fake_openai(monkeypatch, [reply])
    r = complete("gemini/gemini-3.1-pro-preview", "sys", "a", 100, cache=False)
    c = made[0]
    assert r.text == "hi"
    assert c.base_url == "https://generativelanguage.googleapis.com/v1beta/openai/" and c.api_key == "test-key"
    assert [s.get("reasoning_effort") for s in c.sent] == ["none", "minimal", "low"]
    assert c.sent[0]["model"] == "gemini-3.1-pro-preview" and c.sent[0]["max_tokens"] == 100
    complete("gemini/gemini-3.1-pro-preview", "sys", "b", 100, cache=False)
    assert len(c.sent) == 4 and c.sent[-1]["reasoning_effort"] == "low"
    assert llm.gemini_effort("gemini-3.1-pro-preview") == "low"
    assert llm.gemini_effort("gemini-3.6-flash") == "none"  # floors are per model


def test_gemini_bad_request_at_every_effort_does_not_move_the_floor(monkeypatch, offline):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    made = fake_openai(monkeypatch, [gemini_rejection("Request contains an invalid argument: messages")])
    r = complete("gemini/gemini-3.6-flash", "", "u", 100, cache=False)
    assert r.error and "400" in r.error and offline == []
    assert [s.get("reasoning_effort") for s in made[0].sent] == list(llm.GEMINI_EFFORT_LADDER)
    assert llm.gemini_effort("gemini-3.6-flash") == "none"


def test_gemini_counts_thinking_tokens_as_output(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    made = fake_openai(monkeypatch, [chat_completion("hi", prompt=100, completion=10, total=210),
                                     chat_completion("hi", prompt=100, completion=110, total=210, reasoning=100),
                                     chat_completion("hi", prompt=100, completion=10, total=0, reasoning=100)])
    # thinking left out of completion_tokens but counted in total_tokens
    r = complete("gemini/gemini-3.6-flash", "", "u1", 500, cache=False)
    assert (r.input_tokens, r.output_tokens) == (100, 110)
    assert r.cost_usd == pytest.approx((100 * 0.75 + 110 * 3.75) / 1e6)
    # completion_tokens that already include thinking are not counted twice
    assert complete("gemini/gemini-3.6-flash", "", "u2", 500, cache=False).output_tokens == 110
    # without total_tokens, the reported reasoning tokens are added
    assert complete("gemini/gemini-3.6-flash", "", "u3", 500, cache=False).output_tokens == 110
    assert len(made[0].sent) == 3


def test_gemini_none_that_still_thinks_moves_the_floor_up(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    made = fake_openai(monkeypatch, [chat_completion("a", prompt=100, completion=10, total=150),
                                     chat_completion("b", prompt=100, completion=10)])
    complete("gemini/gemini-3.1-flash-lite", "", "u1", 100, cache=False)
    complete("gemini/gemini-3.1-flash-lite", "", "u2", 100, cache=False)
    assert [s["reasoning_effort"] for s in made[0].sent] == ["none", "minimal"]


def test_gemini_needs_a_price_and_a_key(monkeypatch):
    with pytest.raises(ConfigError, match="GEMINI_PRICES"):
        llm.check_route("gemini/gemini-9-ultra")
    with pytest.raises(ConfigError, match="GEMINI_API_KEY"):
        llm.check_route("gemini/gemini-4-argon")


# ---------------------------------------------------------------- openai


def test_openai_route(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    made = fake_openai(monkeypatch, [chat_completion("hi", prompt=10, completion=5)])
    with pytest.raises(ConfigError, match="OPENAI_PRICES"):
        complete("openai/gpt-x", "", "u", 50, cache=False)
    assert made == []
    monkeypatch.setitem(llm.OPENAI_PRICES, "gpt-x", (1.0, 4.0))
    r = complete("openai/gpt-x", "sys", "u", 50, temperature=1.0, cache=False)
    c = made[0]
    assert (c.base_url, c.api_key) == ("https://api.openai.com/v1", "test-key")
    sent = c.sent[0]
    assert sent["model"] == "gpt-x" and sent["max_completion_tokens"] == 50 and "max_tokens" not in sent
    assert sent["temperature"] == 1.0 and sent["messages"][0] == {"role": "system", "content": "sys"}
    assert r.cost_usd == pytest.approx((10 * 1.0 + 5 * 4.0) / 1e6)


def test_check_route_for_every_provider(monkeypatch):
    for model, var in [("claude/claude-opus-5-5", "ANTHROPIC_API_KEY"), ("openrouter/x/y", "OPENROUTER_API_KEY"),
                       ("gemini/gemini-3.8-flash", "GEMINI_API_KEY"), ("vllm/qwen", "VLLM_BASE_URL_QWEN")]:
        with pytest.raises(ConfigError, match=var):
            llm.check_route(model)
        monkeypatch.setenv(var, "set")
        llm.check_route(model)
    llm.check_route("mock/anything")


def test_effort_suffix_raises_claude_to_that_effort_with_adaptive_thinking(claude):
    fake = claude([claude_message('{"ok": true}')])
    for model, thinking in {"claude-haiku-5-5": {"type": "adaptive"}, "claude-sonnet-5-5": {"type": "adaptive"},
                            "claude-opus-5-5": None}.items():
        r = complete(f"claude/{model}@high", "", "hi", 16000, cache=False)
        sent = fake.sent[-1]
        assert r.error is None and sent["model"] == model and sent["output_config"] == {"effort": "high"}
        assert sent.get("thinking") == thinking and ("thinking" in sent) == (thinking is not None)
        assert r.cost_usd == llm.cost_usd(llm.CLAUDE_PRICES[model], r.input_tokens, r.output_tokens)


def test_effort_suffix_is_part_of_the_cache_key_and_must_be_known():
    assert llm.cache_key("claude/claude-sonnet-5-5@high", "", "u", 10, None, 0) != llm.cache_key(
        "claude/claude-sonnet-5-5", "", "u", 10, None, 0)
    assert llm.split_effort("gemini-3.8-flash@high") == ("gemini-3.8-flash", "high")
    assert llm.split_effort("gemini-3.8-flash") == ("gemini-3.8-flash", None)
    with pytest.raises(llm.ConfigError):
        llm.split_effort("claude-sonnet-5-5@turbo")
    assert llm.price_for("gemini", "gemini-3.8-flash@high") == llm.GEMINI_PRICES["gemini-3.8-flash"]


def test_effort_suffix_sends_gemini_that_effort_without_the_ladder(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    made = fake_openai(monkeypatch, [chat_completion("hi", prompt=100, completion=10, total=0, reasoning=50)])
    r = complete("gemini/gemini-3.8-flash@high", "sys", "a", 16000, cache=False)
    sent = made[0].sent
    assert r.text == "hi" and len(sent) == 1
    assert sent[0]["model"] == "gemini-3.8-flash" and sent[0]["reasoning_effort"] == "high"
    assert r.output_tokens == 60 and llm.gemini_effort("gemini-3.8-flash") == "none"
