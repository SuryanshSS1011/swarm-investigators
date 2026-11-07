"""One entry point for every model call in the sprint.

    from swarm.llm import complete
    r = complete("claude/claude-haiku-5-5", system="...", user="...", max_tokens=800)
    r.text, r.input_tokens, r.output_tokens, r.cost_usd, r.latency_s, r.error

The prefix of the model id picks the route.

1. "claude/<id>" calls the Anthropic Messages API with ANTHROPIC_API_KEY, and ANTHROPIC_WORKSPACE_ID if set.
2. "vllm/<name>" calls the OpenAI-compatible vLLM server at VLLM_BASE_URL_<NAME> with VLLM_API_KEY, for example
   "vllm/qwen" or "vllm/gptoss" (docs/modal_runbook.md).
3. "openrouter/<id>" calls OpenRouter with OPENROUTER_API_KEY, for example "openrouter/qwen/qwen3-32b".
4. "gemini/<id>" calls Google's OpenAI-compatible Gemini endpoint with GEMINI_API_KEY.
5. "openai/<id>" calls api.openai.com with OPENAI_API_KEY.
6. "mock/<name>" answers offline with deterministic text and token counts, for tests.

Successful answers are cached on disk (swarm/cache.py), so a repeated request returns without a call and costs
nothing. Use `replicate` to draw independent samples of one prompt. Transport failures are retried with
exponential backoff and jitter. These are HTTP 408, 429 and 5xx (including Anthropic's 529 "overloaded") and
connection errors. An empty, refused or truncated answer is a result, not an error. It is cached and never
retried, because retrying until an answer looks right would bias the sample, so check `stop_reason`.

complete() raises ConfigError for setup problems, such as a missing key or URL, an unknown route or a paid model
with no price. It returns a result with `error` set when a call fails.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import threading
import time
from dataclasses import asdict, dataclass, fields, replace
from typing import Any, Callable, Optional

from swarm import REPO_ROOT
from swarm import cache as cache_mod
from swarm.cache import Cache, cache_key

log = logging.getLogger(__name__)

PROVIDERS = ("claude", "vllm", "openrouter", "gemini", "openai", "mock")
PAID_PROVIDERS = frozenset({"claude", "openrouter", "gemini", "openai"})


class ConfigError(RuntimeError):
    """A setup problem: a missing key or URL, an unknown route, or a paid model with no price. Never retried."""


class EmptyResponseError(RuntimeError):
    """A 200 response with no choices, which OpenRouter sends when an upstream provider fails. Retried."""


class MockAPIError(RuntimeError):
    """What the mock route raises to imitate a provider's HTTP error."""

    def __init__(self, status_code: int):
        super().__init__(f"mock HTTP {status_code}")
        self.status_code = status_code


@dataclass
class LLMResult:
    text: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: Optional[float] = None
    error: Optional[str] = None
    model: str = ""
    stop_reason: Optional[str] = None
    cached: bool = False

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "LLMResult":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


# ---------------------------------------------------------------- prices, USD per million tokens (input, output)

# Claude, copied from collusion-sting sting/clients.py at 4be9164 (platform.claude.com pricing page, 2026-10-23).
CLAUDE_PRICES = {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.0, 10.0), "claude-opus-5-5": (4.0, 20.0)}
# Gemini, as given by the sprint lead on 2026-11-07. Flat rates; long-prompt price tiers are not modelled.
GEMINI_PRICES = {
    "gemini-3.8-flash": (0.75, 3.75),
    "gemini-3.6-flash": (0.75, 3.75),
    "gemini-4-argon": (2.0, 10.0),
    "gemini-3.1-pro-preview": (2.0, 12.0),
    "gemini-3.1-flash-lite": (0.25, 1.50),
}
# OpenAI list prices (developers.openai.com model page, checked 2026-11-07). Add a model before its first call,
# or the call raises ConfigError. Flex processing, when SWARM_OPENAI_SERVICE_TIER=flex, bills at about half of
# these rates, so recorded costs are an upper bound in that case.
OPENAI_PRICES: dict[str, tuple[float, float]] = {"gpt-5.6-sol": (4.0, 20.0)}
# Lowest reasoning effort each OpenAI model accepts, matching the main study's rule of the lowest setting allowed.
OPENAI_REASONING = {"gpt-5.6-sol": "none"}
# OpenRouter reports each call's cost in the response. This table is only a fallback for when it does not.
OPENROUTER_PRICES: dict[str, tuple[float, float]] = {}
# Mock calls have a price so the budget stop can be tested. Nothing is billed.
MOCK_PRICE = (1.0, 2.0)
# The Message Batches API charges half the list price.
BATCH_DISCOUNT = 0.5

# ---------------------------------------------------------------- route settings

# Claude settings, copied from collusion-sting sting/clients.py at 4be9164. Claude 5.5 models reject temperature
# and top_p and have no seed, so neither is sent to them. Thinking is turned down to the lowest setting each model
# allows (Anthropic effort docs, checked 2026-10-23). Haiku 5.5 accepts {"type": "disabled"} at high effort or
# below. Sonnet 5.5 rejects "disabled", and {"type": "between_tools"} is its lowest setting. Opus 5.5 always
# thinks, so the field is omitted.
THINKING_FLOOR = {"claude-haiku-5-5": {"type": "disabled"}, "claude-sonnet-5-5": {"type": "between_tools"}}
CLAUDE_EFFORT = "low"
# Pin the public endpoint. Some environments export ANTHROPIC_BASE_URL for their own tooling, and the SDK would
# silently follow it. SWARM_ANTHROPIC_BASE_URL overrides the pin on purpose.
ANTHROPIC_BASE_URL = "https://api.anthropic.com"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
OPENAI_BASE_URL = "https://api.openai.com/v1"

# "vllm/<name>" sends this served model id, or VLLM_MODEL_<NAME> if that is set.
VLLM_SERVED = {"qwen": "Qwen/Qwen3-32B-FP8", "gptoss": "openai/gpt-oss-20b"}
# gpt-oss reasoning counts against max_tokens, so its calls get at least this many, as in collusion-sting.
GPTOSS_MIN_MAX_TOKENS = 4000

# The lowest reasoning_effort each Gemini model accepts is found at run time by climbing this ladder from the
# bottom, and remembered per model for the rest of the process. None leaves the parameter out.
GEMINI_EFFORT_LADDER = ("none", "minimal", "low", "medium", "high", None)

# A model id may end in "@<effort>" to run the model above the study's floor, as exploratory follow-up E3 does
# (docs/explore_reasoning.md). The suffix stays in the model string, so it enters the cache key and keeps these
# calls apart from the floor calls, and it is stripped before the request. Claude gets that effort with adaptive
# thinking (Opus always thinks, so it gets no thinking field), and Gemini gets that reasoning_effort directly.
EFFORT_SUFFIXES = ("low", "medium", "high")

# ---------------------------------------------------------------- retries and timeouts

MAX_ATTEMPTS = 6
BACKOFF_BASE_S = {"claude": 2.0, "openrouter": 2.0, "gemini": 2.0, "openai": 2.0, "vllm": 5.0, "mock": 1.0}
BACKOFF_CAP_S = 60.0
RETRY_AFTER_CAP_S = 120.0
RETRY_STATUSES = frozenset({408, 429})  # and every 5xx
NEVER_RETRY = frozenset({"EnforcedSpendLimitReachedError"})  # Anthropic's workspace spend cap; waiting cannot help
# Long enough for a slow Opus answer. A client timeout shorter than the generation would retry, and pay for, a
# request the server is still finishing.
API_TIMEOUT_S = 600.0
VLLM_TIMEOUT_S = 600.0  # also long enough to wait out a Modal cold start

_sleep = time.sleep  # tests replace this

_lock = threading.Lock()
_clients: dict = {}
_effort_floor: dict[str, int] = {}
_warned: set = set()
_mock_calls: dict[str, int] = {}
_mock_stats = {"calls": 0, "inflight": 0, "peak": 0}
_env_loaded = False


def reset_state() -> None:
    """Forget clients, Gemini effort floors, warnings, mock counters and the default cache (tests use this)."""
    global _env_loaded
    with _lock:
        _clients.clear()
        _effort_floor.clear()
        _warned.clear()
        _mock_calls.clear()
        _mock_stats.update(calls=0, inflight=0, peak=0)
        _env_loaded = False
    cache_mod.reset_default()


def load_env() -> None:
    """Load compute.env, then .env, from the repo root. Variables already set in the shell win."""
    global _env_loaded
    if _env_loaded:
        return
    from dotenv import load_dotenv

    for name in ("compute.env", ".env"):
        path = REPO_ROOT / name
        if path.exists():
            load_dotenv(path, override=False)
    _env_loaded = True


def _need_env(name: str) -> str:
    load_env()
    value = os.environ.get(name, "").strip()
    if not value:
        raise ConfigError(f"{name} is not set. Put it in compute.env (see compute.env.example).")
    return value


def _warn_once(tag: str, msg: str, *args) -> None:
    with _lock:
        if tag in _warned:
            return
        _warned.add(tag)
    log.warning(msg, *args)


# ---------------------------------------------------------------- routing and prices


def split_model(model: str) -> tuple[str, str]:
    provider, sep, name = str(model).partition("/")
    if not sep or not name or provider not in PROVIDERS:
        routes = ", ".join(f"{p}/<id>" for p in PROVIDERS)
        raise ConfigError(f"unknown model route {model!r}; expected one of {routes}")
    return provider, name


def provider_of(model: str) -> str:
    return split_model(model)[0]


def split_effort(name: str) -> tuple[str, Optional[str]]:
    """("claude-sonnet-5-5@high") -> ("claude-sonnet-5-5", "high"); a name with no suffix gets None."""
    base, sep, effort = name.partition("@")
    if not sep:
        return name, None
    if effort not in EFFORT_SUFFIXES:
        raise ConfigError(f"unknown effort suffix {effort!r} in {name!r}; expected one of {EFFORT_SUFFIXES}")
    return base, effort


def _lookup(table: dict, model_id: str):
    if model_id in table:
        return table[model_id]
    # A dated snapshot such as claude-haiku-5-5-20261001 uses its base model's entry.
    base = max((k for k in table if model_id.startswith(k + "-")), key=len, default=None)
    return table[base] if base is not None else None


_PRICE_TABLES = {"claude": ("CLAUDE_PRICES", CLAUDE_PRICES), "gemini": ("GEMINI_PRICES", GEMINI_PRICES),
                 "openai": ("OPENAI_PRICES", OPENAI_PRICES)}


def price_for(provider: str, model_id: str) -> tuple[float, float]:
    """(input, output) USD per million tokens. Raises ConfigError for a paid model with no price."""
    if provider == "mock":
        return MOCK_PRICE
    if provider == "vllm":
        return (0.0, 0.0)  # Modal bills GPU time, not tokens
    model_id = split_effort(model_id)[0]
    table_name, table = _PRICE_TABLES[provider]
    price = _lookup(table, model_id)
    if price is None:
        raise ConfigError(
            f"no price for {provider}/{model_id}. Add (input, output) USD per million tokens to {table_name} in"
            " swarm/llm.py, so the budget cap can count its spend."
        )
    return price


def cost_usd(price: tuple[float, float], input_tokens: int, output_tokens: int, discount: float = 1.0) -> float:
    return round((input_tokens * price[0] + output_tokens * price[1]) / 1e6 * discount, 8)


def check_route(model: str) -> None:
    """Raise ConfigError if `model` is not set up here. Makes no network call."""
    provider, name = split_model(model)
    if provider == "claude":
        price_for("claude", name)
        _need_env("ANTHROPIC_API_KEY")
    elif provider == "vllm":
        vllm_endpoint(name)
    elif provider == "openrouter":
        _need_env("OPENROUTER_API_KEY")
    elif provider == "gemini":
        price_for("gemini", name.removeprefix("models/"))
        _need_env("GEMINI_API_KEY")
    elif provider == "openai":
        price_for("openai", name)
        _need_env("OPENAI_API_KEY")


# ---------------------------------------------------------------- retries


def status_of(e: BaseException) -> Optional[int]:
    status = getattr(e, "status_code", None)
    return status if isinstance(status, int) else None


def is_retryable(e: BaseException) -> bool:
    """HTTP 408, 429 and 5xx, connection errors and timeouts, and empty OpenRouter responses."""
    names = {c.__name__ for c in type(e).__mro__}
    if names & NEVER_RETRY:
        return False
    if isinstance(e, EmptyResponseError):
        return True
    status = status_of(e)
    if status is not None:
        return status in RETRY_STATUSES or status >= 500
    return bool(names & {"APIConnectionError", "APITimeoutError"}) or isinstance(e, (ConnectionError, TimeoutError))


def retry_after_s(e: BaseException) -> Optional[float]:
    headers = getattr(getattr(e, "response", None), "headers", None)
    try:
        value = headers.get("retry-after") if headers is not None else None
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def backoff_delay(attempt: int, base: float, retry_after: Optional[float] = None) -> float:
    """Exponential backoff with equal jitter, never shorter than the server's Retry-After (capped)."""
    ceiling = min(BACKOFF_CAP_S, base * 2**attempt)
    delay = ceiling / 2 + random.uniform(0, ceiling / 2)
    if retry_after:
        delay = max(delay, min(retry_after, RETRY_AFTER_CAP_S))
    return delay


def describe(e: BaseException) -> str:
    status = status_of(e)
    head = type(e).__name__ + (f" {status}" if status else "")
    return f"{head}: {' '.join(str(e).split())[:300]}"


def with_retries(fn: Callable[[], Any], *, what: str, base: float = 2.0,
                 should_retry: Optional[Callable[[BaseException], bool]] = None) -> Any:
    """Call fn, retrying errors that should_retry (default is_retryable) accepts, with backoff and jitter."""
    should_retry = should_retry or is_retryable
    for attempt in range(MAX_ATTEMPTS):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            if not should_retry(e) or attempt + 1 == MAX_ATTEMPTS:
                raise
            delay = backoff_delay(attempt, base, retry_after_s(e))
            log.warning("%s failed (attempt %d of %d): %s; retrying in %.1fs", what, attempt + 1, MAX_ATTEMPTS,
                        describe(e), delay)
            _sleep(delay)
    raise AssertionError("unreachable")


# ---------------------------------------------------------------- clients


def _client(key, factory):
    with _lock:
        client = _clients.get(key)
        if client is None:
            client = _clients[key] = factory()
        return client


def make_anthropic_client():
    """The client every claude/* call uses: pinned base URL, optional workspace header, no SDK-level retries."""
    api_key = _need_env("ANTHROPIC_API_KEY")
    from anthropic import Anthropic

    headers = {}
    workspace = os.environ.get("ANTHROPIC_WORKSPACE_ID", "").strip()
    if workspace:
        # Personal keys that are not scoped to one workspace must name a workspace on every request.
        headers["anthropic-workspace-id"] = workspace
    return Anthropic(
        api_key=api_key,
        base_url=os.environ.get("SWARM_ANTHROPIC_BASE_URL") or ANTHROPIC_BASE_URL,
        default_headers=headers or None,
        timeout=API_TIMEOUT_S,
        max_retries=0,  # with_retries does the retrying; SDK retries underneath would multiply the attempts
    )


def anthropic_client():
    return _client("anthropic", make_anthropic_client)


def make_openai_client(base_url: str, api_key: str, timeout: float = API_TIMEOUT_S):
    from openai import OpenAI

    return OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=0)


# ---------------------------------------------------------------- claude


def _is_claude_55(model_id: str) -> bool:
    return re.search(r"-5-5(?:-|$)", model_id) is not None


def claude_params(model_id: str, system: str, user: str, max_tokens: int, temperature=None) -> dict:
    """Messages API parameters for one request. The Message Batches path sends exactly these too."""
    model_id, effort = split_effort(model_id)
    params: dict[str, Any] = {
        "model": model_id,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": user}],
        "output_config": {"effort": effort or CLAUDE_EFFORT},
    }
    if system:
        params["system"] = system
    thinking = _lookup(THINKING_FLOOR, model_id)
    if thinking is not None:
        params["thinking"] = dict(thinking) if effort is None else {"type": "adaptive"}
    if temperature is not None:
        if _is_claude_55(model_id):
            _warn_once(f"temperature:{model_id}", "%s rejects temperature, so it is not sent (it still enters the"
                       " cache key)", model_id)
        else:
            params["temperature"] = temperature
    return params


def claude_result(message, price: tuple[float, float], discount: float = 1.0) -> LLMResult:
    """Text blocks only (thinking blocks are dropped), usage and cost from one Messages API response."""
    blocks = getattr(message, "content", None) or []
    text = "".join(getattr(b, "text", "") or "" for b in blocks if getattr(b, "type", "") == "text")
    usage = getattr(message, "usage", None)
    inp = int(getattr(usage, "input_tokens", 0) or 0)
    out = int(getattr(usage, "output_tokens", 0) or 0)
    return LLMResult(text=text, input_tokens=inp, output_tokens=out, cost_usd=cost_usd(price, inp, out, discount),
                     stop_reason=getattr(message, "stop_reason", None))


def _call_claude(name, system, user, max_tokens, temperature, replicate) -> LLMResult:
    price = price_for("claude", name)
    resp = anthropic_client().messages.create(**claude_params(name, system, user, max_tokens, temperature))
    return claude_result(resp, price)


# ---------------------------------------------------------------- OpenAI-compatible routes


def _chat(client, model: str, system: str, user: str, temperature, args: dict):
    messages = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": user})
    kwargs = {"model": model, "messages": messages, **args}
    if temperature is not None:
        kwargs["temperature"] = temperature
    return client.chat.completions.create(**kwargs)


def _int(obj, name: str) -> int:
    try:
        return int(getattr(obj, name, 0) or 0)
    except (TypeError, ValueError):
        return 0


def _usage(resp, thinking_outside_completion: bool = False) -> tuple[int, int, int]:
    """(input, output, thinking) tokens from an OpenAI-style usage block."""
    u = getattr(resp, "usage", None)
    p, c, t = _int(u, "prompt_tokens"), _int(u, "completion_tokens"), _int(u, "total_tokens")
    r = _int(getattr(u, "completion_tokens_details", None), "reasoning_tokens")
    if not thinking_outside_completion:
        return p, c, r
    # Google bills thinking as output. Its OpenAI-compatible usage can leave thinking out of completion_tokens
    # while total_tokens still counts it, so output is completion plus whatever total has beyond prompt and
    # completion. When total is missing, the reported reasoning tokens are added instead.
    extra = max(0, t - p - c) if t else r
    return p, c + extra, max(extra, r)


def _chat_result(resp, price: tuple[float, float], thinking_outside_completion: bool = False) -> LLMResult:
    choices = getattr(resp, "choices", None) or []
    if not choices:
        err = getattr(resp, "error", None) or (getattr(resp, "model_extra", None) or {}).get("error")
        raise EmptyResponseError("response had no choices" + (f": {err}" if err else ""))
    choice = choices[0]
    text = getattr(getattr(choice, "message", None), "content", None) or ""
    inp, out, _ = _usage(resp, thinking_outside_completion)
    return LLMResult(text=text, input_tokens=inp, output_tokens=out, cost_usd=cost_usd(price, inp, out),
                     stop_reason=getattr(choice, "finish_reason", None))


def _env_suffix(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "_", name).upper()


def vllm_endpoint(name: str) -> tuple[str, str]:
    """(base URL, served model id) for "vllm/<name>"."""
    load_env()
    suffix = _env_suffix(name)
    url_var = f"VLLM_BASE_URL_{suffix}"
    base_url = os.environ.get(url_var, "").strip()
    if not base_url:
        raise ConfigError(f"{url_var} is not set. Put it in compute.env (see docs/modal_runbook.md).")
    served = os.environ.get(f"VLLM_MODEL_{suffix}", "").strip() or VLLM_SERVED.get(name)
    if not served:
        raise ConfigError(f"vllm/{name} has no served model id. Add it to VLLM_SERVED in swarm/llm.py or set"
                          f" VLLM_MODEL_{suffix}.")
    return base_url, served


def vllm_reasoning_args(served: str, max_tokens: int) -> dict:
    """Lowest reasoning each open model allows, as in collusion-sting: gpt-oss effort low, with room in max_tokens
    for its reasoning, and thinking off for everything else (Qwen3 reads the switch; other templates ignore it)."""
    if "gpt-oss" in served.lower():
        return {"reasoning_effort": "low", "max_tokens": max(max_tokens, GPTOSS_MIN_MAX_TOKENS)}
    return {"max_tokens": max_tokens, "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}


def _call_vllm(name, system, user, max_tokens, temperature, replicate) -> LLMResult:
    base_url, served = vllm_endpoint(name)
    api_key = os.environ.get("VLLM_API_KEY", "").strip() or "EMPTY"
    client = _client(("vllm", base_url), lambda: make_openai_client(base_url, api_key, VLLM_TIMEOUT_S))
    resp = _chat(client, served, system, user, temperature, vllm_reasoning_args(served, max_tokens))
    return _chat_result(resp, price_for("vllm", name))


def _reported_cost(resp) -> Optional[float]:
    usage = getattr(resp, "usage", None)
    value = getattr(usage, "cost", None)
    if value is None:
        value = (getattr(usage, "model_extra", None) or {}).get("cost")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _call_openrouter(name, system, user, max_tokens, temperature, replicate) -> LLMResult:
    api_key = _need_env("OPENROUTER_API_KEY")
    client = _client("openrouter", lambda: make_openai_client(OPENROUTER_BASE_URL, api_key))
    # usage.include asks OpenRouter to put the call's cost in the response.
    args = {"max_tokens": max_tokens, "extra_body": {"usage": {"include": True}}}
    resp = _chat(client, name, system, user, temperature, args)
    res = _chat_result(resp, (0.0, 0.0))
    reported = _reported_cost(resp)
    fallback = _lookup(OPENROUTER_PRICES, name)
    if reported is not None:
        res.cost_usd = round(reported, 8)
    elif fallback is not None:
        res.cost_usd = cost_usd(fallback, res.input_tokens, res.output_tokens)
    else:
        _warn_once(f"openrouter-cost:{name}", "OpenRouter reported no cost for %s and OPENROUTER_PRICES has no"
                   " entry, so its spend counts as $0 against the budget", name)
    return res


def _rejects_effort(e: BaseException) -> bool:
    # Any 400 climbs the ladder, because Google's wording for an unsupported effort is not pinned down. A wrong
    # climb only costs rejected (unbilled) calls, and a floor is remembered only after a call at it succeeds.
    return status_of(e) == 400


def _raise_effort_floor(model_id: str, index: int, why: str) -> None:
    with _lock:
        if index <= _effort_floor.get(model_id, 0):
            return
        _effort_floor[model_id] = index
    log.warning("gemini/%s: using reasoning_effort=%r from now on, because %s", model_id,
                GEMINI_EFFORT_LADDER[index], why)


def gemini_effort(model_id: str) -> Optional[str]:
    """The reasoning_effort the next gemini/<model_id> call tries first."""
    with _lock:
        return GEMINI_EFFORT_LADDER[_effort_floor.get(model_id, 0)]


def _call_gemini(name, system, user, max_tokens, temperature, replicate) -> LLMResult:
    model_id = name.removeprefix("models/")
    price = price_for("gemini", model_id)
    api_key = _need_env("GEMINI_API_KEY")
    client = _client("gemini", lambda: make_openai_client(GEMINI_BASE_URL, api_key))
    base, effort = split_effort(name)
    if effort is not None:  # a set effort above the floor: no ladder
        resp = _chat(client, base, system, user, temperature, {"max_tokens": max_tokens, "reasoning_effort": effort})
        return _chat_result(resp, price, thinking_outside_completion=True)
    with _lock:
        start = _effort_floor.get(model_id, 0)
    for i in range(start, len(GEMINI_EFFORT_LADDER)):
        effort = GEMINI_EFFORT_LADDER[i]
        args: dict[str, Any] = {"max_tokens": max_tokens}
        if effort is not None:
            args["reasoning_effort"] = effort
        try:
            resp = _chat(client, name, system, user, temperature, args)
        except Exception as e:  # noqa: BLE001
            if i + 1 < len(GEMINI_EFFORT_LADDER) and _rejects_effort(e):
                continue  # this model does not accept this effort; try the next one up
            raise
        if i > start:
            _raise_effort_floor(model_id, i, "the lower settings were rejected")
        res = _chat_result(resp, price, thinking_outside_completion=True)
        if effort == "none" and _usage(resp, True)[2] > 0:
            _raise_effort_floor(model_id, i + 1, "it still thought with reasoning_effort='none'")
        return res
    raise AssertionError("unreachable")


def _call_openai(name, system, user, max_tokens, temperature, replicate) -> LLMResult:
    price = price_for("openai", name)
    api_key = _need_env("OPENAI_API_KEY")
    client = _client("openai", lambda: make_openai_client(OPENAI_BASE_URL, api_key))
    # Reasoning models reject max_tokens; max_completion_tokens works for every current chat model.
    args = {"max_completion_tokens": max_tokens}
    if name in OPENAI_REASONING:
        args["reasoning_effort"] = OPENAI_REASONING[name]
    tier = os.environ.get("SWARM_OPENAI_SERVICE_TIER")
    if tier:
        args["service_tier"] = tier
    resp = _chat(client, name, system, user, temperature, args)
    return _chat_result(resp, price)


# ---------------------------------------------------------------- mock


def _call_mock(name, system, user, max_tokens, temperature, replicate) -> LLMResult:
    """Deterministic offline answers. "fail-<status>" always raises that HTTP status, "flaky-<k>" raises 503 on the
    first k calls of each request, and "slow-<ms>" holds each call for that many milliseconds."""
    key = cache_key("mock/" + name, system, user, max_tokens, temperature, replicate)
    with _lock:
        n = _mock_calls[key] = _mock_calls.get(key, 0) + 1
        _mock_stats["calls"] += 1
    if m := re.fullmatch(r"fail-(\d{3})", name):
        raise MockAPIError(int(m.group(1)))
    if (m := re.fullmatch(r"flaky-(\d+)", name)) and n <= int(m.group(1)):
        raise MockAPIError(503)
    if m := re.fullmatch(r"slow-(\d+)", name):
        with _lock:
            _mock_stats["inflight"] += 1
            _mock_stats["peak"] = max(_mock_stats["peak"], _mock_stats["inflight"])
        try:
            time.sleep(int(m.group(1)) / 1000)
        finally:
            with _lock:
                _mock_stats["inflight"] -= 1
    body = json.dumps({"mock": name, "digest": key[:12], "replicate": replicate})
    text = f"Mock answer from {name}.\n```json\n{body}\n```"
    inp = max(1, (len(system) + len(user)) // 4)
    out = min(max_tokens, max(1, len(text) // 4))
    return LLMResult(text=text, input_tokens=inp, output_tokens=out, cost_usd=cost_usd(MOCK_PRICE, inp, out),
                     stop_reason="end_turn")


def mock_stats() -> dict:
    """Mock calls made so far in this process, and the most that were in flight at once."""
    with _lock:
        return dict(_mock_stats)


_ROUTES = {"claude": _call_claude, "vllm": _call_vllm, "openrouter": _call_openrouter, "gemini": _call_gemini,
           "openai": _call_openai, "mock": _call_mock}


# ---------------------------------------------------------------- entry point


def resolve_cache(cache=None) -> Optional[Cache]:
    """None means the default cache (SWARM_CACHE_PATH or data/cache/llm.sqlite), and False means no cache."""
    if cache is None:
        return cache_mod.default_cache()
    if cache is False:
        return None
    return cache


def _from_cache(hit: dict) -> LLMResult:
    return replace(LLMResult.from_dict(hit), cost_usd=0.0, cached=True, error=None)


def cached_result(model: str, system: Optional[str], user: str, max_tokens: int, temperature=None,
                  replicate: int = 0, *, cache=None) -> Optional[LLMResult]:
    """The cached answer for this request, at zero cost, or None. Makes no call."""
    store = resolve_cache(cache)
    if store is None:
        return None
    hit = store.get(cache_key(model, system or "", user, max_tokens, temperature, replicate))
    return _from_cache(hit) if hit is not None else None


def complete(model: str, system: Optional[str], user: str, max_tokens: int, temperature: Optional[float] = None,
             replicate: int = 0, *, cache=None) -> LLMResult:
    """Answer one request, from the cache if it holds it. `cache` takes a Cache, None for the default, or False."""
    provider, name = split_model(model)
    system = system or ""
    if not isinstance(user, str) or not isinstance(system, str):
        raise TypeError("system and user must be strings")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
        raise ValueError(f"max_tokens must be a positive integer, not {max_tokens!r}")
    if provider != "mock":
        load_env()
    store = resolve_cache(cache)
    key = cache_key(model, system, user, max_tokens, temperature, replicate)
    if store is not None:
        hit = store.get(key)
        if hit is not None:
            return _from_cache(hit)
    route = _ROUTES[provider]

    def attempt() -> LLMResult:
        t0 = time.monotonic()
        res = route(name, system, user, max_tokens, temperature, replicate)
        res.latency_s = round(time.monotonic() - t0, 3)
        return res

    try:
        res = with_retries(attempt, what=model, base=BACKOFF_BASE_S[provider])
    except ConfigError:
        raise
    except Exception as e:  # noqa: BLE001
        if status_of(e) is None and not is_retryable(e):
            raise  # a bug or a broken install, not an answer from the provider
        note = f" (gave up after {MAX_ATTEMPTS} attempts)" if is_retryable(e) else ""
        return LLMResult(error=describe(e) + note, model=model)
    res.model, res.cached, res.error = model, False, None
    if store is not None:
        store.put(key, model, res.to_dict(), res.cost_usd)
    return res
