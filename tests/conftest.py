"""Every test runs offline. Sockets are blocked, the cache is a temp file, compute.env is never read, provider
variables are cleared, and backoff sleeps are recorded instead of slept."""

import os
import socket

import pytest

from swarm import llm

PROVIDER_ENV = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_WORKSPACE_ID",
    "SWARM_ANTHROPIC_BASE_URL",
    "GEMINI_API_KEY",
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "VLLM_API_KEY",
    "SWARM_BUDGET_USD",
)


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    def blocked(*args, **kwargs):
        raise RuntimeError("tests must not open network connections")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    for name in PROVIDER_ENV:
        monkeypatch.delenv(name, raising=False)
    for name in list(os.environ):
        if name.startswith(("VLLM_BASE_URL_", "VLLM_MODEL_")):
            monkeypatch.delenv(name)
    monkeypatch.setenv("SWARM_CACHE_PATH", str(tmp_path / "cache" / "llm.sqlite"))
    monkeypatch.setattr(llm, "load_env", lambda: None)
    sleeps: list[float] = []
    monkeypatch.setattr(llm, "_sleep", sleeps.append)
    llm.reset_state()
    yield sleeps
    llm.reset_state()
