"""Find the answer object in model output with parse_json_block(text), which returns a dict or None.

The rule is that the last JSON-looking value in the text is the answer.

1. Reasoning is dropped first. That covers anything up to a closing </think> tag, anything from an unclosed
   <think> tag on, and anything before gpt-oss's final-channel marker when vLLM's reasoning parser was off.
2. A JSON-looking value starts at "{" followed by a quote or "}", or at "[" followed by "{". It ends at its
   matching bracket, and brackets inside strings and comments do not count. Prose braces such as "{1, 2}" are
   ignored, and an object nested inside another value is never returned on its own.
3. The last such value is parsed as JSON, then with light repairs (trailing commas, comments, raw newlines in
   strings, Python's True, False and None), then as a Python dict literal, which covers single quotes.
4. The answer is None when the text has no JSON-looking value, when the last one is cut off (the usual sign of
   hitting max_tokens), when it cannot be repaired, or when it is an array. An earlier object is never returned
   in place of a broken last one, because an earlier object is usually a draft or an example.
"""

from __future__ import annotations

import ast
import io
import json
import re
import tokenize
import warnings
from typing import Any, Optional

_FINAL_MARK = "<|channel|>final<|message|>"
_THINK_OPEN = re.compile(r"<think\s*>", re.I)
_THINK_CLOSE = re.compile(r"</think\s*>", re.I)
_VALUE_START = re.compile(r"\{\s*[\"'}]|\[\s*\{")
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_TO_JSON = {"True": "true", "False": "false", "None": "null"}
_TO_PYTHON = {"true": "True", "false": "False", "null": "None"}
_MAX_PYTHON_LITERAL = 200_000
_FAIL = object()


def parse_json_block(text: Any) -> Optional[dict]:
    """The last top-level JSON object in `text`, or None. See the module docstring for the exact rule."""
    if not isinstance(text, str):
        return None
    text = _drop_reasoning(text)
    span = _last_value(text)
    if span is None or span[1] is None:
        return None
    value = _load(text[span[0]:span[1]])
    return value if isinstance(value, dict) else None


def _drop_reasoning(text: str) -> str:
    if _FINAL_MARK in text:
        text = text.rsplit(_FINAL_MARK, 1)[1]
    closes = list(_THINK_CLOSE.finditer(text))
    if closes:
        text = text[closes[-1].end():]
    opened = _THINK_OPEN.search(text)
    return text[:opened.start()] if opened else text


def _last_value(text: str) -> Optional[tuple[int, Optional[int]]]:
    """(start, end) of the last top-level JSON-looking value. end is None when it never closes."""
    last, i = None, 0
    while True:
        m = _VALUE_START.search(text, i)
        if m is None:
            return last
        end = _match(text, m.start())
        last = (m.start(), end)
        if end is None:
            return last
        i = end


def _string_end(text: str, start: int) -> Optional[int]:
    quote, i, n = text[start], start + 1, len(text)
    while i < n:
        c = text[i]
        if c == "\\":
            i += 2
            continue
        if c == quote:
            return i + 1
        i += 1
    return None


def _skip_comment(text: str, i: int) -> Optional[int]:
    """Index after a // or /* */ comment starting at i, or None if there is no comment there."""
    if text.startswith("//", i):
        nl = text.find("\n", i)
        return len(text) if nl == -1 else nl
    if text.startswith("/*", i):
        close = text.find("*/", i + 2)
        return len(text) if close == -1 else close + 2
    return None


def _match(text: str, start: int) -> Optional[int]:
    """Index just past the bracket that closes the one at `start`, or None if the text ends first."""
    depth, i, n = 0, start, len(text)
    while i < n:
        c = text[i]
        if c in "\"'":
            end = _string_end(text, i)
            if end is None:
                return None
            i = end
            continue
        if c == "/":
            after = _skip_comment(text, i)
            if after is not None:
                i = after
                continue
        if c in "{[":
            depth += 1
        elif c in "}]":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return None


def _rewrite(s: str, quotes: str, words: dict) -> str:
    """Copy s with strings in `quotes` untouched, comments and trailing commas removed, and bare words mapped."""
    out, i, n = [], 0, len(s)
    while i < n:
        c = s[i]
        if c in quotes:
            end = _string_end(s, i) or n
            out.append(s[i:end])
            i = end
            continue
        if c == "/":
            after = _skip_comment(s, i)
            if after is not None:
                i = after
                continue
        if c == ",":
            k = i + 1
            while k < n and s[k].isspace():
                k += 1
            if k < n and s[k] in "}]":
                i += 1
                continue
        m = _WORD.match(s, i)
        if m:
            out.append(words.get(m.group(0), m.group(0)))
            i = m.end()
            continue
        out.append(c)
        i += 1
    return "".join(out)


def _load(span: str) -> Any:
    for candidate in (span, _rewrite(span, '"', _TO_JSON)):
        try:
            return json.loads(candidate, strict=False)
        except (ValueError, RecursionError):
            pass
    return _python_literal(span)


def _adjacent_strings(source: str) -> bool:
    """True if two string tokens follow each other, which Python would silently join (a missing comma in JSON)."""
    skip = {tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.COMMENT, tokenize.ENDMARKER}
    prev = None
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type in skip:
                continue
            if tok.type == tokenize.STRING and prev == tokenize.STRING:
                return True
            prev = tok.type
    except (tokenize.TokenError, SyntaxError):
        return True
    return False


def _plain(v: Any) -> bool:
    if v is None or isinstance(v, (bool, int, float, str)):
        return True
    if isinstance(v, list):
        return all(_plain(x) for x in v)
    if isinstance(v, dict):
        return all(isinstance(k, str) and _plain(x) for k, x in v.items())
    return False


def _python_literal(span: str) -> Any:
    if len(span) > _MAX_PYTHON_LITERAL:
        return _FAIL
    source = _rewrite(span, "\"'", _TO_PYTHON)
    if _adjacent_strings(source):
        return _FAIL
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # JSON escapes such as \/ are invalid in Python and would warn
            value = ast.literal_eval(source)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return _FAIL
    return value if _plain(value) else _FAIL
