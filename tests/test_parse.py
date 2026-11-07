"""parse_json_block: the last JSON-looking value in model text is the answer."""

import pytest

from swarm.parse import parse_json_block


@pytest.mark.parametrize("text, expected", [
    ('{"a": 1}', {"a": 1}),
    ('  {"a": 1}  \n', {"a": 1}),
    ('Here you go:\n```json\n{"verdict": "yes", "ids": [3, 7]}\n```\nLet me know if you need more.',
     {"verdict": "yes", "ids": [3, 7]}),
    ('```\n{"a": 1}\n```', {"a": 1}),
    ('First thought: {"verdict": "no"} looked wrong.\nFinal answer:\n{"verdict": "yes"}', {"verdict": "yes"}),
    ('<think>maybe {"a": 1}</think>\n{"a": 2}', {"a": 2}),
    ('reasoning that mentions {"a": 1}...</think>{"a": 3}', {"a": 3}),
    ('<|channel|>analysis<|message|>try {"a": 1}<|end|><|start|>assistant<|channel|>final<|message|>{"a": 4}',
     {"a": 4}),
    ('{"outer": {"inner": {"deep": 1}}, "list": [{"x": 1}]}', {"outer": {"inner": {"deep": 1}}, "list": [{"x": 1}]}),
    ('{"text": "braces } { and \\"quotes\\" inside"}', {"text": 'braces } { and "quotes" inside'}),
    ('{"note": "it\'s fine"}', {"note": "it's fine"}),
    ('The set {1, 2} is small, see [1]. Answer: {"a": 1}', {"a": 1}),
    ('{"a": [1, 2]} as cited in [1]', {"a": [1, 2]}),
    ('{"a": [1, 2,], "b": 3,}', {"a": [1, 2], "b": 3}),
    ('{"a": True, "b": None, "c": False}', {"a": True, "b": None, "c": False}),
    ("{'a': 'single', 'b': True, 'c': None, 'd': [1, 2,]}", {"a": "single", "b": True, "c": None, "d": [1, 2]}),
    ("{'a': true, 'b': null}", {"a": True, "b": None}),
    ('{"a": 1, // a comment\n "b": 2 /* another */}', {"a": 1, "b": 2}),
    ('{"note": "line one\nline two"}', {"note": "line one\nline two"}),
    ('{"url": "http://example.com/a//b", "path": "x/*y*/z"}', {"url": "http://example.com/a//b", "path": "x/*y*/z"}),
    ('{"esc": "a\\/b \\u00e9"}', {"esc": "a/b é"}),
    ('{}', {}),
    ('{"a": 1}{"b": 2}', {"b": 2}),
    ('{"n": -1.5e3, "big": 12345678901234567890}', {"n": -1500.0, "big": 12345678901234567890}),
])
def test_finds_the_answer(text, expected):
    assert parse_json_block(text) == expected


@pytest.mark.parametrize("text", [
    None,
    "",
    "   ",
    42,
    ["{\"a\": 1}"],
    "no json here",
    "{not json}",
    "The set {1, 2} has two members.",
    '{"a": 1, "b": [1, 2',                                  # cut off by max_tokens
    'Draft: {"a": 1}\nFinal: {"a": 2, "b": "unterminated',  # a cut-off answer is not replaced by the draft
    '{"a": {"b": 1}, "c": "unterminated}',                  # nor by a fragment of itself
    '{"a": 1}\nThen: {"b": oops}',                          # a broken last answer is not replaced by an earlier one
    '[{"a": 1}, {"b": 2}]',                                 # an array is not an object
    '<think>{"a": 1} and then the output stopped',          # stopped inside the reasoning
    '{"a": "x" "b": "y"}',                                  # a missing comma is not silently repaired
    "{'list': ['a' 'b']}",                                  # nor by Python's implicit string joining
    "{'a': (1, 2)}",                                        # a tuple is not JSON
])
def test_returns_none(text):
    assert parse_json_block(text) is None


def test_large_and_deep_inputs_do_not_crash():
    big = "word {x} " * 50_000 + '{"answer": "' + "x" * 10 + '"}'
    assert parse_json_block(big) == {"answer": "x" * 10}
    deep = '{"a": ' * 5000 + "1" + "}" * 5000
    assert parse_json_block(deep) is None or isinstance(parse_json_block(deep), dict)
