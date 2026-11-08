"""The storybook tour runs offline on the committed boards and verdicts."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STORIES = ["neutral", "retraction", "fake-record", "note", "alibi", "logged", "attacker"]


def tour(*args):
    return subprocess.run([sys.executable, str(ROOT / "scripts/tour.py"), *args], capture_output=True, text=True,
                          check=True, cwd=ROOT).stdout


def test_contents_lists_every_story():
    out = tour()
    for s in STORIES:
        assert s in out


@pytest.mark.parametrize("story", STORIES[:-1])
def test_story_shows_board_and_verdicts(story):
    out = tour(story)
    assert ">>" in out and "How the seven investigators ruled" in out


def test_attacker_story_shows_both_conditions():
    out = tour("attacker")
    assert "Against the standard log" in out and "Against effect logging" in out


def test_chosen_board():
    out = tour("retraction", "--seed", "1009", "--author", "claude/claude-haiku-5-5")
    assert "seed 1009" in out
