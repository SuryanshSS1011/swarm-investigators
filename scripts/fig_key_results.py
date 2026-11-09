"""The headline chart: the panel under the abstract on page 1 of the report, and the project image.

    python scripts/fig_key_results.py      (reads results/main/secondary.json, writes results/main/fig0_key_results.png
                                            and .pdf for the report, and results/main/fig0_headline.png for the project page)

Every bar uses the same boards and the same seven investigators: the five retraction cells written by Haiku and
Gemini 3.8 Flash, from the alibi and effect-logging follow-ups.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import figures as F  # noqa: E402

plt = F.plt


def bars():
    S = json.loads((F.OUT / "secondary.json").read_text())
    return [("Neutral post", 100 * S["e1"]["standard"]["neutral"], F.REF),
            ("The culprit retracts its admission", 100 * S["e1"]["standard"]["spoof"], F.ORANGE),
            ("Retraction plus a peer's alibi", 100 * S["e1"]["standard"]["self_vouch"], "#b8461c"),
            ("Retraction, with effect logging", 100 * S["e2"]["logged"]["spoof"], F.AQUA)]


def draw(w, h, s, title_lines, sub, label_w, pad_top, margin=0.02, footer=None):
    """w and h in inches, s scales every font, label_w is the label column and pad_top the space for the titles."""
    fig = plt.figure(figsize=(w, h))
    rows = bars()
    x0, x1 = label_w, w - 0.55 * s
    row_h = (h - pad_top - (0.08 if s < 1.5 else 0.34) * s) / len(rows)
    T = lambda x, y, t, **k: fig.text(x / w, y / h, t, **k)  # noqa: E731
    y = h - 0.05 * s
    for line in title_lines:
        y -= 0.19 * s
        T(margin, y, line, fontsize=9 * s, fontweight="semibold", color=F.INK, va="center")
    if sub:
        T(margin, y - 0.17 * s, sub, fontsize=7.4 * s, color=F.MUTED, va="center")
    if footer:
        T(margin, 0.13 * s, footer, fontsize=6.4 * s, color=F.MUTED, va="center")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    X = lambda p: x0 + (x1 - x0) * p / 100  # noqa: E731
    top = h - pad_top
    ax.plot([X(0), X(0)], [top - row_h * len(rows), top], color=F.BASE, lw=0.8 * s, zorder=3)
    for i, (label, v, c) in enumerate(rows):
        yc = top - row_h * (i + 0.5)
        T(x0 - 0.1 * s, yc, label, fontsize=8.6 * s, color=F.INK, ha="right", va="center")
        ax.add_patch(F.FancyBboxPatch((X(0), yc - row_h * 0.3), X(v) - X(0), row_h * 0.6,
                                      boxstyle=f"round,pad=0,rounding_size={0.03 * s}", color=c, lw=0, zorder=2))
        T(X(v) + 0.07 * s, yc, f"{F.half_up(v)}%", fontsize=9 * s, fontweight="bold", color=F.INK, va="center")
    return fig


def main() -> None:
    # In the report the caption carries the title, so the panel has none.
    fig = draw(F.WIDTH, 0.82, 1.0, [], "", 2.35, 0.06)
    F.save(fig, "fig0_key_results")
    # The project image is the same chart with a title on top, at the same row height as the report version.
    fig = draw(F.WIDTH, 1.33, 1.0, ["One retraction halves detection, and logging what tool calls did restores it"],
               "Detection of the planted violation by seven LLM investigators, on the same boards in every row", 2.35, 0.57, margin=0.08)
    fig.savefig(F.OUT / "fig0_headline.png", dpi=400)
    plt.close(fig)
    for label, v, _ in bars():
        print(f"{label:38s} {v:5.1f}%")


if __name__ == "__main__":
    main()
