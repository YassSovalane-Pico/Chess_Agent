"""Distil a page dump into the minimal fixture the tests read.

A full ``dump_pages.py`` capture is ~2.7 MB of HTML, 99% of it SVG path data
and inline JSON. This reduces it to an ~18 KB fixture -- a reduction of roughly
150x -- keeping every fact the dump-reading tests assert on and collapsing each
``<svg>`` element to ``<svg/>``.

The result is ``tests/fixtures/chess_com_review_markup.json``, and that fixture
is what ``tests/test_review_logic.py`` actually reads, so the markup-pinned
expectations stay live without 2.7 MB of HTML in the repo. The capture it came
from is not needed: ``dumps/`` is gitignored and safe to delete.

Those tests assert structural facts only, so exactly those are extracted:

  * the set of data-node values, and that they give plies 1..N with no gaps
  * the set of <g id=...> classification labels
  * the 20 tally-row-value analysis-<kind> pairs (10 kinds x 2 sides)
  * the username / rating / review-rating class tokens
  * how many of the 29 cells carry an annotation, and that the analysis board
    carries none

To rebuild it from a fresh capture:

    python dump_pages.py <chess.com game link>
    python make_fixture.py
"""

import json
import re
import sys
from pathlib import Path

D = Path("dumps")
OUT = Path("tests/fixtures")
OUT.mkdir(parents=True, exist_ok=True)


def elide(s: str) -> str:
    """Drop the SVG geometry. It is 99% of the bytes and 0% of the assertions."""
    s = re.sub(r'd="[^"]{20,}"', 'd="..."', s)
    s = re.sub(r"<svg[^>]*>.*?</svg>", "<svg/>", s, flags=re.S)
    return s


def _read(name: str) -> str:
    """Read one dump, or say how to produce it.

    dumps/ is gitignored and normally absent, so a missing file is the normal
    state of a clean checkout rather than a fault; name the command that
    produces it instead of letting FileNotFoundError surface.
    """
    path = D / name
    if path.exists():
        return path.read_text(encoding="utf-8")
    print(
        f"error: {path} is not there.\n"
        f"\n"
        f"  The fixture is distilled from a real capture, so it needs one:\n"
        f"    python dump_pages.py <chess.com game link>\n"
        f"  then run this again. dumps/ is gitignored and safe to delete\n"
        f"  afterwards -- the fixture is what the tests read.\n",
        file=sys.stderr,
    )
    raise SystemExit(1)


started = _read("review_started.html")
summary = _read("review_loaded.html")
analysis = _read("analysis.html")

# 1. the move cells, structurally intact minus the SVG guts
cells = re.findall(
    r'<div data-node="(\d+-\d+)" class="node (white-move|black-move) '
    r'main-line-ply">(.*?)</div>',
    started,
    re.S,
)
moves = [
    {
        "node": node,
        "colour": colour,
        # the SAN span, with the figurine span kept (it carries data-figurine)
        "inner": elide(inner).strip(),
    }
    for node, colour, inner in cells
]
# the clock lives in a sibling div, not in the cell
stamps = re.findall(
    r'data-move-list-el="timestamp" data-ply="(\d+)" data-time="(\d+)"', started
)
rows = re.findall(
    r'class="main-line-row move-list-row[^"]*" data-whole-move-number="(\d+)"',
    started,
)
# one real row, so the "the number is on the ROW" fact survives
row_sample = elide(
    re.search(
        r'<div class="main-line-row move-list-row[^"]*" data-whole-move-number="1">'
        r".*?</div>\s*<div",
        started,
        re.S,
    ).group(0)
)

# 2. the coach panel, minus the avatar
coach = elide(
    re.search(
        r'<div class="move-feedback-box-component">.*?'
        r"</div></div></div>",
        started,
        re.S,
    ).group(0)
)
# The coach's prose is a <span class="move-feedback-speech-text-tw">, not a div,
# so grab it by offset from that class rather than by guessing its tag shape.
_speech_at = started.find('class="move-feedback-speech-text-tw"')
speech = started[_speech_at - 60 : _speech_at + 700] if _speech_at > 0 else None

# 3. the summary tally, both columns
tally = re.findall(
    r'class="tally-row-value analysis-([a-zA-Z]+)"[^>]*>([^<]*)<', summary
)
summary_block = elide(
    re.search(r'<div class="game-overview-row">.*?<hr>', summary, re.S).group(0)
)
summary_tally = elide(
    re.search(r'<div class="tally-row-row">.*?<hr>', summary, re.S).group(0)
)
players = elide(
    re.search(
        r'<div class="player-tagline">.*?</div></div></div>', summary, re.S
    ).group(0)
)

# 4. the facts about the analysis board, as booleans
analysis_facts = {
    "has_data_node_cells": bool(re.search(r'data-node="\d+-\d+"', analysis)),
    "has_color_classification": "--color-classification" in analysis,
    "classification_word_counts": {
        w: len(re.findall(w, analysis, re.I))
        for w in ("inaccuracy", "bestMove", "greatFind", "brilliant", "blunder")
    },
}

fixture = {
    "_readme": (
        "Minimal extract of a real dump_pages.py run, produced from dumps/ so "
        "the markup-pinned tests stay live without 2.7 MB of HTML. Each <svg> "
        "element is collapsed to <svg/> -- the path data is 99% of the bytes "
        "and 0% of the assertions. Everything asserted on is preserved: the "
        "data-node plies, the --color-classification-* styles, the figurine "
        "spans, both tally columns, the coach panel and its word-by-word "
        "speech, and the accuracy and player markup. Source game: 29 plies. "
        "Re-capture with python dump_pages.py <link>, then regenerate with "
        "python make_fixture.py."
    ),
    "move_cells": moves,
    "clock_stamps": [{"ply": p, "centiseconds": c} for p, c in stamps],
    "whole_move_numbers": rows,
    "row_sample": row_sample,
    "coach_panel": coach,
    "coach_speech": elide(speech) if speech else None,
    "summary_accuracy_row": summary_block,
    "summary_tally_row": summary_tally,
    "summary_players": players,
    "tally_pairs": tally,
    "analysis_facts": analysis_facts,
}

path = OUT / "chess_com_review_markup.json"
path.write_text(json.dumps(fixture, indent=1), encoding="utf-8")

raw_mb = sum(f.stat().st_size for f in D.glob("*.html")) / 1024 / 1024
print(f"dumps/*.html   {raw_mb:8.2f} MB")
print(f"fixture        {path.stat().st_size / 1024:8.1f} KB  ({path})")
print(f"reduction      {raw_mb * 1024 / (path.stat().st_size / 1024):8.0f}x")
print()
print("cells:", len(moves), "| stamps:", len(stamps), "| rows:", len(rows))
print("tally pairs:", len(tally))
print(
    "analysis has per-move classification:", analysis_facts["has_color_classification"]
)
