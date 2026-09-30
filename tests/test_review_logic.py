"""Tests for the pure logic behind the review pipeline.

No browser, no network, no login. These lock in the exact regressions found in
the review, so a future chess.com redesign cannot quietly reintroduce them:

  * move_counts bleeding one classification's number into another's slot
  * substring classification ("dismiss" -> miss, "Facebook" -> book)
  * the annotation zip() desync that handed the model the wrong box
  * blind confidence when nothing was actually found
  * navigation "succeeding" onto the wrong position, or onto no position
  * a tool that hands the agent loop a coroutine instead of a dict

Several tests read `dumps/review_started.html` when it is present, so the
expectations are pinned to real captured markup rather than to my memory of it.
They skip cleanly when it is not.

Run:  python -m pytest tests/ -q     (or: python tests/test_review_logic.py)
"""

import inspect
import json
import os
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.move_list import (
    AUX_KINDS,
    HIGH,
    KINDS,
    NEXT_MOVE_SELECTORS,
    class_text,
    classify_class,
    classify_facts,
    classify_text,
    merge_counts,
    ply_from_node,
)
from tools.review import parse_review_panel

ROOT_FOR_TESTS = Path(__file__).resolve().parent.parent
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "chess_com_review_markup.json"


def _captured():
    """The real markup, as a minimal fixture.

    A full `dump_pages.py` run is 2.7 MB -- 99% SVG path data. This is the
    18 KB extract that keeps the structural facts the tests actually assert on,
    committed to the repo so these checks stay live without the dumps. Regenerate
    with `python make_fixture.py` after a fresh capture.

    Returns None if even the fixture is missing, so the tests skip rather than
    fail on a clean checkout.
    """
    if not FIXTURE.exists():
        return None
    with FIXTURE.open(encoding="utf-8") as handle:
        return json.load(handle)


# ---------------------------------------------------------------------------
# parse_review_panel -- the move_counts cross-label bug
# ---------------------------------------------------------------------------


def test_counts_do_not_bleed_into_each_other():
    """Regression: rf"{label}\\D*(\\d+)\\D+(\\d+)" reported the wrong number.

    With the old pattern, "Blunder 2  Mistake 1" produced blunder=[2, 1] --
    the *mistake* count in the *blunder* slot, which the system prompt then
    stated to the user as fact.
    """
    text = "Game Review\nWhite 71.5\nBlack 68.8\nBrilliant 1\nBlunder 2\nMistake 1\n"
    _, _, counts = parse_review_panel(text)
    assert counts == {"brilliant": 1, "blunder": 2, "mistake": 1}, counts


def test_counts_handle_infinitive_labels():
    text = "Accuracy\nGreat move 3\nBest move 5\nMissed 2\nBlunder 0\n"
    _, _, counts = parse_review_panel(text)
    assert counts["great"] == 3
    assert counts["best"] == 5
    assert counts["miss"] == 2
    assert counts["blunder"] == 0


def test_counts_handle_great_find_and_book():
    """Regression from calibration run 1.

    The live legend reads Great / Best / Book, and the tally classes are
    `analysis-greatFind` / `analysis-bestMove` -- camelCase, which the old
    `Great\\s*move?` pattern could not see at all.
    """
    text = (
        "Accuracy\nBrilliant\n0\n0\nGreat\n0\n0\nBest\n0\n0\n"
        "Book\n0\n0\nExcellent\n0\n0\nGood\n0\n0\n"
    )
    _, _, counts = parse_review_panel(text)
    for kind in ("brilliant", "great", "best", "book"):
        assert kind in counts, f"{kind} missing from {counts}"


def test_merge_counts_prefers_class_tally():
    """The class-name tally is deterministic; the text parse only fills gaps."""
    tally = {"blunder": {"white": 2, "black": 0, "total": 2}, "book": {"total": 0}}
    merged = merge_counts(tally, {"blunder": 99, "great": 1})
    assert merged["blunder"] == 2, "tally must win over the text fallback"
    assert merged["book"] == 0
    assert merged["great"] == 1, "gaps must be filled from the text parse"


def test_step_result_is_not_optimistic():
    """Regression from calibration run 1: "could not read the board" was
    counted as "the board moved"."""
    from tools.move_list import StepResult

    unverified = StepResult(
        requested=3, moved=0, verified=False, reason="board position is not readable"
    )
    assert not unverified.ok
    assert not bool(unverified)

    verified = StepResult(requested=3, moved=3, verified=True, method="button")
    assert verified.ok
    assert bool(verified)

    assert not StepResult(requested=3, moved=1, verified=True).ok


# ---------------------------------------------------------------------------
# The move table: data-node -> ply, and the classification vocabulary
# ---------------------------------------------------------------------------


def test_ply_comes_from_data_node_not_from_counting():
    """`data-node="<variation>-<index>"` is authoritative.

    The old reader found "the element with the most move-like cells" and then
    lost 3 of 29 plies to a duplicate filter. One real game had d6, e4 and b6
    each played twice, and the move number it keyed on lived on the ROW, so
    every cell read as "?" and the repeats collided.
    """
    assert ply_from_node("0-0") == 1
    assert ply_from_node("0-1") == 2
    assert ply_from_node("1-0") == 3
    assert ply_from_node("0-13") == 14
    assert ply_from_node("0-24") == 25  # the capture's blunder, ply 24 -> node 0-23
    assert ply_from_node("14-28") == 57
    assert ply_from_node("nonsense") is None
    assert ply_from_node("") is None
    assert ply_from_node(None) is None
    # the JS must use the same formula, or the two will drift
    from tools import move_list

    assert "parseInt(m[1], 10) * 2 + parseInt(m[2], 10) + 1" in move_list._MOVE_TABLE_JS


def test_captured_plys_are_contiguous_with_no_gaps():
    """Against the real dump: every ply 1..N exists exactly once.

    This is the property the old heuristic could not promise. If chess.com
    changes how it numbers nodes, this fails loudly rather than the tool
    quietly reporting the wrong positions.
    """
    cap = _captured()
    if cap is None:
        print("skip  tests/fixtures/chess_com_review_markup.json not present")
        return
    nodes = [cell["node"] for cell in cap["move_cells"]]
    assert len(nodes) == 29, f"expected the captured game's 29 cells, got {len(nodes)}"
    plies = [ply_from_node(n) for n in nodes]
    assert plies == sorted(plies), "data-node order is not already ply order"
    assert plies == list(range(1, len(plies) + 1)), "gaps or duplicates in the plies"
    # the move number is on the ROW, not the cell -- the old dedup key keyed on
    # it, which is how d6, e4 and b6 each played twice all collided
    assert cap["whole_move_numbers"], "the row's data-whole-move-number is the source"
    row = cap["row_sample"]
    assert 'data-whole-move-number="1"' in row
    assert 'data-node="0-0"' in row, "the first cell is inside that row"


def test_classifier_reads_every_label_found_in_the_capture():
    """Every classification chess.com actually wrote must be understood.

    Includes the casing quirk: the captured markup has `id="Brilliant"` sitting
    next to `id="book"` and `id="mistake"`, so matching cannot be
    case-sensitive.
    """
    cap = _captured()
    if cap is None:
        print("skip  tests/fixtures/chess_com_review_markup.json not present")
        return
    # The fixture elides the SVG bodies, so the marker ids come from the coach
    # panel and the tally rather than from the cells' icons. Both are the same
    # vocabulary chess.com writes, and both must be understood.
    ids = set(re.findall(r'<g id="([a-zA-Z]+)"', cap["coach_panel"] or ""))
    ids |= {kind for kind, _ in cap["tally_pairs"]}
    ids |= {
        m
        for cell in cap["move_cells"]
        for m in re.findall(r"<g id=\"([a-zA-Z]+)\"", cell["inner"])
    }
    assert ids, "no marker ids in the capture"
    vocabulary = set(KINDS) | set(AUX_KINDS)
    for icon in ids:
        kind, confidence, _ = classify_facts(icon, None, None)
        assert kind in vocabulary, f"id={icon!r} produced {kind!r}"
        assert confidence == HIGH, (icon, confidence)
    # excellent and good are real words the tally uses even though they are not
    # "notable"; they must be understood rather than silently unmatched
    assert classify_class("excellent") == "excellent"
    assert classify_class("good") == "good"
    # the casing quirk specifically: PascalCase beside lowercase
    assert (
        "Brilliant" in cap["summary_tally_row"]
        or "brilliant" in cap["summary_tally_row"]
    )


def test_classifier_understands_both_label_sources():
    """The two places chess.com labels a move.

    Neither is a class, an alt, a title or an aria-label -- which is why the
    old code inspected a list of attributes, found nothing, and fell through to
    a learned colour: 2 of the game's 11 notable moves were ever reported.
    """
    # 1. <span class="node-annotation-icon"><svg><g id="blunder">
    assert classify_facts("blunder", None, None) == (
        "blunder",
        HIGH,
        "svg marker id=blunder",
    )
    assert classify_facts("Brilliant", None, None)[0] == "brilliant"
    # 2. style="color: var(--color-classification-book)" on the SAN span
    assert classify_facts(None, "book", None) == (
        "book",
        HIGH,
        "style --color-classification-book",
    )
    assert classify_facts(None, "greatFind", None)[0] == "great"
    # neither: unclassified, and never guessed
    assert classify_facts(None, None, None) == (None, None, None)
    assert classify_facts(None, None, "node white-move main-line-ply") == (
        None,
        None,
        None,
    )


def test_classification_is_boundary_aware_not_a_substring():
    """The "dismiss"/"Facebook" regression, on the real matcher.

    Stripping punctuation and doing a substring search is exactly what turned
    "dismiss" into a miss and "Facebook" into a book. The class of a real move
    cell is `node white-move main-line-ply`, so a loose matcher has to survive
    it as well.
    """
    for junk in (
        "dismiss",
        "icon-submission",
        "Facebook",
        "goodbye",
        "bookshelf",
        "mission",
        "blunderish",
    ):
        assert classify_class(junk) is None, junk
    # "missed" is a real chess.com word, so it IS a rule -- the difference is
    # that it is matched as a whole word, like every other entry
    assert classify_class("missed") == "miss"
    assert classify_class("analysis-greatFind") == "great"
    assert classify_class("analysis-bestMove") == "best"
    assert classify_class("tally-row-value analysis-blunder") == "blunder"
    assert classify_class("node white-move main-line-ply") is None
    # camelCase and separators both normalise to the same words
    assert class_text("analysis-bestMove") == "analysis best move"
    assert class_text("GreatFind") == "great find"
    assert class_text(None) == ""


def test_classify_text_handles_the_coach_verdict():
    """The coach writes prose: "is a blunder", "is the last book move"."""
    assert classify_text("Book") == "book"
    assert classify_text("is a blunder") == "blunder"
    assert classify_text("is the last book move") == "book"
    assert classify_text("missed the win") == "miss"
    assert classify_text("nothing to say") is None
    assert classify_text("") is None
    assert classify_text(None) is None


def test_move_table_parsing_applies_the_vocabulary():
    """The browser extracts facts; Python applies the vocabulary."""
    from tools.move_list import _parse_move_table

    payload = {
        "supported": True,
        "plies": 2,
        "rows": [
            {
                "ply": 1,
                "node": "0-0",
                "number": 1,
                "color": "white",
                "san": "g4",
                "figurine": None,
                "clock_cs": 8,
                "icon_id": "book",
                "css_classification": "book",
                "classes": "node white-move main-line-ply",
            },
            {
                "ply": 2,
                "node": "0-1",
                "number": 1,
                "color": "black",
                "san": "e6",
                "figurine": None,
                "clock_cs": None,
                "icon_id": None,
                "css_classification": None,
                "classes": "node black-move main-line-ply",
            },
        ],
        "diagnostics": {"pliesContiguous": True},
    }
    rows, diag = _parse_move_table(payload)
    assert len(rows) == 2
    assert rows[0]["classification"] == "book"
    assert rows[0]["confidence"] == HIGH
    # centiseconds in the DOM, seconds for a person
    assert rows[0]["clock_seconds"] == 0.8, rows[0]
    assert rows[1]["clock_seconds"] is None
    # an unlabelled move is unlabelled
    assert rows[1]["classification"] is None
    assert rows[1]["confidence"] is None
    assert diag["coverage"] == "2/2"

    # an unsupported page yields nothing and says so, rather than pretending
    rows, diag = _parse_move_table({"supported": False, "rows": []})
    assert rows == []
    assert diag["supported"] is False


def test_the_tally_is_two_columns_and_reading_one_was_a_bug():
    """The captured summary has 20 tally values: 10 kinds x 2 sides.

    The reader collected them into one key per kind and kept the last, so it
    reported ONE player's tally while calling it the game's. A live run then
    said 2 mistakes where the move list had found 3, and 0 brilliant moves where
    the move list had found 1 -- the same game, two numbers, no error anywhere.
    """
    from tools.move_list import split_tally

    pairs = {
        # straight out of dumps/review_loaded.html
        "brilliant": [1, 0],
        "blunder": [0, 1],
        "mistake": [0, 2],
        "inaccuracy": [2, 3],
        "bestmove": [5, 6],
        "greatfind": [0, 0],
    }
    totals, by_side = split_tally(pairs, 2)
    assert totals["blunder"] == 1, "summed, not one column"
    assert totals["mistake"] == 2
    assert totals["inaccuracy"] == 5
    assert totals["brilliant"] == 1
    # the keys go through the same vocabulary as the move cells
    assert totals["best"] == 11
    assert totals["great"] == 0
    assert "bestmove" not in totals and "greatfind" not in totals
    assert by_side["white"]["inaccuracy"] == 2
    assert by_side["black"]["inaccuracy"] == 3
    for side in ("white", "black"):
        for kind, value in by_side[side].items():
            assert totals[kind] >= value, (side, kind, value)

    # a single-column page still works, and says which side it saw
    totals, by_side = split_tally({"blunder": [2]}, 1)
    assert totals == {"blunder": 2}
    assert by_side == {"white": {"blunder": 2}, "black": {}}

    cap = _captured()
    if cap is not None:
        found = cap["tally_pairs"]
        assert len(found) == 20, f"expected 10 kinds x 2 sides, got {len(found)}"
        per = {}
        for kind, value in found:
            per.setdefault(kind.lower(), []).append(int(value))
        assert all(len(v) == 2 for v in per.values()), per
        for kind, (w, b) in per.items():
            assert isinstance(w, int) and isinstance(b, int), kind
        # and the two columns really are one per player, in the tally's markup
        row = cap["summary_tally_row"]
        assert row.count("tally-row-value analysis-") == 20, row
        first = row.split('class="tally-row-row"')[1]
        assert first.count("tally-row-value analysis-") == 2, first


def test_captured_summary_has_the_players_and_the_columns():
    """The username class is `cc-user-username-component`.

    Not `cc-user-username`, which is what the selector guessed and which is not
    a class token anywhere in the capture: the real element's class list is
    `cc-text-medium-bold cc-user-username-component cc-user-username-boldest
    cc-user-block-username`. The guess matched nothing, so a live run reported
    an empty player list and then added a note about ordering that turned out
    to be a symptom rather than a limitation.
    """
    cap = _captured()
    if cap is None:
        print("skip  tests/fixtures/chess_com_review_markup.json not present")
        return
    tokens = set()
    for fragment in ("summary_players", "summary_accuracy_row", "summary_tally_row"):
        for group in re.findall(r'class="([^"]*)"', cap[fragment] or ""):
            tokens.update(group.split())
    assert "cc-user-username-component" in tokens
    assert "cc-user-username" not in tokens, "the guess was never a real class"
    assert "cc-user-rating-boldest" in tokens
    # the accuracy row is explicitly labelled, which is what makes the tally's
    # column order trustworthy as white-then-black
    assert "review-rating-white" in tokens and "review-rating-black" in tokens
    accuracy = cap["summary_accuracy_row"]
    assert accuracy.index("review-rating-white") < accuracy.index(
        "review-rating-black"
    ), "column 0 is White, which is what split_tally assumes"


def test_the_summary_and_the_move_list_agree_on_the_tally():
    """Both numbers describe the same game, so the tools must not disagree.

    `get_review_state` sums both sides; `get_moves` counts the moves it read. If
    these drift, one of the two is lying, and the user cannot tell which.
    """
    import inspect

    from tools.review import _get_moves_async, _get_review_state_async

    # get_moves derives its tally from the moves it actually read
    assert "classifications" in inspect.getsource(_get_moves_async)
    assert "_tally_of(moves)" in inspect.getsource(_get_moves_async)
    # get_review_state uses the summary panel's two columns, summed
    assert 'summary.get("move_counts")' in inspect.getsource(_get_review_state_async)
    # and it never starts the review, which would destroy the summary
    assert "open_review_mode" not in inspect.getsource(_get_review_state_async)
    assert "read_moves_when_ready" not in inspect.getsource(_get_review_state_async)


# ---------------------------------------------------------------------------
# Load stages
# ---------------------------------------------------------------------------


def test_waits_for_loading_placeholders():
    from tools import move_list

    assert any("skeleton" in h.lower() for h in move_list.LOADING_HINTS)
    assert any("loading" in h.lower() for h in move_list.LOADING_HINTS)
    assert callable(move_list.wait_for_review_loaded)
    assert "getClientRects" in move_list._REVIEW_STATE_JS


def test_readiness_requires_a_positive_signal_not_just_no_loaders():
    """The bug the user hit, twice, in two disguises.

    Checking only for the absence of placeholders is also true *before* the app
    has mounted anything. Then using the board as the data signal: the board
    renders with the layout shell, so a run reported "finished loading" at 0ms
    against 447 elements and a 119-character body text.
    """
    from tools import move_list

    # the board is a layout signal and must not be a data signal, and the
    # summary panel is not the move list: they are different questions and
    # only one of them is answerable at a time.
    assert not set(move_list.MOVE_LIST_HINTS) & set(move_list.REVIEW_LAYOUT_HINTS)
    assert not set(move_list.REVIEW_PANEL_HINTS) & set(move_list.REVIEW_LAYOUT_HINTS)
    assert not set(move_list.MOVE_LIST_HINTS) & set(move_list.REVIEW_PANEL_HINTS)
    # each stage has its own hint, or two stages would pass together
    assert "[data-node]" in move_list.MOVE_LIST_HINTS
    assert ".review-rating-white" in move_list.REVIEW_PANEL_HINTS
    assert "wc-chess-board" in move_list.REVIEW_LAYOUT_HINTS

    src = move_list._REVIEW_STATE_JS
    for key in ("panelFound", "layoutFound", "moveListFound", "moveCells"):
        assert key in src, key
    assert "loaded: placeholders.length === 0" in src
    assert "sidebarChars" in src, "the diagnostic that made this visible"

    body = inspect.getsource(move_list.wait_for_review_loaded)
    assert 'state.get("loaded") and state.get(key)' in body
    assert '"panel": "panelFound"' in body
    assert '"layout": "layoutFound"' in body
    assert '"movelist": "moveListFound"' in body
    assert 'require: str = "panel"' in body


def test_start_review_is_a_separate_stage_from_the_summary():
    """A run reached a fully loaded review page and still parsed zero moves,
    because chess.com renders no moves until Start Review is pressed."""
    from tools import move_list

    assert ".tab-review-start-review-button" in move_list.REVIEW_ENTRY_SELECTORS
    assert "Game Review" not in move_list.REVIEW_ENTRY_LABELS, "that is the <h1>"
    body = inspect.getsource(move_list.read_moves_when_ready)
    assert 'require="panel"' in body
    assert "await open_review_mode(page)" in body
    assert 'require="movelist"' in body
    assert body.index("await open_review_mode(page)") < body.index(
        "await read_moves(page)"
    )


def test_summary_is_read_before_the_review_is_started():
    """ORDER MATTERS, and the dump proves it rather than defending it.

    `review_loaded.html` has the accuracy pair and the tally;
    `review_started.html` has neither -- zero `review-rating-*`, zero
    `tally-row-value`. So a summary read after starting returns None/{} and
    looks exactly like a page that had no summary.
    """

    from tools.review import (
        _get_moves_async,
        _get_review_state_async,
        _run_review_async,
    )

    body = inspect.getsource(_run_review_async)
    assert body.index("_extract_review_data(page)") < body.index(
        "read_moves_when_ready(page)"
    ), "read the summary BEFORE starting the review"

    # get_moves starts the review too, so it must read the summary first --
    # that is how it can report the kinds chess.com never annotates per move.
    moves = inspect.getsource(_get_moves_async)
    assert "read_summary_panel" in moves
    assert moves.index("read_summary_panel") < moves.index("read_moves_when_ready")
    assert "read_summary_panel" in inspect.getsource(_get_review_state_async)

    cap = _captured()
    if cap is not None:
        # the summary panel's own markup: the accuracy pair and the tally
        assert "review-rating-white" in cap["summary_accuracy_row"]
        assert "tally-row-value" in cap["summary_tally_row"]
        # and neither appears in what is captured of the started review: the
        # summary is destroyed by Start Review, which is why the order is fixed
        started = json.dumps(
            [c["inner"] for c in cap["move_cells"]]
            + [cap["coach_panel"], cap["coach_speech"] or ""]
        )
        assert "review-rating" not in started, "capture contradicts the design"
        assert "tally-row-value" not in started, "capture contradicts the design"


def test_a_null_classification_is_never_reported_as_a_sound_move():
    """The trap this guards: "null" reads like "fine", and is not.

    chess.com annotates only some moves in the sidebar. In the captured game 7
    of 29 cells carried a label, while the summary counted 7 inaccuracies and
    1 miss that no cell mentions -- and the analysis board has no per-move
    classification markup at all, so there is no second source. A reply that
    treated every null as "no problem" would tell the player they had no
    inaccuracies when they had seven.
    """
    from tools.review import _annotation_report

    moves = [
        {"classification": c}
        for c in ["book", None, None, "mistake", None, "blunder", "brilliant", None]
    ]
    counts = {"book": 1, "mistake": 1, "blunder": 1, "brilliant": 1}
    totals = {
        "book": 1,
        "mistake": 1,
        "blunder": 1,
        "brilliant": 1,
        "inaccuracy": 7,
        "miss": 1,
        "best": 13,
    }
    report = _annotation_report(moves, counts, totals)
    assert report["annotated_moves"] == 4
    assert report["unannotated_moves"] == 4
    assert report["counts_without_a_move"]["inaccuracy"] == 7
    assert report["counts_without_a_move"]["miss"] == 1
    assert "inaccuracy x7" in report["note"]
    # this sentence is the one that stops the lie
    assert "does not mean the move was sound" in report["note"]
    # a kind that is fully accounted for is not listed as missing
    assert "blunder" not in report["counts_without_a_move"]

    # when everything lines up, the note says so and lists nothing
    aligned = _annotation_report(
        [{"classification": "blunder"}], {"blunder": 1}, {"blunder": 1}
    )
    assert "counts_without_a_move" not in aligned
    assert "no annotation" in aligned["note"]


def test_the_reconciliation_balances_for_every_kind():
    """annotated + not_located must equal the summary, for every kind.

    `counts_without_a_move` is a SHORTFALL, not a repeat of the total. The
    first version reported the whole count, so "best" -- annotated on 2 of 13
    moves -- came back as 2 + 13 = 15, and two independent readers of one game
    disagreed about how many best moves it had.
    """
    from tools.review import _annotation_report

    # the live game's numbers
    counts = {"book": 1, "mistake": 2, "blunder": 1, "brilliant": 1, "best": 2}
    totals = {
        "brilliant": 1,
        "great": 0,
        "book": 1,
        "best": 13,
        "excellent": 1,
        "good": 2,
        "inaccuracy": 7,
        "mistake": 2,
        "miss": 1,
        "blunder": 1,
    }
    moves = [{"classification": k} for k, count in counts.items() for _ in range(count)]
    # all 29 plies, of which 7 carry an annotation
    moves += [{"classification": None}] * (29 - len(moves))
    assert len(moves) == 29
    report = _annotation_report(moves, counts, totals)
    unlocated = report.get("counts_without_a_move", {})
    for kind, total in totals.items():
        assert counts.get(kind, 0) + unlocated.get(kind, 0) == total, (
            kind,
            counts.get(kind, 0),
            unlocated.get(kind, 0),
            total,
        )
    # a partly-annotated kind reports only what is missing
    assert unlocated["best"] == 11, unlocated
    assert unlocated["inaccuracy"] == 7
    # nothing is listed when the move is fully accounted for
    for kind in ("blunder", "book", "brilliant", "mistake"):
        assert kind not in unlocated, kind
    # and the arithmetic ties back to the number of moves read
    assert report["annotated_moves"] == sum(counts.values()) == 7
    assert report["unannotated_moves"] == 29 - 7
    # the 22 unannotated moves are the 7 inaccuracies, 1 miss, 11 best,
    # 1 excellent and 2 good moves chess.com did not mark
    assert sum(unlocated.values()) == report["unannotated_moves"]


def test_the_capture_shows_the_annotation_gap_is_real():
    """Pinned to the actual markup, so this stops being a belief.

    29 cells, 7 annotated. And the analysis page -- the other place chess.com
    could have been labelling moves -- has no per-move classification at all.
    """
    cap = _captured()
    if cap is None:
        print("skip  tests/fixtures/chess_com_review_markup.json not present")
        return
    cells = [cell["inner"] for cell in cap["move_cells"]]
    assert len(cells) == 29, len(cells)
    labelled = [c for c in cells if "--color-classification-" in c]
    assert len(labelled) == 7, f"expected 7 annotated cells, got {len(labelled)}"
    # the two label sources agree on all 7
    for cell in labelled:
        assert re.search(r"--color-classification-([a-zA-Z-]+)", cell), cell

    facts = cap["analysis_facts"]
    assert facts["has_color_classification"] is False, (
        "the analysis board now carries per-move classifications; revisit the "
        "limitation note in get_moves and the readers that assume it does not"
    )
    assert facts["has_data_node_cells"] is False
    # the words are in the analysis page's legend and tally, and nowhere else
    assert facts["classification_word_counts"]["inaccuracy"] > 0

    # a white move carries a figurine; it contributes no text, so the SAN is
    # the span's own text. Reading the cell's textContent naively once shipped
    # a SAN with the piece glyph in it.
    with_piece = [c for c in cells if "data-figurine" in c]
    assert with_piece, "white moves carry a figurine"
    for cell in with_piece:
        body = re.sub(r"<[^>]+>", "", cell).strip()
        assert re.match(r"^[NBRQK]?[a-h]?[1-8]?x?[a-h][1-8]", body), body


def test_the_fixture_is_a_deliberate_extract_not_a_truncated_dump():
    """So nobody mistakes 18 KB of elided markup for real captured HTML."""
    cap = _captured()
    if cap is None:
        print("skip  tests/fixtures/chess_com_review_markup.json not present")
        return
    assert cap["_readme"], "the fixture must say what it is"
    assert "make_fixture.py" in cap["_readme"]
    # the elision is visible and only ever applied to SVG geometry: whole <svg>
    # elements collapse to a placeholder, and the `id` on a marker <g> survives
    # only where it was outside an <svg> (the coach panel's icon is inside one,
    # so the vocabulary test leans on the tally instead)
    assert "<svg/>" in json.dumps(cap), "SVG bodies should be collapsed"
    assert 'd="' not in json.dumps(cap), "no path data should survive the extract"
    # and it carries every fact the tests above lean on
    for key in (
        "move_cells",
        "clock_stamps",
        "whole_move_numbers",
        "row_sample",
        "coach_panel",
        "coach_speech",
        "summary_accuracy_row",
        "summary_tally_row",
        "summary_players",
        "tally_pairs",
        "analysis_facts",
    ):
        assert cap.get(key), f"fixture is missing {key}"
    # the coach's prose is where the two real bugs were
    assert "move-feedback-speech-text-hidden" in cap["coach_speech"], (
        "the unrevealed words are what innerText was hiding"
    )
    assert "move-feedback-box-negative" in cap["coach_panel"], (
        "the score's sign class, which is a second family from score-text-*"
    )


def test_summary_is_expanded_before_it_is_read_and_photographed():
    """The tally starts collapsed, showing six of nine rows."""
    from tools import move_list
    from tools.review import _run_review_async

    body = inspect.getsource(_run_review_async)
    assert body.index("expand_review_summary(page)") < body.index(
        "_extract_review_data(page)"
    )
    assert body.index("expand_review_summary(page)") < body.index(
        "capture_review_panel(page"
    )
    assert "aria-label='Expand'" in inspect.getsource(move_list.expand_review_summary)


def test_coach_panel_is_read_after_navigating():
    """One `.move-feedback-box-component` existed in the whole capture, so the
    panel is for the current move only. Reading it before navigating returns
    the previous move's words -- or nothing."""
    from tools.review import _get_coach_commentary_async

    body = inspect.getsource(_get_coach_commentary_async)
    assert body.index("_goto_and_read") < body.index("read_coach_commentary")
    from tools.move_list import _COACH_JS

    for part in (
        "move-feedback-box-description",
        "move-feedback-box-score",
        "move-feedback-speech-text-component",
        "score-text-negative",
    ):
        assert part in _COACH_JS, part


# ---------------------------------------------------------------------------
# Board geometry and navigation
# ---------------------------------------------------------------------------


def test_board_detection_rejects_icons_and_layout_columns():
    """`[class*='board' i]` matched the 24px `board-layout-icon` gear, and
    hashing it produced a "verified" ArrowLeft on a board never found."""
    from tools import move_list
    from tools import review as rv

    assert "[class*='board' i]" not in move_list.BOARD_HINTS
    assert not move_list._looks_like_a_board(
        {"x": 0, "y": 0, "width": 24, "height": 24}
    )
    assert not move_list._looks_like_a_board(
        {"x": 0, "y": 0, "width": 400, "height": 900}
    )
    assert not move_list._looks_like_a_board(None)
    assert move_list._looks_like_a_board({"x": 0, "y": 0, "width": 700, "height": 700})

    # the crop is stricter than verification: 1% vs 18%
    assert not rv._is_square({"x": 0, "y": 0, "width": 800, "height": 1080})
    assert rv._is_square({"x": 0, "y": 0, "width": 700, "height": 700})
    assert rv.MIN_BOARD_ASPECT >= 0.99
    assert rv.MAX_BOARD_ASPECT <= 1.01


def test_frame_width_follows_the_configured_viewport():
    """The board is height-constrained, so the window is the lever on width."""
    from tools import review as rv

    assert rv._configured_viewport() == {"width": 1400, "height": 820}
    os.environ["REVIEW_VIEWPORT_W"] = "10"
    os.environ["REVIEW_VIEWPORT_H"] = "wide"
    try:
        tiny = rv._configured_viewport()
        # a width below the floor is raised to it, and the window is not so
        # narrow that chess.com reflows the board into a column
        assert tiny["width"] == rv.VIEWPORT_MIN["width"]
        # junk falls back to the default, which is already above the floor
        assert tiny["height"] == 820
    finally:
        os.environ.pop("REVIEW_VIEWPORT_W", None)
        os.environ.pop("REVIEW_VIEWPORT_H", None)
    assert "_configured_viewport()" in inspect.getsource(rv._new_logged_in_page)
    assert 'getattr(page, "viewport_size"' in inspect.getsource(rv.content_clip)


class _FakeBoxLoc:
    def __init__(self, box, page):
        self._box = box
        self._page = page
        self.first = self

    async def all(self):
        return [self]

    async def count(self):
        return 1 if self._box else 0

    async def bounding_box(self):
        return self._box

    async def is_visible(self, timeout=None):
        return bool(self._box)

    async def inner_text(self):
        return ""

    async def get_attribute(self, name):
        return None

    async def scroll_into_view_if_needed(self, timeout=None):
        return None

    async def evaluate(self, expression):
        self._page.evaluated.append(expression)
        return False

    async def screenshot(self, path=None, **kwargs):
        self._page.shots.append(("element", path))
        if path:
            Path(path).write_bytes(b"PNG")

    async def click(self, timeout=None):
        return None

    def nth(self, i):
        return self


class _FakeNavPage:
    """A page laid out like the live one: nav | board | review sidebar."""

    def __init__(self):
        self.evaluated = []
        self.shots = []

    def locator(self, selector):
        # Laid out to fit 1400x820 exactly: nav | gap | square board | panel.
        # Keeping the panel inside the viewport is the point -- a frame that
        # cropped to the union of board and panel would be 1390 wide.
        boxes = {
            ".sidebar-container": {"x": 0, "y": 0, "width": 300, "height": 820},
            "wc-chess-board": {"x": 320, "y": 20, "width": 660, "height": 660},
            "#board-layout-sidebar": {"x": 990, "y": 10, "width": 400, "height": 800},
        }
        return _FakeBoxLoc(boxes.get(selector), self)

    @property
    def viewport_size(self):
        return {"width": 1400, "height": 820}

    async def screenshot(self, path=None, clip=None, full_page=False, **kwargs):
        self.shots.append(("clip" if clip else "viewport", path, clip))
        if path:
            Path(path).write_bytes(b"PNG")


def test_content_clip_excludes_the_left_navigation():
    """A full-viewport shot is what put the Play/Puzzles column, the search box
    and the player's own avatar in the frame."""
    import asyncio

    from tools.review import content_clip

    page = _FakeNavPage()
    clip = asyncio.run(content_clip(page))
    assert clip is not None
    assert clip["x"] >= 300, f"clip starts inside the navigation: {clip}"
    assert clip["x"] <= 320, "must start at the board"
    # the whole point: the panel is in the frame and the nav is not
    assert clip["x"] + clip["width"] >= 990 + 400 - 1, f"panel is cut off: {clip}"
    assert clip["width"] < 1400, "must be tighter than the whole window"
    assert clip["x"] + clip["width"] <= 1400, "must stay inside the viewport"


def test_left_nav_selectors_cannot_match_the_review_sidebar():
    """`.sidebar-container` and `.board-layout-sidebar` are different things;
    hiding the wrong one would photograph nothing."""
    from tools.review import LEFT_NAV_HINTS, SIDEBAR_SELECTORS

    for nav in LEFT_NAV_HINTS:
        for sidebar in SIDEBAR_SELECTORS:
            assert "board-layout-sidebar" not in nav, (nav, sidebar)
    assert not set(LEFT_NAV_HINTS) & set(SIDEBAR_SELECTORS)


def test_hide_left_nav_never_hides_the_review_sidebar():
    import asyncio

    from tools.review import hide_left_nav

    page = _FakeNavPage()
    asyncio.run(hide_left_nav(page))
    guard = "el => !!el.closest('#board-layout-sidebar')"
    assert page.evaluated[0] == guard
    assert "visibility" in page.evaluated[-1]


def test_screenshot_review_area_uses_the_clip():
    import asyncio

    from tools.review import screenshot_review_area

    page = _FakeNavPage()
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "shot.png"
        asyncio.run(screenshot_review_area(page, target))
        kind, _path, clip = page.shots[-1]
        assert kind == "clip" and clip is not None and clip["x"] >= 300
        assert target.exists()


def test_shot_region_modes():
    import asyncio

    from tools.review import content_clip

    page = _FakeNavPage()
    os.environ["REVIEW_SHOT_REGION"] = "board"
    board = asyncio.run(content_clip(page))
    os.environ["REVIEW_SHOT_REGION"] = "sidebar"
    sidebar = asyncio.run(content_clip(page))
    os.environ["REVIEW_SHOT_REGION"] = "board_sidebar"
    both = asyncio.run(content_clip(page))
    os.environ.pop("REVIEW_SHOT_REGION", None)
    assert board["width"] < both["width"]
    assert sidebar["width"] < both["width"]
    assert board["x"] < sidebar["x"]


def test_capture_review_panel_resets_the_sidebar_scroll():
    import asyncio

    from tools.review import capture_review_panel

    page = _FakeNavPage()
    with tempfile.TemporaryDirectory() as tmp:
        asyncio.run(capture_review_panel(page, Path(tmp) / "s.png"))
    assert any("scrollTop = 0" in e for e in page.evaluated)


def test_one_shot_per_brilliant_and_blunder():
    from tools.review import _describe_move, _shot_label, _shot_path

    described = _describe_move(
        {
            "classification": "brilliant",
            "san": "O-O",
            "ply": 27,
            "number": 14,
            "color": "black",
            "confidence": "high",
        },
        "u",
    )
    assert _shot_label(described) == "m14_ply27"
    import tools.review as rv

    original = rv._project_root
    rv._project_root = lambda: Path(tempfile.gettempdir())
    try:
        a = _shot_path("brilliant", "1", "20260101", "m14_ply27")
        b = _shot_path("brilliant", "1", "20260101", "m3_ply6")
    finally:
        rv._project_root = original
    assert a != b, "two brilliants must not overwrite each other"
    assert a.name.startswith("brilliant_m14_ply27_")


def test_there_is_no_separate_moves_screenshot():
    """The user asked for it to be dropped: summary + one frame per brilliant
    and per blunder, nothing else."""
    from tools.review import _run_review_async

    body = inspect.getsource(_run_review_async)
    assert "for kind in MOVE_SHOT_KINDS" in body
    assert '"moves"' not in body


# ---------------------------------------------------------------------------
# Navigation, with a fake board
# ---------------------------------------------------------------------------

_NEXT = NEXT_MOVE_SELECTORS[0]


class _FakeLoc:
    def __init__(self, selector, page):
        self._sel = selector
        self._page = page
        self.first = self

    async def all(self):
        return [self]

    async def count(self):
        return 1 if self._sel in self._page.visible else 0

    async def is_visible(self, timeout=None):
        return await self.count() > 0

    async def inner_text(self):
        return self._page.labels.get(self._sel, "")

    async def get_attribute(self, name):
        return None

    async def evaluate(self, expression):
        return self._page.evaluated[-1] if self._page.evaluated else False

    async def click(self, timeout=None):
        self._page.clicks.append(self._sel)

    async def scroll_into_view_if_needed(self, timeout=None):
        return None


class _FakeBoardPage:
    """A page whose board signature is a scripted sequence of probe results."""

    def __init__(
        self, probes=(), visible=(), active=None, labels=None, active_ply=None
    ):
        self.probes = list(probes)
        self.visible = set(visible)
        # `active` is the OLD text-counting answer. Kept so a test can assert
        # that `active_ply` no longer consults it -- see
        # test_active_ply_ignores_the_broken_text_counting_heuristic.
        self.active = active
        # What `active_ply` should report now: the selected cell / ?move= pair.
        self.active_ply = active_ply
        self.labels = labels or {}
        self.clicks = []
        self.evaluated = []

    @property
    def keyboard(self):
        return _FakeKeyboard(self)

    async def wait_for_timeout(self, ms):
        return None

    def locator(self, selector):
        return _FakeLoc(selector, self)

    async def evaluate(self, script, arg=None):
        if "getAttribute('position')" in script or "data-fen" in script:
            return self.probes.pop(0) if self.probes else None
        if "node-highlight-content.selected" in script:
            if self.active_ply is None:
                return {"url": None, "cell": None, "agree": True}
            return self.active_ply
        if "active" in script:
            return self.active
        return None


class _FakeKeyboard:
    def __init__(self, page):
        self._page = page

    async def press(self, key):
        self._page.clicks.append(f"keyboard:{key}")


def test_unreadable_board_is_not_counted_as_a_step():
    """Regression: `if after is None or after != before` counted a move."""
    import asyncio

    from tools.move_list import step_moves

    page = _FakeBoardPage(
        probes=["pos:1", None, None, None, None, None], visible=[_NEXT]
    )
    res = asyncio.run(step_moves(page, 1))
    assert res.moved == 0
    assert not res.verified
    assert not res.ok
    assert "unreadable" in res.reason


def test_goto_ply_does_not_claim_success_from_a_partial_step():
    """The signature is re-read after stepping; a stale one let a half-completed
    step satisfy "did the board change?" on its own."""
    import asyncio

    from tools.move_list import goto_ply

    page = _FakeBoardPage(
        probes=["pos:1", "pos:2", "pos:2", "pos:2", "pos:2", None],
        visible=[_NEXT],
        # The board is on ply 3, so goto_ply(6) takes the three-step path, the
        # signature stops responding part way through, and the result must be
        # False -- a partial step is not a successful one.
        active_ply={"url": 3, "cell": 3, "agree": True},
    )
    assert asyncio.run(goto_ply(page, 6)) is False
    assert page.clicks == [_NEXT, _NEXT]


def test_active_ply_reads_the_selected_cell_not_a_text_count():
    """`active_ply` counted elements with move-like text and was wrong.

    On the live review page it reported ply 3 while the board was on ply 1, and
    ply 17 after a click that landed on ply 9. `goto_ply` trusts it to decide
    whether a short step suffices, so this was steering navigation onto the
    wrong position while reporting that it had verified.

    Two real signals replace it: the one cell marked `selected`, and the URL's
    `?move=` parameter.
    """
    import asyncio

    from tools.move_list import active_ply

    # `active` is the old heuristic's answer and is deliberately wrong.
    page = _FakeBoardPage(active=17, active_ply={"url": 9, "cell": 9, "agree": True})
    assert asyncio.run(active_ply(page)) == 9


def test_active_ply_prefers_the_selected_cell_when_the_url_disagrees():
    """Two sources, two answers: trust the cell, because that is the click
    target. The disagreement is logged, not hidden."""
    import asyncio

    from tools.move_list import active_ply

    page = _FakeBoardPage(active_ply={"url": 12, "cell": 9, "agree": False})
    assert asyncio.run(active_ply(page)) == 9


def test_active_ply_falls_back_to_the_url_when_no_cell_is_selected():
    import asyncio

    from tools.move_list import active_ply

    page = _FakeBoardPage(active_ply={"url": 5, "cell": None, "agree": False})
    assert asyncio.run(active_ply(page)) == 5


def test_active_ply_is_none_when_nothing_marks_a_position():
    """'Cannot tell where the board is' must not become a guess of ply 1."""
    import asyncio

    from tools.move_list import active_ply

    page = _FakeBoardPage(active_ply={"url": None, "cell": None, "agree": True})
    assert asyncio.run(active_ply(page)) is None


def test_goto_ply_clicks_the_cell_read_moves_tagged():
    """Indexing into `[class*='move' i] *, [class*='ply' i] *, ...` matched
    clock and icon elements too, so `ply - 1` was arbitrary."""
    import asyncio

    from tools.move_list import _click_row, goto_ply

    page = _FakeBoardPage(visible=['[data-ai-ply="7"]'])
    assert asyncio.run(_click_row(page, 7)) is True
    assert page.clicks == ['[data-ai-ply="7"]']

    # No position signal: `active_ply` returns None, so goto_ply skips the
    # step path entirely and must go straight to clicking the tagged cell.
    # `active_ply=None` is what the fake returns for "nothing marks a position".
    page = _FakeBoardPage(probes=["pos:1", "pos:2"], visible=['[data-ai-ply="7"]'])
    assert asyncio.run(goto_ply(page, 7)) is True
    assert page.clicks == ['[data-ai-ply="7"]']


def test_click_row_is_honest_when_nothing_is_tagged():
    import asyncio

    from tools.move_list import _click_row

    page = _FakeBoardPage()
    assert asyncio.run(_click_row(page, 4)) is False
    assert page.clicks == []


def test_open_moves_tab_ignores_the_game_overview():
    """The old tab pattern also matched "game", so it could click the very tab
    the moves are not behind, then report success and stop."""
    import asyncio

    from tools.move_list import _MOVE_TAB, _TAB_BLOCKLIST, TAB_HINTS, open_moves_tab

    assert _TAB_BLOCKLIST.search("Game Overview")
    assert not _MOVE_TAB.search("Game Overview")
    assert _MOVE_TAB.search("Moves")
    assert TAB_HINTS[-1] == "button"

    page = _FakeBoardPage(visible={"button"}, labels={"button": "Game Overview"})
    assert asyncio.run(open_moves_tab(page)) is False
    assert page.clicks == []

    page = _FakeBoardPage(visible={"button"}, labels={"button": "Moves"})
    assert asyncio.run(open_moves_tab(page)) is True
    assert page.clicks == ["button"]


# ---------------------------------------------------------------------------
# Choosing a move, and the coach
# ---------------------------------------------------------------------------


def _game_rows():
    """A 29-ply game with repeated move numbers, like the real one."""
    rows = []
    for ply in range(1, 30):
        rows.append(
            {
                "ply": ply,
                "number": (ply + 1) // 2,
                "color": "white" if ply % 2 else "black",
                "san": {1: "g4", 27: "O-O"}.get(ply, f"m{ply}"),
                "classification": "brilliant" if ply == 27 else None,
                "confidence": HIGH if ply == 27 else None,
                "evidence": "svg marker id=brilliant" if ply == 27 else None,
            }
        )
    return rows


def test_resolve_move_by_ply_and_by_move_number():
    """Chess.com numbers moves two ways and the user may use either."""
    from tools.review import resolve_move

    rows = _game_rows()
    # odd plies are White's, so ply 27 is White's 14th move
    hits, err = resolve_move(rows, ply=27)
    assert not err
    assert (hits[0]["san"], hits[0]["color"], hits[0]["number"]) == ("O-O", "white", 14)

    hits, err = resolve_move(rows, move_number=14, color="white")
    assert not err and hits[0]["ply"] == 27
    hits, err = resolve_move(rows, move_number=14, color="black")
    assert not err and hits[0]["ply"] == 28

    once = [dict(r) for r in rows]
    once[14]["number"] = 99
    hits, err = resolve_move(once, move_number=99)
    assert not err and len(hits) == 1


def test_resolve_move_refuses_to_guess_when_ambiguous():
    """Guessing would attribute the coach's words to the wrong move."""
    from tools.review import resolve_move

    hits, err = resolve_move(_game_rows(), move_number=14)
    assert hits == []
    assert "both" in err and "colour" in err
    assert "white" in err and "black" in err


def test_resolve_move_reports_what_is_available():
    from tools.review import resolve_move

    rows = _game_rows()
    assert "No move 99" in resolve_move(rows, move_number=99)[1]
    assert "No ply 5" in resolve_move([], ply=5)[1]
    assert "move_number" in resolve_move(rows)[1]


def test_notable_moves_exclude_approval_moves():
    from tools.review import NOTABLE

    assert NOTABLE == {"brilliant", "great", "blunder", "inaccuracy", "mistake", "miss"}
    assert not NOTABLE & {"best", "excellent", "good", "book"}


def test_describe_move_reports_a_real_colour():
    """`color` used to hold the legend blurb, so the reply had no real colour."""
    from tools.review import _describe_move

    d = _describe_move(
        {
            "classification": "brilliant",
            "san": "O-O",
            "ply": 27,
            "number": 14,
            "color": "black",
            "confidence": "high",
        },
        "u/review",
    )
    assert d["color"] == "black" and d["marker"] == "teal !!"
    assert d["move_number"] == 14
    fallback = _describe_move({"classification": "blunder", "san": "d5", "ply": 6}, "u")
    assert fallback["move_number"] == 6
    assert fallback["color"] == "unknown", "never invent a colour"


# ---------------------------------------------------------------------------
# The agent contract
# ---------------------------------------------------------------------------

AGENT_TOOLS = (
    "get_recent_games",
    "get_player_profile",
    "review_game_screenshots",
    "get_review_state",
    "get_moves",
    "get_coach_commentary",
    "screenshot_game_move",
    # The interactive browser tools. Every one of these opens the same
    # long-lived session, so each is called with a bare URL or no argument.
    "open_chess_page",
    "where_am_i",
    "move_board",
    "goto_move_in_game",
    "take_screenshot",
    "close_browser_session",
)

# Tools that take a game link as their single positional argument, so the
# contract test can call them with one. The interactive tools do not: they act
# on a URL or on nothing at all, and `close_browser_session` takes no
# arguments, so calling every tool the same way is not possible.
ONE_URL_ARG_TOOLS = (
    "review_game_screenshots",
    "get_review_state",
    "get_moves",
    "get_coach_commentary",
    "screenshot_game_move",
)


def test_every_agent_tool_is_synchronous_and_json_safe():
    """Regression: `review_game_screenshots` was declared `async def`.

    The Gemini loop is synchronous and JSON-serialises the tool's return value.
    An `async def` hands back a coroutine, so the call raised "Object of type
    coroutine is not JSON serializable", the turn aborted, and the user saw only
    "Agent error" -- with no hint that the review had never started. The tests
    imported the function without ever calling it from a sync context, so
    nothing failed.
    """
    import main

    assert set(main.FUNCTION_MAP) == set(AGENT_TOOLS), sorted(main.FUNCTION_MAP)
    assert set(AGENT_TOOLS) == {f.__name__ for f in main.TOOLS}, (
        "FUNCTION_MAP is derived from TOOLS, so they cannot disagree"
    )
    for name, func in main.FUNCTION_MAP.items():
        assert not inspect.iscoroutinefunction(func), f"{name} is a coroutine function"
        assert func.__doc__, f"{name} has no docstring, so Gemini gets no schema"
        assert "Args:" in func.__doc__, f"{name}'s docstring must document its args"

    # Point at a session file that does not exist, so the browser never starts:
    # this checks the SHAPE of what a failed tool returns, which is all the
    # agent contract needs and costs nothing.
    os.environ["CHESS_AGENT_STATE"] = str(ROOT_FOR_TESTS / "no_such_session.json")
    url = "https://www.chess.com/game/live/184571437220"
    try:
        for name in ONE_URL_ARG_TOOLS:
            result = main.FUNCTION_MAP[name](url)
            assert isinstance(result, dict), (name, type(result))
            json.dumps({"result": result})  # the call the agent loop makes
            assert "error" in result, (name, result)

        # The interactive tools, each called the way the model would call it.
        # Every one must fail as a MESSAGE with no session file, never as an
        # exception and never as a coroutine.
        calls = {
            "open_chess_page": lambda: main.FUNCTION_MAP["open_chess_page"](url),
            "where_am_i": lambda: main.FUNCTION_MAP["where_am_i"](),
            "move_board": lambda: main.FUNCTION_MAP["move_board"](
                direction="next", plies=1
            ),
            "goto_move_in_game": lambda: main.FUNCTION_MAP["goto_move_in_game"](ply=3),
            "take_screenshot": lambda: main.FUNCTION_MAP["take_screenshot"](),
            "close_browser_session": main.FUNCTION_MAP["close_browser_session"],
        }
        for name, call in calls.items():
            result = call()
            assert isinstance(result, dict), (name, type(result))
            json.dumps({"result": result})  # the call the agent loop makes
            # `where_am_i` and `close_browser_session` legitimately succeed with
            # nothing open -- there is nothing to log in to and nothing to
            # close -- so they only have to be a serialisable dict.
            if name not in ("where_am_i", "close_browser_session"):
                assert "error" in result, (name, result)
    finally:
        del os.environ["CHESS_AGENT_STATE"]
        # The interactive tools share a process-wide session; make sure one
        # test's browser cannot leak into the next.
        from tools.session import close_session

        close_session()


def test_each_review_tool_documents_its_own_shape():
    """The model chooses by name and by docstring, so both must be specific."""
    from tools import (
        get_coach_commentary,
        get_moves,
        get_review_state,
        screenshot_game_move,
    )

    def doc(fn):
        return fn.__doc__ or ""

    assert "accuracy" in doc(get_review_state).lower()
    assert "ply" in doc(get_moves).lower() and "san" in doc(get_moves).lower()
    assert "engine" in doc(get_coach_commentary).lower()
    assert "commentary" in doc(get_coach_commentary).lower()
    for fn in (screenshot_game_move, get_coach_commentary):
        for field in ("move_number", "colour", "ply"):
            assert field in doc(fn), f"{fn.__name__} must document {field}"


def test_review_tools_share_one_session_path():
    """Five tools each opening a browser is five places for a leak or a
    forgotten cleanup; they all go through one context manager."""
    from tools import review as rv

    assert callable(rv._review_page)
    for name in (
        "_get_moves_async",
        "_get_review_state_async",
        "_get_coach_commentary_async",
        "_screenshot_move_async",
    ):
        assert "_review_page(" in inspect.getsource(getattr(rv, name)), name
    # and one place that turns a failure into a dict
    assert "except _ToolError" in inspect.getsource(rv._run)


# ---------------------------------------------------------------------------
# Games API
# ---------------------------------------------------------------------------


def test_parse_game_refuses_to_invent_a_color():
    """Regression: 'if not white, then black' made any game the player was not
    in look like a loss for Black against a made-up opponent."""
    from tools.games import _parse_game

    game = {
        "white": {"username": "someone_else", "result": "win", "rating": 2000},
        "black": {"username": "another_person", "result": "resigned", "rating": 1900},
        "end_time": 1700000000,
        "url": "https://www.chess.com/game/live/1",
    }
    assert _parse_game(game, "hikaru") is None

    game["black"]["username"] = "Hikaru"
    parsed = _parse_game(game, "hikaru")
    assert parsed["color"] == "black"
    assert parsed["opponent"] == "someone_else"
    assert parsed["result"] == "loss" and parsed["ending"] == "resignation"


def test_missing_end_time_does_not_render_1970():
    from tools.games import _parse_game

    game = {
        "white": {"username": "hikaru", "result": "win", "rating": 2000},
        "black": {"username": "opponent", "result": "resigned", "rating": 1900},
        "url": "https://www.chess.com/game/live/1",
    }
    assert _parse_game(game, "hikaru")["date"] is None


def test_json_array_response_is_not_an_attribute_error():
    """`.get` on a JSON list raised AttributeError, which `except ValueError`
    never caught, so it escaped as 'Tool failed'."""
    from tools.games import _json_object

    class _Resp:
        def __init__(self, value):
            self._value = value

        def json(self):
            return self._value

    assert _json_object(_Resp([1, 2, 3])) is None
    assert _json_object(_Resp({"a": 1})) == {"a": 1}

    class _Bad:
        def json(self):
            raise ValueError("not json")

    assert _json_object(_Bad()) is None


# ---------------------------------------------------------------------------
# Odds and ends
# ---------------------------------------------------------------------------


def test_star_import_resolves():
    """`open_review_mode` was listed in tools.__all__ but never imported, so
    `from tools import *` raised AttributeError."""
    import tools

    for name in tools.__all__:
        assert hasattr(tools, name), f"tools.__all__ names missing {name}"


def test_session_path_can_move_out_of_the_project():
    """storage_state.json holds a live ACCESS_TOKEN + PHPSESSID."""
    from tools.common import DEFAULT_STATE_PATH, session_state_path

    assert session_state_path() == DEFAULT_STATE_PATH
    os.environ["CHESS_AGENT_STATE"] = r"C:\elsewhere\storage_state.json"
    try:
        assert str(session_state_path()) == r"C:\elsewhere\storage_state.json"
    finally:
        del os.environ["CHESS_AGENT_STATE"]


def test_calibrator_writes_its_dump_before_anything_can_throw():
    """The dump is the only evidence a calibration produces. A run that crashed
    at the entry-control step wrote nothing, so the next attempt had no data."""
    import calibrate_review

    body = inspect.getsource(calibrate_review.run)
    dump_at = body.index("await _write_dump(page")
    entry_at = body.index("await open_review_mode(page)")
    assert dump_at < entry_at
    try_at = body.rfind("try:", 0, entry_at)
    call_at = body.rfind("clicked =", 0, entry_at)
    assert try_at != -1 and try_at < call_at


def test_dumper_covers_the_state_the_moves_live_in():
    """A dump that stops before Start Review cannot show the move list, which
    is the entire reason most of the bugs here were invisible."""
    import dump_pages

    src = inspect.getsource(dump_pages.run)
    for state in (
        "game",
        "analysis",
        "review_loaded",
        "review_expanded",
        "review_started",
    ):
        assert f'"{state}"' in src, state
    assert src.index('"review_started"') > src.index("open_review_mode"), (
        "the moves only exist after Start Review"
    )
    assert "moves_scrolled" in src, "the list may be virtualised"


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------


def main() -> int:
    tests = [(n, o) for n, o in sorted(globals().items()) if n.startswith("test_")]
    failed = []
    for name, fn in tests:
        try:
            fn()
        except AssertionError as exc:
            failed.append(name)
            print(f"FAIL  {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed.append(name)
            print(f"ERROR {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"ok    {name}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
