"""Prove that move classification and navigation work on a real game.

Run this against a game that HAS at least one blunder or brilliant move --
a clean game proves nothing, because there is nothing to find.

    python calibrate_review.py <game link>
    python calibrate_review.py <link> --expect blunder=2,brilliant=1
    python calibrate_review.py <link> --strict      # also fail on unlisted kinds
    python calibrate_review.py <link> --headed        # watch it work

Ten checks, in the order they print
-----------------------------------
  1. page load state      element count, body text length, visible loaders and
                          the layout/panel/move-list readiness signals.
  2. review data          the panel is computed server-side and arrives long
                          after the layout; this is the wait that matters.
  3. summary panel        the accuracy pair and the tally, read BEFORE the
                          review is started.
  4. starting the review  the entry control, then the move list it produces.
  5. summary panel again  expected to be empty once the review is running, so
                          the pre-start numbers above are the real ones.
  6. move list            every parsed ply with its classification, confidence
                          and evidence, plus coverage against the plies the
                          DOM knows about.
  7. colour palette       what the review legend taught the reader, and which
                          colours were discarded as ambiguous.
  8. detection            at least one brilliant or blunder, and at least one
                          of them labelled by chess.com rather than inferred.
  9. ground truth         --expect compared against what was read.
 10. navigation           forward 3 and back 3, each verified against the
                          board signature, with the board proven idle first.

A DOM dump goes to screenshots/calibration.json: written immediately after the
load wait, before anything else can throw, and re-written if a later stage
gets further. A failed run with a dump is worth far more than a
successful-looking run without one.

Reading the output
------------------
Read the "page load state" output before anything else. A page that never
mounted has a few hundred elements and a body text of a few hundred
characters; a mounted app has thousands of elements and many thousands of
characters. Every check after that point is measuring a blank page, and the
run says so rather than quietly reporting success.

--expect and --strict
---------------------
--expect is the ground truth you can see for yourself: --expect blunder=2
compares only blunder. --strict additionally fails on any classification not
in the list, which is what a full ground-truth table and a unit test want and
what a partial list cannot use, because every real game also contains book,
great and best moves: --expect blunder=2 --strict would fail on the kinds you
never wrote down.

Exit codes
----------
  0  every check passed
  1  a check failed, and the output names the stage and what it expected
  2  the run could not start: no Playwright, an unusable link, or no session
"""

import argparse
import asyncio
import json
import os
import sys

from tools.common import (
    PROJECT_ROOT,
    __version__,
    configure_logging,
    load_browser_profile,
    session_state_path,
)
from tools.move_list import (
    HIGH,
    KINDS,
    board_is_stable,
    find_classified,
    inspect_dom,
    open_moves_tab,
    open_review_mode,
    read_moves,
    read_moves_full,
    read_ratings,
    read_review_state,
    read_tally,
    review_panel_present,
    step_moves,
    summarise,
    validate_against,
    wait_for_review_loaded,
)
from tools.review import _build_review_url, _open_session, parse_review_panel


def _parse_expect(raw: str | None) -> dict[str, int]:
    if not raw:
        return {}
    out: dict[str, int] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise SystemExit(f"--expect entry {part!r} must look like blunder=1")
        key, _, value = part.partition("=")
        key = key.strip().lower()
        if key not in KINDS:
            raise SystemExit(f"unknown classification {key!r}; pick from {KINDS}")
        out[key] = int(value)
    return out


def _report_moves(rows: list[dict], diag: dict) -> None:
    print(f"  parsed {len(rows)} moves")
    print(
        f"  move list root   : {diag.get('moveListRootClass')!r} "
        f"({diag.get('moveListRoot')} cells under it)"
    )
    known = diag.get("pliesInDom")
    if known:
        print(
            f"  ply coverage     : {diag.get('coverage') or f'{len(rows)}/{known}'}"
            f"   (virtualised list: only the rendered window is in the DOM)"
        )
    print(
        f"  scanned        : {diag.get('elementsScanned')} elements "
        f"(+{diag.get('shadowRootsEntered', 0)} shadow roots)"
    )
    print(f"  raw cells      : {diag.get('rawCells')}")
    print(f"  hidden dropped : {diag.get('hiddenCellsDropped')}")
    print(f"  dupes dropped    : {diag.get('duplicateCellsDropped')}")
    if diag.get("loadingPlaceholders"):
        print(f"  STILL LOADING    : {diag['loadingPlaceholders'][:6]}")
    print(f"  classifications  : {summarise(rows)}")
    if diag.get("sampleRowText"):
        print(f"  sample rows      : {diag['sampleRowText'][:4]}")
    if diag.get("moveLikeElsewhere"):
        print("  move-like text found elsewhere on the page:")
        for item in diag["moveLikeElsewhere"][:8]:
            print(f"    {item['text']!r}  in  {item['chain']!r}")
    if diag.get("tabs"):
        print("  sidebar tabs:")
        for tab in diag["tabs"][:12]:
            print(f"    {tab['label']!r}  {tab['cls']!r}")


def _print_state(state: dict) -> None:
    """Print what the page actually looks like right now.

    Read `bodyChars` next to `elements`: a mounted app has thousands of
    elements and many thousands of characters of body text, and the two
    together separate "the review data has not arrived" from "there is nothing
    on this page at all".
    """
    print(f"  elements in DOM  : {state.get('elements')}")
    print(f"  body text length : {state.get('bodyChars')}")
    print(f"  sidebar text     : {state.get('sidebarChars')} chars")
    placeholders = state.get("placeholders") or []
    print(f"  still loading    : {placeholders[:4] if placeholders else 'no'}")
    print(
        f"  layout mounted   : {bool(state.get('layoutFound'))} {state.get('layout') or ''}"
    )
    print(
        f"  review data      : {bool(state.get('panelFound'))} {state.get('panel') or 'NONE'}"
    )
    print(
        f"  move list        : {bool(state.get('moveListFound'))} "
        f"{state.get('movelist') or 'none (review not started?)'}"
    )


async def _write_dump(page, label: str) -> dict:
    """Write the DOM dump to screenshots/calibration.json; never fail the run.

    The dump is the only evidence a calibration produces, so it is written as
    soon as the page is ready and rewritten by any later stage that gets
    further. Capture failures are reported and swallowed: a missing dump is
    bad, but a run that dies before printing its findings is worse.
    """
    out = PROJECT_ROOT / "screenshots" / "calibration.json"
    out.parent.mkdir(exist_ok=True)
    try:
        dump = await inspect_dom(page)
    except Exception as exc:  # noqa: BLE001 - diagnostics must never fail the run
        print(f"  could not capture the DOM dump: {exc}")
        return {}
    try:
        out.write_text(json.dumps(dump, indent=2), encoding="utf-8")
        print(f"  wrote {out} ({label})")
    except OSError as exc:
        print(f"  could not write {out}: {exc}")
    return dump


async def run(game_url: str, expect: dict[str, int], headed: bool, strict: bool) -> int:
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        print("Playwright is not installed. Run: pip install -r requirements.txt")
        return 2

    try:
        _, _, review_url = _build_review_url(game_url)
    except ValueError as exc:
        print(f"error: {exc}")
        return 2

    state_path = session_state_path()
    if not state_path.exists():
        print(f"No session at {state_path}. Run: python login_chess.py")
        return 2
    if not load_browser_profile().get("user_agent"):
        print(
            "NOTE: no recorded user agent. Re-run login_chess.py so the login "
            "and review fingerprints match.\n"
        )
    if headed:
        os.environ["REVIEW_HEADLESS"] = "0"

    exit_code = 0
    async with async_playwright() as p:
        session = await _open_session(p, state_path)
        try:
            page = session.page
            print(f"-> {review_url}")
            await page.goto(review_url, wait_until="domcontentloaded", timeout=30000)

            # --- loading state -------------------------------------------
            print("\n=== page load state ===")
            _print_state(await read_review_state(page))
            layout_ready = await wait_for_review_loaded(page, require="layout")
            print(f"\n  layout mounted   : {layout_ready}")
            if not layout_ready:
                print("  FAIL: the app shell never mounted. Everything below would")
                print("        only measure a blank page.")
                exit_code = 1

            # The review data is computed server-side and takes far longer than
            # the layout. This is the wait that actually matters.
            print("\n=== waiting for the review data ===")
            panel_ready = await wait_for_review_loaded(page, require="panel")
            state = await read_review_state(page)
            _print_state(state)
            print(f"  review data ready: {panel_ready}")

            # Capture the DOM now, before anything else can throw. A failed run
            # with a dump is worth ten successful-looking runs without one.
            dump = await _write_dump(page, "after load wait")

            # --- summary panel ------------------------------------------
            # Read this BEFORE starting the review. Starting it replaces the
            # summary with the move-by-move view, after which both accuracies
            # read None and the tally reads {} -- which looks exactly like a
            # page that had no summary.
            print("\n=== summary panel (pre-start) ===")
            try:
                text = await page.evaluate("() => document.body.innerText || ''")
            except Exception as exc:  # noqa: BLE001
                text = ""
                print(f"could not read page text: {exc}")
            white, black, fallback = parse_review_panel(text)
            tally = await read_tally(page)
            by_class_white, by_class_black = await read_ratings(page)
            print(f"  white accuracy : {white}")
            print(f"  black accuracy : {black}")
            print(
                f"  by .review-rating-*: white={by_class_white} black={by_class_black}"
            )
            print(f"  tally (by class): {tally or '(none found)'}")
            if fallback:
                print(f"  counts (text)   : {fallback}")

            # --- starting the review ---------------------------------------
            # The page serves the summary only: no move, no move-list-row and
            # no main-line-ply exists until this control is used, so skipping
            # it leaves a game full of blunders looking like a clean one.
            print("\n=== starting the review ===")
            if panel_ready:
                try:
                    clicked = await open_review_mode(page)
                except Exception as exc:  # noqa: BLE001
                    clicked = False
                    print(f"  entry control raised: {exc}")
                print(f"  'Start Review' clicked: {clicked}")
                move_list_ready = await wait_for_review_loaded(
                    page, require="movelist", timeout_ms=30000
                )
                _print_state(await read_review_state(page))
                print(f"  move list ready: {move_list_ready}")
                if clicked and move_list_ready:
                    dump = await _write_dump(page, "after Start Review")
                if not move_list_ready:
                    print(
                        "  FAIL: the move list never appeared even after starting"
                        " the review. The dump above shows whether chess.com"
                        " serves moves under different class names."
                    )
                    exit_code = 1
            else:
                print("  skipped: the review data never arrived")
                print(f"  review panel present: {await review_panel_present(page)}")
                print(
                    "  FAIL: no review data after waiting. The dump above shows"
                    " what chess.com actually served. If bodyChars is tiny the"
                    " app did not boot (check the session, or try --headed to"
                    " see the browser); if bodyChars is large but the panel"
                    " signals are absent, the panel selectors need re-pinning"
                    " against the new dump."
                )
                exit_code = 1

            print("\n=== summary panel (post-start, expected empty) ===")
            post_white, post_black = await read_ratings(page)
            print(f"  white accuracy : {post_white}")
            print(f"  black accuracy : {post_black}")
            if post_white is None and post_black is None and white is not None:
                print(
                    "  note: the summary is gone because the review is running."
                    " The pre-start numbers above are the real ones."
                )

            # --- move list -------------------------------------------------
            print("\n=== move list (sweeping the virtualised list) ===")
            rows, diag = await read_moves_full(page)
            _report_moves(rows, diag)

            if not rows:
                print("\n  no move list in the default tab -- trying sidebar tabs")
                clicked = await open_moves_tab(page)
                print(f"  tab clicked: {clicked}")
                if clicked:
                    await wait_for_review_loaded(page, timeout_ms=20000)
                    rows, diag = await read_moves(page)
                    _report_moves(rows, diag)

            classified = [r for r in rows if r.get("classification")]
            if classified:
                print("\n  ply  num  san    kind        conf    evidence")
                for r in classified:
                    print(
                        f"  {r['ply']:>3}  {r['number'] or 0:>3}  "
                        f"{r['san']:<6} {r['classification']:<10} "
                        f"{r['confidence']:<7} {r['evidence'] or ''}"
                    )
            else:
                print("\n  NO CLASSIFIED MOVES FOUND.")
                print("  If this game genuinely has no errors, that is correct.")
                print("  Otherwise check the DOM dump at the end of this output.")

            # --- palette ---------------------------------------------------
            palette = diag.get("palette") or {}
            if palette:
                print("\n=== learned colour palette (from the review legend) ===")
                for kind, colours in sorted(palette.items()):
                    print(f"  {kind:<10} {colours}")
                if diag.get("ambiguousColors"):
                    print("  discarded as ambiguous:")
                    for entry in diag["ambiguousColors"][:8]:
                        print(f"    {entry}")

            # --- detection --------------------------------------------------
            print("\n=== brilliant / blunder detection ===")
            hits, _ = await find_classified(page, ("brilliant", "blunder"))
            if not hits:
                print("  FAIL: no brilliant or blunder detected.")
                exit_code = 1
            for h in hits:
                print(
                    f"  ply {h['ply']:>3} ({h['san']}) -> {h['classification']} "
                    f"[{h['confidence']}] via {h['evidence']}"
                )
            explicit = [h for h in hits if h.get("confidence") == HIGH]
            inferred = [h for h in hits if h.get("confidence") != HIGH]
            if inferred:
                print(
                    f"  note: {len(inferred)} hit(s) were inferred from a glyph or a"
                    " colour, not explicitly labelled by chess.com."
                )
            if hits and not explicit:
                # An inferred hit is a glyph or a colour, not a label chess.com
                # wrote. If every hit is inferred, the page may not be the
                # review view at all, so this must not be reported as a pass.
                print(
                    "  FAIL: every hit was inferred. chess.com did not label any"
                    " move, which usually means the Game Review view never opened."
                )
                exit_code = 1
            if white is None and black is None and not (tally or fallback):
                print(
                    "  FAIL: no accuracy panel and no tally were read. The page"
                    " looks like the plain analysis board, not a Game Review."
                )
                exit_code = 1

            # --- ground truth ------------------------------------------------
            print("\n=== ground truth check ===")
            if expect:
                problems = validate_against(rows, expect, strict=strict)
                if problems:
                    print("  FAIL:")
                    for item in problems:
                        print(f"    - {item}")
                    exit_code = 1
                else:
                    print(f"  PASS: matches {expect}")
            else:
                print("  skipped (pass --expect blunder=2,brilliant=1)")

            # --- navigation ---------------------------------------------------
            print("\n=== forward / backward navigation ===")
            # The coach auto-plays once the review is started, so a board that
            # moves on its own would make every step below look verified.
            stable, sig = await board_is_stable(page)
            print(
                f"  board signature readable : {bool(sig)}"
                f"{' (image hash fallback)' if sig and sig.startswith('img:') else ''}"
            )
            print(f"  board idle on its own   : {stable}")
            if sig is None:
                print(
                    "  FAIL: no board element could be located, so a step cannot be"
                    " verified. This is not a pass -- reporting it as one is how a"
                    " run 'verified' ArrowLeft on a 24px sidebar icon."
                )
                exit_code = 1
            elif not stable:
                print(
                    "  NOTE: the board changes without any input, so the coach is"
                    " auto-playing. The step results below may reflect its moves"
                    " rather than the controls."
                )
            fwd = await step_moves(page, 3)
            back = await step_moves(page, -3)
            print(
                f"  stepped forward 3 : moved {fwd.moved}, verified {fwd.verified}"
                f"{' via ' + fwd.method if fwd.method else ''}"
            )
            print(
                f"  stepped back 3    : moved {back.moved}, verified {back.verified}"
                f"{' via ' + back.method if back.method else ''}"
            )
            for res in (fwd, back):
                if res.reason:
                    print(f"  note: {res.reason}")
            if not fwd.ok or not back.ok:
                print("  FAIL: navigation did not verifiably move the board.")
                exit_code = 1

            # --- DOM dump ------------------------------------------------------
            print("\n=== raw DOM dump ===")
            if not dump:
                dump = await _write_dump(page, "final")
            if not dump:
                print("  no dump captured")
            else:
                sidebar = dump.get("sidebarHTML", "")
                print(
                    f"  sidebar markup     : {len(sidebar)} chars (full, untruncated)"
                )
                print("  container candidates:")
                for c in dump.get("containerCandidates", []):
                    mark = "OK " if c["found"] else "-- "
                    print(f"   {mark}{c['selector']}  children={c['childCount']}")
                sigs = dump.get("signals", [])
                print(f"  classification-ish elements: {len(sigs)}")
                for s in sigs[:20]:
                    print(
                        f"   <{s['tag']}> class={s['class'][:52]!r} text={s['text']!r}"
                    )
                if len(sigs) > 20:
                    print(f"   ... and {len(sigs) - 20} more (see the JSON file)")
        finally:
            await session.close()

    print(
        "\n"
        + ("RESULT: all checks passed" if exit_code == 0 else "RESULT: problems found")
    )
    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # The version goes into the output so a run can be traced back to the build
    # that produced it.
    print(f"chess-agent review calibrator v{__version__}\n")
    parser.add_argument("game_url", help="a chess.com /game/ link")
    parser.add_argument(
        "--expect",
        help="ground truth, e.g. blunder=2,brilliant=1",
    )
    parser.add_argument("--headed", action="store_true", help="show the browser window")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="also fail on classifications that are not in --expect",
    )
    parser.add_argument("--dump", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    configure_logging(os.getenv("CHESS_AGENT_LOG", "info"))
    return asyncio.run(
        run(args.game_url, _parse_expect(args.expect), args.headed, args.strict)
    )


if __name__ == "__main__":
    sys.exit(main())
