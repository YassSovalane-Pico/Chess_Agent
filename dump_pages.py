"""Dump the full HTML of every chess.com page state, so the DOM can be studied offline.

    python dump_pages.py https://www.chess.com/game/live/184571437220
    python dump_pages.py <link> --out dumps --headed

Why this exists
---------------
Most of the bugs in this project were invisible because only a fragment of the
page ever reached the code. A `class` lookup failed, the reader fell through to
a weaker signal, and the tool reported "no blunders" for a game that had one --
and nothing in the output said the real markup was sitting right there. The
following were all only discovered by finally looking at the whole document:

  * the move list lives in `wc-move-list` / `chessboard-pkg-move-list-component`
  * a move is classified by `<g id="blunder">` and a
    `var(--color-classification-book)` custom property, not by any class
  * the move number is on the row, as `data-whole-move-number`
  * the sidebar title is an `<h1>Game Review</h1>`, not a button
  * `tab-review-start-review-button` is what actually starts the review
  * the tally is collapsed behind a `button[aria-label="Expand"]` chevron

This writes the full document for each state, plus the review sidebar's markup
on its own, plus a manifest describing what each file is. Run it when a
selector stops working, paste the manifest and the relevant file, and the
selectors can be re-pinned against real markup instead of guessed at.

What it captures, in order
--------------------------
    game            the /game/... page
    analysis        the /analysis/game/... board
    review_loaded   /analysis/.../review with the summary panel loaded
    review_expanded the same, after clicking the tally's expand chevron
    review_started  after clicking Start Review -- this is where moves appear
    moves_scrolled  the move list at several scroll offsets

``moves_scrolled`` exists because this was ASSUMED to be a virtualised list.
The capture says otherwise: all 29 ``[data-node]`` cells were present at every
one of the five offsets, so the sweep is a no-op on the current layout. It is
kept because a future layout may virtualise it, and because finding that out
requires the evidence rather than an assumption.

Run ``python make_fixture.py`` afterwards to distil the dump into the 18 KB
fixture the tests actually read, then ``dumps/`` can be deleted.

Everything is written under ``--out`` (default ``dumps/``) and is git-ignored
by the ``dumps/`` entry in .gitignore.
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from tools.common import (
    __version__,
    configure_logging,
    load_browser_profile,
    session_state_path,
)
from tools.move_list import (
    _scroll_to_fraction,
    describe_state,
    expand_review_summary,
    open_review_mode,
    read_moves_full,
    read_review_state,
    review_panel_present,
    wait_for_review_loaded,
)
from tools.review import (
    _build_review_url,
    _launch_browser,
    _new_logged_in_page,
    _Session,
)

# Keys to also save the review sidebar's outerHTML for, in a companion file.
SIDEBAR_KEYS = ("analysis", "review_loaded", "review_expanded", "review_started")

# How many scroll offsets to sample the move list at. The capture shows the list
# is NOT virtualised on the current layout -- all 29 cells at every offset -- so
# this is insurance, not a known requirement.
SCROLL_STEPS = 4


async def _dump(
    page, name: str, out_dir: Path, manifest: list[dict], note: str
) -> None:
    """Write the full document plus, for the review states, the sidebar alone."""
    try:
        html = await page.content()
    except Exception as exc:  # noqa: BLE001 - diagnostics must not abort the dump
        print(f"  ! could not read content for {name}: {exc}")
        return
    path = out_dir / f"{name}.html"
    path.write_text(html, encoding="utf-8")
    entry = {
        "state": name,
        "note": note,
        "url": page.url,
        "file": path.name,
        "bytes": len(html),
    }

    if name in SIDEBAR_KEYS:
        try:
            sidebar = await page.evaluate(
                "() => { const el = document.querySelector('#board-layout-sidebar');"
                " return el ? el.outerHTML : ''; }"
            )
        except Exception:  # noqa: BLE001
            sidebar = ""
        if sidebar:
            spath = out_dir / f"{name}.sidebar.html"
            spath.write_text(sidebar, encoding="utf-8")
            entry["sidebar_file"] = spath.name
            entry["sidebar_bytes"] = len(sidebar)

    # A small state snapshot makes each file self-describing.
    state = await read_review_state(page)
    entry["state_snapshot"] = {
        "elements": state.get("elements"),
        "bodyChars": state.get("bodyChars"),
        "sidebarChars": state.get("sidebarChars"),
        "layout": state.get("layout"),
        "panel": state.get("panel"),
        "movelist": state.get("movelist"),
        "placeholders": (state.get("placeholders") or [])[:4],
    }

    manifest.append(entry)
    print(
        f"  wrote {path.name:<34} {entry['bytes']:>8,} bytes   "
        f"elements={state.get('elements')}"
    )


async def run(game_url: str, out_dir: Path, headed: bool) -> int:
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        print("Playwright is not installed. Run: pip install -r requirements.txt")
        return 2

    try:
        kind, game_id, review_url = _build_review_url(game_url)
    except ValueError as exc:
        print(f"error: {exc}")
        return 2

    state_path = session_state_path()
    if not state_path.exists():
        print(f"No session at {state_path}. Run: python login_chess.py")
        return 2
    if not load_browser_profile().get("user_agent"):
        print(
            "NOTE: no recorded user agent. Re-run login_chess.py so the login\n"
            "      and the dump use the same fingerprint.\n"
        )
    if headed:
        os.environ["REVIEW_HEADLESS"] = "0"

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    analysis_url = f"https://www.chess.com/analysis/game/{kind}/{game_id}"

    async with async_playwright() as p:
        # Built explicitly rather than via _open_session, because the game and
        # analysis pages are visited first on the same tab and _open_session
        # hardcodes the review URL flow.
        browser = await _launch_browser(p)
        try:
            context, page = await _new_logged_in_page(p, browser, state_path)
        except Exception:
            await browser.close()
            raise
        session = _Session(p, browser, context, page)
        try:
            print(f"-> dumping to {out_dir.resolve()}")

            # 1. The game page.
            await page.goto(
                f"https://www.chess.com/game/{kind}/{game_id}",
                wait_until="domcontentloaded",
                timeout=30000,
            )
            await page.wait_for_timeout(4000)
            await _dump(page, "game", out_dir, manifest, "the /game/ page")

            # 2. The analysis board.
            await page.goto(analysis_url, wait_until="domcontentloaded", timeout=30000)
            await page.wait_for_timeout(6000)
            await _dump(page, "analysis", out_dir, manifest, "the /analysis/ board")

            # 3. The review page as it first loads, with the summary.
            await page.goto(review_url, wait_until="domcontentloaded", timeout=30000)
            await wait_for_review_loaded(page, require="layout")
            loaded = await wait_for_review_loaded(page, require="panel")
            print(f"  review panel ready: {loaded}")
            if not loaded:
                print(f"  {describe_state(await read_review_state(page), 'panel')}")
            await _dump(
                page, "review_loaded", out_dir, manifest, "review page, summary panel"
            )

            # 4. The same page with the collapsed tally expanded.
            if await expand_review_summary(page):
                print("  expanded the collapsed tally")
            await _dump(
                page, "review_expanded", out_dir, manifest, "after the expand chevron"
            )

            # 5. After pressing Start Review -- where the moves finally exist.
            started = await open_review_mode(page)
            print(f"  Start Review clicked: {started}")
            move_ready = await wait_for_review_loaded(
                page, require="movelist", timeout_ms=20000
            )
            print(f"  move list ready: {move_ready}")
            await _dump(
                page,
                "review_started",
                out_dir,
                manifest,
                "after Start Review: the move list exists here",
            )

            # 6. The move list at several scroll offsets, to check whether it is
            #    virtualised. It is not, on the current layout.
            if move_ready:
                rows, diag = await read_moves_full(page)
                total = diag.get("pliesKnown") or len(rows)
                print(
                    f"  parsed {len(rows)} moves "
                    f"({diag.get('coverage') or 'coverage unknown'}); "
                    f"dumping {SCROLL_STEPS + 1} scroll offsets"
                )
                for i in range(SCROLL_STEPS + 1):
                    fraction = i / SCROLL_STEPS
                    await _scroll_to_fraction(page, fraction)
                    await page.wait_for_timeout(400)
                    pct = int(fraction * 100)
                    await _dump(
                        page,
                        f"moves_scrolled_{pct:03d}",
                        out_dir,
                        manifest,
                        f"move list scrolled to {pct}% (plies known: {total})",
                    )
            else:
                print("  no move list; skipping the scrolled dumps")

            print(f"\n  review panel present: {await review_panel_present(page)}")
        finally:
            await session.close()

    stamp = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "generated": stamp,
                "tool_version": __version__,
                "game_url": game_url,
                "review_url": review_url,
                "state_path": str(state_path),
                "states": manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n  wrote {manifest_path.name} ({len(manifest)} states)")
    print(f"  total: {sum(s['bytes'] for s in manifest):,} bytes of HTML")
    print(
        "\n  The manifest lists every file. When a selector breaks, send\n"
        "  manifest.json plus the state where it failed and the selectors can\n"
        "  be re-pinned against real markup."
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    print(f"chess-agent page dumper v{__version__}\n")
    parser.add_argument("game_url", help="a chess.com /game/ link")
    parser.add_argument(
        "--out", default="dumps", help="output directory (default: dumps/)"
    )
    parser.add_argument("--headed", action="store_true", help="show the browser window")
    args = parser.parse_args()
    configure_logging(os.getenv("CHESS_AGENT_LOG", "info"))
    return asyncio.run(run(args.game_url, Path(args.out), args.headed))


if __name__ == "__main__":
    sys.exit(main())
