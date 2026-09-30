"""S0.3 -- exercise BrowserSession against the live site.

Phase 1's unit tests cover the allow-list, which is the part with no browser in
it. Everything else -- the private loop, the lock, the TTL, the snapshot -- only
means something against a real page. This drives the real session end to end and
prints what it found, so the interactive tools are built against observed
behaviour rather than against the design sketch.

    python spikes/s0_3_session_live.py [game_url]

Exercises, in order:
  1. open -> home, confirm the session reports itself open
  2. navigate to the game review URL (allow-list must accept it)
  3. snapshot the review summary BEFORE starting the review (the ordering
     tools/move_list.py calls load-bearing)
  4. start the review, read the move list, read the engine eval at a ply
  5. step forward and back with the board-signature verification
  6. screenshot the board+panel region
  7. navigate to a refused URL and confirm the guard refuses it
  8. close, and confirm a second close reports nothing was open

Exit 0 if every step behaved. Not a substitute for the real tool tests -- it is
the measurement that said which of them were worth writing.
"""

import asyncio
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.common import configure_logging, session_state_path  # noqa: E402
from tools.review import _build_review_url, content_clip, screenshot_review_area  # noqa: E402
from tools.session import BrowserSession, SessionClosed, url_allowed  # noqa: E402

GAME = "https://www.chess.com/game/live/184571437220"


def _step(n: int, label: str) -> None:
    print(f"\n[{n}] {label}")


def main() -> int:
    configure_logging("warning")
    game_url = sys.argv[1] if len(sys.argv) > 1 else GAME
    _, game_id, review_url = _build_review_url(game_url)
    if not session_state_path().exists():
        print("No session. Run: python login_chess.py")
        return 2

    problems: list[str] = []
    session = BrowserSession(idle_seconds=600)  # long: this is a one-shot probe
    shots = ROOT / "screenshots" / "s0_3"
    shots.mkdir(parents=True, exist_ok=True)

    try:
        _step(1, "open")
        page = session.open()
        print(f"    open={session.is_open}")
        title = session.submit(lambda: _title())
        print(f"    title={title[:60]!r}")
        if not session.is_open or "chess.com" not in session.current_url:
            problems.append("session did not open on chess.com")

        _step(2, f"navigate to the review page\n    {review_url}")
        allowed, reason = url_allowed(review_url)
        print(f"    allow-list: {allowed} ({reason or 'no reason'})")
        final = session.navigate(review_url, wait="load")
        print(f"    landed on {final}")

        _step(3, "snapshot the review summary BEFORE Start Review")
        summary = session.snapshot_review_summary() or {}
        game_key = session._game_id_from_url(final)
        print(f"    white={summary.get('white')} black={summary.get('black')}")
        print(f"    tally keys={sorted((summary.get('move_counts') or {}).keys())}")
        print(f"    cached for {game_key}: {bool(session.cached_summary(game_key))}")
        if not summary:
            problems.append("no review summary captured before starting")

        _step(4, "start the review and read the move list")
        info = session.submit(lambda: _read_moves())
        print(f"    moves_read={info.get('coverage')} rows={info.get('rows')}")
        if not info.get("rows"):
            problems.append("no move rows read after starting the review")

        _step(5, "engine eval at the current position")
        engine = session.submit(lambda: _read_engine())
        print(f"    eval={engine.get('eval')}  source={engine.get('source')}")

        _step(6, "step forward and back, with board verification")
        fwd = session.submit(lambda: _step_moves(1))
        print(f"    forward: {fwd}")
        back = session.submit(lambda: _step_moves(-1))
        print(f"    backward: {back}")
        if not fwd.get("verified"):
            problems.append("a forward step could not be verified")

        _step(7, "screenshot the board+panel region")
        path = shots / f"session_live_{game_id}.png"
        session.submit(lambda: _shoot(path))
        exists = path.exists() and path.stat().st_size > 0
        print(f"    {path.name}: exists={exists} "
              f"bytes={path.stat().st_size if exists else 0}")
        if not exists:
            problems.append("no screenshot written")

        _step(8, "the allow-list refuses an off-limits page")
        for bad in ("https://www.chess.com/settings",
                    "https://www.chess.com/logout"):
            try:
                session.navigate(bad)
                print(f"    {bad}: NOT REFUSED  <-- problem")
                problems.append(f"navigated to {bad} when it should be refused")
            except SessionClosed as exc:
                print(f"    {bad}: refused ({str(exc)[:60]})")

        _step(9, "close, then close again")
        was = session.close()
        again = session.close()
        print(f"    first close={was} second close={again}")
        if not was or again:
            problems.append("close() did not report open-then-closed correctly")

    except Exception as exc:  # noqa: BLE001
        print(f"\nEXCEPTION: {type(exc).__name__}: {exc}")
        import traceback

        traceback.print_exc()
        problems.append(f"raised {type(exc).__name__}")
    finally:
        try:
            session.close()
        except Exception:  # noqa: BLE001
            pass

    print("\n" + "=" * 66)
    if problems:
        print(f"  {len(problems)} PROBLEM(S):")
        for p in problems:
            print(f"    - {p}")
        return 1
    print("  Every step behaved. The session is usable.")
    print(f"  screenshot: {shots}")
    return 0


# -- coroutines run on the session's private loop ---------------------------


async def _title() -> str:
    from tools.session import current_page

    return await current_page().evaluate("() => document.title")


async def _read_moves() -> dict:
    from tools import move_list
    from tools.session import current_page

    rows, diag = await move_list.read_moves_when_ready(current_page())
    return {"rows": len(rows), "coverage": diag.get("coverage")}


async def _read_engine() -> dict:
    from tools.session import current_page

    return await current_page().evaluate(
        "() => {"
        " const el = document.querySelector('.evaluation-bar-score')"
        "   || document.querySelector('#board-layout-evaluation');"
        " return {eval: el ? el.textContent.trim() : null,"
        "         source: el ? el.className : null};"
        "}"
    )


async def _step_moves(n: int) -> dict:
    from tools import move_list
    from tools.session import current_page

    result = await move_list.step_moves(current_page(), n)
    return {
        "requested": result.requested,
        "moved": result.moved,
        "verified": result.verified,
        "reason": result.reason,
    }


async def _shoot(path: Path) -> None:
    from tools.session import current_page

    page = current_page()
    await screenshot_review_area(page, path, await content_clip(page))


if __name__ == "__main__":
    sys.exit(main())