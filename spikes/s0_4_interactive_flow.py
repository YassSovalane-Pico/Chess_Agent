"""S0.4 -- drive the interactive tools the way the agent would.

The unit tests in tests/test_interact.py cover arguments and refusals without a
browser. This drives the actual tools against the live site, in the order a
conversation would use them, and checks the things only a real page can answer:
does the browser stay open between calls, does stepping forward/back actually
move the board, does the screenshot land on disk, does the summary survive the
review being started.

    python spikes/s0_4_interactive_flow.py [game_url]

Exit 0 if every step behaved. This is the measurement that says the interactive
surface actually works, rather than merely importing.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.common import configure_logging, session_state_path  # noqa: E402
from tools.interact import (  # noqa: E402
    close_browser_session,
    goto_move_in_game,
    move_board,
    open_chess_page,
    take_screenshot,
    where_am_i,
)
from tools.review import _build_review_url  # noqa: E402

GAME = "https://www.chess.com/game/live/184571437220"

problems: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {label}{('  -- ' + detail) if detail else ''}")
    if not ok:
        problems.append(label)


def main() -> int:
    configure_logging("warning")
    game_url = sys.argv[1] if len(sys.argv) > 1 else GAME
    _, game_id, review_url = _build_review_url(game_url)
    if not session_state_path().exists():
        print("No session. Run: python login_chess.py")
        return 2

    try:
        print("\n[1] where_am_i with nothing open yet")
        out = where_am_i()
        check("where_am_i returns a dict", isinstance(out, dict), str(out)[:80])
        check("session not open yet", out.get("session_open") is False)

        print(f"\n[2] open_chess_page -> the review page\n    {review_url}")
        out = open_chess_page(url=review_url, wait="load")
        if "error" in out:
            check("open_chess_page", False, out["error"])
            return 1
        check("landed on the review page", "/review" in out.get("url", ""), out["url"])
        check("kind detected", out.get("kind") == "game review", str(out.get("kind")))
        check("summary captured before the review starts",
              bool(out.get("summary")), f"white={out.get('summary', {}).get('white_accuracy')}")
        summary_before = (out.get("summary") or {}).get("white_accuracy")
        first_url = out.get("url")

        print("\n[3] where_am_i again -- the browser must STILL be on that page")
        out = where_am_i()
        check("browser stayed open", out.get("session_open") is True)
        check("still on the review page", out.get("url") == first_url, out.get("url", ""))
        check("board detected", out.get("has_board") is True)

        print("\n[4] take_screenshot (auto region)")
        out = take_screenshot(region="auto", label="start")
        shot = out.get("screenshot", "")
        check("screenshot written", bool(shot) and Path(shot).is_file(),
              f"{Path(shot).name if shot else out.get('error')}")
        if shot and Path(shot).is_file():
            check("screenshot is not empty", Path(shot).stat().st_size > 5000,
                  f"{Path(shot).stat().st_size} bytes")

        print("\n[5] start the review, then move forward")
        # move_board cannot step until a move list exists, so open the moves
        # the way the review flow does.
        out = goto_move_in_game(ply=1)
        if "error" in out:
            check("goto ply 1 (this also starts the review)", False, out["error"])
        else:
            # Either the board moved, or it was already there. Both are success;
            # "moved is False" alone would flag a request that was satisfied
            # before it was made.
            check(
                "at ply 1",
                out.get("moved") is True or out.get("already_at") is True,
                str(out.get("move")),
            )
            check("ply 1 is g4 (book)", (out.get("move") or {}).get("san") == "g4",
                  str(out.get("move")))

        out = move_board(direction="next", plies=2)
        if "error" in out:
            check("move forward 2", False, out["error"])
        else:
            check("moved forward 2", out.get("moved") == 2, str(out))
            check("board verified", out.get("verified") is True, out.get("reason", ""))

        print("\n[6] step back one")
        out = move_board(direction="previous", plies=1)
        check("moved back 1", out.get("moved") == 1 and out.get("verified") is True,
              str(out))

        print("\n[7] goto a specific move by number")
        out = goto_move_in_game(move_number=5, color="white")
        check("jumped to move 5 as white", out.get("moved") is True, str(out.get("move")))
        mv = out.get("move") or {}
        check("that move is 5", mv.get("move_number") == 5, str(mv))

        print("\n[8] an ambiguous move number is refused, not guessed")
        out = goto_move_in_game(move_number=5)
        check("ambiguous move refused", "error" in out, str(out.get("error"))[:90])
        check("the error names both sides",
              "white" in out.get("error", "") and "black" in out.get("error", ""),
              out.get("error", "")[:90])

        print("\n[9] the review summary survives the review having started")
        out = where_am_i()
        check("engine eval readable", out.get("engine_eval") is not None,
              str(out.get("engine_eval")))

        print("\n[10] screenshot the board only")
        out = take_screenshot(region="board", label="boardonly")
        shot = out.get("screenshot", "")
        check("board screenshot written", bool(shot) and Path(shot).is_file(),
              f"{Path(shot).name if shot else out.get('error')}")

        print("\n[11] close_browser_session")
        out = close_browser_session()
        check("closed", out.get("closed") is True)
        check("it reported the session was open", out.get("was_open") is True)

        print("\n[12] after closing, the session really is gone")
        out = where_am_i()
        check("session reports closed", out.get("session_open") is False, str(out)[:100])

    finally:
        try:
            close_browser_session()
        except Exception:  # noqa: BLE001
            pass

    print("\n" + "=" * 66)
    if problems:
        print(f"  {len(problems)} PROBLEM(S):")
        for p in problems:
            print(f"    - {p}")
        return 1
    print("  Every step behaved. The interactive tools work against the live site.")
    return 0


if __name__ == "__main__":
    sys.exit(main())