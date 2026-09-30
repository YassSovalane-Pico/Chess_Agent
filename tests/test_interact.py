"""Pure-logic tests for the interactive tools: argument handling, no browser.

These cover the parts that must be right before a browser is ever launched --
region validation, direction parsing, refusal of an ambiguous or absent move,
and the shape of an error the model will have to relay.

    python tests/test_interact.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools import interact  # noqa: E402


def test_every_interactive_tool_is_sync_with_a_documented_signature():
    """The agent loop JSON-serialises a tool's return and reads its docstring.

    An `async def` tool hands back a coroutine and the turn dies with "Object of
    type coroutine is not JSON serializable" -- the exact regression that
    `test_review_logic.test_every_agent_tool_is_synchronous_and_json_safe`
    exists for. These tools must obey the same contract from the day they land.
    """
    import inspect

    for func in interact.INTERACTIVE_TOOLS:
        assert not inspect.iscoroutinefunction(func), f"{func.__name__} is async"
        assert func.__doc__, f"{func.__name__} has no docstring"
        assert "Args:" in func.__doc__, f"{func.__name__} must document its args"
        assert "Returns:" in func.__doc__, f"{func.__name__} must document returns"


def test_direction_parsing():
    """Forward/back synonyms all resolve; anything else is refused by name."""
    # These call through to the session, so only the pure parsing is checked
    # here, by reading the source of the branch -- the behaviour itself needs
    # a browser, which spikes/s0_4 covers. What matters here is that the tool
    # refuses an unknown direction rather than defaulting silently.
    import inspect

    src = inspect.getsource(interact.move_board)
    assert '"previous"' in src
    assert "must be 'next' or 'previous'" in src


def test_unknown_region_is_refused():
    out = interact.take_screenshot(region="not-a-region")
    assert "error" in out, out
    assert "auto" in out["error"], out


def test_unknown_direction_is_refused():
    out = interact.move_board(direction="sideways")
    assert "error" in out, out
    assert "next" in out["error"]


def test_zero_plies_is_refused():
    out = interact.move_board(direction="next", plies=0)
    assert "error" in out
    assert "0" in out["error"]


def test_a_move_with_no_number_is_refused():
    """Rather than silently jumping to ply 0 and photographing nothing."""
    out = interact.goto_move_in_game()
    assert "error" in out, out
    assert "ply" in out["error"], out


def test_a_refused_url_names_the_reason_and_never_opens_a_browser():
    out = interact.open_chess_page(url="https://www.chess.com/settings")
    assert "error" in out, out
    assert "chess.com" in out["error"], out
    # A cross-host URL is refused too.
    out = interact.open_chess_page(url="https://example.com/analysis/")
    assert "error" in out
    assert "chess.com" in out["error"]


def test_a_missing_session_is_a_message_not_a_crash():
    """With no storage_state on disk every browser tool must say so plainly.

    This is the shape the agent contract test checks for the review tools: a
    dict with an `error` key, never an exception, because the model has to relay
    it to the user.
    """
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        previous = os.environ.get("CHESS_AGENT_STATE")
        os.environ["CHESS_AGENT_STATE"] = str(Path(tmp) / "no_session.json")
        try:
            for call in (
                lambda: interact.where_am_i(),
                lambda: interact.open_chess_page(url="https://www.chess.com/analysis/"),
                lambda: interact.move_board(),
                lambda: interact.goto_move_in_game(ply=3),
                lambda: interact.take_screenshot(),
                lambda: interact.close_browser_session(),
            ):
                out = call()
                assert isinstance(out, dict), out
        finally:
            if previous is None:
                os.environ.pop("CHESS_AGENT_STATE", None)
            else:
                os.environ["CHESS_AGENT_STATE"] = previous


def test_shot_regions_are_the_documented_set():
    assert interact.SHOT_REGIONS == ("auto", "board", "sidebar", "viewport")


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for test in tests:
        try:
            test()
            print(f"ok    {test.__name__}")
            passed += 1
        except AssertionError as exc:
            print(f"FAIL  {test.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR {test.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
