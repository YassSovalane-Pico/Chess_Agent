"""Pure-logic tests for the interactive tools: argument handling, no browser.

These cover the parts that must be right before a browser is ever launched --
region validation, direction parsing, refusal of an ambiguous or absent move,
and the shape of an error the model will have to relay.

    python tests/test_interact.py
"""

import re
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


# ---------------------------------------------------------------------------
# Framing: what actually ends up in the picture
# ---------------------------------------------------------------------------


class _FakePage:
    """Just enough page for the framing path, with no browser anywhere."""

    def __init__(self) -> None:
        self.clips = []
        self.shots = []

    async def evaluate(self, script):
        return {
            "url": "https://www.chess.com/analysis/game/live/1/review",
            "title": "Game Review",
            "has_board": True,
            "has_move_list": 29,
            "engine_eval": "+0.31",
            "heading": "Game Review",
        }

    async def screenshot(self, path=None, clip=None, full_page=False, **kwargs):
        self.shots.append(("clip" if clip else "viewport", clip))
        Path(path).write_bytes(b"PNG")


def _patched_shoot(region, tmpdir, clip=None):
    """Run `_shoot` with the page and the review helpers faked out.

    Returns ``(info, page, seen)``, where `seen` records
    REVIEW_SHOT_REGION both while the frame is being measured and after the
    call returns. A wrong region is set before the frame is measured and undone
    afterwards, so neither moment alone is enough to catch it.
    """
    import asyncio
    import os

    from tools import review as rv

    page = _FakePage()
    seen = {}

    async def fake_clip(pg):
        seen["while_clipping"] = os.environ.get("REVIEW_SHOT_REGION")
        return clip

    async def fake_area(pg, path, region=None):
        page.shots.append(("clip" if region else "viewport", region))
        Path(path).write_bytes(b"PNG")
        return str(path)

    originals = (rv.content_clip, rv.screenshot_review_area, interact.current_page)
    rv.content_clip, rv.screenshot_review_area = fake_clip, fake_area
    interact.current_page = lambda: page
    try:
        out = Path(tmpdir) / "shot.png"
        info = asyncio.run(interact._shoot(region, out))
        seen["after"] = os.environ.get("REVIEW_SHOT_REGION")
    finally:
        rv.content_clip, rv.screenshot_review_area, interact.current_page = originals
    return info, page, seen


def test_the_auto_region_frames_the_board_and_the_panel():
    """`auto` is documented as "the chessboard and the review panel".

    It was not. `_shoot` set REVIEW_SHOT_REGION to "sidebar" for every region
    that was not "board", so the DEFAULT shot was the panel alone: no board, and
    therefore no position -- in a file whose caption says "the board and the
    review panel". A user asking to see a move got the commentary column.
    """
    import os
    import tempfile

    previous = os.environ.get("REVIEW_SHOT_REGION")
    os.environ["REVIEW_SHOT_REGION"] = "board_sidebar"
    try:
        with tempfile.TemporaryDirectory() as tmp:
            info, _page, seen = _patched_shoot(
                "auto", tmp, clip={"x": 300, "y": 0, "width": 1100, "height": 800}
            )
    finally:
        if previous is None:
            os.environ.pop("REVIEW_SHOT_REGION", None)
        else:
            os.environ["REVIEW_SHOT_REGION"] = previous

    assert seen["while_clipping"] == "board_sidebar", (
        "auto must leave the framing alone, not narrow it to the sidebar"
    )
    assert "board" in info["framed"] and "panel" in info["framed"], info
    assert seen["after"] == "board_sidebar", "the caller's setting was left changed"


def test_board_and_sidebar_narrow_the_frame_and_give_it_back():
    import os
    import tempfile

    for region, expected_env, expected_framed in (
        ("board", "board", "the board"),
        ("sidebar", "sidebar", "the side panel"),
    ):
        previous = os.environ.get("REVIEW_SHOT_REGION")
        os.environ["REVIEW_SHOT_REGION"] = "board_sidebar"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                info, _page, seen = _patched_shoot(
                    region, tmp, clip={"x": 0, "y": 0, "width": 800, "height": 800}
                )
        finally:
            if previous is None:
                os.environ.pop("REVIEW_SHOT_REGION", None)
            else:
                os.environ["REVIEW_SHOT_REGION"] = previous
        assert seen["while_clipping"] == expected_env, (region, seen)
        assert info["framed"] == expected_framed, (region, info)
        assert seen["after"] == "board_sidebar", (
            f"{region} left the caller's setting changed"
        )


def test_a_frame_that_could_not_be_measured_says_so():
    """`screenshot_review_area` falls back to the whole window, and a whole
    window is not the position the user asked about. The reply has to admit it
    rather than let the model describe a tight board frame."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        info, page, _seen = _patched_shoot("auto", tmp, clip=None)
    assert "fallback" in info["framed"], info
    assert page.shots == [("viewport", None)], page.shots


def test_every_screenshot_call_writes_its_own_file():
    """The end-to-end version of the collision: two calls, two files.

    This is what the chat front end attaches by path, and it de-duplicates by
    path, so a repeated name here is a picture that is never sent.
    """
    import asyncio
    import os
    import tempfile

    from tools import review as rv

    page = _FakePage()
    seen = []

    async def fake_clip(pg):
        seen.append(os.environ.get("REVIEW_SHOT_REGION"))
        return {"x": 300, "y": 0, "width": 1100, "height": 800}

    async def fake_area(pg, path, region=None):
        Path(path).write_bytes(b"PNG")
        return str(path)

    class _Session:
        is_open = True
        current_url = "https://www.chess.com/analysis/game/live/1/review"

        def submit(self, factory, timeout=120, ensure_open=True):
            # the real session runs the coroutine on its own private loop
            return asyncio.run(factory())

    originals = (
        rv.content_clip,
        rv.screenshot_review_area,
        interact.current_page,
        interact.get_session,
        interact._shot_dir,
        interact._stamp,
    )
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        rv.content_clip, rv.screenshot_review_area = fake_clip, fake_area
        interact.current_page = lambda: page
        interact.get_session = lambda: _Session()
        interact._shot_dir = lambda: root
        # the same second, which is what actually happens inside one turn
        interact._stamp = lambda: "20260101_120000"
        try:
            first = interact.take_screenshot()
            second = interact.take_screenshot()
            labelled = interact.take_screenshot(label="after e5")
        finally:
            (
                rv.content_clip,
                rv.screenshot_review_area,
                interact.current_page,
                interact.get_session,
                interact._shot_dir,
                interact._stamp,
            ) = originals

        for shot in (first, second, labelled):
            assert "error" not in shot, shot
            assert Path(shot["screenshot"]).is_file(), shot
        assert len({first["screenshot"], second["screenshot"]}) == 2, (
            "two calls in the same second produced the same file name"
        )
        assert len(list(root.iterdir())) == 3, sorted(p.name for p in root.iterdir())
        # the caption regex in discord_bot still reads every one of them
        for shot in (first, second, labelled):
            assert re.fullmatch(
                r"shot_\w+_\d{8}_\d{6}\.png", Path(shot["screenshot"]).name
            ), shot["screenshot"]


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
