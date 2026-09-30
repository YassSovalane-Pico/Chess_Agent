"""Tests for the Discord front end and the agent core it wraps.

No network, no browser, no Discord connection. What is worth testing here is
the logic that is easy to get subtly wrong and invisible when it is:

  * message chunking that must not slice a table in half and lose games
  * path-stripping that must not eat a sentence that mentions a path
  * the artifact walk, including the tool result shapes that differ from each
    other and the file that a failed tool named but never wrote
  * access control, which is the only thing standing between a stranger and
    somebody's logged-in chess.com account
  * one-browser-at-a-time accounting, where a double decrement would make the
    queue counter go negative and lie about how busy the bot is
  * command registration and dispatch, which failed on the first real run: a
    plain coroutine passed to `add_command` raises during login, and an
    overridden `on_message` silently stops `process_commands` from ever running
  * the agent tools staying synchronous, which is what makes running them in a
    worker thread work at all

Run:  python -m pytest tests/ -q     (or: python tests/test_discord_bot.py)
"""

import asyncio
import inspect
import json
import os
import re
import shutil
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import discord
from discord.ext import commands

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agent
import discord_bot
from tools import interact as tools_interact

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Message chunking
# ---------------------------------------------------------------------------


def test_short_message_is_one_chunk():
    assert discord_bot.chunk_message("hello") == ["hello"]
    assert discord_bot.chunk_message("") == []
    assert discord_bot.chunk_message(None) == []


def test_no_chunk_ever_exceeds_the_discord_limit():
    """2000 is Discord's hard limit; over it the send raises and the answer is
    lost, so this is the assertion that matters most here."""
    limit = discord_bot.MAX_MESSAGE_CHARS
    assert limit <= 2000
    cases = {
        "many paragraphs": "\n\n".join(f"paragraph {i} " * 40 for i in range(40)),
        "one huge line": "x" * 9000,
        "one huge word": "y" * 9000,
        "long code block": "```\n"
        + "\n".join("def f(): pass" for _ in range(300))
        + "\n```",
        "many short lines": "\n".join(f"line {i}" for i in range(2000)),
        "mixed": ("a" * 1900 + "\n\n" + "b" * 1900 + "\n\n" + "c" * 50),
    }
    for name, text in cases.items():
        chunks = discord_bot.chunk_message(text)
        assert chunks, name
        for chunk in chunks:
            assert len(chunk) <= limit, f"{name}: {len(chunk)} chars"


def test_chunking_does_not_lose_or_duplicate_content():
    """A naive `text[:1900]` silently drops the tail -- which is how a bot
    lists six of nine games. Every character must survive the split."""
    text = "\n\n".join(f"Game {i}: won as white vs e{i} (1500)" for i in range(30))
    joined = " ".join(discord_bot.chunk_message(text))
    for i in range(30):
        assert f"Game {i}:" in joined, f"game {i} was lost in the split"
    assert joined.count("Game ") == 30, "a game was duplicated"


def test_chunking_prefers_paragraph_then_line_boundaries():
    """Splitting mid-table turns a list of games into nonsense, so the splitter
    must break on a blank line before it breaks anywhere else."""
    para = "A" * 1000
    text = f"{para}\n\n{para}\n\n{para}"
    chunks = discord_bot.chunk_message(text, limit=2100)
    assert len(chunks) == 2, chunks
    # every paragraph is intact -- 1000 characters, never a partial one. This
    # is the property that stops a table of games being sliced mid-row.
    for chunk in chunks:
        assert chunk.count("A") in (1000, 2000), chunk.count("A")
        assert chunk.replace("\n\n", "").count("A") == chunk.count("A")
    assert chunks[0].count("A") == 2000, "should have packed two paragraphs"
    assert chunks[1].count("A") == 1000, "the remainder is the third paragraph"

    lines = "\n".join(f"{i:04d}" for i in range(1000))
    chunks = discord_bot.chunk_message(lines, limit=100)
    for chunk in chunks:
        for line in chunk.split("\n"):
            assert len(line) <= 100, line
            assert line.isdigit(), f"line was cut: {line!r}"


# ---------------------------------------------------------------------------
# Path stripping
# ---------------------------------------------------------------------------


def test_only_bare_path_lines_are_stripped():
    """A path on its own line is noise on Discord; a path inside a sentence is
    the model's answer and must survive."""
    text = "\n".join(
        [
            "I found one blunder.",
            r"`D:\Python\Ai\chess-agent\screenshots\blunder_m12_ply24_2026.png`",
            "The image above shows the board after 12...e4.",
            "It is saved at D:\\photos\\board.png if you want it.",
            "*C:/Users/me/Pictures/x.jpg*",
        ]
    )
    kept = discord_bot.strip_path_lines(text)
    assert "I found one blunder." in kept
    assert "The image above shows the board after 12...e4." in kept
    # a sentence that merely mentions a path is still an answer
    assert "It is saved at" in kept
    assert "blunder_m12_ply24" not in kept
    assert "board.png if you want it" in kept, "the sentence was eaten"
    assert "Pictures/x.jpg" not in kept, "the decorated path line should go"
    assert discord_bot.strip_path_lines("") == ""


# ---------------------------------------------------------------------------
# Artifact collection
# ---------------------------------------------------------------------------


def test_artifacts_are_collected_from_both_tool_shapes():
    """The two screenshot tools return completely different shapes. A collector
    that reads known keys breaks the moment a third tool is added; this one
    walks the value, and the test pins both real shapes."""
    with tempfile.TemporaryDirectory() as tmp:
        a = Path(tmp) / "brilliant_m14_ply27_20260101_120000.png"
        b = Path(tmp) / "summary_20260101_120000.png"
        a.write_bytes(b"PNG")
        b.write_bytes(b"PNG")

        # review_game_screenshots: a dict of dicts keyed by shot name
        review = {
            "white_accuracy": 94.1,
            "screenshots": {
                "summary": str(b),
                "brilliant_m14_ply27_20260101_120000": str(a),
            },
        }
        found = agent.collect_artifacts(review, "review_game_screenshots")
        assert {p for _t, p in found} == {a, b}, found
        assert {t for t, _p in found} == {"review_game_screenshots"}

        # screenshot_game_move: one path, nested under `move`
        single = {"move": {"ply": 27, "san": "O-O", "screenshot": str(a)}}
        found = agent.collect_artifacts(single, "screenshot_game_move")
        assert [(t, p) for t, p in found] == [("screenshot_game_move", a)]


def test_artifact_collection_drops_paths_that_do_not_exist():
    """A tool that failed halfway can name a file it never wrote. Posting that
    is a 404 in a channel, and worse, it looks like a successful review."""
    with tempfile.TemporaryDirectory() as tmp:
        real = Path(tmp) / "summary.png"
        real.write_bytes(b"PNG")
        result = {"screenshots": {"summary": str(real), "gone": "C:/nope/blunder.png"}}
        found = agent.collect_artifacts(result, "review_game_screenshots")
        assert [p for _t, p in found] == [real], found


def test_artifact_collection_ignores_non_images_and_dedupes():
    with tempfile.TemporaryDirectory() as tmp:
        png = Path(tmp) / "a.png"
        cfg = Path(tmp) / "calibration.json"
        png.write_bytes(b"PNG")
        cfg.write_bytes(b"{}")
        result = {
            "screenshots": {"summary": str(png), "calibration": str(cfg)},
            "again": str(png),  # same file reached twice
        }
        found = agent.collect_artifacts(result, "t")
        assert [p for _t, p in found] == [png], found


def test_artifact_collection_survives_garbage():
    """The result comes from a tool that already catches its own failures, but
    it can still be a list, a string, or None. None of those may raise."""
    for junk in (None, [], "a string", 42, {"a": {"b": [None, 1, {"c": None}]}}):
        assert agent.collect_artifacts(junk, "t") == []
    # a self-referential structure must not hang the walk
    loop: dict = {}
    loop["self"] = loop
    loop["png"] = "x.png"
    assert agent.collect_artifacts(loop, "t") == []


def test_artifact_walk_is_bounded():
    """A cap and a depth limit, so a tool that returned a directory listing
    cannot try to post 400 images or recurse forever."""
    assert agent.MAX_ARTIFACTS <= 20
    deep: dict = {}
    node = deep
    for _ in range(30):
        node["next"] = {}
        node = node["next"]
    node["x.png"] = "deep.png"
    assert agent.collect_artifacts(deep, "t") == []

    wide = {str(i): f"{i}.png" for i in range(200)}
    assert len(agent.collect_artifacts(wide, "t")) <= agent.MAX_ARTIFACTS


# ---------------------------------------------------------------------------
# Artifact captions
# ---------------------------------------------------------------------------


def test_captions_describe_the_frame_without_guessing():
    """The filenames already say what each frame is, so the caption is a
    formatting job. Unrecognised names must not be invented into a claim."""
    cases = {
        "brilliant_m14_ply27_20260101_120000.png": "brilliant -- move 14 (ply 27)",
        "blunder_m12_ply24_20260101_120000.png": "blunder -- move 12 (ply 24)",
        "summary_20260101_120000.png": (
            "review summary -- accuracy, full tally, coach commentary"
        ),
        "move_m2_ply3_20260101_120000.png": "the position after move 2 (ply 3)",
        # take_screenshot's naming: shot_<region>[_<label>]_<stamp>. Without a
        # branch for these, every interactive frame was captioned "review
        # screenshot" -- a lie, since they are whatever the user asked to see,
        # possibly on a puzzle page or a lesson.
        "shot_auto_20260101_120000.png": "the board and the review panel",
        "shot_board_20260101_120000.png": "the board",
        "shot_sidebar_20260101_120000.png": "the side panel",
        "shot_viewport_20260101_120000.png": "the whole page",
        "shot_board_after_e5_20260101_120000.png": "the board -- after e5",
        # An unrecognised name says nothing about what is in it. "review
        # screenshot" was a guess, and a wrong one for any interactive frame.
        "mystery_20260101.png": "screenshot",
    }
    for name, expected in cases.items():
        assert discord_bot.describe_artifact(Path(name)) == expected, name


# ---------------------------------------------------------------------------
# Access control
# ---------------------------------------------------------------------------


class _User:
    def __init__(self, uid: int) -> None:
        self.id = uid


class _Guild:
    def __init__(self, gid: int) -> None:
        self.id = gid


def test_only_the_configured_user_ids_may_drive_the_browser():
    """The tools log into chess.com as one person and launch a real browser on
    one machine. This is the only gate on that, so an empty config must mean
    "nobody", not "everybody"."""
    original = discord_bot.ALLOWED_USERS
    everyone = discord_bot.OPEN_TO_EVERYONE
    try:
        discord_bot.ALLOWED_USERS = {111, 222}
        discord_bot.OPEN_TO_EVERYONE = False
        assert discord_bot.permitted(_User(111))
        assert not discord_bot.permitted(_User(999))

        # the dangerous default: no allow-list configured at all
        discord_bot.ALLOWED_USERS = set()
        assert not discord_bot.permitted(_User(111))
        assert not discord_bot.permitted(_User(999))

        # opt in on purpose
        discord_bot.OPEN_TO_EVERYONE = True
        assert discord_bot.permitted(_User(999))
    finally:
        discord_bot.ALLOWED_USERS = original
        discord_bot.OPEN_TO_EVERYONE = everyone


def test_the_allowlist_parser_ignores_junk_rather_than_crashing():
    """A typo in .env must not take the bot down, and must not silently widen
    access to a value that was not a user ID."""
    import importlib

    for raw, expected in (
        ("111", {111}),
        ("111, 222 ; 333", {111, 222, 333}),
        ("  111  ", {111}),
        ("", set()),
        ("all", set()),
    ):
        os.environ["DISCORD_ALLOWED_USER_IDS"] = raw
        importlib.reload(discord_bot)
        assert discord_bot.ALLOWED_USERS == expected, (raw, discord_bot.ALLOWED_USERS)
        assert discord_bot.OPEN_TO_EVERYONE == (raw.lower() in {"all", "*"})

    os.environ["DISCORD_ALLOWED_USER_IDS"] = "not-a-number, 444"
    importlib.reload(discord_bot)
    assert discord_bot.ALLOWED_USERS == {444}, discord_bot.ALLOWED_USERS

    del os.environ["DISCORD_ALLOWED_USER_IDS"]
    importlib.reload(discord_bot)


# ---------------------------------------------------------------------------
# The queue
# ---------------------------------------------------------------------------


def test_only_one_browser_at_a_time_and_the_queue_counter_balances():
    """Two reviews at once would fight over one Chromium profile and one
    storage_state.json. The counter is what the user is told, so it has to
    return to zero -- a double decrement makes it lie."""
    from discord_bot import ChessBot

    async def scenario():
        bot = ChessBot.__new__(ChessBot)  # no gateway, no token
        bot._browser_lock = asyncio.Lock()
        bot._queued = 0
        order: list[str] = []
        release = asyncio.Event()

        async def worker(name: str, hold: bool) -> None:
            queued = bot._browser_lock.locked()
            if queued:
                bot._queued += 1
            try:
                async with bot._browser_lock:
                    order.append(f"start:{name}")
                    if hold:
                        await release.wait()
                    else:
                        await asyncio.sleep(0)
                    order.append(f"end:{name}")
            finally:
                if queued:
                    bot._queued = max(0, bot._queued - 1)

        first = asyncio.create_task(worker("first", hold=True))
        await asyncio.sleep(0)  # let it take the lock
        second = asyncio.create_task(worker("second", hold=False))
        third = asyncio.create_task(worker("third", hold=False))
        await asyncio.sleep(0)
        # nobody else has started yet: the lock is doing its job
        assert order == ["start:first"], order
        release.set()
        await asyncio.gather(first, second, third)
        return order, bot._queued

    order, remaining = asyncio.run(scenario())
    # strictly serialised, never interleaved
    assert order == [
        "start:first",
        "end:first",
        "start:second",
        "end:second",
        "start:third",
        "end:third",
    ], order
    assert remaining == 0, f"queue counter did not balance: {remaining}"


# ---------------------------------------------------------------------------
# The agent core
# ---------------------------------------------------------------------------


def test_the_agent_keeps_the_thirteen_tools_and_the_export_contract():
    """main.py re-exports these; the CLI and the bot must not drift apart.

    The count was 7 and is now 13: the six interactive browser tools joined the
    seven review/HTTP ones. It is asserted as a number because that is what
    catches a tool being added without the front end being updated -- but the
    name list below is the real check, and it is what a future addition has to
    touch.
    """
    import main

    expected = {
        "get_recent_games",
        "get_player_profile",
        "review_game_screenshots",
        "get_review_state",
        "get_moves",
        "get_coach_commentary",
        "screenshot_game_move",
        "open_chess_page",
        "where_am_i",
        "move_board",
        "goto_move_in_game",
        "take_screenshot",
        "close_browser_session",
    }
    assert set(main.FUNCTION_MAP) == set(agent.FUNCTION_MAP)
    assert {f.__name__ for f in agent.TOOLS} == set(agent.FUNCTION_MAP)
    assert set(agent.FUNCTION_MAP) == expected, sorted(agent.FUNCTION_MAP)
    assert len(agent.TOOLS) == 13, agent.TOOLS
    for name, func in agent.FUNCTION_MAP.items():
        assert not inspect.iscoroutinefunction(func), (
            f"{name} is async; a bot runs it in a worker thread where the tools' "
            f"internal asyncio.run works, but an async tool would return a "
            f"coroutine and JSON-serialise to nothing"
        )
        assert func.__doc__ and "Args:" in func.__doc__, name


def test_every_interactive_tool_is_in_browser_tools():
    """Each one drives the shared browser, so each takes the gate.

    Without this a tool that launches a Chromium would run outside the lock and
    the Discord bot's promise -- one browser at a time -- would quietly stop
    holding.
    """
    interactive = {f.__name__ for f in tools_interact.INTERACTIVE_TOOLS}
    assert interactive <= set(agent.FUNCTION_MAP), interactive - set(agent.FUNCTION_MAP)
    missing = interactive - set(agent.BROWSER_TOOLS)
    assert not missing, f"not gated behind the browser lock: {sorted(missing)}"
    # The HTTP tools must stay outside it, or a two-second lookup queues behind
    # a two-minute page load.
    assert "get_recent_games" not in agent.BROWSER_TOOLS
    assert "get_player_profile" not in agent.BROWSER_TOOLS


def test_the_system_prompt_teaches_the_model_when_to_use_each_half():
    """The prompt was built entirely on "the review tools answer three
    questions -- pick the narrowest one". Adding six interactive tools without
    saying how they differ produces a model that still calls get_review_state
    for everything, which is the failure this whole addition risks."""
    instruction = agent.SYSTEM_INSTRUCTION
    for tool in ("open_chess_page", "move_board", "take_screenshot", "where_am_i"):
        assert tool in instruction, f"{tool} is not mentioned in the prompt"
    assert "one-shot" in instruction.lower()
    # The honesty rules the interactive tools depend on.
    assert "verified" in instruction
    assert "already_at" in instruction
    assert "engine_eval" in instruction
    # And it must not promise a multi-move engine line the site does not show.
    assert "does NOT expose a multi-move engine line" in instruction


def test_the_discord_platform_note_stops_the_model_printing_paths():
    """The CLI instruction says "include every path in `screenshots`". On
    Discord that produces a wall of D:\\...\\...png the user cannot open, so the
    platform note has to override it explicitly."""
    note = agent.DISCORD_PLATFORM_NOTE
    assert "NEVER print a" in note
    assert "attached" in note.lower()
    assert "table" in note.lower(), "pipe tables do not render in Discord"
    full = agent.system_instruction(agent.DISCORD_PLATFORM_NOTE)
    assert full.startswith(agent.SYSTEM_INSTRUCTION)
    assert "Discord" in full
    # the CLI gets the plain instruction, with no Discord text leaking in
    assert "Discord" not in agent.system_instruction()


def test_run_agent_reports_progress_and_collects_artifacts():
    """on_tool fires before each tool, from whatever thread this runs on -- that
    is how the bot updates a Discord message mid-review."""
    calls: list[tuple] = []

    class _Call:
        def __init__(self, name, args):
            self.name = name
            self.args = args

    class _Response:
        def __init__(self, text=None, calls=()):
            self.text = text
            self.function_calls = list(calls)

    class _Chat:
        def __init__(self, script, tmp):
            self.script = script
            self.tmp = tmp
            self.sent = 0

        def send_message(self, message):
            self.sent += 1
            return self.script.pop(0)

    with tempfile.TemporaryDirectory() as tmp:
        shot = Path(tmp) / "blunder_m12_ply24_x.png"
        shot.write_bytes(b"PNG")

        real = agent.FUNCTION_MAP["screenshot_game_move"]
        agent.FUNCTION_MAP["screenshot_game_move"] = lambda **kw: {
            "move": {"ply": 24, "screenshot": str(shot)}
        }
        try:
            chat = _Chat(
                [
                    _Response(calls=[_Call("screenshot_game_move", {"ply": 24})]),
                    _Response(text="Here is the blunder at move 12."),
                ],
                tmp,
            )
            text, tools, artifacts = agent.run_agent(
                chat,
                "show me move 12 as black",
                on_tool=lambda n, a: calls.append((n, a)),
            )
        finally:
            agent.FUNCTION_MAP["screenshot_game_move"] = real

    assert text == "Here is the blunder at move 12."
    assert tools == ["screenshot_game_move"]
    assert calls == [("screenshot_game_move", {"ply": 24})], "progress not reported"
    assert artifacts == [("screenshot_game_move", shot)], artifacts
    # the whole thing is JSON-safe, which is what the model is handed
    json.dumps({"result": {"artifacts": [str(p) for _t, p in artifacts]}})


def test_run_agent_stops_at_the_round_cap():
    """A model that keeps re-requesting a tool must not loop forever. The
    original had no bound and would spend tokens until something else stopped it.
    """
    calls: list[str] = []

    class _Call:
        def __init__(self):
            self.name = "get_review_state"
            self.args = {"game_url": "u"}

    class _Response:
        def __init__(self):
            self.text = "still going"
            self.function_calls = [_Call()]

    class _Chat:
        def send_message(self, message):
            calls.append("x")
            return _Response()

    original = agent.FUNCTION_MAP["get_review_state"]
    agent.FUNCTION_MAP["get_review_state"] = lambda **kw: {"error": "no"}
    try:
        text, _tools, _art = agent.run_agent(_Chat(), "go")
    finally:
        agent.FUNCTION_MAP["get_review_state"] = original

    # one call for the user message, then one per tool round
    assert len(calls) == agent.MAX_TOOL_ROUNDS + 1, (
        f"expected 1 user call + {agent.MAX_TOOL_ROUNDS} tool rounds, got {len(calls)}"
    )
    assert agent.MAX_TOOL_ROUNDS < 20, "the cap is the only thing bounding spend"
    # and the user is TOLD the turn was cut off, rather than being handed the
    # model's in-flight text as if it were a finished answer
    assert "stopped after" in text, text
    assert "still going" in text, "the partial answer should be kept, not dropped"
    assert text.index("still going") < text.index("stopped after"), (
        "the notice must come after what was actually answered"
    )


def test_an_unknown_tool_is_reported_not_called():
    class _Call:
        def __init__(self):
            self.name = "delete_everything"
            self.args = {}

    class _Response:
        def __init__(self, text=None):
            self.text = text
            # only keep asking for tools while there is no answer yet
            self.function_calls = [_Call()] if text is None else []

    class _Chat:
        def __init__(self):
            self.sent = 0

        def send_message(self, message):
            self.sent += 1
            # ask again only the first time, so the loop ends naturally
            return _Response() if self.sent == 1 else _Response(text="done")

    _text, tools, _art = agent.run_agent(_Chat(), "go")
    assert tools == ["delete_everything"], tools
    assert "delete_everything" not in agent.FUNCTION_MAP, (
        "the test is only meaningful if that tool really does not exist"
    )


def test_a_session_starts_over_rather_than_growing_forever():
    """Gemini chat history cannot be truncated, so a long-lived bot would
    eventually pay for -- and be distracted by -- an entire old conversation.
    A silent reset would look like the bot forgetting mid-sentence, so the
    result says it happened."""
    created: list[object] = []

    def fake_create_chat(client, platform_note=""):
        class _Chat:
            def __init__(self):
                self.turns = 0

            def send_message(self, message):
                self.turns += 1
                return type("R", (), {"text": "ok", "function_calls": []})()

        chat = _Chat()
        created.append(chat)
        return chat

    original = agent.create_chat
    agent.create_chat = fake_create_chat
    try:
        session = agent.AgentSession(client=object(), max_turns=2)
        assert session.ask("one").history_reset is False
        assert session.ask("two").history_reset is False
        third = session.ask("three")
        assert third.history_reset is True, "the cap was not enforced"
        assert "fresh conversation" in third.text
        assert len(created) == 2, "a new chat should have been created once"
        assert third.text.endswith("ok"), "the answer was replaced by the notice"

        session.reset()
        assert session.turns == 0
        session.ask("four")
        assert session.turns == 1
    finally:
        agent.create_chat = original


def test_a_session_turn_never_raises():
    """A front end wants a sentence to post, not an exception to convert."""

    class _Chat:
        def send_message(self, message):
            raise RuntimeError("the model is on fire")

    session = agent.AgentSession.__new__(agent.AgentSession)
    session._client = object()
    session._note = ""
    session._max_turns = 5
    session.turns = 0
    session._chat = _Chat()

    result = session.ask("hello")
    assert result.error is not None
    assert "on fire" in result.text, result.text
    assert result.text  # postable as-is


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


def test_the_bot_declares_only_the_intents_and_scopes_it_needs():
    """Message content is a PRIVILEGED intent: off by default, and the single
    most common reason a bot like this is online but replies to nothing."""
    assert discord_bot.INTENTS.message_content is True
    assert discord_bot.INTENTS.members is False, "not needed, and it is privileged"
    assert discord_bot.INTENTS.presences is False
    # view channel, send, embed, attach, read history -- nothing more
    assert discord_bot.BOT_PERMISSIONS == 1024 + 2048 + 16384 + 32768 + 65536
    assert discord_bot.BOT_PERMISSIONS == 117760
    # and none of the dangerous ones: create invite, kick, ban, admin,
    # manage channels, manage server, manage messages, mention everyone
    for bit, name in (
        (0x1, "CREATE_INSTANT_INVITE"),
        (0x2, "KICK_MEMBERS"),
        (0x4, "BAN_MEMBERS"),
        (0x8, "ADMINISTRATOR"),
        (0x10, "MANAGE_CHANNELS"),
        (0x20, "MANAGE_GUILD"),
        (0x2000, "MANAGE_MESSAGES"),
        (0x20000, "MENTION_EVERYONE"),
    ):
        assert not discord_bot.BOT_PERMISSIONS & bit, name


def test_attachment_and_typing_limits_are_under_discord_hard_limits():
    assert discord_bot.MAX_MESSAGE_CHARS <= 2000
    assert discord_bot.MAX_ATTACHMENTS <= 10
    # Discord's typing state lasts 10s; refresh under that or the indicator
    # visibly lapses mid-review.
    assert discord_bot.TYPING_REFRESH_SECONDS < 10


def test_the_mention_is_stripped_before_the_text_reaches_the_model():
    """Leaving "<@123>" in the message makes the model address the user as if
    they had typed their own Discord tag, and wastes tokens on it."""

    for raw, expected in (
        ("<@123456789> how was my game?", "how was my game?"),
        ("<@!123456789> review this", "review this"),
        ("<@123> <@456> hello", "hello"),
    ):
        assert discord_bot._MENTION_STRIP.sub("", raw).strip() == expected


def test_conversations_are_kept_per_person_per_place():
    """Keyed on the channel alone, two people in one channel inherit each
    other's "the opponent in game 2" and the bot looks confused rather than
    busy."""

    class _G:
        def __init__(self, gid):
            self.id = gid

    class _C:
        def __init__(self, cid):
            self.id = cid

    from discord_bot import ChessBot

    key = ChessBot._key_for(_G(1), _C(10), _User(111))
    assert key == (1, 10, 111)
    # a DM has no guild
    assert ChessBot._key_for(None, _C(10), _User(111))[0] == 0
    # different people in the same room are different conversations
    assert key != ChessBot._key_for(_G(1), _C(10), _User(222))
    # same person, different rooms are different conversations too
    assert key != ChessBot._key_for(_G(1), _C(11), _User(111))


def test_batching_never_exceeds_discord_file_limit():
    items = list(range(25))
    batches = list(discord_bot._batched(items, discord_bot.MAX_ATTACHMENTS))
    assert [len(b) for b in batches] == [10, 10, 5], batches
    assert [x for b in batches for x in b] == items, "batching lost or reordered"


# ---------------------------------------------------------------------------
# Command registration
# ---------------------------------------------------------------------------


def test_the_commands_are_real_Command_objects_and_register_cleanly():
    """`add_command` raises `TypeError: The command passed must be a subclass of
    Command` on a plain coroutine, and it does so inside `setup_hook`, which runs
    during login -- so the bot printed its config line, connected, and then died
    with a traceback instead of starting.

    Built here with no gateway and no token, which is the point: this needs no
    network to be a real check.
    """
    import discord.ext.commands as cmds

    bot = discord_bot.ChessBot()
    for attribute in ("cmd_reset", "cmd_status", "cmd_help"):
        command = getattr(bot, attribute)
        assert isinstance(command, cmds.Command), (
            f"{attribute} is {type(command).__name__}, not a Command -- it is "
            f"missing @commands.command(name=...)"
        )
    # and registering them, which is what actually raised
    asyncio.run(bot.setup_hook())
    names = {c.name for c in bot.commands}
    assert names == {"reset", "status", "help"}, names
    for command in bot.commands:
        # `ctx` is supplied by the framework and never appears in `params`; what
        # must not happen is a command demanding an argument the user cannot
        # sensibly supply. `!status` correctly takes none.
        required = [
            n
            for n, p in command.params.items()
            if p.default is p.empty and p.annotation is not bool
        ]
        assert not required, (command.name, required)


def test_command_names_are_the_documented_ones():
    """The decorator defaults the name to the *function* name, so `cmd_reset`
    would have become the command `!cmd_reset` -- documented as `!reset`."""
    bot = discord_bot.ChessBot()
    asyncio.run(bot.setup_hook())
    by_name = {c.name: c for c in bot.commands}
    assert "cmd_reset" not in by_name, "the name came from the method name"
    for name in ("reset", "status", "help"):
        assert name in by_name, sorted(by_name)
        assert by_name[name].help, f"!{name} has no help text"


def test_on_message_dispatches_prefix_commands():
    """The subtle one.

    `commands.Bot.on_message` is what normally calls `process_commands`. This
    class *overrides* `on_message`, so without an explicit call the commands
    would be registered, documented in the README, and silently never fire --
    and every other test would still pass.
    """
    body = inspect.getsource(discord_bot.ChessBot.on_message)
    assert "process_commands" in body, (
        "overriding on_message bypasses the command dispatch"
    )
    assert body.index("process_commands") < body.index("self.user not in"), (
        "commands must be tried before the mention gate, or `!reset` in a "
        "channel would be ignored"
    )
    assert "ctx.invoked_with" in body, "a matched command must stop the turn"


def test_one_access_check_serves_both_the_chat_and_the_command_paths():
    """Two copies of the same gate is two rules, and the one that drifts is the
    one nobody tests."""
    for name in ("cmd_reset", "cmd_status", "cmd_help"):
        # the decorator replaces the method with a Command, so the source of
        # the real function is on .callback
        command = getattr(discord_bot.ChessBot, name)
        body = inspect.getsource(command.callback)
        assert "_refuse_if_not_permitted" in body, name
        # and no command decides access for itself any more
        assert "permitted(ctx.author)" not in body, (
            f"{name} has its own copy of the access rule"
        )
    chat = inspect.getsource(discord_bot.ChessBot.on_message)
    assert "_refuse_if_not_permitted" in chat
    assert "permitted(message.author)" not in chat


def test_the_help_text_advertises_commands_that_actually_exist():
    """A documented `!command` that does not exist is worse than an undocumented
    one, because the user is told it is there. So `!help` is *generated* from the
    registry rather than written out, and this checks the generation."""
    bot = discord_bot.ChessBot()
    asyncio.run(bot.setup_hook())
    real = {c.name for c in bot.commands}
    text = discord_bot.help_text({c.name: c.help for c in bot.commands})
    advertised = {m.lstrip("!") for m in re.findall(r"!(?:[a-z_]+)", text)}
    assert advertised, "the help text mentions no commands at all"
    assert not (advertised - real), f"help advertises {advertised - real}"
    for name in real:
        assert f"!{name}" in text, f"!{name} is registered but not documented"
    # and the examples are still there, since that is the part a generated list
    # cannot produce
    assert "how was my last game" in text.lower()


def test_our_own_help_replaces_the_builtin_one():
    """`!help` collides with the built-in command, which cannot be overridden --
    only removed. Registering ours without removing that one raised
    `CommandRegistrationError: The command help is already an existing command`."""
    bot = discord_bot.ChessBot()
    asyncio.run(bot.setup_hook())
    assert bot.help_command is None, (
        "the built-in help must be removed, or ours cannot register"
    )
    names = [c.name for c in bot.commands]
    assert names.count("help") == 1, names
    assert "help" in names


def test_add_command_is_not_awaited():
    """`Bot.add_command` is a plain sync method. `await`ing it raises
    `TypeError: 'NoneType' object can't be awaited` from inside `setup_hook`,
    which runs during login -- so the bot printed its config, connected, and
    then died again. `process_commands` a few lines below IS async, which is
    what makes this easy to get wrong in both directions."""
    import inspect

    from discord.ext import commands as cmds

    assert not inspect.iscoroutinefunction(cmds.Bot.add_command), (
        "discord.py changed: add_command is now a coroutine"
    )
    assert inspect.iscoroutinefunction(cmds.Bot.process_commands), (
        "discord.py changed: process_commands is no longer a coroutine"
    )
    body = inspect.getsource(discord_bot.ChessBot.setup_hook)
    assert "await self.add_command" not in body
    assert "self.add_command(" in body
    assert "await self.process_commands" in inspect.getsource(
        discord_bot.ChessBot.on_message
    )


def test_two_people_run_at_once_and_only_reviews_queue():
    """The end-to-end shape of the concurrency fix, with a fake model and fake
    tools so it needs neither network nor a browser.

    Before: the whole turn sat behind one global lock, so a two-second
    `get_recent_games` waited out a two-minute review. Now the per-session lock
    keeps one person's turns apart, and only `BROWSER_TOOLS` take the shared
    browser.
    """

    class _Msg(_FakeMessage):
        def __init__(self, user_id, channel_id=1, **kw):
            super().__init__(**kw)
            self.author = _User(user_id)
            self.guild = _Guild(7)
            self.mentions = ["the bot"]
            self.channel = _FakeChannel(channel_id=channel_id)

    async def scenario():
        bot = discord_bot.ChessBot()
        loop = asyncio.get_running_loop()

        # a "review" that takes a moment, and an HTTP tool that does not
        started = asyncio.Event()
        release = asyncio.Event()

        async def _hold():
            started.set()
            await release.wait()

        def slow_review(**kwargs):
            # block this worker thread the way a real browser tool does
            asyncio.run_coroutine_threadsafe(_hold(), loop).result()
            return {"error": "no session"}

        # 1 holds the browser, 2 needs no browser, 3 needs the browser
        scripts = {
            1: ["review_game_screenshots"],
            2: ["get_recent_games"],
            3: ["review_game_screenshots"],
        }
        bot.session_for = lambda key: _session_with(scripts[key[2]], answer="answered")

        original = dict(agent.FUNCTION_MAP)
        agent.FUNCTION_MAP["review_game_screenshots"] = slow_review
        agent.FUNCTION_MAP["get_recent_games"] = lambda **kw: {"games": []}
        try:
            reviewer = _Msg(1)
            review_task = asyncio.create_task(bot._answer(reviewer, "review it"))
            await asyncio.wait_for(started.wait(), 5)

            # a different person asks something that needs no browser at all
            asker = _Msg(2)
            await asyncio.wait_for(bot._answer(asker, "how was my last game?"), 5)
            assert bot._browser_lock.locked(), "the review let go of the browser"

            # and a second review cannot start until the browser is free
            third = asyncio.create_task(bot._answer(_Msg(3, 2), "review this too"))
            await asyncio.sleep(0.05)
            assert not third.done(), "a queued review ran before its turn"

            release.set()
            await asyncio.wait_for(review_task, 5)
            await asyncio.wait_for(third, 5)
        finally:
            agent.FUNCTION_MAP.clear()
            agent.FUNCTION_MAP.update(original)
        return asker

    asker = asyncio.run(scenario())
    answers = [c for c, _k in asker.edits if c and "Working on it" not in c]
    assert answers, f"the HTTP-only question was never answered: {asker.edits}"
    assert any("answered" in c for c in answers), asker.edits
    assert not any("Queued" in str(c) for c, _k in asker.edits), (
        "an HTTP question was told it was queued behind the browser"
    )


def test_an_urgent_status_line_is_not_delayed_or_lost():
    """The line that says a tool really started must not sit behind the rate
    limit after a wait, and it must not clobber a newer ordinary line on its
    way through -- one queue, so both hold at once."""

    async def scenario():
        bot = discord_bot.ChessBot()
        loop = asyncio.get_running_loop()
        status = _FakeMessage()
        gate = discord_bot._BrowserGate(bot, loop, status, loop.time())
        await gate._queue("queued")
        # two urgent lines land back to back with no gap allowed
        await asyncio.gather(
            gate._queue("running A", urgent=True),
            gate._queue("running B", urgent=True),
        )
        return status

    status = asyncio.run(scenario())
    lines = [c for c, _a in status.edits]
    assert lines[0] == "queued", lines
    assert lines[-1] == "running B", f"the newest line was lost: {lines}"
    assert "running A" in lines, f"an urgent line was dropped: {lines}"


def test_a_queued_reviewer_is_told_where_it_stands():
    """A request that has to wait for the browser says so, at the moment it
    starts waiting. The old code checked once, at the start of the turn, so it
    was wrong whenever the queue formed later and silent whenever it did not.
    """

    class _Msg(_FakeMessage):
        def __init__(self, user_id, channel_id=1, **kw):
            super().__init__(**kw)
            self.author = _User(user_id)
            self.guild = _Guild(7)
            self.mentions = ["the bot"]
            self.channel = _FakeChannel(channel_id=channel_id)

    async def scenario():
        bot = discord_bot.ChessBot()
        loop = asyncio.get_running_loop()
        held = asyncio.Event()
        release = asyncio.Event()

        async def _hold():
            held.set()
            await release.wait()

        def slow_review(**kwargs):
            asyncio.run_coroutine_threadsafe(_hold(), loop).result()
            return {"error": "no session"}

        bot.session_for = lambda key: _session_with(["review_game_screenshots"])

        original = dict(agent.FUNCTION_MAP)
        agent.FUNCTION_MAP["review_game_screenshots"] = slow_review
        try:
            first = _Msg(1, 11)
            first_task = asyncio.create_task(bot._answer(first, "review"))
            await asyncio.wait_for(held.wait(), 5)

            second = _Msg(2, 12)
            second_task = asyncio.create_task(bot._answer(second, "review"))
            for _ in range(100):
                if bot._queued:
                    break
                await asyncio.sleep(0.02)
            assert bot._queued == 1, "the waiter was never counted"
            # past the status line's own rate limit
            await asyncio.sleep(discord_bot.STATUS_EDIT_MIN_INTERVAL * 2)
            release.set()
            await asyncio.wait_for(asyncio.gather(first_task, second_task), 15)
        finally:
            agent.FUNCTION_MAP.clear()
            agent.FUNCTION_MAP.update(original)
        return second, bot._queued

    second, remaining = asyncio.run(scenario())
    said = " ".join(str(c) for c, _k in second.edits)
    assert "Queued" in said, f"the waiter was not told anything: {second.edits}"
    assert "in line" in said, second.edits
    assert remaining == 0, f"the queue counter did not balance: {remaining}"


# ---------------------------------------------------------------------------
# The progress line is for the user, not for the log
# ---------------------------------------------------------------------------


def test_the_progress_line_never_names_a_tool_or_its_arguments():
    """`_running `screenshot_game_move` gameurl=https://www.chess.com/...`_
    was being edited into the one message the user is reading.

    Three things wrong with it: it is code in the middle of an answer, it pastes
    half a URL into the channel, and it says nothing the user asked to know.
    What they want is "is it working, and on what", in words.

    Every registered tool is checked, including with arguments full of things
    that must never be printed, and an unrecognised name must not leak either --
    the mapping is a lookup with a safe default, not a fallback to the name.
    """
    args = {
        "game_url": "https://www.chess.com/game/live/184575245570?token=abc",
        "ply": 27,
        "label": "after e5",
    }

    async def scenario():
        bot = discord_bot.ChessBot()
        loop = asyncio.get_running_loop()
        gate = discord_bot._BrowserGate(bot, loop, _FakeMessage(), loop.time())
        lines = {name: gate._describe(name, args) for name in agent.FUNCTION_MAP}
        lines["a_tool_from_the_future"] = gate._describe("a_tool_from_the_future", args)
        return lines

    lines = asyncio.run(scenario())
    for name, line in lines.items():
        assert name not in line, f"{name} leaked into the status line: {line}"
        assert "chess.com" not in line, line
        assert "ply" not in line and "after e5" not in line, line
        assert line.startswith("_") and line.endswith("_"), line
    # nearly every tool has its own phrase, so the line is actually informative
    # rather than a single "working on it" for the whole turn
    assert len(set(lines.values())) >= len(agent.FUNCTION_MAP) - 1, lines
    # a tool we do not have a phrase for is described, not named
    assert "a_tool_from_the_future" not in lines["a_tool_from_the_future"]


def test_the_technical_progress_line_is_still_available_for_the_owner():
    """Quiet by default, not gone: DISCORD_SHOW_TOOLS=1 puts the old line back
    for whoever is watching the bot and needs to see which step is slow."""

    async def scenario():
        bot = discord_bot.ChessBot()
        loop = asyncio.get_running_loop()
        gate = discord_bot._BrowserGate(bot, loop, _FakeMessage(), loop.time())
        return gate._describe("get_moves", {"game_url": "u"})

    original = discord_bot.SHOW_TOOL_NAMES
    discord_bot.SHOW_TOOL_NAMES = True
    try:
        line = asyncio.run(scenario())
    finally:
        discord_bot.SHOW_TOOL_NAMES = original
    assert "`get_moves`" in line, line

    discord_bot.SHOW_TOOL_NAMES = False
    assert "get_moves" not in asyncio.run(scenario())


# ---------------------------------------------------------------------------
# A picture that was never produced
# ---------------------------------------------------------------------------


def test_a_screenshot_that_was_never_produced_says_so():
    """The screenshot tool ran, returned an error, and the model's own text is
    the only thing the user gets.

    A model handed {"error": ...} will still write "here is the screenshot", and
    then the picture is simply absent -- which reads as the bot ignoring the
    request. So the front end says it, and says WHY, from the tool's own words.
    """
    result = agent.AgentResult(
        text="Here is the screenshot of move 12.",
        tools_used=["screenshot_game_move"],
        tool_errors=[
            (
                "screenshot_game_move",
                "The move list could not be read, so no move could be "
                "photographed. Run: python dump_pages.py https://x",
            )
        ],
    )
    status = _FakeMessage("working")
    asyncio.run(discord_bot.ChessBot()._post(status, result))
    posted = " ".join(str(c) for c, _k in status.every_write())
    assert "No picture came through" in posted, status.every_write()
    assert "move list could not be read" in posted, (
        "the reason is what the user can act on; the generic line is a dead end"
    )
    # the fix belongs to the owner, and the reader cannot run it
    assert "dump_pages" not in posted, posted
    assert "Here is the screenshot" in posted, "the answer itself was replaced"

    # and it stays quiet when a picture really was attached, or when no
    # screenshot was involved at all
    # not TemporaryDirectory: `discord.File` still holds the PNG open
    tmp = tempfile.mkdtemp()
    try:
        shot = Path(tmp) / "shot.png"
        shot.write_bytes(b"PNG")
        for with_artifact in (True, False):
            status = _FakeMessage("working")
            asyncio.run(
                discord_bot.ChessBot()._post(
                    status,
                    agent.AgentResult(
                        text="Here it is.",
                        tools_used=(
                            ["take_screenshot"] if with_artifact else ["get_moves"]
                        ),
                        artifacts=[("take_screenshot", shot)] if with_artifact else [],
                    ),
                )
            )
            posted = " ".join(str(c) for c, _k in status.every_write())
            assert "No picture came through" not in posted, posted
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_an_expired_chess_com_session_is_named_as_the_reason():
    """The failure that actually happens, and the one that is actionable.

    Every review tool reports a missing or expired sign-in the same way, and it
    is the most common reason a picture does not arrive. Naming it turns "the
    bot is broken" into "the sign-in needs doing again once" -- which is true,
    and which the user can pass to the owner.
    """
    for message in (
        "Not logged in. Run 'python login_chess.py' once to save your chess.com "
        "session, then ask again.",
        "The session browser could not be opened.",
    ):
        note = discord_bot.picture_failure_note([("screenshot_game_move", message)])
        if "not logged in" in message.lower():
            assert note == discord_bot._EXPIRED_NOTE, note
            assert "expired" in note, note
            assert "login" not in note.lower() or "sign-in" in note, note
        else:
            assert "No picture" in note, note

    # the note must not tell the reader to run something on the owner's machine
    assert "python" not in discord_bot._EXPIRED_NOTE.lower()


def test_the_failure_note_falls_back_rather_than_inventing_a_reason():
    assert "No picture came through" in discord_bot.picture_failure_note([])
    assert "No picture came through" in discord_bot.picture_failure_note(
        [("take_screenshot", "")]
    )
    # two different reasons, both said, each once
    note = discord_bot.picture_failure_note(
        [
            ("get_moves", "the page did not load"),
            ("take_screenshot", "the page did not load"),
            ("get_review_state", "board not found"),
        ]
    )
    assert note.count("did not load") == 1, note
    assert "board not found" in note, note


def test_the_sign_in_file_age_is_reported_as_an_age_not_a_verdict():
    """`!status` says how old the saved session file is.

    It cannot know whether chess.com still accepts the cookie without loading a
    page, so it must not claim the session is expired -- that would be a
    plausible wrong value, which is the one thing this project keeps designing
    against. Age, and a hint.
    """
    with tempfile.TemporaryDirectory() as tmp:
        state = Path(tmp) / "storage_state.json"
        state.write_text("{}")
        original = os.environ.get("CHESS_AGENT_STATE")
        os.environ["CHESS_AGENT_STATE"] = str(state)
        try:
            assert "days old" in discord_bot.chess_com_sign_in_age()
            fresh = discord_bot.chess_com_sign_in_age()
            assert "login_chess" not in fresh, fresh
            # a genuinely old file earns the warning
            old = time.time() - 90 * 86400
            os.utime(state, (old, old))
            stale = discord_bot.chess_com_sign_in_age()
            assert "login_chess" in stale, stale
            # and no file at all is not a claim either way
            os.environ["CHESS_AGENT_STATE"] = str(Path(tmp) / "missing.json")
            assert discord_bot.chess_com_sign_in_age() == ""
        finally:
            if original is None:
                os.environ.pop("CHESS_AGENT_STATE", None)
            else:
                os.environ["CHESS_AGENT_STATE"] = original


def test_the_turn_records_why_a_tool_failed():
    """A missing picture is only reportable if the reason survived the turn.

    The tools report failure as an `error` key, the model is handed that key and
    writes prose about it, and the front end never sees a tool's return value --
    so without this the only thing anyone can say is "no picture came through",
    which is the dead end this whole path exists to avoid.
    """

    class _Call:
        def __init__(self, name, args=None):
            self.name = name
            self.args = args or {}

    class _Response:
        def __init__(self, text=None, calls=()):
            self.text = text
            self.function_calls = list(calls)

    class _Chat:
        def __init__(self):
            self.sent = 0

        def send_message(self, message):
            self.sent += 1
            if self.sent == 1:
                return _Response(
                    calls=[
                        _Call("screenshot_game_move", {"game_url": "u"}),
                        _Call("get_moves", {"game_url": "u"}),
                        _Call("where_am_i"),
                    ]
                )
            return _Response(text="Here it is.")

    original = dict(agent.FUNCTION_MAP)
    agent.FUNCTION_MAP["screenshot_game_move"] = lambda **kw: {
        "error": "Not logged in. Run 'python login_chess.py' once."
    }
    agent.FUNCTION_MAP["get_moves"] = lambda **kw: {"moves": []}
    agent.FUNCTION_MAP["where_am_i"] = lambda **kw: {"session_open": False}
    try:
        errors: list = []
        _text, tools, artifacts = agent.run_agent(_Chat(), "show me", errors=errors)
    finally:
        agent.FUNCTION_MAP.clear()
        agent.FUNCTION_MAP.update(original)

    assert not artifacts, "no file was written, so nothing should be collected"
    assert errors == [
        ("screenshot_game_move", "Not logged in. Run 'python login_chess.py' once.")
    ], errors
    # a tool that succeeded contributes nothing, even though it ran
    assert "get_moves" not in [name for name, _m in errors]
    assert "get_moves" in tools, "tools_used is what the model asked for"

    # a turn that dies mid-way still carries what the tools had already said
    class _Dying:
        def __init__(self):
            self.sent = 0

        def send_message(self, message):
            self.sent += 1
            if self.sent == 1:
                return _Response(calls=[_Call("screenshot_game_move")])
            raise RuntimeError("the socket went away")

    original = agent.FUNCTION_MAP["screenshot_game_move"]
    agent.FUNCTION_MAP["screenshot_game_move"] = lambda **kw: {"error": "nope"}
    try:
        session = agent.AgentSession.__new__(agent.AgentSession)
        session._client = object()
        session._note = ""
        session._max_turns = 5
        session.turns = 0
        session._chat = _Dying()
        result = session.ask("show me the brilliant move")
    finally:
        agent.FUNCTION_MAP["screenshot_game_move"] = original
    assert result.error is not None
    assert result.tool_errors == [("screenshot_game_move", "nope")], result.tool_errors


def test_several_pictures_in_one_turn_are_all_collected():
    """The user asks for three moves; three distinct files must all be attached.

    `seen` is shared for the whole turn, so a repeated path is attached once --
    which is why every screenshot tool has to write a NEW file each time. That
    pairing is what this pair of tests pins.
    """
    with tempfile.TemporaryDirectory() as tmp:
        shots = []
        for i in range(3):
            path = Path(tmp) / f"shot_board_{i}.png"
            path.write_bytes(b"PNG")
            shots.append(path)
        seen: set = set()
        found = []
        for path in shots:
            found.extend(
                agent.collect_artifacts(
                    {"screenshot": str(path)}, "take_screenshot", seen=seen
                )
            )
        assert [p for _t, p in found] == shots, found
        # and one path reached twice is still one attachment
        again = agent.collect_artifacts(
            {"a": str(shots[0]), "b": str(shots[0])}, "take_screenshot", seen=seen
        )
        assert again == [], again


# ---------------------------------------------------------------------------
# Memory: the conversation outlives the model's own history
# ---------------------------------------------------------------------------


class _RecordingChat:
    """A Gemini chat that keeps every message it was asked to send."""

    def __init__(self, answer="ok"):
        self.messages = []
        self.answer = answer

    def send_message(self, message):
        self.messages.append(message)
        return type("R", (), {"text": self.answer, "function_calls": []})()


@contextmanager
def _recording_chats(answer="ok", created=None):
    """Patch chat creation for the whole block, not just the constructor.

    A session creates its chat lazily, on the first turn -- so a patch that is
    undone when the constructor returns leaves every `ask` running the real
    `create_chat` against a bare `object()`.
    """
    original = agent.create_chat

    def fake_create_chat(client, platform_note=""):
        chat = _RecordingChat(answer)
        if created is not None:
            created.append(chat)
        return chat

    agent.create_chat = fake_create_chat
    try:
        yield
    finally:
        agent.create_chat = original


def test_the_conversation_is_remembered_across_a_reset():
    """The complaint: "I just sent you a link, asked about it later, and you
    asked me for the link again".

    The turn cap replaces the Gemini chat, and the old chat is unrecoverable --
    so the memory has to live on the session, not in the chat. The new chat's
    first message must therefore carry what the user already said.
    """
    created: list = []
    link = "https://www.chess.com/game/live/184575245570"
    with _recording_chats(answer="White 69.3 - Black 79.9", created=created):
        session = agent.AgentSession(client=object(), max_turns=2)

        session.ask(f"analyse {link}")
        # the first turn of a conversation carries no preamble at all
        assert created[0].messages[0] == f"analyse {link}"

        session.ask("how accurate was I?")
        assert agent.MEMORY_FOOTER.lower() in created[0].messages[1].lower()
        assert link in created[0].messages[1]

        third = session.ask("which game was that?")
    assert third.history_reset is True, "the cap was not enforced"
    assert len(created) == 2, "a fresh chat was not started"
    sent = created[1].messages[0]
    assert link in sent, "the link from two turns ago is gone after the reset"
    assert "how accurate was I?" in sent
    assert sent.rstrip().endswith("which game was that?"), (
        "the new message must come last, or the model answers the memory"
    )
    # and the user is told the conversation carries over, so the reset does not
    # read as the bot losing the thread
    assert "fresh conversation" in third.text
    assert "do not have to repeat yourself" in third.text


def test_the_memory_is_bounded_and_keeps_the_newest_turns():
    """It is prepended to every message, so it cannot grow without limit -- and
    a half-remembered exchange is worse than none, so the budget drops whole
    turns from the OLD end rather than cutting one in half."""
    memory = agent.ConversationMemory(exchanges=3, item_chars=60, total_chars=2000)
    for i in range(9):
        memory.remember(f"question {i}", f"answer {i}")
    assert len(memory) == 3
    block = memory.as_preamble()
    assert "question 8" in block and "question 6" in block
    assert "question 5" not in block, "the oldest turn was not dropped"
    assert block.count("- user:") == 3
    assert block.rstrip().endswith(agent.MEMORY_FOOTER)

    # a turn too big for the budget is not partially remembered
    tight = agent.ConversationMemory(exchanges=4, item_chars=400, total_chars=200)
    for i in range(3):
        tight.remember("x" * 300, "y" * 300)
    assert tight.lines() == [], tight.lines()
    assert tight.as_preamble() == "", "a memory with no room must say nothing"
    assert tight.decorate("hello") == "hello"

    # and whitespace is collapsed: this is carried on every turn
    wide = agent.ConversationMemory(exchanges=5, item_chars=200, total_chars=2000)
    wide.remember("a\n\nb   c", "x\n| y | z")
    assert wide.lines() == ["- user: a b c\n- you: x | y | z"], wide.lines()

    # a turn with no answer is not worth remembering
    blank = agent.ConversationMemory()
    blank.remember("question", "")
    blank.remember("", "answer")
    assert len(blank) == 0


def test_a_session_built_without_init_still_has_a_memory():
    """`AgentSession.__new__` is how the tests reach one failing turn without a
    client, so the memory cannot be an attribute that only `__init__` sets."""
    session = agent.AgentSession.__new__(agent.AgentSession)
    session._client = object()
    session._note = ""
    session._max_turns = 5
    session.turns = 0
    session._chat = _RecordingChat("hi")
    result = session.ask("hello")
    assert result.error is None, result.text
    assert len(session.memory) == 1


def test_an_explicit_reset_forgets_the_memory_too():
    """`!reset` is the user asking for a clean slate, so the memory goes with
    the chat -- the opposite of the turn-cap reset, which nobody asked for."""
    created: list = []
    with _recording_chats(created=created):
        session = agent.AgentSession(client=object())
        session.ask("remember this link")
        assert len(session.memory) == 1
        session.reset()
        assert session.turns == 0
        assert len(session.memory) == 0
        assert session.memory.as_preamble() == ""
        session.ask("what did I just send?")
    assert created[-1].messages[0] == "what did I just send?", (
        "a remembered exchange survived an explicit reset"
    )


# ---------------------------------------------------------------------------
# Nothing this bot posts can ping anybody
# ---------------------------------------------------------------------------


class _ToolCall:
    def __init__(self, name, args=None):
        self.name = name
        self.args = args or {}


class _GenResponse:
    def __init__(self, text=None, calls=()):
        self.text = text
        self.function_calls = list(calls)


class _FakeChat:
    """A Gemini chat that asks for a scripted list of tools, then answers."""

    def __init__(self, tool_calls=(), answer="done"):
        self.tool_calls = list(tool_calls)
        self.answer = answer
        self.sent = 0

    def send_message(self, message):
        self.sent += 1
        if self.tool_calls and isinstance(message, str):
            return _GenResponse(calls=[_ToolCall(self.tool_calls.pop(0))])
        return _GenResponse(text=self.answer)


def _session_with(tool_calls=(), answer="done"):
    """A real AgentSession, already holding a fake chat.

    A `__new__`-ed one would fail in `_ensure_chat` and every turn would come
    back as an error, which would make these tests pass for the wrong reason.
    """
    session = agent.AgentSession(client=object())
    session._chat = _FakeChat(tool_calls, answer)
    return session


class _FakeResponse:
    status = 429
    reason = "Too Many Requests"


def _http_error(message="rate limited"):
    return discord.HTTPException(_FakeResponse(), message)


class _FakeChannel:
    """Records what was sent, and can be told to start refusing."""

    def __init__(self, fail_from=None, fail_with=None, channel_id=1):
        self.sent = []
        self.fail_from = fail_from
        self.fail_with = fail_with or _http_error()
        self.id = channel_id

    async def send(self, content=None, **kwargs):
        if self.fail_from is not None and len(self.sent) >= self.fail_from:
            raise self.fail_with
        self.sent.append((content, kwargs))
        return _FakeMessage(content or "")

    async def typing(self):
        return None


class _FakeMessage:
    def __init__(self, content="status", channel=None):
        self.content = content
        self.channel = channel or _FakeChannel()
        self.edits = []

    async def edit(self, *, content=None, allowed_mentions=None):
        self.edits.append((content, allowed_mentions))
        self.content = content or ""
        return self

    async def reply(self, content=None, **kwargs):
        self.edits.append((content, kwargs))
        return self

    def every_write(self):
        """(content, allowed_mentions) for every write, however it was made."""
        return self.edits + [
            (content, kwargs.get("allowed_mentions"))
            for content, kwargs in self.channel.sent
        ]


def test_nothing_this_bot_posts_can_ping_anyone():
    """The reply text is written by a language model, so Discord's default --
    parse everyone, users and roles -- means a reply containing `@everyone`
    actually pings, in somebody else's server, with a permission set that
    deliberately does not include MENTION_EVERYONE.

    `AllowedMentions()` really does default to parsing all three; the empty
    `parse` list below is the only thing that stops it.
    """
    assert discord.AllowedMentions().to_dict()["parse"] == [
        "everyone",
        "users",
        "roles",
    ], "discord.py changed: the default is no longer the dangerous one"
    assert discord_bot.NO_MENTIONS.to_dict() == {"parse": []}
    # on the Bot, so anything the bot sends inherits it
    assert discord_bot.ChessBot().allowed_mentions.to_dict() == {"parse": []}

    # not TemporaryDirectory: `discord.File` still holds the PNG open afterwards
    tmp = tempfile.mkdtemp()
    try:
        shot = Path(tmp) / "summary.png"
        shot.write_bytes(b"PNG")
        status = _FakeMessage("working")
        result = agent.AgentResult(
            text="here you go @everyone <@123> -- and a second chunk",
            artifacts=[("review_game_screenshots", shot)],
        )
        asyncio.run(discord_bot.ChessBot()._post(status, result))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    writes = status.every_write()
    assert writes, "nothing was posted"
    for content, allowed in writes:
        assert allowed is not None, f"a write passed no allowed_mentions: {content!r}"
        assert allowed.to_dict() == {"parse": []}, content


# ---------------------------------------------------------------------------
# A hung turn cannot take the bot down with it
# ---------------------------------------------------------------------------


def test_a_turn_that_never_finishes_is_reported_not_waited_on():
    """`wait_for` stops US waiting; it cannot stop a thread inside Playwright.
    Without a deadline the user stares at "working..." forever, and without the
    watchdog the browser lock stays held by a thread that will never return, so
    every later request queues behind it in silence. That is the whole failure
    this pair of tests pins.
    """
    bot = discord_bot.ChessBot()
    original = discord_bot.TURN_TIMEOUT_SECONDS
    discord_bot.TURN_TIMEOUT_SECONDS = 0.05
    try:

        async def scenario():
            async def wedged():
                await asyncio.sleep(30)

            return await bot._bounded(wedged(), 0.0, 42)

        result = asyncio.run(scenario())
    finally:
        discord_bot.TURN_TIMEOUT_SECONDS = original

    assert result.error is not None
    assert "gave up" in result.text, result.text
    assert result.text, "the user must get something postable"
    assert 42 in bot._stalled, "a stalled turn must be visible in !status"


def test_the_browser_lock_is_reclaimed_after_an_abandoned_turn():
    bot = discord_bot.ChessBot()
    original = discord_bot.BROWSER_LOCK_GRACE_SECONDS
    discord_bot.BROWSER_LOCK_GRACE_SECONDS = 0.01
    try:

        async def scenario():
            await bot._browser_lock.acquire()
            assert bot._browser_lock.locked()
            bot._schedule_watchdog(7)
            await asyncio.sleep(0.08)
            return bot._browser_lock.locked()

        still_locked = asyncio.run(scenario())
    finally:
        discord_bot.BROWSER_LOCK_GRACE_SECONDS = original

    assert not still_locked, (
        "the browser lock was never taken back, so every later request is stuck "
        "behind a thread that will never return"
    )
    assert 7 not in bot._stalled, "a reclaimed turn should stop being reported"


# ---------------------------------------------------------------------------
# Only the browser is serialised
# ---------------------------------------------------------------------------


def test_only_browser_tools_take_the_browser_lock():
    """`get_recent_games` is an HTTPS call with a 15s timeout. It shares no
    Chromium profile and no `storage_state.json` with a review, so putting it
    behind the same lock made a two-second question wait two minutes -- and the
    user was told, wrongly, that it was queued behind the browser.
    """
    assert "get_recent_games" not in agent.BROWSER_TOOLS
    assert "get_player_profile" not in agent.BROWSER_TOOLS
    assert set(agent.BROWSER_TOOLS) <= set(agent.FUNCTION_MAP)
    assert {
        "review_game_screenshots",
        "screenshot_game_move",
        "get_moves",
        "get_review_state",
        "get_coach_commentary",
        # The interactive tools take the same gate: each drives the shared
        # browser session. The gate covers the CALL, not the session -- the
        # session outlives the call deliberately, and serialises itself.
        "open_chess_page",
        "where_am_i",
        "move_board",
        "goto_move_in_game",
        "take_screenshot",
        "close_browser_session",
    } == set(agent.BROWSER_TOOLS)

    bot = discord_bot.ChessBot()

    async def scenario():
        loop = asyncio.get_running_loop()
        status = _FakeMessage()
        gate = discord_bot._BrowserGate(bot, loop, status, loop.time())

        # a non-browser tool: the lock must not even be looked at
        await asyncio.to_thread(gate.acquire, "get_recent_games", {"username": "x"})
        assert not bot._browser_lock.locked(), "an HTTP call took the browser lock"
        await asyncio.to_thread(gate.release, "get_recent_games")

        # a browser tool: it must be held for the call and released after
        await asyncio.to_thread(gate.acquire, "get_review_state", {"game_url": "u"})
        assert bot._browser_lock.locked(), "a review ran without the lock"
        await asyncio.to_thread(gate.release, "get_review_state")
        await asyncio.sleep(0)  # let call_soon_threadsafe run
        assert not bot._browser_lock.locked(), "the lock was not released"
        return status

    asyncio.run(scenario())


def test_a_gate_is_released_even_when_the_tool_raises():
    """A gate released only on success deadlocks every later request behind
    whatever the tool raised -- and the tools are the part of this project most
    likely to raise, since they drive a browser."""

    class _Call:
        def __init__(self, name):
            self.name = name
            self.args = {}

    class _Response:
        def __init__(self, calls=(), text=None):
            self.function_calls = list(calls)
            self.text = text

    class _Chat:
        def __init__(self):
            self.sent = 0
            self.parts = []

        def send_message(self, message):
            self.sent += 1
            if self.sent == 1:
                return _Response(calls=[_Call("get_review_state")])
            if not isinstance(message, str):
                self.parts = message
            return _Response(text="done")

    order = []

    class _Gate(agent.ToolGate):
        def acquire(self, name, args):
            order.append(f"acquire:{name}")

        def release(self, name):
            order.append(f"release:{name}")

    def boom(**kwargs):
        raise RuntimeError("the browser fell over")

    original = agent.FUNCTION_MAP["get_review_state"]
    agent.FUNCTION_MAP["get_review_state"] = boom
    chat = _Chat()
    try:
        _text, _tools, _art = agent.run_agent(chat, "go", gate=_Gate())
    finally:
        agent.FUNCTION_MAP["get_review_state"] = original

    assert order == ["acquire:get_review_state", "release:get_review_state"], order
    # the tool's failure still reached the model as a function response
    assert chat.parts, "no tool result was sent back to the model"
    assert "the browser fell over" in str(chat.parts[0])


# ---------------------------------------------------------------------------
# Artifacts survive a turn that dies
# ---------------------------------------------------------------------------


def test_a_turn_that_dies_midway_still_returns_its_screenshots():
    """A review that fails on round 4 of 6 has three good frames on disk. The
    user should get them with the error, not an error and nothing else."""

    class _Call:
        def __init__(self, name):
            self.name = name
            self.args = {}

    class _Response:
        def __init__(self, calls=(), text=None):
            self.function_calls = list(calls)
            self.text = text

    class _Chat:
        """Screenshot on round 1, then the model connection dies."""

        def __init__(self):
            self.sent = 0

        def send_message(self, message):
            self.sent += 1
            if self.sent == 1:
                return _Response(calls=[_Call("screenshot_game_move")])
            if self.sent == 2:
                raise RuntimeError("the socket went away")
            return _Response(text="never reached")

    with tempfile.TemporaryDirectory() as tmp:
        shot = Path(tmp) / "blunder_m12_ply24_x.png"
        shot.write_bytes(b"PNG")
        real = agent.FUNCTION_MAP["screenshot_game_move"]
        agent.FUNCTION_MAP["screenshot_game_move"] = lambda **kw: {
            "move": {"ply": 24, "screenshot": str(shot)}
        }
        try:
            session = agent.AgentSession.__new__(agent.AgentSession)
            session._client = object()
            session._note = ""
            session._max_turns = 5
            session.turns = 0
            session._chat = _Chat()
            result = session.ask("show me the blunder")
        finally:
            agent.FUNCTION_MAP["screenshot_game_move"] = real

    assert result.error is not None
    assert "socket went away" in result.text, result.text
    assert result.artifacts == [("screenshot_game_move", shot)], (
        "the screenshot written before the failure was thrown away"
    )


def test_a_progress_callback_that_raises_does_not_kill_the_turn():
    """`on_tool` runs outside the tool try/except, and the Discord one schedules
    a coroutine on the event loop -- which raises outright if the loop is
    closing. A glitched status line must not be reported as a model failure."""

    class _Call:
        def __init__(self):
            self.name = "get_review_state"
            self.args = {"game_url": "u"}

    class _Response:
        def __init__(self, text=None, calls=()):
            self.text = text
            self.function_calls = list(calls)

    class _Chat:
        def __init__(self):
            self.sent = 0

        def send_message(self, message):
            self.sent += 1
            if self.sent == 1:
                return _Response(calls=[_Call()])
            return _Response(text="all done")

    def grumpy(name, args):
        raise RuntimeError("Event loop is closed")

    original = agent.FUNCTION_MAP["get_review_state"]
    agent.FUNCTION_MAP["get_review_state"] = lambda **kw: {"ok": True}
    try:
        text, tools, _art = agent.run_agent(_Chat(), "go", on_tool=grumpy)
    finally:
        agent.FUNCTION_MAP["get_review_state"] = original

    assert text == "all done", text
    assert tools == ["get_review_state"]


def test_a_hallucinated_tool_is_not_announced_as_running():
    """`on_tool` fires after the FUNCTION_MAP lookup, so a front end never
    renders "running delete_everything" for a tool that does not exist."""
    seen: list[str] = []

    class _Call:
        def __init__(self):
            self.name = "delete_everything"
            self.args: dict = {}

    class _Response:
        def __init__(self, text=None, calls=()):
            self.text = text
            self.function_calls = list(calls)

    class _Chat:
        def __init__(self):
            self.sent = 0

        def send_message(self, message):
            self.sent += 1
            return (
                _Response(calls=[_Call()])
                if self.sent == 1
                else _Response(text="I cannot do that")
            )

    _text, tools, _art = agent.run_agent(
        _Chat(), "delete everything", on_tool=lambda n, a: seen.append(n)
    )
    assert seen == [], seen
    assert tools == ["delete_everything"], "it must still be recorded as requested"
    assert "delete_everything" not in agent.FUNCTION_MAP


# ---------------------------------------------------------------------------
# Tool errors say the right thing
# ---------------------------------------------------------------------------


def test_a_typeerror_inside_a_tool_is_not_reported_as_bad_arguments():
    """Both are TypeError. Catching TypeError first meant a crash deep inside
    `review_game_screenshots` was reported to the model as "Bad tool arguments"
    AND, since that branch has no exc_info, never logged -- so the real
    traceback was gone and the message pointed at the wrong thing."""

    class _Call:
        def __init__(self, name, args):
            self.name = name
            self.args = args

    class _Response:
        def __init__(self, text=None, calls=()):
            self.text = text
            self.function_calls = list(calls)

    def run(name, args):
        class _Chat:
            def __init__(self):
                self.sent = 0
                self.parts = []

            def send_message(self, message):
                self.sent += 1
                if self.sent == 1:
                    return _Response(calls=[_Call(name, args)])
                if not isinstance(message, str):
                    self.parts = message
                return _Response(text="done")

        chat = _Chat()
        agent.run_agent(chat, "go")
        return chat.parts[0].function_response.response if chat.parts else {}

    def crash(**kwargs):
        # a TypeError from the tool's own body, not from the call
        raise TypeError("unsupported operand str + int")

    original = agent.FUNCTION_MAP["get_review_state"]
    agent.FUNCTION_MAP["get_review_state"] = crash
    try:
        reported = str(run("get_review_state", {"game_url": "u"}))
    finally:
        agent.FUNCTION_MAP["get_review_state"] = original
    assert "Tool failed" in reported, reported
    assert "Bad tool arguments" not in reported, reported
    assert "unsupported operand" in reported, reported

    # a genuinely wrong argument is still called out as one. The stub needs a
    # real signature: `lambda **kw` accepts anything, so binding never fails.
    def only_game_url(game_url):
        return {"ok": True}

    original = agent.FUNCTION_MAP["get_review_state"]
    agent.FUNCTION_MAP["get_review_state"] = only_game_url
    try:
        reported = str(run("get_review_state", {"nope": 1}))
    finally:
        agent.FUNCTION_MAP["get_review_state"] = original
    assert "Bad tool arguments" in reported, reported


# ---------------------------------------------------------------------------
# Screenshots: batched by count AND by size
# ---------------------------------------------------------------------------


def test_screenshots_are_batched_by_size_as_well_as_count():
    """Ten board crops measured ~620KB each -- ~6MB against Discord's 8MB
    ceiling for a server with no boost, and REVIEW_VIEWPORT_W can push a frame
    well past that. One oversized file used to fail its own batch and every
    batch after it, with nothing catching the 413."""

    # the file count limit alone is not enough. Ten board crops measured ~620KB
    # each -- ~6MB against Discord's 8MB ceiling for a server with no boost, and
    # REVIEW_VIEWPORT_W can push a frame well past that. One oversized file used
    # to fail its own batch and every batch after it, with nothing catching the
    # 413. Not TemporaryDirectory: `discord.File` still holds the PNG open.
    tmp = tempfile.mkdtemp()
    try:
        root = Path(tmp)

        def png(path, size):
            path.write_bytes(b"\0" * size)
            return path

        small = [png(root / f"s{i}.png", 1000) for i in range(25)]
        batches = list(discord_bot._upload_batches([("t", p) for p in small]))
        assert [len(b) for b in batches] == [10, 10, 5], [len(b) for b in batches]
        assert [p for b in batches for _t, p in b] == small, "lost or reordered"

        original = discord_bot.MAX_UPLOAD_BYTES
        discord_bot.MAX_UPLOAD_BYTES = 2500
        try:
            batches = list(discord_bot._upload_batches([("t", p) for p in small]))
        finally:
            discord_bot.MAX_UPLOAD_BYTES = original
        # 2 files per message at 1000 bytes each, and the count limit still holds
        assert all(len(b) <= 10 for b in batches)
        assert all(sum(p.stat().st_size for _t, p in b) <= 2500 for b in batches)
        assert [p for b in batches for _t, p in b] == small, "a file was dropped"

        # one file too big on its own is skipped, not posted
        original = discord_bot.MAX_ATTACHMENT_BYTES
        discord_bot.MAX_ATTACHMENT_BYTES = 1500
        try:
            huge = png(root / "huge.png", 9000)
            batches = list(
                discord_bot._upload_batches([("t", p) for p in (small[0], huge)])
            )
        finally:
            discord_bot.MAX_ATTACHMENT_BYTES = original
        assert [p for b in batches for _t, p in b] == [small[0]], batches
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_a_frame_the_budget_refuses_is_told_to_the_user():
    """The other way to lose a picture, and the quieter one.

    A frame over the per-file limit is dropped before Discord is even asked, so
    there is no HTTP error, nothing in the channel, and nothing but a log line
    to say the blunder frame did not arrive. The user asked for pictures and
    gets the text with no explanation, which is the same complaint as a
    screenshot tool that failed.
    """
    # not TemporaryDirectory: `discord.File` still holds the PNG open
    tmp = tempfile.mkdtemp()
    original = discord_bot.MAX_ATTACHMENT_BYTES
    discord_bot.MAX_ATTACHMENT_BYTES = 1500
    try:
        root = Path(tmp)
        ok = root / "summary.png"
        huge = root / "viewport.png"
        ok.write_bytes(b"\0" * 1000)
        huge.write_bytes(b"\0" * 9000)

        async def scenario():
            bot = discord_bot.ChessBot()
            status = _FakeMessage("working", channel=_FakeChannel())
            await bot._attach(
                status,
                [("take_screenshot", ok), ("take_screenshot", huge)],
            )
            return status

        status = asyncio.run(scenario())
        sent = [c for c, _kw in status.channel.sent if c]
        assert any("too large" in c for c in sent), sent
        # and the one that fitted was still posted
        attached = [c for c, kw in status.channel.sent if kw.get("files")]
        assert len(attached) == 1, attached
    finally:
        discord_bot.MAX_ATTACHMENT_BYTES = original
        shutil.rmtree(tmp, ignore_errors=True)


def test_a_failed_upload_does_not_lose_the_answer():
    """`_post` used to await an unguarded `channel.send` per chunk, so one rate
    limit threw out of `_post`, out of `_answer`, and into discord.py's default
    handler -- and the user lost the rest of what they were told, having been
    told nothing about why."""

    async def scenario():
        bot = discord_bot.ChessBot()
        # small chunks, so the answer really is more than one message
        original = discord_bot.MAX_MESSAGE_CHARS
        discord_bot.MAX_MESSAGE_CHARS = 40
        try:
            status = _FakeMessage("working", channel=_FakeChannel(fail_from=1))
            await bot._post(
                status, agent.AgentResult(text="\n\n".join(["para " * 20] * 4))
            )
        finally:
            discord_bot.MAX_MESSAGE_CHARS = original
        return status

    status = asyncio.run(scenario())  # must not raise
    assert status.edits, "the first chunk never reached the channel"
    assert status.channel.sent, "no follow-up chunk was even attempted"
    assert len(status.channel.sent) == 1, (
        "the loop kept sending after Discord refused one message"
    )
    for _content, allowed in status.every_write():
        assert allowed is not None and allowed.to_dict() == {"parse": []}


# ---------------------------------------------------------------------------
# Path stripping, everywhere a path can be written
# ---------------------------------------------------------------------------


def test_path_stripping_covers_every_shape_a_path_is_written_in():
    """The original only matched a Windows drive letter, so `screenshots/x.png`
    and `/home/.../x.png` -- the same noise, different font -- went straight
    into the channel. Still narrow: a sentence that mentions a path is an
    answer, not noise."""

    text = "\n".join(
        [
            "I found one blunder.",
            r"`D:\Python\Ai\chess-agent\screenshots\blunder_m12_ply24_x.png`",
            "screenshots/summary_20260101_120000.png",
            "/home/ubuntu/chess-agent/screenshots/summary_20260101_120000.png",
            "- screenshots/blunder_m14_ply27_20260101.png",
            "**summary_20260101_120000.png**",
            "The image above shows the board after 12...e4.",
            "It is saved at D:\\photos\\board.png if you want it.",
            "You can also see C:/Users/me/Pictures/x.jpg in the other tab.",
        ]
    )
    kept = discord_bot.strip_path_lines(text)
    assert "I found one blunder." in kept
    assert "The image above shows the board after 12...e4." in kept
    assert "It is saved at" in kept
    assert "Pictures/x.jpg in the other tab" in kept, "a sentence was eaten"
    for gone in (
        "blunder_m12_ply24",
        "screenshots/summary_20260101",
        "/home/ubuntu",
        "blunder_m14_ply27",
        "summary_20260101_120000.png**",
    ):
        assert gone not in kept, f"{gone!r} should have been stripped"


def test_chunking_does_not_cut_an_emoji_in_half():
    """`line[:limit]` can land between a base character and its combining mark
    or zero-width joiner, and Discord then renders a replacement box in the
    middle of a sentence."""

    # family emoji, joined by ZWJ, with a skin-tone modifier
    text = "\U0001f468\u200d\U0001f469\u200d\U0001f467" * 900
    for chunk in discord_bot.chunk_message(text, limit=10):
        assert len(chunk) <= 10
        assert not chunk.endswith("\u200d"), repr(chunk)
        assert not chunk[0].startswith("\ufe0f")

    text2 = "e\u0301" * 4000  # e + combining acute
    for chunk in discord_bot.chunk_message(text2, limit=7):
        assert len(chunk) <= 7
        assert chunk[0] != "\u0301", repr(chunk)


# ---------------------------------------------------------------------------
# Bounded state
# ---------------------------------------------------------------------------


def test_conversations_are_bounded_and_idle_ones_dropped():
    """A session is a live Gemini chat holding up to AGENT_MAX_TURNS turns, and
    the key is per (guild, channel, user) -- so on a busy server the map grew
    for ever. Idle first, then oldest, so a cap never costs somebody their
    place mid-conversation."""

    bot = discord_bot.ChessBot()
    bot._client = object()
    cap = discord_bot.MAX_SESSIONS
    try:
        discord_bot.MAX_SESSIONS = 5
        for i in range(20):
            bot.session_for((0, 100 + i, 200 + i))
        assert len(bot._sessions) <= 6, len(bot._sessions)
        assert len(bot._session_locks) == len(bot._sessions)
        assert len(bot._session_seen) == len(bot._sessions)

        # touching an old one keeps it alive past the cap
        oldest = next(iter(bot._sessions))
        bot.session_for(oldest)
        assert oldest in bot._sessions
    finally:
        discord_bot.MAX_SESSIONS = cap

    # an idle conversation is dropped, lock and all
    bot2 = discord_bot.ChessBot()
    bot2._client = object()
    bot2.session_for((0, 1, 2))
    bot2.session_for((0, 3, 4))
    assert len(bot2._sessions) == 2
    # age both entries rather than shortening the window, which would also
    # prune the session being created
    bot2._session_seen = {k: 0.0 for k in bot2._session_seen}
    bot2._prune_sessions()
    assert not bot2._sessions, "an idle conversation was kept"
    assert not bot2._session_locks, "a lock leaked"
    assert not bot2._session_seen, "a timestamp leaked"

    assert bot2.forget_session((0, 3, 4)) is False
    bot2.session_for((0, 5, 6))
    assert bot2.forget_session((0, 5, 6)) is True
    assert bot2.forget_session((0, 5, 6)) is False


def test_one_turn_per_user_per_cooldown():
    """A review is a real browser against the owner's chess.com session, so a
    user who fires off fifty mentions spends the owner's tokens, occupies the
    browser for an hour, and walks that account towards a Cloudflare
    challenge."""
    bot = discord_bot.ChessBot()
    original = discord_bot.USER_COOLDOWN_SECONDS
    try:
        discord_bot.USER_COOLDOWN_SECONDS = 60
        assert bot._may_start_turn(1) is True
        bot._last_turn[1] = time.monotonic()
        assert bot._may_start_turn(1) is False, "a second turn was let through"
        assert bot._may_start_turn(2) is True, "one user blocked another"

        discord_bot.USER_COOLDOWN_SECONDS = 0
        assert bot._may_start_turn(1) is True, "0 must disable the check"
    finally:
        discord_bot.USER_COOLDOWN_SECONDS = original


def test_status_lines_are_coalesced_and_never_go_backwards():
    """Discord allows about 5 edits per 5s per channel and a six-round turn
    outruns that. Posting every line as it arrived also reordered them -- the
    status could end on `get_recent_games - 3s` after `get_moves - 12s` --
    because each waiting coroutine woke in an arbitrary order.

    A progress line is a snapshot of "what is happening now", so only the
    newest one is worth writing: that is rate limiting and correct ordering in
    the same move.
    """

    class _Timed(_FakeMessage):
        def __init__(self):
            super().__init__()
            self.at = []

        async def edit(self, *, content=None, allowed_mentions=None):
            self.at.append(asyncio.get_running_loop().time())
            await super().edit(content=content, allowed_mentions=allowed_mentions)

    async def scenario():
        bot = discord_bot.ChessBot()
        loop = asyncio.get_running_loop()
        status = _Timed()
        gate = discord_bot._BrowserGate(bot, loop, status, loop.time())
        # concurrent tasks on one loop: what a burst of tool calls looks like
        await asyncio.gather(*(gate._queue(f"_running step {i}_") for i in range(5)))
        return status

    status = asyncio.run(scenario())
    assert status.edits, "nothing was posted at all"
    lines = [c for c, _a in status.edits]
    # the first line goes up at once, the burst collapses into the newest
    assert lines[0] == "_running step 0_", lines
    assert lines[-1] == "_running step 4_", (
        f"the status line did not end on the newest update: {lines}"
    )
    # and nothing ever went backwards
    order = [int(line.split("step ")[1].rstrip("_")) for line in lines]
    assert order == sorted(order), f"progress lines arrived out of order: {order}"
    assert len(lines) < 5, f"no coalescing happened: {lines}"
    for a, b in zip(status.at, status.at[1:]):
        assert b - a >= discord_bot.STATUS_EDIT_MIN_INTERVAL * 0.9, (
            f"two edits {b - a:.2f}s apart; the limiter was not applied"
        )
    for content, allowed in status.edits:
        assert allowed is not None and allowed.to_dict() == {"parse": []}, content


def test_status_lines_are_spaced_out_over_time():
    """The rate limit still has to hold when the lines are seconds apart."""

    class _Timed(_FakeMessage):
        def __init__(self):
            super().__init__()
            self.at = []

        async def edit(self, *, content=None, allowed_mentions=None):
            self.at.append(asyncio.get_running_loop().time())
            await super().edit(content=content, allowed_mentions=allowed_mentions)

    async def scenario():
        bot = discord_bot.ChessBot()
        loop = asyncio.get_running_loop()
        status = _Timed()
        gate = discord_bot._BrowserGate(bot, loop, status, loop.time())
        for i in range(3):
            await gate._queue(f"_running step {i}_")
            await asyncio.sleep(0)  # let the edit land
        return status

    status = asyncio.run(scenario())
    assert len(status.at) == 3, f"only {len(status.at)} of 3 lines were posted"
    for a, b in zip(status.at, status.at[1:]):
        assert b - a >= discord_bot.STATUS_EDIT_MIN_INTERVAL * 0.9, (
            f"two edits {b - a:.2f}s apart; the limiter was not applied"
        )
    assert [c for c, _a in status.edits] == [f"_running step {i}_" for i in range(3)]


# ---------------------------------------------------------------------------
# Configuration that used to be silently discarded
# ---------------------------------------------------------------------------


def test_logging_configuration_is_not_silently_discarded():
    """`import agent` had already installed a root handler, and
    `logging.basicConfig` is a documented no-op once one exists -- so the bot's
    level and format were thrown away without a word."""

    import logging

    root = logging.getLogger()
    before = [h.formatter._fmt if h.formatter else None for h in root.handlers]
    discord_bot.configure_logging("info", fmt="TEST-FORMAT %(message)s")
    try:
        after = [h.formatter._fmt if h.formatter else None for h in root.handlers]
        assert all(f == "TEST-FORMAT %(message)s" for f in after), after
        assert root.level == logging.INFO
    finally:
        discord_bot.configure_logging("warning")
        for handler, fmt in zip(root.handlers, before):
            handler.setFormatter(
                logging.Formatter(fmt) if fmt else logging.Formatter("%(message)s")
            )
    # and the module must not be calling basicConfig behind configure_logging's back
    source = inspect.getsource(discord_bot)
    assert "logging.basicConfig(" not in source, (
        "a basicConfig call in this module is a no-op and hides the real config"
    )
    assert "configure_logging(" in source


def test_the_access_rule_docstring_matches_the_rule():
    """`permitted()`'s docstring used to claim that DMs are let through when the
    list is empty. The code does not, the code is right, and a docstring that
    describes a more permissive rule than the code is how a real hole gets
    introduced later."""
    doc = inspect.getdoc(discord_bot.permitted).lower()
    assert "nobody" in doc, doc
    assert "dm" in doc
    # the rule itself: an empty list refuses everyone
    original, everyone = discord_bot.ALLOWED_USERS, discord_bot.OPEN_TO_EVERYONE
    try:
        discord_bot.ALLOWED_USERS, discord_bot.OPEN_TO_EVERYONE = set(), False
        assert not discord_bot.permitted(_User(1))
    finally:
        discord_bot.ALLOWED_USERS = original
        discord_bot.OPEN_TO_EVERYONE = everyone


def test_a_missing_api_key_is_a_message_not_a_dead_process():
    """`get_api_key` calls `sys.exit`, and the bot built its client from inside
    a coroutine on the event loop, where `SystemExit` is a BaseException that
    no `except Exception` catches. A key that went bad between startup and the
    first message would have taken the process down."""

    def exits():
        try:
            agent.get_api_key()
        except SystemExit:
            return True
        return False

    original = os.environ.pop("GEMINI_API_KEY", None)
    try:
        os.environ["GEMINI_API_KEY"] = "your_key_here"
        try:
            agent.require_api_key()
            raise AssertionError("a placeholder was accepted")
        except agent.MissingSetting as exc:
            assert exc.name == "GEMINI_API_KEY"
            assert "placeholder" in exc.detail
        os.environ["GEMINI_API_KEY"] = "AIza-real-looking-key"
        assert agent.require_api_key() == "AIza-real-looking-key"
    finally:
        if original is None:
            os.environ.pop("GEMINI_API_KEY", None)
        else:
            os.environ["GEMINI_API_KEY"] = original

    # the CLI wrapper still exits, because a program should stop
    try:
        os.environ["GEMINI_API_KEY"] = "your_key_here"
        assert exits() is True
    finally:
        if original is None:
            os.environ.pop("GEMINI_API_KEY", None)
        else:
            os.environ["GEMINI_API_KEY"] = original


def test_a_mistyped_command_gets_an_answer():
    """With no `on_command_error`, discord.py logs the traceback and the user
    sees nothing, so `!statuss` read as a broken bot rather than a typo."""

    assert hasattr(discord_bot.ChessBot, "on_command_error")

    class _Ctx:
        invoked_with = "statuss"
        command = None

        def __init__(self):
            self.replies = []

        async def reply(self, content=None, **kwargs):
            self.replies.append((content, kwargs))

    async def scenario():
        bot = discord_bot.ChessBot()
        ctx = _Ctx()
        await bot.on_command_error(ctx, commands.UserInputError("no such command"))
        return ctx.replies

    replies = asyncio.run(scenario())
    assert replies, "the user was told nothing"
    assert "!statuss" in replies[0][0]
    assert replies[0][1]["allowed_mentions"].to_dict() == {"parse": []}


def test_the_model_error_hint_names_the_fallback():
    """`main.py` has always said "try GEMINI_MODEL=..." when a turn fails. The
    Discord path returned a bare 404 to the least technical person who can
    reach it, and a retired model is the most common cause."""

    for text in ("404 NOT_FOUND", "model not found", "no longer available"):
        hint = agent.model_error_hint(text)
        assert hint, f"no hint for {text!r}"
        assert agent.MODEL in hint, hint
        assert agent.MODEL_FALLBACK_HINT in hint, hint
        assert "restart" in hint.lower(), hint
    for text in ("something else went wrong", "", "the browser fell over"):
        assert agent.model_error_hint(text) == "", text


def test_the_package_ships_the_entry_points():
    """`packages = ["tools"]` built a wheel with no agent.py, no discord_bot.py
    and no project_env.py, and discord.py was in requirements.txt but not in the
    dependencies -- so `pip install .` produced something that could not run."""

    try:
        import tomllib
    except ImportError:  # pragma: no cover - 3.11+
        return
    root = ROOT
    config = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    assert "discord.py>=2.4" in config["project"]["dependencies"]
    setuptools_cfg = config["tool"]["setuptools"]
    modules = set(setuptools_cfg.get("py-modules", []))
    assert {"agent", "discord_bot", "main", "project_env"} <= modules, modules
    for name in modules:
        assert (root / f"{name}.py").exists(), name
    assert "tools" in setuptools_cfg["packages"]


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
