"""Discord front end for the chess agent.

Same agent as the terminal (`agent.py`), different manners. The things that had
to be designed rather than copied:

**Nothing blocks the event loop.** A review drives a real Chromium and takes
30 seconds to two minutes. Calling that inline would freeze the bot for every
channel it is in, so the agent runs in a worker thread (`asyncio.to_thread`).
This is only safe because the tools are *synchronous*: they call `asyncio.run`
internally, which is fine in a thread with no running loop and would be a
`RuntimeError` on the loop itself.

**One browser at a time -- and only the browser.** The tools share one
`storage_state.json` and one Chromium profile, so two concurrent reviews would
fight over it. But only five of the seven tools touch either; the other two are
plain HTTPS calls to chess.com, and a two-second "how was my last game?" must
not wait out a two-minute review. So the lock is taken per tool call, through
the `ToolGate` hook in `agent.py`, and only for `agent.BROWSER_TOOLS`.

**A hung turn cannot take the bot down with it.** A tool that never returns
would otherwise hold the browser lock for ever and every later request would
queue behind it silently. The turn is bounded, the user is told, and a
watchdog frees the lock afterwards so the queue drains.

**Screenshots are attached, not described.** The model cannot see images, so it
writes prose about them -- correct for a terminal, useless on Discord. The
files are collected at the tool boundary in `agent.py` and attached to the
reply, and the model is told not to print paths.

**The progress line is written for the person reading it.** One message is
edited as the turn proceeds, and it says what is happening in words -- "reading
the move list", "taking a screenshot" -- never the tool name and never its
arguments. `DISCORD_SHOW_TOOLS=1` brings the technical line back for the owner
watching the bot, which is the only audience it was ever for.

**A picture that was not produced is announced.** A model handed `{"error":
...}` will still write "here is the screenshot", and the user then finds nothing
attached. If a screenshot tool ran and the turn carries no file, `_post` says so
itself, because that is a fact about the turn rather than about the model's
willingness to admit a failure.

**Nothing this bot posts can ping anyone.** Replies carry
`AllowedMentions.none()`. `discord.py` defaults to parsing everyone, users and
roles, and the bot posts text a language model wrote, so without this a reply
containing `@everyone` would actually ping it -- in somebody else's server, with
a permission set that deliberately does not include `MENTION_EVERYONE`.

Setup is in the README: create an application, enable the MESSAGE CONTENT
INTENT, put the token and your user ID in `.env`, run this, and open the invite
URL it prints.
"""

import asyncio
import logging
import os
import re
import sys
import time
import unicodedata
from pathlib import Path

import discord
from discord.ext import commands
from google import genai

import agent
import project_env
from tools import configure_logging
from tools.common import session_state_path
from tools.session import get_session

project_env.load_env()
# `configure_logging`, not `logging.basicConfig`: importing `agent` above has
# already run it, and `basicConfig` is a documented no-op once the root logger
# has a handler. The timestamps are the point -- this process runs for days.
configure_logging(fmt="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
_LOG = logging.getLogger("discord_bot")

# Discord's hard limit. Chunk to a bit under it so a trailing line never tips
# the message over and throws.
MAX_MESSAGE_CHARS = 1900
# Discord's hard limit on files per message.
MAX_ATTACHMENTS = 10
# How often to refresh the typing indicator. Discord's own typing state lasts
# 10s, so go under that.
TYPING_REFRESH_SECONDS = 8

# Discord allows about 5 message edits per 5s per channel. Six tool rounds plus
# a final answer can outrun that, and a throttled edit is indistinguishable from
# a failed one from the user's side, so status lines are spaced out instead. The
# FINAL edit in `_post` is never throttled -- by then nothing else is competing.
STATUS_EDIT_MIN_INTERVAL = 1.0

# What the progress line says while a tool runs.
#
# NOT the tool name and NOT its arguments. The line used to be
# `_running `screenshot_game_move` gameurl=https://www.chess.com/game/live/...`_
# -- a log line, edited into the one message the user is reading, in the middle
# of the answer they asked for. It reads as the bot talking to itself in code,
# it leaks a half-URL into the channel, and it says nothing the user wanted to
# know. What they want to know is whether it is working, and on what.
#
# Set DISCORD_SHOW_TOOLS=1 to get the technical line back (useful when the owner
# is watching the bot and needs to see which step is slow).
SHOW_TOOL_NAMES = os.getenv("DISCORD_SHOW_TOOLS", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

# One plain-English phrase per tool. A tool missing here falls back to
# DEFAULT_PROGRESS rather than leaking its name, which is the whole point.
TOOL_PROGRESS = {
    "get_recent_games": "looking up recent games",
    "get_player_profile": "looking up that player",
    "review_game_screenshots": "reading the game review",
    "get_review_state": "reading the review summary",
    "get_moves": "reading the move list",
    "get_coach_commentary": "reading the coach's notes",
    "screenshot_game_move": "taking the picture of that move",
    "open_chess_page": "opening the page",
    "where_am_i": "checking what is on screen",
    "move_board": "moving the board",
    "goto_move_in_game": "going to that move",
    "take_screenshot": "taking a screenshot",
    "close_browser_session": "closing the browser",
}
DEFAULT_PROGRESS = "working on it"

# How long one turn may take, browser queue included. Long enough for a real
# review (the Cloudflare wait alone is 90s), short enough that a wedged tool is
# reported rather than left to hang. Set to 0 to disable the bound.
TURN_TIMEOUT_SECONDS = float(os.getenv("DISCORD_TURN_TIMEOUT", "600"))

# After a turn is abandoned, how long the browser lock is left alone before a
# watchdog takes it back. The worker thread cannot be killed, so the only way
# out of a wedged tool is to stop waiting for it -- and then the lock has to be
# recovered, or every later request is stuck behind a thread that will never
# return. Generous, because reclaiming it early means two Chromiums at once.
BROWSER_LOCK_GRACE_SECONDS = float(os.getenv("DISCORD_BROWSER_GRACE", "120"))

# One turn per user per this many seconds. The per-session lock already makes
# a second question wait; what it does not stop is a user queueing fifty of
# them, each spending tokens and driving a real browser against the owner's
# chess.com session -- which is also a good way to get that account
# Cloudflare-flagged. 0 disables it.
USER_COOLDOWN_SECONDS = float(os.getenv("DISCORD_USER_COOLDOWN", "5"))

# A session per (guild, channel, user) is the right key, and also an unbounded
# map: every person who ever speaks in every channel leaves a live Gemini chat
# holding up to AGENT_MAX_TURNS turns of history, and `get_moves` results are a
# few hundred plies of dicts. Longest-idle-out, then oldest-out, at this cap.
SESSION_IDLE_SECONDS = float(os.getenv("DISCORD_SESSION_IDLE", "3600")) * 60
MAX_SESSIONS = int(os.getenv("DISCORD_MAX_SESSIONS", "200"))

# Uploads are budgeted by BYTES, not by file count. Ten 1400x820 board crops
# measured ~620KB each, so a full batch is ~6MB against Discord's 8MB ceiling
# for a server with no boost -- close enough that a wider REVIEW_VIEWPORT tips
# it over and the whole batch is refused with a 413. Splitting by size means
# the answer and the frames that do fit still arrive.
MAX_UPLOAD_BYTES = int(os.getenv("DISCORD_MAX_UPLOAD_BYTES", str(7 * 1024 * 1024)))
# A file bigger than this on its own is skipped rather than posted, so one
# runaway screenshot cannot fail its own batch and every batch after it.
MAX_ATTACHMENT_BYTES = int(
    os.getenv("DISCORD_MAX_ATTACHMENT_BYTES", str(6 * 1024 * 1024))
)

# The bot needs to read message text to know it was mentioned, and that is a
# PRIVILEGED intent -- off by default, and the single most common reason a bot
# like this appears to be online but replies to nothing.
INTENTS = discord.Intents.default()
INTENTS.message_content = True

# Nothing this bot posts may ping. Set on the Bot so every send and edit
# inherits it, and passed explicitly at the call sites as well so the intent is
# visible where the text is written.
NO_MENTIONS = discord.AllowedMentions.none()

# The tools whose job is to produce a picture. If one of these ran and the turn
# carries no artifact, the user has been told nothing about why the picture they
# asked for did not arrive -- and the model's own text is the only thing they
# get, and a model that was handed {"error": ...} will happily still write
# "here is the screenshot".
SHOT_TOOLS = frozenset(
    {
        "take_screenshot",
        "screenshot_game_move",
        "review_game_screenshots",
    }
)
MISSING_SHOT_NOTE = (
    "_(No picture came through: the screenshot step did not produce one. "
    "Everything above is what I could read from chess.com.)_"
)

# The one failure the user can do something about, and the one that happens
# most: chess.com's saved session expires. Every review tool says the same
# thing about it, so it is matched on rather than on the tool name.
_EXPIRED_SESSION = ("not logged in", "login_chess")
_EXPIRED_NOTE = (
    "_(No picture: my chess.com sign-in has expired, or was never saved. It "
    "has to be done again once -- the owner has been told, and after that "
    "this works.)_"
)


def _plain_reason(message: str) -> str:
    """A tool's error text, minus the instructions meant for the owner.

    The tools are careful to name a fix ("Run: python dump_pages.py <link>"), and
    that is right for the terminal. On Discord the reader cannot run anything on
    the owner's machine, so the command is dropped and the reason kept -- the
    part that tells them whether to try again.
    """
    text = " ".join((message or "").split())
    for marker in ("Run: python", "run 'python", "run `python"):
        head, sep, _tail = text.partition(marker)
        if sep:
            text = head.strip().rstrip(":.").strip()
    return text


# Above this age the sign-in is worth a warning. chess.com expires these, and an
# expired one fails every review tool with a message about logging in -- which is
# a confusing thing to be told by a bot, at the moment you asked for a picture.
SIGN_IN_WARN_DAYS = 30


def chess_com_sign_in_age() -> str:
    """How old the saved chess.com session file is, or "" if there is none.

    Deliberately the file's AGE and not a verdict. Nothing here can know whether
    chess.com still accepts the cookie without a page load, and claiming
    "expired" from a timestamp would be a plausible wrong value -- the exact
    thing this project keeps designing against. So it states the age and lets
    the reader connect it.
    """
    try:
        path = session_state_path()
        age_days = (time.time() - path.stat().st_mtime) / 86400
    except OSError:
        return ""
    if age_days < SIGN_IN_WARN_DAYS:
        return f"{age_days:.0f} days old"
    return (
        f"**{age_days:.0f} days old** -- chess.com expires these, so reviews "
        f"may fail until the owner re-runs `python login_chess.py`"
    )


def picture_failure_note(errors: list) -> str:
    """Why the picture did not arrive, in one line, from what the tools said.

    The model is handed the same failure and writes prose about it, but it is
    under no obligation to repeat the reason -- and when it does, it is a
    paragraph, not something the user can act on. The tool's own words are the
    honest source, and the front end is the only place that still has them.
    """
    for _tool, message in errors or []:
        low = (message or "").lower()
        if any(word in low for word in _EXPIRED_SESSION):
            return _EXPIRED_NOTE
    reasons = []
    for _tool, message in errors or []:
        reason = _plain_reason(message)
        if reason and reason not in reasons:
            reasons.append(reason)
    if not reasons:
        return MISSING_SHOT_NOTE
    return (
        "_(No picture came through: "
        + "; ".join(reasons[:2])
        + ". Send me the game link and I will try again -- the owner has been "
        "told.)_"
    )


# Minimal permission set for the invite link. ViewChannel 1024, SendMessages
# 2048, EmbedLinks 16384, AttachFiles 32768, ReadMessageHistory 65536.
BOT_PERMISSIONS = 1024 + 2048 + 16384 + 32768 + 65536

USAGE_EXAMPLES = """\
**Ask me anything, by mention or DM.** No prefix needed.

  `@ChessAgent how was my last game?`
  `@ChessAgent review https://www.chess.com/game/live/184571437220`
  `@ChessAgent what did the coach say about move 12 as black?`
  `@ChessAgent show me the last 5 games of hikaru`
  `@ChessAgent how accurate was I?`

**I can drive the browser too**, so you can walk through a game a move at a
time -- the page stays open between messages:

  `@ChessAgent open the review of https://www.chess.com/game/live/184571437220`
  `@ChessAgent go forward 3 moves`
  `@ChessAgent what does the engine think of this position?`
  `@ChessAgent screenshot that`
  `@ChessAgent step back two`
  `@ChessAgent close the browser`

Reviews and page loads launch a real browser and take 30s-2min. I edit one
message as I go, and attach the screenshots -- you never need to open a file
path.
"""

# Why the command list is generated rather than written down: a help text that
# lists commands by hand drifts the moment one is renamed, and then tells users
# to type something that does not exist. `help_text` reads the real registry.
HELP_FOOTER = "Prefix commands work in any channel, no mention needed."


def help_text(commands_by_name: dict) -> str:
    """Build `!help` from the commands that are actually registered."""
    lines = [USAGE_EXAMPLES, "", "**Commands**"]
    for name in sorted(commands_by_name):
        summary = (commands_by_name[name] or "").strip()
        lines.append(f"  `!{name}` — {summary}" if summary else f"  `!{name}`")
    lines += ["", HELP_FOOTER]
    return "\n".join(lines)


_MENTION_STRIP = re.compile(r"<@!?\d+>")
# A line that is NOTHING BUT a screenshot path, in any of the shapes a model
# actually emits. The Discord platform note already tells it not to print paths,
# and this catches the leftovers.
#
# Structure: a path, then an image extension, then end of line. Nothing else
# may appear on the line, which is what keeps "It is saved at D:\photos\x.png if
# you want it." -- which is the model's answer, not noise -- intact. Leading
# markdown decoration and a leading list bullet are peeled off first, since
# both are also noise rather than content.
_PATHISH = re.compile(
    r"""^\s*[`*_]*
        (?:
            [A-Za-z]:[\\/][^\n]*?      # C:\...\x.png  /  C:/.../x.png
          | /[^\n]*?                   # /home/.../x.png
          | [^\s`*/<>|][^\n]*?         # screenshots/x.png, or just x.png
        )
        \.(?:png|jpe?g|webp|gif)
        [`*_]*\s*$""",
    re.IGNORECASE | re.VERBOSE,
)
# `- `, `* `, `1. ` in front of a path.
_BULLET = re.compile(r"^\s*(?:[-*+\u2022]|\d+[.)])\s+")


def _load_allowed_users() -> set[int]:
    """The user IDs permitted to make the bot drive their browser.

    Required, and refusing to start without it is deliberate. The tools log into
    chess.com as one person and launch a real browser on one machine; "everyone
    in the server" is a much worse default than "you have to configure something
    first". Set DISCORD_ALLOWED_USER_IDS=all to open it up on purpose.
    """
    raw = os.getenv("DISCORD_ALLOWED_USER_IDS", "").strip()
    if not raw:
        return set()
    if raw.lower() in {"all", "*"}:
        return set()  # sentinel handled by permitted()
    ids = set()
    for part in raw.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if not part.isdigit():
            _LOG.warning(
                "DISCORD_ALLOWED_USER_IDS: %r is not a user ID, ignoring", part
            )
            continue
        ids.add(int(part))
    return ids


ALLOWED_USERS = _load_allowed_users()
OPEN_TO_EVERYONE = os.getenv("DISCORD_ALLOWED_USER_IDS", "").strip().lower() in {
    "all",
    "*",
}


def permitted(user: discord.abc.User) -> bool:
    """May this user make the bot run a tool?

    No, unless they are on the list. An empty list means NOBODY, in a DM just as
    much as in a channel: the tools log into chess.com as the owner and drive a
    real browser on the owner's machine, and a DM is not a smaller risk than a
    channel, it is the same risk with fewer witnesses. `!status` and the startup
    banner say what is configured, so an owner who has not set an ID yet finds
    out at once.
    """
    if OPEN_TO_EVERYONE:
        return True
    if not ALLOWED_USERS:
        return False
    return user.id in ALLOWED_USERS


def describe_artifact(path: Path) -> str:
    """A human caption for a screenshot file, from its name.

    The review pipeline already names frames after what they show --
    `brilliant_m14_ply27_<stamp>.png`, `summary_<stamp>.png` -- so the caption
    is a formatting job rather than a second source of truth. Anything
    unrecognised is described as a screenshot rather than guessed at.
    """
    stem = path.stem
    # `move_m12_ply24_...` is the generic single-move shot, and needs no kind
    # prefix; `brilliant_m14_ply27_...` names the classification.
    generic = re.match(r"move_m(\d+)_ply(\d+)", stem)
    if generic:
        return f"the position after move {generic.group(1)} (ply {generic.group(2)})"
    kind = re.match(r"([a-z_]+)_m(\d+)_ply(\d+)", stem)
    if kind:
        return (
            f"{kind.group(1).replace('_', ' ')} -- move {kind.group(2)} "
            f"(ply {kind.group(3)})"
        )
    if stem.startswith("summary"):
        return "review summary -- accuracy, full tally, coach commentary"
    # `shot_<region>[_<label>]_<stamp>` -- take_screenshot. These are whatever
    # the user asked to see, on a puzzle page or a lesson, so they must not be
    # captioned as a review screenshot.
    shot = re.match(r"shot_([a-z]+)(?:_(.+?))?_\d{8}_\d{6}$", stem)
    if shot:
        region = shot.group(1)
        label = (shot.group(2) or "").replace("_", " ").strip()
        where = {
            "auto": "the board and the review panel",
            "board": "the board",
            "sidebar": "the side panel",
            "viewport": "the whole page",
        }.get(region, "the page")
        return f"{where}{' -- ' + label if label else ''}"
    return "screenshot"


# Codepoints that must never be the FIRST character of a chunk: a combining
# mark belongs to the letter before it, a ZWJ belongs to the emoji before it,
# a variation selector and a skin-tone modifier belong to the pictograph
# before them. `unicodedata.combining` catches the accents but reports 0 for the
# other three, so they are listed.
_ORPHANS = frozenset({"\u200d", "\ufe0f", "\ufe0e"} | set(range(0x1F3FB, 0x1F400)))


def _orphaned(text: str, index: int) -> bool:
    """Would `text[index]` be separated from the character it modifies?"""
    if index >= len(text):
        return False
    char = text[index]
    return char in _ORPHANS or bool(unicodedata.combining(char))


def _safe_cut(text: str, limit: int) -> int:
    """The largest cut point <= limit that leaves nothing orphaned.

    `text[:limit]` is fine for prose and wrong for anything else: land between
    a base character and its combining mark, or between an emoji and its
    zero-width joiner, and Discord renders a replacement box in the middle of a
    sentence.

    Only the character that would START the remainder is checked. A combining
    mark at the END of a chunk is correct -- that is where it belongs -- and
    treating it as a reason to back off is what produces chunks that begin with
    a bare accent.
    """
    cut = min(limit, len(text))
    while cut > 0 and _orphaned(text, cut):
        cut -= 1
    return max(1, cut)


def chunk_message(text: str, limit: int | None = None) -> list[str]:
    """Split a long answer into Discord-sized pieces.

    Split on blank lines first so a table of games stays one item per line, then
    on newlines, and only hard-split a single line that is longer than the limit
    on its own. Not a naive `text[:2000]`: that slices a table in half and drops
    the second half's meaning, which is how a bot ends up listing six of nine
    games.

    `limit` defaults to `MAX_MESSAGE_CHARS` at CALL time, not at def time, so
    the constant stays tunable and the multi-chunk path is testable without
    building a 2000-character string.
    """
    limit = MAX_MESSAGE_CHARS if limit is None else limit
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    current = ""
    for block in re.split(r"\n\s*\n", text):
        candidate = f"{current}\n\n{block}" if current else block
        if len(candidate) <= limit:
            current = candidate
            continue
        if current:
            chunks.append(current)
            current = ""
        if len(block) <= limit:
            current = block
            continue
        # one oversized block: fall back to line breaks, then hard cuts
        for line in block.split("\n"):
            candidate = f"{current}\n{line}" if current else line
            if len(candidate) <= limit:
                current = candidate
                continue
            if current:
                chunks.append(current)
                current = ""
            while len(line) > limit:
                cut = _safe_cut(line, limit)
                chunks.append(line[:cut])
                line = line[cut:]
            current = line
    if current:
        chunks.append(current)
    return [c for c in chunks if c.strip()]


def strip_path_lines(text: str) -> str:
    """Drop lines that are nothing but a screenshot path.

    The Discord platform note already tells the model not to print paths, and
    when it does this catches the leftovers. Deliberately narrow: a line must be
    *only* a path, so a sentence that happens to mention one survives.
    """
    kept = []
    for line in (text or "").splitlines():
        if _PATHISH.match(_BULLET.sub("", line)):
            continue
        kept.append(line)
    return "\n".join(kept).strip()


class _Ctx:
    """The attributes the access check needs, built from a plain message.

    Real `commands.Context` objects arrive from the command handlers; this is
    the same duck type constructed from a `discord.Message`, so ONE access check
    serves both entry points instead of two copies of the rule. Two copies is
    how the gate and the commands end up disagreeing about who may use the bot.
    """

    def __init__(self, message: discord.Message) -> None:
        self.author = message.author
        self.channel = message.channel
        self.guild = message.guild
        self._message = message

    async def reply(self, content: str, **kwargs) -> None:
        await self._message.reply(content, **kwargs)


def _spawn(coro, loop: asyncio.AbstractEventLoop, where: str):
    """Schedule a coroutine from a worker thread. Never raises, never silent.

    `run_coroutine_threadsafe` raises `RuntimeError` outright if the loop is
    closed or shutting down, and the future it returns holds any exception the
    coroutine raised -- which, ignored, becomes an "exception was never
    retrieved" warning on stderr and no log line of our own. Both failure modes
    are handled here so a status-line problem can never be mistaken for a
    broken turn.
    """
    try:
        future = asyncio.run_coroutine_threadsafe(coro, loop)
    except RuntimeError as exc:
        # The coroutine was created but will never be awaited; closing it stops
        # the interpreter warning about exactly that.
        coro.close()
        _LOG.debug("could not schedule %s: %s", where, exc)
        return None

    def _log_result(fut) -> None:
        if fut.cancelled():
            return
        exc = fut.exception()
        if exc is not None:
            _LOG.debug("%s failed: %s", where, exc)

    future.add_done_callback(_log_result)
    return future


class _BrowserGate(agent.ToolGate):
    """Holds the shared browser for one browser tool call.

    `agent.run_agent` calls `acquire` from a worker thread, so the asyncio lock
    is taken with `run_coroutine_threadsafe` and the thread waits on the
    future. That is the whole trick: the wait happens on a thread that has
    nothing else to do, while the event loop stays free to answer other people.

    Only `agent.BROWSER_TOOLS` come through here. A `get_recent_games` HTTP
    call does not touch Chromium or `storage_state.json`, so it never queues
    behind a review -- which is the entire reason this is a gate on the tool
    call rather than a lock around the turn.

    While it waits it says so, and afterwards it reports the real elapsed
    time, so the queue position is taken when the queue actually forms rather
    than guessed at the start of the turn.
    """

    def __init__(
        self,
        bot: "ChessBot",
        loop: asyncio.AbstractEventLoop,
        status: discord.Message,
        started: float,
    ) -> None:
        self._bot = bot
        self._loop = loop
        self._status = status
        self._started = started
        # All four are touched only on the event loop.
        self._last_edit = 0.0
        self._pending: str | None = None
        self._urgent = False
        self._draining = False

    # -- progress ----------------------------------------------------------

    async def _queue(self, content: str, urgent: bool = False) -> None:
        """Take the newest progress line and write it, rate-limited.

        Coalescing, not queueing. Discord allows about 5 edits per 5s per
        channel and a six-round turn outruns that, and a burst of updates that
        each waited their own turn would come out in an arbitrary order -- so
        the status line could end on "looking up recent games · 3s" after
        "reading the move list · 12s".

        Keeping only the newest line fixes both. A progress line is a snapshot
        of "what is happening now", so the ones it replaces carry no
        information by the time anyone could read them, and the last one is
        always the truth.

        `urgent` skips the rate limit, for the line that says a tool has
        actually started after a wait -- a stale "waiting for the browser" is
        worse than no line at all. One queue, so an urgent line can never
        discard a newer ordinary one on its way past.
        """
        self._pending = content
        self._urgent = self._urgent or urgent
        if self._draining:
            return
        self._draining = True
        try:
            while self._pending is not None:
                urgent, self._urgent = self._urgent, False
                gap = self._loop.time() - self._last_edit
                if not urgent and self._last_edit and gap < STATUS_EDIT_MIN_INTERVAL:
                    # Re-read `_pending` on the way round: a newer line may have
                    # arrived while we waited, and posting the stale one first
                    # would put the line back out of order.
                    await asyncio.sleep(STATUS_EDIT_MIN_INTERVAL - gap)
                    continue
                content, self._pending = self._pending, None
                self._last_edit = self._loop.time()
                await _safe_edit(self._status, content)
        finally:
            self._draining = False

    def _show(self, content: str) -> None:
        _spawn(self._queue(content), self._loop, "status edit")

    def _show_now(self, content: str) -> None:
        """The line that says a tool has really started, past the rate limit."""
        _spawn(self._queue(content, urgent=True), self._loop, "status edit")

    def _describe(self, name: str, args: dict) -> str:
        """The line shown while `name` runs. Never the tool name, by default.

        `args` is only used by the opt-in technical mode, and is truncated
        there: a game URL is long, a whole argument dump is not something to
        put in front of a user who asked a question about chess.
        """
        if SHOW_TOOL_NAMES:
            shown = ", ".join(f"{k}={str(v)[:60]}" for k, v in list(args.items())[:2])
            where = f"`{name}`" + (f" {shown}" if shown else "")
            return f"_running {where}_ \u00b7 {self._loop.time() - self._started:.0f}s"
        what = TOOL_PROGRESS.get(name, DEFAULT_PROGRESS)
        return f"_{what.capitalize()}\u2026 \u00b7 {self._loop.time() - self._started:.0f}s_"

    # -- the gate ----------------------------------------------------------

    def acquire(self, name: str, args: dict) -> None:
        if name not in agent.BROWSER_TOOLS:
            return
        lock = self._bot._browser_lock
        if lock.locked():
            self._bot._queued += 1
            try:
                # No tool name here either, for the same reason as in
                # `_describe`: this line goes into the user's channel.
                self._show(
                    "_Queued: someone else is using the browser for now. "
                    f"You are number {self._bot._queued} in line._"
                )
                # Blocking here is the point: this worker thread has nothing
                # else to do, and the event loop is free meanwhile.
                asyncio.run_coroutine_threadsafe(lock.acquire(), self._loop).result()
            finally:
                self._bot._queued = max(0, self._bot._queued - 1)
        else:
            # Fast path. If somebody beats us to it between the check and the
            # acquire we simply block, which is correct -- we just miss saying
            # so, and the user sees "working on it" instead of their place in
            # line. Better than holding the lock across the gap.
            asyncio.run_coroutine_threadsafe(lock.acquire(), self._loop).result()
        # Now it really is running, so the elapsed time shown is work, not queue.
        self._show_now(self._describe(name, args))

    def release(self, name: str) -> None:
        if name not in agent.BROWSER_TOOLS:
            return
        try:
            self._loop.call_soon_threadsafe(self._release_browser)
        except RuntimeError as exc:  # loop already gone; nothing to release onto
            _LOG.debug("could not release the browser lock: %s", exc)

    def _release_browser(self) -> None:
        """On the loop, and idempotent, so a watchdog release cannot double up.

        `asyncio.Lock.release` raises `RuntimeError` when the lock is not held.
        Both the holder and the abandonment watchdog can end up calling this --
        the holder when the tool finally returns, minutes after the watchdog
        already took the lock back -- and the check plus the release happen with
        no `await` between them, so together they are atomic on this thread.
        """
        if self._bot._browser_lock.locked():
            self._bot._browser_lock.release()


class ChessBot(commands.Bot):
    def __init__(self) -> None:
        # help_command=None: the built-in `help` cannot be replaced, only
        # removed, and it collides with a command of our own. Removing it lets
        # `!help` be our text, which is generated from the real command registry
        # so it cannot advertise something that does not exist.
        super().__init__(command_prefix="!", intents=INTENTS, help_command=None)
        # Belt and braces with the explicit `allowed_mentions` at each call
        # site: this is the one setting that reaches sends and edits the bot
        # does not make itself, and its default pings.
        self.allowed_mentions = NO_MENTIONS
        self._client: genai.Client | None = None
        # One conversation per (guild, channel, user), so two people in one
        # channel do not inherit each other's "the opponent in game 2".
        self._sessions: dict[tuple, agent.AgentSession] = {}
        # When each session was last used, for the idle sweep.
        self._session_seen: dict[tuple, float] = {}
        # One lock per session. An `AgentSession` owns a Gemini chat and is not
        # thread-safe, so two messages from one person must not run at once --
        # and since the browser lock is taken per tool call, this is the only
        # thing serialising a given conversation.
        self._session_locks: dict[tuple, asyncio.Lock] = {}
        # The tools share one browser profile and one chess.com session file.
        self._browser_lock = asyncio.Lock()
        self._queued = 0
        # Per-user cooldown, so one person cannot queue fifty two-minute turns.
        self._last_turn: dict[int, float] = {}
        # Abstractions whose worker thread is still stuck, for `!status`.
        self._stalled: dict[int, str] = {}
        # Watchdog tasks, kept referenced so they survive to do their job.
        self._watchdogs: set[asyncio.Task] = set()

    async def close(self) -> None:
        for task in list(self._watchdogs):
            task.cancel()
        self._watchdogs.clear()
        # Shut the browser down on the way out. The idle TTL would get there
        # eventually, but the process is ending: leaving a Chromium and its
        # driver to be reaped by the OS on every restart is a slow leak of
        # processes, and the owner stops noticing.
        try:
            await asyncio.to_thread(get_session().close)
        except Exception:  # noqa: BLE001 - shutdown must not fail
            _LOG.debug("closing the session browser during shutdown", exc_info=True)
        await super().close()

    # -- lifecycle ---------------------------------------------------------

    async def setup_hook(self) -> None:
        # The `@commands.command()` decorator is what makes these `Command`
        # objects. Without it they are plain coroutine methods, and
        # `add_command` raises `TypeError` before the bot ever connects.
        #
        # `add_command` is a plain synchronous method, NOT a coroutine, so there
        # is no `await` on it. `process_commands` a few lines down IS async.
        for command in (self.cmd_reset, self.cmd_status, self.cmd_help):
            self.add_command(command)

    async def on_ready(self) -> None:
        _LOG.info("logged in as %s (%s)", self.user, self.user.id)
        print("=" * 68)
        print(f"  Discord bot ready: {self.user}")
        print(f"  Client ID:        {self.user.id}")
        print(f"  Guilds:           {len(self.guilds)}")
        print("=" * 68)
        print("\nAdd it to your servers with this link:\n")
        print(f"  https://discord.com/oauth2/authorize?client_id={self.user.id}")
        print(f"  &scope=bot&permissions={BOT_PERMISSIONS}\n")
        print(
            f"  permissions: {BOT_PERMISSIONS} "
            "(view channel, send, embed, attach, read history)"
        )
        if OPEN_TO_EVERYONE:
            print("\n  ACCESS: open to everyone who can see me.")
        else:
            print(f"\n  ACCESS: {sorted(ALLOWED_USERS) or 'NOBODY -- see below'}")
            if not ALLOWED_USERS:
                print("  No DISCORD_ALLOWED_USER_IDS in .env, so I will refuse")
                print("  everyone. Put your own ID there, or 'all' to open up.")
        print("\n  If I never answer, enable SERVER SETTINGS > MESSAGES >")
        print("  MESSAGE CONTENT INTENT in the Developer Portal.\n")
        print("  Ctrl-C to stop.\n")

    def session_for(self, key: tuple) -> agent.AgentSession:
        """The conversation for this (guild, channel, user), creating it if new.

        Raises `agent.MissingSetting` if the API key is unusable. The key is
        read here, from inside a coroutine on the event loop, so it has to be
        `require_api_key` (which raises) and not the exiting kind: a
        `SystemExit` there is a `BaseException` no `except Exception` catches,
        and a key that goes bad between startup and the first message would
        take the whole bot down instead of posting a sentence about `.env`.
        """
        if self._client is None:
            self._client = genai.Client(api_key=agent.require_api_key())
        if key not in self._sessions:
            self._sessions[key] = agent.AgentSession(
                self._client, platform_note=agent.DISCORD_PLATFORM_NOTE
            )
            self._session_locks[key] = asyncio.Lock()
            self._session_seen[key] = time.monotonic()
        else:
            # Re-insert so the dict stays in least-recently-used order.
            self._session_seen[key] = time.monotonic()
        self._prune_sessions()
        return self._sessions[key]

    def forget_session(self, key: tuple) -> bool:
        """Drop a conversation and its lock. True if there was one."""
        self._session_seen.pop(key, None)
        self._session_locks.pop(key, None)
        return self._sessions.pop(key, None) is not None

    def _prune_sessions(self) -> None:
        """Keep the session map bounded.

        A session is a live Gemini chat with up to `AGENT_MAX_TURNS` turns of
        history, and `get_moves` alone returns a few hundred plies of dicts. The
        key is per (guild, channel, user), so on a busy server -- or with
        `DISCORD_ALLOWED_USER_IDS=all` -- that map grew for ever, and the
        process with it. Idle conversations go first, then the oldest, so a
        cap never costs somebody their place mid-conversation.
        """
        now = time.monotonic()
        stale = [
            key
            for key, seen in self._session_seen.items()
            if now - seen > SESSION_IDLE_SECONDS
        ]
        for key in stale:
            _LOG.info("dropping idle conversation %s", key)
            self.forget_session(key)
        while len(self._sessions) > MAX_SESSIONS:
            oldest = next(iter(self._sessions))
            _LOG.info("dropping oldest conversation %s (over cap)", oldest)
            self.forget_session(oldest)

    # -- commands ----------------------------------------------------------

    @staticmethod
    def _key_for(guild, channel, author) -> tuple:
        """One conversation per person per place.

        Keying on the channel alone would let two people in the same channel
        inherit each other's "the opponent in game 2", which reads as the bot
        getting confused rather than as two conversations.
        """
        return (
            guild.id if guild else 0,
            channel.id,
            author.id,
        )

    @commands.command(name="reset")
    async def cmd_reset(self, ctx: commands.Context) -> None:
        """Start a fresh conversation."""
        if not await self._refuse_if_not_permitted(ctx):
            return
        had = self.forget_session(self._key_for(ctx.guild, ctx.channel, ctx.author))
        await ctx.reply(
            "Fresh conversation. Ask me anything."
            if had
            else "You did not have a conversation going here anyway. Ask me anything."
        )

    @commands.command(name="status")
    async def cmd_status(self, ctx: commands.Context) -> None:
        """What the bot is doing right now."""
        if not await self._refuse_if_not_permitted(ctx):
            return
        busy = self._browser_lock.locked()
        # The session outlives the gate on purpose, so "the lock is free" and
        # "no browser is running" are different questions and reporting only the
        # first is misleading: a page can be open and waiting while the gate sits
        # idle. Report both.
        session = get_session().describe()
        lines = [
            f"model: `{agent.MODEL}`",
            f"browser: {'BUSY (one review at a time)' if busy else 'free'}",
            (
                f"session: {'OPEN on ' + session['url'] if session['open'] else 'closed'}"
                + (
                    f" (idle {session['idle_seconds_ago']}s of "
                    f"{session['idle_ttl']}s, closes itself)"
                    if session["open"]
                    else ""
                )
            ),
            f"queued for the browser: {self._queued}",
            f"conversations: {len(self._sessions)} (cap {MAX_SESSIONS})",
            f"turn limit: {int(TURN_TIMEOUT_SECONDS)}s"
            if TURN_TIMEOUT_SECONDS
            else "turn limit: none",
        ]
        sign_in = chess_com_sign_in_age()
        if sign_in:
            lines.append(f"chess.com sign-in file: {sign_in}")
        if self._stalled:
            # Worth surfacing: a wedged tool is invisible otherwise, and the
            # owner needs to know a Chromium is still running.
            lines.append(
                f"**stalled: {len(self._stalled)}** -- "
                + ", ".join(sorted(self._stalled.values()))
            )
        await ctx.reply("\n".join(lines), allowed_mentions=NO_MENTIONS)

    @commands.command(name="help")
    async def cmd_help(self, ctx: commands.Context) -> None:
        """How to use me."""
        if not await self._refuse_if_not_permitted(ctx):
            return
        await ctx.reply(
            help_text({c.name: c.help for c in self.commands}),
            allowed_mentions=NO_MENTIONS,
        )

    async def on_command_error(self, ctx: commands.Context, error) -> None:
        """Say something when a command does not work.

        With no handler, discord.py logs the traceback and the user sees
        nothing at all, so `!statuss` reads as a broken bot rather than a typo.
        """
        if isinstance(error, commands.CommandInvokeError):
            # The command itself raised; it is responsible for its own message.
            _LOG.warning("command %s raised", ctx.command, exc_info=error)
            text = "That command ran into a problem. The owner has been told."
        elif isinstance(error, commands.CommandNotFound):
            return  # a plain message that is not a command; not our business
        else:
            _LOG.info("command error: %s: %s", ctx.command, error)
            text = f"I do not know `!{ctx.invoked_with}`. Try `!help`."
        try:
            await ctx.reply(text, allowed_mentions=NO_MENTIONS)
        except discord.HTTPException:
            _LOG.debug("could not report a command error", exc_info=True)

    async def _refuse_if_not_permitted(self, ctx) -> bool:
        """One access check for every entry point. True if allowed to proceed.

        `on_message` gates the chat path and the command handlers gate the
        `!` path. Two copies of the same rule is two rules, and the one that
        drifts is the one nobody tests.
        """
        if permitted(ctx.author):
            return True
        await ctx.reply(
            "I'm set to private -- the owner has to add you before I can run "
            "anything. Ask them to add your user ID to "
            "`DISCORD_ALLOWED_USER_IDS`.",
            allowed_mentions=NO_MENTIONS,
        )
        return False

    # -- message handling --------------------------------------------------

    async def on_message(self, message: discord.Message) -> None:
        # Never talk to ourselves, and never answer another bot.
        if message.author.id == self.user.id or message.author.bot:
            return

        # Prefix commands first, and FIRST because this override replaces
        # `commands.Bot.on_message`, the method that normally calls
        # `process_commands`. Without this line `!reset` and `!status` would be
        # registered, documented, and silently never fire. They work anywhere,
        # no mention needed.
        ctx = await self.process_commands(message)
        if ctx is not None and ctx.invoked_with:
            return

        in_dm = isinstance(message.channel, (discord.DMChannel, discord.GroupChannel))
        if not in_dm and self.user not in getattr(message, "mentions", []):
            return
        # A bare mention with no text: there is nothing to send to the model.
        text = _MENTION_STRIP.sub("", message.content or "").strip()
        if not text:
            await message.reply(
                "Ask me something -- e.g. *how was my last game?*",
                allowed_mentions=NO_MENTIONS,
            )
            return

        if not await self._refuse_if_not_permitted(_Ctx(message)):
            return

        await self._answer(message, text)

    async def _answer(self, message: discord.Message, text: str) -> None:
        loop = asyncio.get_running_loop()
        key = self._key_for(message.guild, message.channel, message.author)
        try:
            session = self.session_for(key)
        except agent.MissingSetting as exc:
            # The key went bad after startup. `require_api_key` raises rather
            # than exiting, so this is a message and not a dead process.
            _LOG.error("%s is unusable: %s", exc.name, exc.detail)
            await message.reply(
                f"I cannot start: {exc.detail}",
                allowed_mentions=NO_MENTIONS,
            )
            return

        if not self._may_start_turn(message.author.id, message):
            return
        self._last_turn[message.author.id] = time.monotonic()
        self._stalled.pop(message.author.id, None)

        started = loop.time()
        status = await message.reply(
            "_Working on it. Reviews drive a real browser, so this can take a "
            "minute. I'll edit this message as I go._",
            allowed_mentions=NO_MENTIONS,
        )

        async def typing_forever() -> None:
            while True:
                try:
                    await message.channel.typing()
                except discord.HTTPException:
                    return
                await asyncio.sleep(TYPING_REFRESH_SECONDS)

        typing = asyncio.create_task(typing_forever())
        gate = _BrowserGate(self, loop, status, started)

        async def work() -> agent.AgentResult:
            # to_thread, not a bare call: a tool opens a browser and blocks. The
            # tools are sync and call asyncio.run themselves, which is exactly
            # right in a thread and exactly wrong on the event loop.
            return await asyncio.to_thread(session.ask, text, None, gate)

        # Per session, not per bot. An AgentSession owns a Gemini chat and is not
        # thread-safe, so one person cannot have two turns interleaved -- but two
        # different people now run at once, and only a browser tool makes them
        # wait for each other.
        lock = self._session_locks.get(key) or asyncio.Lock()
        self._session_locks[key] = lock
        result: agent.AgentResult
        try:
            async with lock:
                result = await self._bounded(work(), started, message.author.id)
        except Exception as e:  # noqa: BLE001 - never let one turn kill the bot
            _LOG.warning("turn failed", exc_info=True)
            result = agent.AgentResult(text=f"Something went wrong: {e}", error=str(e))
        finally:
            # Awaited, not just cancelled: a `cancel()` we never wait on can
            # still have a `channel.typing()` request in flight, and the task
            # object is left dangling until the loop notices.
            typing.cancel()
            await asyncio.gather(typing, return_exceptions=True)

        try:
            await self._post(status, result)
        except Exception:  # noqa: BLE001 - last line of defence
            # `_post` guards its own sends, so this is belt and braces. Even so,
            # the user must not be left with a "working on it" message and no
            # answer: fall back to posting the text directly.
            _LOG.warning("could not post the answer", exc_info=True)
            await self._send(status, strip_path_lines(result.text) or "_(no answer)_")

    def _may_start_turn(
        self, user_id: int, message: discord.Message | None = None
    ) -> bool:
        """One turn per user per `USER_COOLDOWN_SECONDS`.

        Not politeness. One review is a real browser against the owner's
        chess.com session, so a user who fires off fifty mentions spends the
        owner's tokens, occupies the browser for an hour, and walks chess.com
        closer to a Cloudflare challenge for the account the tools log in as.
        0 turns the check off.

        Says so when it refuses. A silently dropped message is indistinguishable
        from a bot that has hung, which is the one thing this whole path is
        trying not to be.
        """
        if USER_COOLDOWN_SECONDS <= 0:
            return True
        # Bounded for the same reason `_sessions` is: keyed by user id, so on a
        # large server it would otherwise accumulate for ever. The cap is well
        # above MAX_SESSIONS, so it only ever discards stale entries.
        if len(self._last_turn) > 4 * MAX_SESSIONS:
            cutoff = time.monotonic() - USER_COOLDOWN_SECONDS
            self._last_turn = {
                uid: at for uid, at in self._last_turn.items() if at >= cutoff
            }
        last = self._last_turn.get(user_id)
        if last is None or time.monotonic() - last >= USER_COOLDOWN_SECONDS:
            return True
        if message is not None:
            _spawn(
                message.reply(
                    f"_One question at a time -- give me {USER_COOLDOWN_SECONDS:g}s._",
                    allowed_mentions=NO_MENTIONS,
                ),
                asyncio.get_running_loop(),
                "cooldown notice",
            )
        return False

    async def _bounded(self, coro, started: float, user_id: int):
        """`await coro`, with a deadline and an escape hatch for a wedged tool.

        `asyncio.wait_for` cancels the future, which stops *us* waiting -- it
        cannot stop a thread that is inside Playwright. So the timeout alone
        would hand the user an answer while the browser lock stayed held by a
        thread that may never return, and every later request would queue behind
        it in silence. Hence the watchdog: after a grace period it takes the
        lock back so the queue drains, and says loudly in `!status` that
        something is stuck.
        """
        if TURN_TIMEOUT_SECONDS <= 0:
            return await coro
        try:
            return await asyncio.wait_for(coro, TURN_TIMEOUT_SECONDS)
        except (asyncio.TimeoutError, TimeoutError):
            _LOG.error(
                "turn exceeded %ds; abandoning it (its thread may still be running)",
                TURN_TIMEOUT_SECONDS,
            )
            self._stalled[user_id] = f"turn over {int(TURN_TIMEOUT_SECONDS)}s"
            self._schedule_watchdog(user_id)
            return agent.AgentResult(
                text=(
                    f"_I gave up after {int(TURN_TIMEOUT_SECONDS)}s without an "
                    f"answer. A review drives a real browser and this one stopped "
                    f"responding -- the owner has been told. The browser may stay "
                    f"busy for a minute while it is cleaned up; ask me again in a "
                    f"bit._"
                ),
                error=f"turn timed out after {TURN_TIMEOUT_SECONDS}s",
            )

    def _schedule_watchdog(self, user_id: int) -> None:
        """Take the browser lock back if the abandoned thread never lets go."""

        async def watchdog() -> None:
            await asyncio.sleep(BROWSER_LOCK_GRACE_SECONDS)
            if self._browser_lock.locked():
                _LOG.error(
                    "browser lock still held %ds after a turn was abandoned; "
                    "reclaiming it (a Chromium may need killing by hand)",
                    BROWSER_LOCK_GRACE_SECONDS,
                )
                self._browser_lock.release()
                self._stalled.pop(user_id, None)

        # Held so the task is not garbage collected mid-sleep, and so the
        # shutdown path can cancel it. Done in `close`/`on_close`.
        self._watchdogs.add(asyncio.create_task(watchdog()))
        self._watchdogs = {t for t in self._watchdogs if not t.done()}

    async def _post(self, status: discord.Message, result: agent.AgentResult) -> None:
        body = strip_path_lines(result.text) or "_(the model returned nothing)_"
        # A screenshot tool ran and nothing came back. Say so, and say WHY: the
        # alternative is a paragraph of confident text and silence where the
        # image should be, which is the complaint, not the fix.
        if not result.artifacts and SHOT_TOOLS.intersection(result.tools_used):
            note = picture_failure_note(result.tool_errors)
            _LOG.warning(
                "a screenshot tool ran and produced no picture (%s): %s",
                ", ".join(result.tools_used),
                result.tool_errors or "no reason reported",
            )
            body = f"{body}\n\n{note}"
        chunks = chunk_message(body) or ["_(the model returned nothing)_"]

        # Reuse the placeholder for the first chunk so the channel does not
        # accumulate a "working on it" message above every answer. If the edit
        # fails -- too old, no permission -- fall back to sending it.
        if await _safe_edit(status, chunks[0], force=True):
            rest = chunks[1:]
        else:
            rest = chunks
        # Every send is guarded, so one rate limit or 413 costs a chunk rather
        # than throwing out of `_post`, out of `_answer`, and into discord.py's
        # default handler -- the user would lose the rest of the answer having
        # been told nothing.
        for extra in rest:
            if not await self._send(status, extra):
                _LOG.warning("could not post a chunk of the answer")
                break

        # Attach the real files. Collected at the tool boundary, so this is the
        # actual PNG that was written -- not a path the model remembered.
        await self._attach(status, result.artifacts)

        if result.error:
            _LOG.error("turn error: %s", result.error)

    async def _send(self, status: discord.Message, content: str) -> bool:
        """One message, tolerating every way Discord can refuse it."""
        try:
            await status.channel.send(content, allowed_mentions=NO_MENTIONS)
            return True
        except discord.HTTPException as e:
            _LOG.warning("could not send a message: %s", e)
        except asyncio.TimeoutError:
            _LOG.warning("timed out sending a message")
        return False

    async def _attach(self, status: discord.Message, artifacts: list) -> None:
        """Post the screenshots, batched by file count AND by size.

        Two different things can cost the user a picture, and both are said out
        loud rather than logged: a frame the byte budget refuses outright, and a
        batch Discord rejects. A review that silently loses its blunder frame is
        exactly the failure this project keeps designing against.
        """
        # Batched first, so what the budget refused can be counted before any
        # send: `_upload_batches` is a generator, and a count taken afterwards
        # would conflate "too big to post" with "the upload failed".
        batches = list(_upload_batches(artifacts))
        refused = len(artifacts) - sum(len(batch) for batch in batches)
        posted = 0
        for batch in batches:
            paths = [path for _tool, path in batch]
            caption = "\n".join(f"- {describe_artifact(path)}" for path in paths)
            try:
                files = [discord.File(path, filename=path.name) for path in paths]
                await status.channel.send(
                    content=caption,
                    files=files,
                    reference=status,
                    mention_author=False,
                    allowed_mentions=NO_MENTIONS,
                )
                posted += len(paths)
            except (discord.HTTPException, asyncio.TimeoutError, OSError) as e:
                # Over the upload ceiling, a file the channel cannot read, a
                # rate limit. The answer is already posted, so the worst case
                # here is missing pictures -- but it has to be said, because a
                # review that silently loses its blunder frame is exactly the
                # failure this project keeps designing against.
                _LOG.warning("could not attach %d screenshot(s): %s", len(paths), e)
                await self._send(
                    status,
                    f"_Could not attach {len(paths)} screenshot(s): {e}. The "
                    f"answer above still stands._",
                )
        if refused:
            _LOG.warning(
                "%d screenshot(s) were over the per-file size limit and were "
                "never offered to Discord",
                refused,
            )
            await self._send(
                status,
                f"_{refused} of the {len(artifacts)} screenshot(s) were too large "
                f"for this channel and were left out. Lower REVIEW_VIEWPORT_W to "
                f"get smaller pictures._",
            )
        skipped = len(artifacts) - posted
        if skipped:
            _LOG.warning("%d artifact(s) never made it to Discord", skipped)


def _batched(items: list, size: int | None = None):
    """Split into chunks of at most `size` -- Discord allows 10 files a message."""
    size = MAX_ATTACHMENTS if size is None else size
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _upload_batches(artifacts: list, size: int | None = None):
    """Split screenshots by file count AND by byte size.

    The count limit alone is not enough. Ten board crops at the default viewport
    measured ~620KB each, which is ~6MB against Discord's 8MB ceiling for a
    server with no boost -- and `REVIEW_VIEWPORT_W` can push a frame well past
    that, which gets the whole batch refused with a 413.

    A file over `MAX_ATTACHMENT_BYTES` on its own is dropped with a log line
    rather than posted, so it cannot fail its own batch and every batch after.
    `_attach` counts what came back and tells the user how many were left out,
    because a frame the budget refused never reaches Discord and so produces no
    HTTP error to report -- it would just be a picture that never arrives.
    Files are stat'ed once here rather than by `discord.File`, which would
    otherwise re-read each one on every retry.
    """
    size = MAX_ATTACHMENTS if size is None else size
    batch: list = []
    total = 0
    for tool, path in artifacts:
        try:
            nbytes = path.stat().st_size
        except OSError as e:
            _LOG.warning("skipping unreadable screenshot %s: %s", path, e)
            continue
        if nbytes > MAX_ATTACHMENT_BYTES:
            _LOG.warning(
                "skipping %s: %d bytes is over the %d byte per-file limit",
                path.name,
                nbytes,
                MAX_ATTACHMENT_BYTES,
            )
            continue
        if batch and (len(batch) >= size or total + nbytes > MAX_UPLOAD_BYTES):
            yield batch
            batch = []
            total = 0
        batch.append((tool, path))
        total += nbytes
    if batch:
        yield batch


async def _safe_edit(
    message: discord.Message, content: str, force: bool = False
) -> bool:
    """Edit a message, tolerating the ways that fails. True if it worked.

    Called on every tool call from a worker thread, so it must never raise: an
    identical-content edit is a 400, and a message older than ~15 minutes cannot
    be edited at all. Returns False rather than raising so the caller can fall
    back to sending.

    `force` skips the identical-content short circuit. That check is right for a
    progress line and wrong for the final answer: if the model happened to
    return exactly the text already on screen, the "edit" would report success
    having done nothing -- and the answer would never be sent. `_post` passes
    `force=True` for that one call.

    `allowed_mentions=NO_MENTIONS` on every edit, for the same reason as on
    every send: this is model-authored text, and the library default parses
    `@everyone`, roles and users.
    """
    try:
        if not force and message.content == content:
            return True
        await message.edit(content=content, allowed_mentions=NO_MENTIONS)
        return True
    except discord.HTTPException as e:
        _LOG.debug("could not update a message: %s", e)
        return False
    except asyncio.TimeoutError:
        _LOG.debug("timed out updating a message")
        return False


def main() -> int:
    token = os.getenv("DISCORD_BOT_TOKEN", "").strip().strip('"').strip("'")

    if not token:
        state = project_env.setting_state("DISCORD_BOT_TOKEN")
        hint = ""
        if state == "placeholder":
            hint = "  It is set, but still to the placeholder from .env.example.\n"
        print(
            f"{'DISCORD_BOT_TOKEN is a placeholder' if state == 'placeholder' else 'Missing DISCORD_BOT_TOKEN'}.\n"
            f"{hint}"
            "\n"
            "1. https://discord.com/developers/applications -> New Application\n"
            "2. Bot -> Reset Token -> copy it (shown once)\n"
            "3. Bot -> Privileged Gateway Intents -> turn ON Message Content "
            "Intent\n"
            "4. OAuth2 -> URL Generator: scopes `bot`, permissions 117760\n"
            f"5. Put it in {project_env.PROJECT_ENV}:\n"
            "     DISCORD_BOT_TOKEN=the_token_you_just_copied\n"
            "     DISCORD_ALLOWED_USER_IDS=your_numeric_user_id\n"
            "   (In Discord: Settings -> Advanced -> enable Developer Mode,\n"
            "    then right-click yourself -> Copy User ID.)\n"
            "\n"
            f"  {project_env.describe_sources()}\n"
            "  Run 'python check_env.py' for a full picture of every setting.\n"
        )
        return 1

    if token.lower().startswith("your_") or token.lower().endswith("_here"):
        print(
            "DISCORD_BOT_TOKEN is still the placeholder from .env.example.\n"
            f"  {project_env.PROJECT_ENV} -- paste the real token over it.\n"
        )
        return 1

    # Fail before the browser or the model is touched, so a misconfiguration is
    # a one-line message rather than a stack trace three layers down. This is a
    # program, so exiting is right here -- the RUNTIME path in `_answer` uses
    # `require_api_key`, which raises, because a `SystemExit` on the event loop
    # would take the process down instead of posting a sentence.
    try:
        agent.require_api_key()
    except agent.MissingSetting as exc:
        print(f"{exc.detail}\n")
        return 1

    if not ALLOWED_USERS and not OPEN_TO_EVERYONE:
        print(
            "Note: DISCORD_ALLOWED_USER_IDS is not set, so I will refuse every\n"
            "user. That is the safe default -- the tools log into chess.com as\n"
            "you and launch a browser on this machine. Set your user ID in .env,\n"
            "or 'all' to open it up on purpose.\n"
        )

    print(f"  config: {project_env.describe_sources()}\n")
    if TURN_TIMEOUT_SECONDS > 0:
        print(
            f"  limits: {int(TURN_TIMEOUT_SECONDS)}s per turn, "
            f"{USER_COOLDOWN_SECONDS:g}s per user, "
            f"{MAX_SESSIONS} conversations, "
            f"{MAX_UPLOAD_BYTES // (1024 * 1024)}MB per upload\n"
        )

    bot = ChessBot()
    try:
        bot.run(token, log_handler=None)
    except discord.LoginFailure:
        print(
            "Discord rejected that token. Check DISCORD_BOT_TOKEN in\n"
            f"  {project_env.PROJECT_ENV}\n"
            "  A token that was ever pasted into a chat, a screenshot or a commit\n"
            "  must be Reset in the Developer Portal, not edited here."
        )
        return 1
    except KeyboardInterrupt:
        print("\nBye!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
