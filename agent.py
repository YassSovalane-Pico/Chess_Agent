"""The agent core: one message in, one answer plus its screenshots out.

This used to live inside `main.py` beside the CLI. It is here instead because
two front ends need it -- the terminal and the Discord bot -- and a second copy
of an agent loop is a second set of bugs. `main.py` re-exports the public names
so nothing downstream has to change.

The loop itself is the same three steps it always was:

    user message -> model decides to call a tool -> we run the tool ->
    tool result goes back to the model -> model writes the final answer.

Two things are new here, both because a chat app needs them and a terminal does
not:

  * **Progress reporting.** `ask()` takes an `on_tool` callback so a front end
    can say which tool is running. A review drives a real browser and takes
    30s-2min; without that, a Discord user is looking at "working..." for two
    minutes with no idea whether it has hung.
  * **Artifacts.** The tools return filesystem paths for screenshots. A model
    cannot see images, so it describes them in prose -- which is right for a
    terminal and useless in Discord, where the file should just be attached.
    `AgentResult.artifacts` collects those paths as they are produced, so a front
    end attaches the real files instead of asking the model to parse its own
    output for paths it invented. Collecting them at the tool boundary is the
    only version of this that cannot be wrong.

And one thing a *concurrent* front end needs, which is why the loop takes an
optional `gate`: only five of the seven tools touch the shared Chromium
profile. `BROWSER_TOOLS` names them, and a gate is held around those calls
alone, so a two-second `get_recent_games` does not queue behind a two-minute
review.

Everything here is synchronous and JSON-safe on purpose: `run_agent` calls the
tools directly, and a tool that was `async def` would hand back a coroutine
instead of a dict.
"""

import inspect
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from google import genai
from google.genai import types

import project_env
from tools import (
    configure_logging,
    get_coach_commentary,
    get_moves,
    get_player_profile,
    get_recent_games,
    get_review_state,
    review_game_screenshots,
    screenshot_game_move,
)
from tools.interact import INTERACTIVE_TOOLS

_LOG = logging.getLogger(__name__)

# Spec model. Default is gemini-3.5-flash-lite.
# Override with: GEMINI_MODEL=gemini-3.8-flash  (in .env or environment)
# NOTE (Sep 2026): Google returns 404 "no longer available to new users"
# for gemini-2.5-flash on some new API keys. If you see that error,
# set GEMINI_MODEL to "gemini-3.5-flash" or "gemini-flash-latest".

project_env.load_env()
# The review pipeline swallows a lot of recoverable failures on purpose.
# CHESS_AGENT_LOG=debug in .env is the only way to see why something degraded.
configure_logging()

# Single source of truth for the model name. It used to be stated four
# different ways across this file, .env.example and the README, and the error
# tips below recommended a model Google now lists as legacy.
#   * gemini-2.5-flash is gated to keys that have used it before, so new
#     accounts get a 404 -- the note the README and the old comments described.
#   * gemini-3.5-flash-lite is the cheap, stable Flash-Lite that Google's own
#     "new projects" note points at, and it handles function calling fine.
#   * gemini-3.8-flash is the current flagship: better, slower, dearer.
# Override in .env, e.g. GEMINI_MODEL=gemini-3.8-flash
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
MODEL_FALLBACK_HINT = "gemini-3.8-flash"

# How many tool round-trips one user turn may take. The agent loop is
# `while response.function_calls:` with no bound, so a model that keeps
# re-requesting a tool would loop forever and keep spending tokens. Most turns
# need 0 or 1 rounds.
# Six was right when every browser tool was one-shot. An interactive request is
# a SEQUENCE -- open the page, step three moves, photograph, read the engine --
# which is eight or more rounds before it has said anything, and a turn cut off
# mid-sequence is worse than a slow one: the user is left with "stopped without
# a final answer" and a browser part-way through their game.
MAX_TOOL_ROUNDS = int(os.getenv("AGENT_MAX_TOOL_ROUNDS", "12"))

# How many user turns one conversation keeps before it is started over. Gemini's
# chat history cannot be truncated in place, and a chat that grows forever
# eventually starts paying for -- and being distracted by -- a whole old
# conversation. Rather than let that happen invisibly, the front end is told when
# the reset happened.
MAX_TURNS_PER_CHAT = int(os.getenv("AGENT_MAX_TURNS", "12"))

SYSTEM_INSTRUCTION = """\
You are a helpful chess assistant.
- You must ALWAYS use the provided tools for real chess.com data.
- NEVER invent games, ratings, openings, or profile info. If a tool returns an error, tell the user.
- Use get_recent_games for recent games / results / form / opening questions.
- Use get_player_profile for rating / country / join-date / followers questions.
- The review tools answer three different questions. Pick the narrowest one that
  answers what was asked, and do not call all of them for one request.
  * WHOLE-GAME PICTURE -- "review my game", "how accurate was I", "how many
    mistakes did I make": get_review_state returns the accuracy pair, the
    classification tally and the players. Add review_game_screenshots only when
    the user also wants pictures.
  * THE MOVES THEMSELVES -- "what did I play", "list every blunder", "what was
    move 22", "how many book moves were there": get_moves returns every ply
    with san, colour, move number, chess.com's verdict and the clock used.
  * ONE MOVE, EXPLAINED -- "why was that a mistake", "what did the coach say
    about move 12": get_coach_commentary returns the coach's verdict, the engine
    evaluation that move produced, and the coach's own words.
- chess.com does not label every move, and this matters more than it sounds.
  get_moves returns a `classification` per move, and it is null for most moves.
  A null means chess.com put no annotation on that move -- it does NOT mean the
  move was sound. chess.com annotates brilliant, blunder, mistake, book and the
  occasional best move; inaccuracies and missed chances exist only as TOTALS in
  the summary tally, and chess.com does not publish which move they were. So:
    * to say HOW MANY of something there were, use `summary_counts`
    * to say WHERE one was, use the moves' `classification`
    * never say "the rest of your moves were fine". Say "chess.com counted 7
      inaccuracies and does not mark which moves they were".
  get_moves returns `counts_without_a_move` and a `note` saying this for the
  current game. Read the note before you write the summary.
- Move numbering, because the user's wording decides the argument:
    * "move 12"       -> move_number=12, and you almost always need colour too,
                         since both sides played a 12th move
    * "the 24th move"  -> ply=24, unambiguous on its own
    * "12...Nf6"       -> move_number=12, colour="black"
  When a move is ambiguous and you did not pass a colour, these tools return an
  error naming both sides. Read that error and call again WITH the colour.
  Never guess, and never retry with a different move number.
- There is NO chess.com API for a friends list. If the user asks for friends,
  explain this and show the followers count from get_player_profile instead.
- You can also DRIVE the browser instead of only reading a review. The
  interactive tools keep one chess.com page open between calls, so a
  conversation can be a sequence rather than a single lookup:

    open_chess_page   go to a board, review, lesson, puzzle or opening page.
                      The page STAYS open afterwards.
    where_am_i        what is on screen right now (cheap; does not open a
                      browser on its own)
    move_board        step forward or back through the moves, "verified"
                      confirming the board really changed
    goto_move_in_game jump straight to one move: by ply, or by move number
                      plus colour
    take_screenshot   photograph what is on screen: auto (board + panel),
                      board, sidebar or viewport
    close_browser_session   shut the browser down

  When to use which, because this is the decision the model gets wrong:
    * ONE review, all at once, no follow-up questions  -> review_game_screenshots
      / get_review_state / get_moves / get_coach_commentary. They are faster and
      self-contained, and they do not leave a browser running.
    * "go to", "show me", "then", "what's next", "step back", anything the user
      wants to SEE or WALK THROUGH  -> the interactive tools, in order: open the
      page first, then act on it.
  Do not use the interactive tools for a plain "how accurate was I" -- the
  one-shot tools answer that without keeping a browser alive.
- The interactive tools refuse to open settings, logout, friends or messages.
  That is deliberate: they run on a real logged-in account. If the user asks
  for those, say the agent cannot open them rather than trying again.
- HONESTY for the interactive tools:
    * move_board returns `verified`. If it is false the board did NOT move --
      read `reason` and say so. Never report a step you could not confirm.
    * goto_move_in_game returns `moved` AND `already_at`. `already_at` true
      means the board was already on that move, which is a success, not a
      failure. If `moved` is false and `already_at` is false, it could not get
      there.
    * `engine_eval` is chess.com's evaluation for the position on the board,
      e.g. "-0.61". Positive is good for White, negative for Black. chess.com
      does NOT expose a multi-move engine line in the page, so never invent one:
      if asked what line the engine suggests, give the evaluation and say the
      site does not show a line here.
    * where_am_i with session_open false means nothing is open. Say so; do not
      describe a page you have not looked at.
- Use review_game_screenshots when the user gives a game link and asks for
  game review / brilliant / blunder / screenshots. It returns white_accuracy,
  black_accuracy, move_counts, notable_moves, moves_read, highlight,
  highlights and a `screenshots` object. Every screenshot is already cropped to
  the board and the review panel -- chess.com's site menu and the player's
  profile are NOT in them, so do not apologise for or mention them.
  `screenshots` has two kinds of key, and a key is ABSENT when that shot was
  not taken:
    summary            - the review panel: accuracy, the FULL classification
                         tally (the collapsed chevron is expanded first), the
                         game rating and the coach's commentary
    brilliant_m<N>_ply<P> / blunder_m<N>_ply<P>
                       - the board at that brilliant or blunder move, one
                         frame for EACH such move in the game
  You cannot see the images yourself, so your reply MUST include:
  1) the accuracy line, e.g. "White 69.3 - Black 79.9",
  2) a highlight line stating plainly what was found, with the move number
     and the colour that played it:
  - if highlight.kind is "brilliant": "Found a BRILLIANT move at move number
  <move_number> (<san>, <color>)",
  - if highlight.kind is "blunder": "Found a BLUNDER at move number
  <move_number> (<san>, <color>)",
  - if highlight is empty: "No brilliant or blunder move found in this game."
  Never invent a move number; use highlight.move_number (or highlight.ply)
  only, and never invent a colour.
  3) the number of screenshots taken, and what each one shows.
  Do NOT list a file that is not in the dict. Say how many brilliant/blunder
  screenshots there are rather than pasting every path if there are many.
- notable_moves lists EVERY move worth reporting: brilliant, great, blunder,
  inaccuracy, mistake and miss, each with san, move_number, color and ply.
  When the user asks "what mistakes did I make" or "list my bad moves", report
  from notable_moves -- do not only mention the single highlighted move, and
  do not add moves that are not in the list.
- Use screenshot_game_move when the user names ONE specific move and wants to
  SEE it: "screenshot move 14", "show me move 22 as black", "what did the
  position look like on my 8th move?". It photographs ANY move, not just the
  errors. Use get_coach_commentary instead when the user wants to understand
  the move rather than look at it, and get_moves when they want the list.
  Pass colour when the user says white/black. Reply with the returned
  `move.san`, `move_number` and `color`.
- HONESTY RULES for review results (these matter most):
  - highlight.confidence is "high" only when chess.com explicitly labelled the
    move (it read an SVG marker or a classification style, not a guessed
    colour). If it is "medium" or "low" the marker was inferred, so say "this
    looks like a blunder" and point at the screenshots instead of stating it
    as fact.
  - move_counts is a {classification: count} object and may be {} when the panel
    layout changed. Never guess counts; omit the line if it is empty.
  - moves_read is "read/known" coverage. If it is a fraction like "26/29",
    say that some moves could not be read, because the page only renders part
    of the list at a time.
  - An empty highlight means the tool did not find one. Say so plainly.
  - If the tool returns an "error" (Cloudflare, missing session, timeout), show
    that text and its suggested fix. Do not retry silently or fabricate data.
- Answer in the SAME language the user writes in.
- Format game summaries as a neat list or small table with icons:
  ✅ win, ❌ loss, \u2b96 draw. Include date, color, opponent, ratings,
  time class, ending, opening (name + ECO code if available), and URL per game.
  If opening is None, just omit it, never invent it.
- End with a one-line overall record like "3 wins, 1 loss, 1 draw".
"""

# Appended to SYSTEM_INSTRUCTION for a chat front end. A model told "reply must
# include every path in `screenshots`" will, quite reasonably, print a Windows
# path like D:\Python\Ai\chess-agent\screenshots\blunder_m14_ply27_...png to a
# user who has no way to open it. On Discord the file is attached for them, so
# the paths are pure noise.
DISCORD_PLATFORM_NOTE = """\
- You are replying in Discord, not a terminal.
  - Screenshots are attached to your message automatically. NEVER print a
    filesystem path, a file name or a "saved to:" line. Describe what each image
    shows instead ("the board after 12...e4, a knight is hanging").
  - Do not tell the user to run a command on the owner's machine. If something
    needs fixing, say what broke and that the owner has been told.
  - Markdown tables do not render in Discord. Use one item per line with an
    emoji, not a pipe table.
  - Keep it short. Discord is a chat, not a report.
"""


def system_instruction(platform_note: str = "") -> str:
    """The full system instruction, plus whatever the front end needs to add."""
    return SYSTEM_INSTRUCTION + platform_note


# Map tool names (as seen by the model) to the real Python functions.
#
# Seven tools, so each kind of question has one obvious home. Folding
# "screenshot any move", "list every move" and "what did the coach say" into
# one review tool with a growing pile of optional parameters gave the model
# nothing in the tool name to steer by, and it guessed.
TOOLS = [
    get_recent_games,
    get_player_profile,
    review_game_screenshots,
    get_review_state,
    get_moves,
    get_coach_commentary,
    screenshot_game_move,
    *INTERACTIVE_TOOLS,
]
# Derived from the list, so a tool can never be registered under a name the
# model cannot call.
FUNCTION_MAP = {f.__name__: f for f in TOOLS}

# The subset that drives the shared browser. Everything else is an HTTP call to
# chess.com's public API with its own 15s timeout, and nothing about it is
# shared with a Chromium profile or `storage_state.json`.
#
# This is what lets a front end serialise the browser without serialising
# everything: `get_recent_games` for one user must not wait two minutes behind
# another user's review. A frozenset is right here -- unlike IMAGE_SUFFIXES
# below, this is never handed to `str.endswith`.
BROWSER_TOOLS = frozenset(
    {
        "review_game_screenshots",
        "screenshot_game_move",
        "get_moves",
        "get_review_state",
        "get_coach_commentary",
        # The interactive tools. Each one drives the shared browser session, so
        # each takes the gate -- but note what the gate does and does not now
        # cover. It serialises the CALL, so no interactive operation ever runs
        # beside a review. It does not hold across calls: the session's browser
        # deliberately outlives the gate, which is the whole point, and the
        # session serialises its own operations internally instead.
        "open_chess_page",
        "where_am_i",
        "move_board",
        "goto_move_in_game",
        "take_screenshot",
        "close_browser_session",
    }
)

# Extensions worth attaching to a message. The review pipeline only ever writes
# PNG, but a future tool might not, and a stray .json is not something to post
# in a channel.
#
# A TUPLE, not a set: `str.endswith` accepts a tuple or a single string, and
# raises `TypeError: endswith first arg must be str or a tuple of str` on a
# set or frozenset. As a frozenset this raised on the first string it saw in
# every tool result, which is every result -- so every screenshot would have
# failed to attach. Caught by tests/test_discord_bot.py, which is the only
# reason it was caught before shipping.
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".gif")

# Discord allows 10 files per message; the CLI has no limit. Cap the walk so a
# tool that returned a whole directory listing cannot try to post 400 images.
MAX_ARTIFACTS = 20

# MAX_ARTIFACTS is per tool CALL, and one turn can make several calls, so on its
# own it does not bound a turn: a six-round turn could try to upload 120 files
# in a dozen messages. This is the per-turn ceiling that actually does, and
# `run_agent` stops collecting once it is reached.
#
# 40 was two Discord messages' worth at the 10-file limit, sized for one
# review. An interactive turn can ask for a frame per position, and the user's
# own session is unbounded, so it is raised. The byte budget in discord_bot is
# what actually stops a flood -- it batches uploads by size as well as count --
# and that is the honest limit to lean on.
MAX_ARTIFACTS_PER_TURN = int(os.getenv("AGENT_MAX_ARTIFACTS", "80"))


class AgentTurnFailed(Exception):
    """A turn died, carrying whatever it had already collected.

    `run_agent` needs to re-raise -- `main.py` reports the error itself -- but a
    plain re-raise throws away the tools already run and the screenshots already
    written. A review that fails on round 4 of 6 has three good frames on disk,
    and the user should get them along with the error, not an error and nothing
    else. This is a plain Exception subclass, so every existing `except
    Exception` around a call still catches it.
    """

    def __init__(
        self,
        message: str,
        tools_used: list[str] | None = None,
        artifacts: list[tuple[str, Path]] | None = None,
    ) -> None:
        super().__init__(message)
        self.tools_used = tools_used or []
        self.artifacts = artifacts or []


class ToolGate:
    """Hold a shared resource for the duration of one tool call.

    `run_agent` calls `acquire` immediately before a tool runs and `release`
    immediately after -- both from the worker thread the agent runs on, so an
    implementation is ALLOWED to block in `acquire`. That is the whole point:
    the Discord bot takes its `asyncio.Lock` here by way of
    `run_coroutine_threadsafe`, so a second review waits in line while the
    thread that would run it parks, instead of two Chromium instances fighting
    over one profile and one `storage_state.json`.

    `acquire`/`release` are only ever called for a tool that exists, so a gate
    never has to decide what to do about a hallucinated tool name.
    """

    def acquire(self, name: str, args: dict) -> None:
        """Called before the tool runs. May block until the resource is free."""

    def release(self, name: str) -> None:
        """Called after the tool returns, including when it raised."""


class MissingSetting(Exception):
    """A required setting is unset, or is still the `.env.example` placeholder.

    An exception rather than an exit, because a library function that calls
    `sys.exit` can only be used by a program. The Discord bot builds its client
    on the first message, from inside a coroutine on the event loop, where
    `SystemExit` is a `BaseException` that no `except Exception` catches -- it
    would tear the process down instead of posting a sentence. `get_api_key`
    below wraps this for the CLI, which does want to stop.
    """

    def __init__(self, name: str, detail: str) -> None:
        super().__init__(detail)
        self.name = name
        self.detail = detail


def require_api_key() -> str:
    """Return GEMINI_API_KEY, or raise `MissingSetting`.

    The old version silently fell back to a parent-directory `.env`, which is how
    the key ended up living one directory away from everything that documents
    it. `project_env` now owns that decision and the error message says which
    files were searched, because "Missing GEMINI_API_KEY" alone leaves you
    guessing whether the file is in the wrong place or the key is in the wrong
    file.
    """
    key = os.getenv("GEMINI_API_KEY", "").strip().strip('"').strip("'")
    # `setting_state` is the one place that knows what a placeholder looks
    # like; this function used to re-implement the check and drift from it.
    state = project_env.setting_state("GEMINI_API_KEY")
    if state == "unset":
        raise MissingSetting(
            "GEMINI_API_KEY",
            "Missing GEMINI_API_KEY.\n"
            "\n"
            "1. Get a key from Google AI Studio: "
            "https://aistudio.google.com/apikey\n"
            f"2. Copy {project_env.EXAMPLE_ENV.name} to "
            f"{project_env.PROJECT_ENV.name}\n"
            "3. Put your key in it as: GEMINI_API_KEY=your_key_here\n"
            f"\n"
            f"  {project_env.describe_sources()}\n"
            "  Run 'python check_env.py' to see every setting and where it "
            "comes from.",
        )
    if state == "placeholder":
        raise MissingSetting(
            "GEMINI_API_KEY",
            "GEMINI_API_KEY is still the placeholder from .env.example.\n"
            f"  {project_env.PROJECT_ENV} -- replace the value with your real key.",
        )
    return key


def get_api_key() -> str:
    """`require_api_key`, but print the reason and exit -- for a CLI entry point.

    Kept because `main.py` and `check_env.py` want to stop, and because
    `discord_bot.main` validates at startup the same way. The bot's *runtime*
    path uses `require_api_key` so a bad key becomes a message rather than a
    dead process.
    """
    try:
        return require_api_key()
    except MissingSetting as exc:
        print(f"{exc.detail}\n")
        sys.exit(1)


def model_error_hint(text: str) -> str:
    """One line to add to a model failure, or "" if the text is not about a model.

    `main.py` has always said "try GEMINI_MODEL=..." when a turn fails; the
    Discord path returned a bare 404 to the least technical person who can reach
    it. A retired or not-yet-enabled model is the single most common cause, and
    the fix is one line in `.env`, so it is worth saying out loud.
    """
    lowered = (text or "").lower()
    if not any(
        token in lowered
        for token in ("404", "not found", "no longer available", "is not supported")
    ):
        return ""
    return (
        f"\n\n_(the model `{MODEL}` may be retired or not enabled for this key -- "
        f"the owner has been told to set `GEMINI_MODEL={MODEL_FALLBACK_HINT}` "
        f"in `{project_env.PROJECT_ENV.name}` and restart.)_"
    )


def create_chat(client: genai.Client, platform_note: str = ""):
    """Create a chat session with tools and disabled auto-calling.

    We disable automatic function calling so we can log each tool call
    and run the Python functions ourselves (better for learning).
    """
    config = types.GenerateContentConfig(
        system_instruction=system_instruction(platform_note),
        tools=TOOLS,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    return client.chats.create(model=MODEL, config=config)


def collect_artifacts(
    result: object,
    tool: str = "",
    seen: set[str] | None = None,
) -> list[tuple[str, Path]]:
    """Find screenshot paths anywhere in a tool's return value.

    Walked generically rather than read off known keys, because the shapes
    differ -- `review_game_screenshots` returns a dict of dicts keyed by shot
    name, `screenshot_game_move` returns one nested under `move` -- and a
    third tool will differ again. A key that stops existing should mean "no
    pictures this time", not a KeyError in a chat bot.

    Non-existent paths are dropped rather than reported: a tool that failed
    halfway may name a file it never wrote, and posting it would be a 404 in a
    channel.

    Dropping is only safe while the tool is honest about what it wrote.
    `take_screenshot` therefore returns the path it actually wrote rather than
    one it intends to, and this function verifies the file exists -- so a tool
    that names a missing file loses the picture here instead of raising at the
    front end.

    `seen` is the caller's set of paths already collected this TURN, so the
    same frame reached by two different tool calls is attached once. Passing
    your own set is what makes that work; leaving it None keeps a single call
    self-contained.
    """
    found: list[tuple[str, Path]] = []
    already = set() if seen is None else seen

    def walk(node: object, depth: int = 0) -> None:
        if len(found) >= MAX_ARTIFACTS or depth > 6:
            return
        if isinstance(node, dict):
            for value in node.values():
                walk(value, depth + 1)
        elif isinstance(node, (list, tuple)):
            for item in node:
                walk(item, depth + 1)
        elif isinstance(node, str) and node.lower().endswith(IMAGE_SUFFIXES):
            if node in already:
                return
            already.add(node)
            try:
                path = Path(node)
            except (TypeError, ValueError):  # pragma: no cover - defensive
                return
            if path.is_file():
                found.append((tool, path))

    walk(result)
    return found


def _call_tool(func, args: dict) -> dict:
    """Run one tool and turn any failure into a dict the model can be told.

    The one subtlety is telling a bad ARGUMENT apart from a tool that crashed.
    Both are `TypeError`, and the old code caught `TypeError` first -- so a
    `TypeError` raised deep inside `review_game_screenshots` was reported to the
    model as "Bad tool arguments: ..." AND, because that branch has no
    `exc_info`, never logged. The real traceback was gone and the message
    pointed at the wrong thing. Binding the signature first separates them
    properly: `Signature.bind` fails for a missing, unexpected or misspelled
    keyword and nothing else.
    """
    try:
        inspect.signature(func).bind(**args)
    except TypeError as e:
        return {"error": f"Bad tool arguments: {e}"}
    try:
        return func(**args)
    except Exception as e:  # noqa: BLE001 - tool boundary: never crash the chat
        _LOG.warning("tool %s failed", func.__name__, exc_info=True)
        return {"error": f"Tool failed: {e}"}


@dataclass
class AgentResult:
    """One answered turn.

    `artifacts` is what makes a screenshot tool useful in a chat: the front end
    attaches these files, so the answer never has to carry a path.
    """

    text: str
    tools_used: list[str] = field(default_factory=list)
    artifacts: list[tuple[str, Path]] = field(default_factory=list)
    error: str | None = None
    history_reset: bool = False


def run_agent(
    chat,
    user_message: str,
    on_tool=None,
    gate: ToolGate | None = None,
) -> tuple[str, list[str], list[tuple[str, Path]]]:
    """Send one user message through the agent loop.

    Returns ``(answer, tools_used, artifacts)``.

    `on_tool(name, args)` is called just before each tool runs. It is called
    from whatever thread this runs on, so a front end that wants to update a
    message has to hop back to its own event loop itself -- and it must not
    raise. It runs OUTSIDE the tool try/except on purpose, because a progress
    callback that dies is not a reason to throw away a turn the model already
    answered.

    `gate` is an optional `ToolGate` held around each tool call. Only tools that
    exist are gated, so a front end can serialise the browser (`BROWSER_TOOLS`)
    without serialising the HTTP tools as well.

    Raises `AgentTurnFailed` -- which still carries `tools_used` and
    `artifacts` -- if the model call itself fails, so a caller can report the
    error without discarding screenshots already written to disk.
    """
    tools_used: list[str] = []
    artifacts: list[tuple[str, Path]] = []
    # Shared across every call in this turn, so one frame reached by two tool
    # calls is attached once and the per-turn ceiling is a real ceiling.
    seen_paths: set[str] = set()

    try:
        response = chat.send_message(user_message)
        rounds = 0

        # While the model wants tools, run them and send results back. Most
        # turns need 0 or 1 rounds; MAX_TOOL_ROUNDS stops a model that keeps
        # re-requesting a tool from looping forever and spending forever.
        while response.function_calls and rounds < MAX_TOOL_ROUNDS:
            rounds += 1
            # Build all function responses first (model may request several at
            # once).
            response_parts = []
            for fc in response.function_calls:
                args = fc.args or {}
                _LOG.info("tool %s %s", fc.name, args)

                func = FUNCTION_MAP.get(fc.name)
                if func is None:
                    result = {"error": f"Unknown tool '{fc.name}'."}
                else:
                    # After the lookup, not before: a model that hallucinates
                    # `delete_everything` should not make a front end announce
                    # that it is running `delete_everything`.
                    if on_tool is not None:
                        try:
                            on_tool(fc.name, args)
                        except Exception:  # noqa: BLE001
                            # Progress reporting is best-effort. Swallowing it
                            # here is the difference between "the status line
                            # glitched" and "the whole turn reported a model
                            # error", which is what an exception thrown by a
                            # loop-bound callback used to do.
                            _LOG.debug("on_tool callback failed", exc_info=True)
                    if gate is not None:
                        gate.acquire(fc.name, args)
                    try:
                        result = _call_tool(func, args)
                    finally:
                        # `finally`, not `else`: a gate released only on success
                        # deadlocks every later request behind whatever the tool
                        # raised.
                        if gate is not None:
                            gate.release(fc.name)

                # Deliberately recorded for unknown and failed tools too: this
                # is "what the model asked for", which is the only place a
                # hallucinated tool name becomes visible.
                tools_used.append(fc.name)
                if len(artifacts) < MAX_ARTIFACTS_PER_TURN:
                    artifacts.extend(
                        collect_artifacts(result, fc.name, seen=seen_paths)
                    )
                response_parts.append(
                    types.Part.from_function_response(
                        name=fc.name, response={"result": result}
                    )
                )

            # Tool results -> model -> final answer (or another tool call).
            response = chat.send_message(response_parts)
    except Exception as e:  # turn boundary: re-raise, carrying what we collected
        raise AgentTurnFailed(str(e), tools_used, artifacts) from e

    if response.function_calls:
        # The cap fired. Say so. Returning the model's in-flight text on its own
        # reads as a finished answer, and a user who asked for three things and
        # got one has no way to tell that anything was dropped -- the same class
        # of problem as a screenshot that silently omits the blunder.
        notice = (
            f"_(stopped after {MAX_TOOL_ROUNDS} tool rounds without a final "
            f"answer -- I asked for more tools than the limit allows. I may not "
            f"have finished the question.)_"
        )
        partial = (response.text or "").strip()
        return (
            f"{partial}\n\n{notice}" if partial else notice,
            tools_used,
            artifacts,
        )
    return response.text or "(no answer from model)", tools_used, artifacts


class AgentSession:
    """One conversation: a Gemini chat, a turn count, and a reset policy.

    Why a class and not a bare `chat`: the Discord bot keeps one of these per
    (guild, channel, user) so "what about the opponent in game 2?" has something
    to refer to, and it has to survive a long-lived process. A Gemini chat
    cannot be truncated, so this counts turns and starts a fresh one, reporting
    that it did -- a silent reset would look to the user like the bot forgetting
    mid-sentence.
    """

    def __init__(
        self,
        client: genai.Client,
        platform_note: str = "",
        max_turns: int | None = None,
    ) -> None:
        self._client = client
        self._note = platform_note
        self._max_turns = max_turns if max_turns is not None else MAX_TURNS_PER_CHAT
        self.turns = 0
        self._chat = None

    def _ensure_chat(self):
        if self._chat is None:
            self._chat = create_chat(self._client, self._note)
        return self._chat

    def reset(self) -> None:
        self._chat = None
        self.turns = 0

    def ask(
        self,
        user_message: str,
        on_tool=None,
        gate: ToolGate | None = None,
    ) -> AgentResult:
        """One turn. Never raises for a model or tool failure.

        A front end wants a sentence it can post, not an exception it has to
        turn into one, and a tool that already returns `{"error": ...}` has done
        the reporting.

        A turn that dies part way through still returns the screenshots it had
        already produced: a review that fails on round 4 of 6 has three good
        frames on disk, and throwing them away would leave the user with an
        error and nothing else.
        """
        reset = self.turns >= self._max_turns
        if reset:
            _LOG.info("chat hit %d turns; starting a fresh one", self.turns)
            self._chat = None
        try:
            chat = self._ensure_chat()
        except Exception as e:  # noqa: BLE001 - chat boundary: answer, never crash
            # A bad model name or a rejected key fails HERE, on the very first
            # turn, so this is the path a Discord user actually meets.
            _LOG.warning("could not start a chat", exc_info=True)
            return self._failed(
                f"Something went wrong talking to the model: {e}", e, reset
            )
        self.turns += 1
        try:
            text, tools, artifacts = run_agent(chat, user_message, on_tool, gate)
        except AgentTurnFailed as e:
            _LOG.warning("agent turn failed", exc_info=True)
            body = f"Something went wrong talking to the model: {e}"
            return AgentResult(
                text=body + model_error_hint(body),
                tools_used=e.tools_used,
                artifacts=e.artifacts,
                error=str(e),
                history_reset=reset,
            )
        except Exception as e:  # noqa: BLE001 - chat boundary: answer, never crash
            _LOG.warning("agent turn failed", exc_info=True)
            return self._failed(
                f"Something went wrong talking to the model: {e}", e, reset
            )
        if reset:
            text = (
                f"_(starting a fresh conversation -- this one had reached "
                f"{self._max_turns} turns)_\n\n{text}"
            )
        return AgentResult(
            text=text, tools_used=tools, artifacts=artifacts, history_reset=reset
        )

    def _failed(self, body: str, error: Exception, reset: bool) -> AgentResult:
        """A postable error, plus the model hint when the model is the problem."""
        return AgentResult(
            text=body + model_error_hint(body),
            error=str(error),
            history_reset=reset,
        )
