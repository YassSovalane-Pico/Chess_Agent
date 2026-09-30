"""Interactive chess.com tools: open a page, walk the moves, take a picture.

These are the tools that make the agent a *browser* rather than a reviewer.
Before them, every browser tool took a game link and did one fixed thing with
it. These take a URL or a move and act on whatever is currently open, so a
conversation can go:

    open the review of game X  ->  go forward 3 moves  ->  screenshot that
    ->  what does the engine think of this position  ->  step back 2

They run against the shared session in `tools/session.py`, which is why they
can be several calls long: the browser outlives each call. Everything they do
is still fenced by that session's navigation allow-list, so "go wherever you
like" means board/review/lesson/puzzle pages -- not settings or logout.

Every function here is a *synchronous* entry point returning a JSON-safe dict,
like every other tool the agent calls. The async work is submitted to the
session's private event loop inside, which is what `spikes/s0_1_loop_affinity.py`
established is the only way a page can outlive a tool call.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from . import move_list
from .session import (
    SessionClosed,
    close_session,
    current_page,
    get_session,
    url_allowed,
)

_LOG = logging.getLogger(__name__)

# Screenshot regions. `auto` is the existing board+panel crop; the others are
# for "show me the whole page" and "just the board".
SHOT_REGIONS = ("auto", "board", "sidebar", "viewport")

# A single tool call may not write more frames than this. The turn-level cap in
# agent.py is higher; this is the per-call guard so one ask cannot fill the disk
# from a loop.
MAX_SHOTS_PER_CALL = 8


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _shot_dir() -> Path:
    d = _project_root() / "screenshots"
    d.mkdir(exist_ok=True)
    return d


def _stamp() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")


def _error(msg: str) -> dict:
    return {"error": msg}


def _not_logged_in() -> dict:
    return _error(
        "Not logged in. Run 'python login_chess.py' once to save your chess.com "
        "session, then ask again."
    )


# ---------------------------------------------------------------------------
# Where am I?
# ---------------------------------------------------------------------------


async def _describe_page() -> dict:
    """A small picture of the page: its title, and whether a board is up.

    Deliberately short. The model cannot see the screenshot, so it needs facts
    to talk about; a huge DOM dump would just be tokens it cannot use.
    """
    page = current_page()
    return await page.evaluate(
        """() => {
          // `textContent || ''` is not enough: an element with no children
          // yields null from `|| ''` only if it is null itself, and the
          // heading/eval spans can be present-but-empty, which made a live
          // run die with "Cannot read properties of null (reading 'trim')".
          const t = (el) => (el ? String(el.textContent || '').trim() : '');
          const evalEl = document.querySelector('.evaluation-bar-score')
            || document.querySelector('#board-layout-evaluation');
          const heading = document.querySelector('#board-layout-sidebar h1');
          return {
            url: location.href,
            title: document.title,
            has_board: !!document.querySelector('wc-chess-board'),
            has_move_list: document.querySelectorAll('[data-node]').length,
            engine_eval: t(evalEl) || null,
            heading: t(heading) || null,
          };
        }"""
    )


def where_am_i() -> dict:
    """Say what the browser currently has open.

    Use this when the user asks "where are we", "what page is that", or as a
    cheap check before stepping through moves -- it costs one DOM read and no
    navigation. Returns the url, the page title, whether a board and a move
    list are present, the engine evaluation currently on the eval bar, and the
    sidebar heading.

    This is the Gemini tool entry point, so it is deliberately synchronous.

    Args:
        None. Takes no arguments.

    Returns:
        {"url": str, "title": str, "has_board": bool, "has_move_list": int,
        "engine_eval": str|None, "heading": str|None, "session_open": bool}
        On failure returns {"error": "..."}.
    """
    try:
        session = get_session()
    except Exception:  # noqa: BLE001
        return _not_logged_in()
    if not session.is_open:
        # A question must not open a browser. `where_am_i` is what a user calls
        # to find out whether anything is running, and answering it by starting
        # a Chromium -- then reporting "you are on the home page" -- is exactly
        # the wrong answer.
        return {
            "session_open": False,
            "url": None,
            "title": None,
            "note": "No chess.com page is open. Use open_chess_page to start one.",
        }
    try:
        info = session.submit(_describe_page, 90, ensure_open=False)
    except SessionClosed as exc:
        return _error(str(exc))
    except Exception as exc:  # noqa: BLE001 - tool boundary
        _LOG.warning("where_am_i failed", exc_info=True)
        return _error(f"Could not read the page: {exc}")
    info["session_open"] = session.is_open
    return info


# ---------------------------------------------------------------------------
# Go somewhere
# ---------------------------------------------------------------------------

_PAGE_KINDS = (
    ("/review", "game review"),
    ("/analysis", "analysis board"),
    ("/game/", "game page"),
    ("/lessons", "lesson"),
    ("/puzzles", "puzzle"),
    ("/openings", "opening explorer"),
)


def _page_kind(url: str) -> str:
    for needle, label in _PAGE_KINDS:
        if needle in url:
            return label
    return "page"


def open_chess_page(url: str = "", wait: str = "load") -> dict:
    """Open a chess.com board, review, lesson, puzzle or opening page.

    The interactive entry point: use it when the user names a page ("open the
    review", "go to the analysis board", "show me the puzzle") or hands over a
    link. The browser then STAYS on that page, so the next call can act on it
    -- step through moves, screenshot the position, read the engine. That is the
    whole point: one call to get there, several cheap calls once you are.

    Only these are reachable, by design: the analysis board (/analysis), game
    reviews (/analysis/game/.../review), game pages (/game/...), lessons,
    puzzles and the opening explorer. Settings, logout, friends and messages are
    refused -- the agent runs on a real logged-in account and this tool has no
    business changing it.

    Args:
        url: A chess.com URL, with or without https://. Use a full game link to
            jump straight to a game's analysis or review. Leave blank to just
            open the home page, or to re-read where you already are.
        wait: How long to let the page settle: "load" (default), "ready" or
            "networkidle". "ready" is slower but is what you want for a review
            page whose panel is computed server-side.

    Returns:
        {"url": str, "kind": str, "title": str, "has_board": bool,
        "has_move_list": int, "engine_eval": str|None, "session_open": bool,
        "summary": {...}|None, "note": str}
        `summary` is the review panel's accuracy and classification tally, read
        and cached before the review starts -- chess.com removes it afterwards,
        so this is the only chance to capture it. It is null off a review page.
        On failure returns {"error": "..."}, including when the URL is outside
        the allowed set.
    """
    target = (url or "").strip()
    if target:
        allowed, reason = url_allowed(target)
        if not allowed:
            return _error(reason)
    try:
        session = get_session()
    except Exception:  # noqa: BLE001
        return _not_logged_in()
    try:
        if target:
            session.navigate(target, wait=wait or "load")
        elif not session.is_open:
            session.open()
        # Capture the review summary before anything can start the review.
        summary = session.snapshot_review_summary() or None
        info = session.submit(_describe_page, 90)
    except SessionClosed as exc:
        return _error(str(exc))
    except Exception as exc:  # noqa: BLE001 - tool boundary
        _LOG.warning("open_chess_page failed", exc_info=True)
        return _error(f"Could not open that page: {exc}")

    out = {
        "url": info.get("url", session.current_url),
        "kind": _page_kind(info.get("url", "")),
        "title": info.get("title", ""),
        "has_board": info.get("has_board"),
        "has_move_list": info.get("has_move_list"),
        "engine_eval": info.get("engine_eval"),
        "session_open": session.is_open,
    }
    if summary:
        out["summary"] = {
            "white_accuracy": summary.get("white"),
            "black_accuracy": summary.get("black"),
            "move_counts": summary.get("move_counts") or {},
            "move_counts_by_side": summary.get("move_counts_by_side") or {},
            "players": summary.get("players_in_dom_order") or [],
        }
    out["note"] = (
        "The browser stays on this page, so you can now step through the moves, "
        "take a screenshot or read the engine evaluation without reopening it."
    )
    return out


# ---------------------------------------------------------------------------
# Walk the moves
# ---------------------------------------------------------------------------


async def _step_and_describe(count: int) -> dict:
    """Step `count` plies, then describe where the board ended up."""
    page = current_page()
    result = await move_list.step_moves(page, count)
    ply = await move_list.active_ply(page)
    return {
        "requested": result.requested,
        "moved": result.moved,
        "verified": result.verified,
        "method": result.method,
        "reason": result.reason,
        "current_ply": ply,
    }


def move_board(direction: str = "next", plies: int = 1) -> dict:
    """Step forward or backward through the moves on the open page.

    Use it for "go forward", "next move", "step back three", "rewind", "show me
    what happened next" -- once a page with a move list is open. Every step is
    checked against a board fingerprint, so a click that changes nothing is
    reported as a failure rather than as progress. That is why `verified` is in
    the reply: if it is false the board did not actually move and you should say
    so rather than claim you moved.

    The coach walks the game by itself once a review is running, so a step can
    race with chess.com's own playback; `verified: false` with a `reason` is
    what that looks like, and the honest reply is to say the board would not
    move.

    Args:
        direction: "next" (default) to go forward, or "previous" to go back.
        plies: How many half-moves to step. Default 1. Negative values step
            backwards regardless of `direction`.

    Returns:
        {"requested": int, "moved": int, "verified": bool, "method": str,
        "reason": str, "current_ply": int|None, "engine_eval": str|None}
        `verified` false means the board could not be confirmed as changed --
        read `reason` and tell the user rather than claiming success. On
        failure returns {"error": "..."}.
    """
    want = (direction or "next").strip().lower()
    # `plies or 1` would turn an explicit 0 into 1, so 0 slipped through as a
    # real forward step. Read the argument as given and refuse a zero count
    # rather than quietly doing something the caller did not ask for.
    try:
        raw = int(plies) if plies is not None else 1
    except (TypeError, ValueError):
        return _error(f"plies must be a whole number, not {plies!r}.")
    if raw == 0:
        return _error("plies must not be 0 -- pass 1 or more.")
    if want in ("prev", "back", "previous", "backward"):
        count = -abs(raw)
    elif want in ("next", "forward", "fwd"):
        count = abs(raw)
    else:
        return _error(f"direction must be 'next' or 'previous', not {direction!r}.")

    try:
        session = get_session()
    except Exception:  # noqa: BLE001
        return _not_logged_in()
    try:
        info = session.submit(lambda: _step_and_describe(count), 120)
        engine = session.submit(_describe_page, 90)
    except SessionClosed as exc:
        return _error(str(exc))
    except Exception as exc:  # noqa: BLE001 - tool boundary
        _LOG.warning("move_board failed", exc_info=True)
        return _error(f"Could not move the board: {exc}")

    info["engine_eval"] = engine.get("engine_eval")
    info["url"] = engine.get("url")
    if not info.get("verified"):
        info["note"] = (
            "The board could not be confirmed as changed, so treat this as NOT "
            "moved and say so. reason: " + (info.get("reason") or "unknown")
        )
    return info


async def _goto_ply_and_describe(ply: int) -> dict:
    """Jump to one ply and describe the position there."""
    page = current_page()
    rows, diag = await move_list.read_moves(page)
    if not rows:
        # The move list only exists after "Start Review", so this is where a
        # review page is entered rather than merely read.
        rows, diag = await move_list.read_moves_when_ready(page)
    total = diag.get("pliesKnown") or len(rows)
    await move_list.scroll_to_ply(page, ply, total)
    # Re-read so the per-ply tags navigation clicks are current. Two reads on
    # purpose: the first may be an empty list, the second is the one the click
    # depends on.
    rows, diag = await move_list.read_moves(page)
    ok = await move_list.goto_ply(page, ply)
    row = next((r for r in rows if r.get("ply") == ply), None)
    return {
        "moved": ok,
        "ply": ply,
        "current_ply": await move_list.active_ply(page),
        "move": _move_summary(row) if row else None,
        "moves_read": diag.get("coverage") or len(rows),
    }


def goto_move_in_game(ply: int = 0, move_number: int = 0, color: str = "") -> dict:
    """Jump straight to one move in the game currently open.

    Use it when the user names a move out of order -- "go to move 22",
    "show me 14...Nf6" -- rather than stepping there from wherever you are. It
    scrolls the target into the rendered window, clicks it, and verifies the
    board moved, so it works from any point in the game.

    Number the move the way the user did:

    * "the 27th move"  -> ply=27, unambiguous on its own
    * "move 14"        -> move_number=14, and add color, because both sides
                          played a 14th move
    * "14...Nf6"       -> move_number=14, color="black"

    If the move number is ambiguous and no color is given, this returns an
    error naming both sides rather than guessing -- guessing would show the
    wrong position.

    Args:
        ply: The half-move index counting every move in the game, e.g. 27 for
            the 27th move played. An alternative to move_number.
        move_number: The move number chess.com prints, e.g. 14 for "14. O-O".
        color: Which side played it: "white" or "black". Needed whenever both
            sides played a move with the same number, which is most moves.

    Returns:
        {"moved": bool, "already_at": bool, "ply": int, "current_ply": int|None,
        "move": {"ply", "move_number", "color", "san", "classification",
        "confidence"}|None, "moves_read": str}
        `moved` false with `already_at` false means the board would not go
        there -- say so rather than claiming you got there. `already_at` true
        means the page was already on that position, so nothing had to change;
        that is a success, not a failure. On failure returns {"error": "..."},
        including when the move does not exist or is ambiguous.
    """
    if not ply and not move_number:
        return _error("Give either ply (e.g. 27) or move_number (e.g. 14).")
    try:
        session = get_session()
    except Exception:  # noqa: BLE001
        return _not_logged_in()
    try:
        # Resolve against the live move list so an ambiguous move_number is
        # reported with both sides, exactly as the review tools do.
        resolved = session.submit(lambda: _resolve_live(ply, move_number, color), 120)
        if isinstance(resolved, str) and resolved.startswith("!"):
            return _error(resolved[1:])
        result = session.submit(lambda: _goto_ply_and_describe(resolved), 180)
        # `moved` is goto_ply's own verification: it reports whether the board
        # CHANGED. On a review page that has just been started, the board is
        # already sitting at ply 1, so asking for ply 1 is a correct request
        # that provably changes nothing -- and reporting that as a failure
        # would make the tool lie about a position it is already on.
        at = result.get("current_ply")
        result["already_at"] = at is not None and at == resolved
        engine = session.submit(_describe_page, 90)
    except SessionClosed as exc:
        return _error(str(exc))
    except Exception as exc:  # noqa: BLE001 - tool boundary
        _LOG.warning("goto_move_in_game failed", exc_info=True)
        return _error(f"Could not go to that move: {exc}")
    result["engine_eval"] = engine.get("engine_eval")
    if result.get("already_at"):
        result["note"] = (
            "The board was already on that move, so nothing had to change."
        )
    elif not result.get("moved"):
        result["note"] = (
            "The board would not move to that position. Say so rather than "
            "claiming you got there."
        )
    return result


def _move_summary(row: dict) -> dict:
    """The one shape the model sees for a move."""
    return {
        "ply": row.get("ply"),
        "move_number": row.get("number"),
        "color": row.get("color"),
        "san": row.get("san", ""),
        "classification": row.get("classification"),
        "confidence": row.get("confidence"),
        "clock_seconds": row.get("clock_seconds"),
    }


async def _resolve_live(ply: int, move_number: int, color: str) -> int:
    """Return the ply the user meant, or an error string prefixed with '!'.

    Shares `resolve_move` with the review tools so "move 14" is ambiguous in
    exactly the same way, and refused in exactly the same way, everywhere.
    """
    from .review import resolve_move

    page = current_page()
    rows, _ = await move_list.read_moves(page)
    if not rows:
        rows, _ = await move_list.read_moves_when_ready(page)
    if not rows:
        return "!The move list is not on this page yet. Open a game review or an analysis board first."
    matches, err = resolve_move(rows, ply, move_number, color)
    if err:
        return "!" + err
    return matches[0]["ply"]


# ---------------------------------------------------------------------------
# Take a picture
# ---------------------------------------------------------------------------


async def _shoot(region: str, out: Path) -> str:
    """Screenshot the open page into `out`. Returns the path as a string."""
    from .review import content_clip, screenshot_review_area

    page = current_page()
    if region in ("auto", "board", "sidebar"):
        # `content_clip` already honours REVIEW_SHOT_REGION for the auto case;
        # board/sidebar here narrow it further by choosing which boxes to union.
        import os

        previous = os.environ.get("REVIEW_SHOT_REGION")
        os.environ["REVIEW_SHOT_REGION"] = "board" if region == "board" else "sidebar"
        try:
            await screenshot_review_area(page, out, await content_clip(page))
        finally:
            if previous is None:
                os.environ.pop("REVIEW_SHOT_REGION", None)
            else:
                os.environ["REVIEW_SHOT_REGION"] = previous
    else:
        await page.screenshot(path=str(out), full_page=False)
    return str(out)


def take_screenshot(region: str = "auto", label: str = "") -> dict:
    """Photograph whatever is on screen right now.

    Use it whenever the user asks to *see* something -- "screenshot this",
    "show me the board", "what does it look like", "grab a picture of that
    puzzle". It shoots the page that is currently open, so pair it with
    open_chess_page or move_board: navigate first, then photograph.

    Regions:
      auto (default)  the chessboard and the review panel, cropped tight --
                     no site menu, no search box, no profile. Same framing as
                     the review screenshots.
      board           just the board
      sidebar         just the side panel
      viewport        the whole window as-is

    Args:
        region: One of "auto", "board", "sidebar", "viewport". Default "auto".
        label: An optional short name to identify the shot, e.g. "after e5".
            It only affects the file name.

    Returns:
        {"screenshot": "<path>", "region": str, "url": str, "label": str,
        "note": str}
        The screenshot path is a real file on disk. On a chat front end it is
        attached to your reply automatically -- never print the path. On
        failure returns {"error": "..."}.
    """
    want = (region or "auto").strip().lower()
    if want not in SHOT_REGIONS:
        return _error(
            f"region must be one of {', '.join(SHOT_REGIONS)}, not {region!r}."
        )
    try:
        session = get_session()
    except Exception:  # noqa: BLE001
        return _not_logged_in()
    safe_label = "".join(
        ch for ch in (label or "") if ch.isalnum() or ch in ("_", "-")
    )[:40]
    try:
        page_info = session.submit(_describe_page, 90)
        name_parts = ["shot", want]
        if safe_label:
            name_parts.append(safe_label)
        out = _shot_dir() / (f"{'_'.join(name_parts)}_{_stamp()}.png")
        session.submit(lambda: _shoot(want, out), 120)
    except SessionClosed as exc:
        return _error(str(exc))
    except Exception as exc:  # noqa: BLE001 - tool boundary
        _LOG.warning("take_screenshot failed", exc_info=True)
        return _error(f"Could not take a screenshot: {exc}")
    if not out.exists():
        return _error("The screenshot could not be written.")
    return {
        "screenshot": str(out),
        "region": want,
        "url": page_info.get("url", session.current_url),
        "title": page_info.get("title", ""),
        "label": safe_label,
        "note": "The picture is attached to the reply automatically. Do not "
        "print the file path.",
    }


# ---------------------------------------------------------------------------
# Shut up
# ---------------------------------------------------------------------------


def close_browser_session() -> dict:
    """Close the chess.com browser now and release it.

    Use it when the user says they are done -- "close it", "thanks that's all",
    "shut the browser" -- or when you want to guarantee the session is not
    holding a browser. It is polite, not required: the session closes itself
    after a short idle timeout.

    Args:
        None. Takes no arguments.

    Returns:
        {"closed": bool, "was_open": bool, "note": str}
        `was_open` false means nothing was running, which is not an error.
    """
    try:
        was = close_session()
    except Exception as exc:  # noqa: BLE001 - tool boundary
        _LOG.warning("close_browser_session failed", exc_info=True)
        return _error(f"Could not close the browser cleanly: {exc}")
    return {
        "closed": True,
        "was_open": was,
        "note": "The browser is closed. Opening a page again will start a fresh one.",
    }


# Everything the agent module registers, derived so the two cannot disagree.
INTERACTIVE_TOOLS = (
    open_chess_page,
    where_am_i,
    move_board,
    goto_move_in_game,
    take_screenshot,
    close_browser_session,
)
