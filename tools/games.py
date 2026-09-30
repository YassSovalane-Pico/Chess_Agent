"""Game-history tools (chess.com PubAPI, read-only)."""

import logging
import re

import requests

from .common import ENDING_LABELS, HEADERS, _format_timestamp, _map_result

_LOG = logging.getLogger(__name__)


def _json_object(resp) -> dict | None:
    """Parse a chess.com response body as a JSON object, else None.

    ``.get`` on a JSON *array* raises AttributeError, which the callers' bare
    ``except ValueError`` never caught -- the error then escaped the tool as
    "Tool failed: 'list' object has no attribute 'get'" instead of a message
    about a bad response.
    """
    try:
        data = resp.json()
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def _parse_opening_name(eco_url: str | None) -> str | None:
    """Extract a human-readable opening name from a chess.com ECO URL.

    Example:
        "https://www.chess.com/openings/Alapin-Sicilian-Defense-2...Qa5"
        -> "Alapin Sicilian Defense 2...Qa5"
    Returns None if the URL is missing.
    """
    if not eco_url:
        return None
    # Take the last part after "/openings/".
    slug = eco_url.rstrip("/").split("/")[-1]
    if not slug:
        return None
    # "Sicilian-Defense" -> "Sicilian Defense". Keep dots like "2...Qa5".
    name = slug.replace("-", " ").replace("_", " ").strip()
    return name or None


def _parse_eco_code(pgn: str | None) -> str | None:
    """Extract the ECO code (e.g. "B22", "C50") from a PGN header.

    The PGN contains a line like: [ECO "B22"].
    Returns None if not found.
    """
    if not pgn:
        return None
    m = re.search(r'\[ECO\s+"([^"]+)"\]', pgn)
    return m.group(1).strip() if m else None


def _parse_game(game: dict, username: str) -> dict | None:
    """Extract the fields we care about from one raw chess.com game.

    Args:
        game: One game dict from the chess.com API.
        username: The requested player (any casing).

    Returns:
        A small dict with date, time control, colors, ratings, result,
        opening (name/ECO code/URL), etc. -- or None when the requested player
        is not in this game at all.

    The old version assumed "if not white, then black", so a game the player
        was not part of was reported as a loss for Black against a fabricated
        opponent. Returning None lets the caller count and report the mismatch
        instead of inventing it.
    """
    wanted = username.lower()
    white = game.get("white", {})
    black = game.get("black", {})

    # Decide which color our user played (case-insensitive compare).
    if white.get("username", "").lower() == wanted:
        color = "white"
        me = white
        opponent = black
    elif black.get("username", "").lower() == wanted:
        color = "black"
        me = black
        opponent = white
    else:
        _LOG.info(
            "game %s has neither player named %r (%r / %r); skipping it",
            game.get("url"),
            username,
            white.get("username"),
            black.get("username"),
        )
        return None

    my_result_code = str(me.get("result", "unknown"))
    end_time = game.get("end_time")

    # Opening info: chess.com gives an "eco" URL like
    # "https://www.chess.com/openings/Sicilian-Defense..."
    # and the PGN header has [ECO "B22"].
    eco_url = game.get("eco")
    pgn = game.get("pgn", "")
    return {
        # None rather than a 1970 date when the field is absent.
        "date": _format_timestamp(end_time) if end_time else None,
        "time_control": game.get("time_control"),
        "time_class": game.get("time_class"),
        "color": color,
        "opponent": opponent.get("username"),
        "opponent_rating": opponent.get("rating"),
        "user_rating": me.get("rating"),
        "result": _map_result(my_result_code),
        "ending": ENDING_LABELS.get(my_result_code.lower(), my_result_code),
        "opening": _parse_opening_name(eco_url),
        "eco": _parse_eco_code(pgn),
        "eco_url": eco_url,
        "url": game.get("url"),
    }


def get_recent_games(username: str, limit: int = 5) -> dict:
    """Get the most recent chess.com games for a player.

    Use this when the user asks about recent games, last games, results,
    wins/losses, form, or openings of a chess.com username.

    Args:
        username: The chess.com username (e.g. "hikaru"). Case-insensitive.
        limit: How many recent games to return. Defaults to 5, capped at 20.

    Returns:
        A dict like {"username": ..., "games": [...], "count": n} with the
        newest game first. Each game has: date, time_control, time_class
        (rapid/blitz/bullet/daily), color ("white" or "black"), opponent,
        opponent_rating, user_rating, result ("win", "loss" or "draw"), ending
        (checkmate, resignation, timeout, stalemate, agreement, etc.), opening
        (human readable name, may be None), eco (code like "B22", may be None),
        eco_url and url. If any game did not list the requested player it is
        skipped and reported as skipped_unmatched.
        On failure returns {"error": "human readable message"}.
    """
    username = username.strip()
    if not username:
        return {"error": "Please provide a chess.com username."}
    if limit <= 0:
        limit = 5
    # Safety cap so one request cannot fetch a huge history.
    limit = min(limit, 20)

    try:
        # Step 1: get the list of monthly archive URLs (oldest -> newest).
        archives_resp = requests.get(
            f"https://api.chess.com/pub/player/{username}/games/archives",
            headers=HEADERS,
            timeout=15,
        )
    except requests.RequestException as e:
        return {"error": f"Network error contacting chess.com: {e}"}

    if archives_resp.status_code == 404:
        return {"error": f"Player '{username}' not found on chess.com."}
    if archives_resp.status_code == 429:
        return {
            "error": "Rate limited by chess.com (429). Please wait a bit and try again."
        }
    if archives_resp.status_code != 200:
        return {
            "error": f"chess.com returned HTTP {archives_resp.status_code} for archives."
        }

    archives = _json_object(archives_resp)
    if archives is None:
        return {"error": "Could not parse chess.com response."}
    archive_urls = archives.get("archives")
    if not isinstance(archive_urls, list) or not archive_urls:
        return {"error": f"No games found for '{username}'."}

    # Step 2: walk backwards from the newest month until we have `limit` games.
    collected: list[dict] = []
    for archive_url in reversed(archive_urls):
        if len(collected) >= limit:
            break
        try:
            r = requests.get(archive_url, headers=HEADERS, timeout=15)
        except requests.RequestException:
            continue  # skip a failed month, try the previous one
        if r.status_code == 429:
            return {
                "error": "Rate limited by chess.com (429). Please wait a bit and try again."
            }
        if r.status_code != 200:
            continue
        month = _json_object(r)
        if month is None:
            continue
        month_games = month.get("games")
        if not isinstance(month_games, list):
            continue
        # Newest game in the month first.
        month_games.sort(key=lambda g: g.get("end_time", 0), reverse=True)
        for g in month_games:
            if not isinstance(g, dict):
                continue
            collected.append(g)
            if len(collected) >= limit:
                break

    if not collected:
        return {"error": f"No games found for '{username}'."}

    # Step 3: sort all collected games newest -> oldest, keep only `limit`.
    collected.sort(key=lambda g: g.get("end_time", 0), reverse=True)
    collected = collected[:limit]

    parsed = [_parse_game(g, username) for g in collected]
    games = [g for g in parsed if g is not None]
    if not games:
        return {
            "error": f"chess.com returned {len(parsed)} game(s) for '{username}' "
            "but none of them list that player."
        }
    result = {"username": username, "games": games, "count": len(games)}
    if len(games) < len(parsed):
        # Say so rather than quietly returning a short list.
        result["skipped_unmatched"] = len(parsed) - len(games)
    return result
