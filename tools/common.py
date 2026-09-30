"""Shared constants and small helpers for the chess.com tools."""

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Bumped whenever the review flow changes shape. calibrate_review.py prints
# this on every run: a whole debugging session once turned on "which build did
# that output come from?", because a run was started before an edit landed.
__version__ = "0.3.1"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Written by login_chess.py, read by tools/review.py. Keeps the User-Agent used
# when the session cookie was created identical to the one used afterwards.
# A session minted under Chrome 140 and replayed under a hardcoded "Chrome/126"
# is exactly the kind of inconsistency Cloudflare scores.
BROWSER_PROFILE_PATH = PROJECT_ROOT / ".browser_profile.json"

# storage_state.json holds a live chess.com ACCESS_TOKEN + PHPSESSID: anyone
# holding the file has the account. It is git-ignored, but a git-ignore is not
# much defence for a folder that also gets zipped and backed up, so the location
# is overridable. Point CHESS_AGENT_STATE somewhere outside the checkout
# (e.g. %LOCALAPPDATA%\chess-agent\storage_state.json) to keep the credential
# out of the project tree entirely. The default is kept for compatibility.
STATE_PATH_ENV = "CHESS_AGENT_STATE"
DEFAULT_STATE_PATH = PROJECT_ROOT / "storage_state.json"


def session_state_path() -> Path:
    """Where the logged-in chess.com session lives.

    Override with ``CHESS_AGENT_STATE``. Returns the historical project-root
    location when the variable is unset, so an existing setup keeps working.
    """
    override = (os.getenv(STATE_PATH_ENV) or "").strip().strip('"').strip("'")
    return Path(override) if override else DEFAULT_STATE_PATH


# chess.com requires a User-Agent header, otherwise it may block the request.
HEADERS = {"User-Agent": "chess-agent/1.0 (beginner learning project)"}

# Mapping of chess.com result codes to win/loss/draw from the user's perspective.
# See: https://www.chess.com/news/view/published-data-api
WIN_CODES = {"win"}
LOSS_CODES = {"checkmated", "resigned", "timeout", "abandoned"}
DRAW_CODES = {
    "agreed",
    "stalemate",
    "repetition",
    "insufficient",
    "timevsinsufficient",
    "50move",
}

# Human-readable labels for how the game ended.
ENDING_LABELS = {
    "win": "win",
    "checkmated": "checkmate",
    "resigned": "resignation",
    "timeout": "timeout",
    "abandoned": "abandoned game",
    "agreed": "agreement",
    "stalemate": "stalemate",
    "repetition": "repetition",
    "insufficient": "insufficient material",
    "timevsinsufficient": "time vs insufficient material",
    "50move": "50-move rule",
}


def _format_timestamp(ts: int) -> str:
    """Convert a unix timestamp to a readable UTC date string."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _format_join_date(ts: int) -> str:
    """Convert a unix timestamp to a short YYYY-MM-DD date."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


def _map_result(result_code: str) -> str:
    """Map a chess.com result code to 'win', 'loss' or 'draw'."""
    code = (result_code or "").lower()
    if code in WIN_CODES:
        return "win"
    if code in LOSS_CODES:
        return "loss"
    if code in DRAW_CODES:
        return "draw"
    # Unknown code: be safe and return the raw code so the model can show it.
    return code or "unknown"


def load_browser_profile() -> dict:
    """Read the fingerprint recorded at login time. Empty dict if absent."""
    try:
        with BROWSER_PROFILE_PATH.open(encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_browser_profile(**fields) -> None:
    """Merge ``fields`` into the on-disk fingerprint (best effort)."""
    profile = load_browser_profile()
    profile.update({k: v for k, v in fields.items() if v is not None})
    try:
        BROWSER_PROFILE_PATH.write_text(
            json.dumps(profile, indent=2, sort_keys=True), encoding="utf-8"
        )
    except OSError:
        # A read-only checkout must not break the login flow.
        pass


def configure_logging(level: str | None = None, fmt: str | None = None) -> None:
    """Send module logs somewhere visible.

    Controlled by ``CHESS_AGENT_LOG`` (debug/info/warning/error) because the
    review pipeline swallows a lot of recoverable failures on purpose -- the
    logs are the only way to see *why* something degraded.

    `fmt` replaces the formatter on the handler that is already installed.
    Without it a second caller's `logging.basicConfig` is a documented no-op --
    `basicConfig` does nothing once the root logger has a handler -- so a caller
    that asked for timestamps or a different level silently got the first
    caller's settings instead. Passing `fmt` here is the way to actually mean
    it; setting the level works either way.
    """
    resolved = (level or os.getenv("CHESS_AGENT_LOG", "warning")).lower()
    numeric = getattr(logging, resolved.upper(), None)
    if not isinstance(numeric, int):
        numeric = logging.WARNING
    root = logging.getLogger()
    root.setLevel(numeric)
    if root.handlers:
        if fmt:
            formatter = logging.Formatter(fmt)
            for handler in root.handlers:
                handler.setFormatter(formatter)
        return
    logging.basicConfig(
        level=numeric,
        format=fmt or "%(levelname)-7s %(name)s: %(message)s",
        stream=sys.stderr,
    )
