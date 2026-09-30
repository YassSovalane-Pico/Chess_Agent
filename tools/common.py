"""Shared constants and small helpers for the chess.com tools."""

import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Bumped whenever the review flow changes shape. calibrate_review.py prints
# this on every run so it is always clear which build produced an output.
__version__ = "0.3.1"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Written by login_chess.py, read by tools/review.py. Keeps the User-Agent used
# when the session cookie was created identical to the one used afterwards; a
# mismatch is something Cloudflare scores.
BROWSER_PROFILE_PATH = PROJECT_ROOT / ".browser_profile.json"

# storage_state.json holds a live chess.com ACCESS_TOKEN + PHPSESSID, so anyone
# holding the file has the account. A git-ignore is thin protection for a
# folder that also gets zipped and backed up, hence the override.
STATE_PATH_ENV = "CHESS_AGENT_STATE"
DEFAULT_STATE_PATH = PROJECT_ROOT / "storage_state.json"


def session_state_path() -> Path:
    """Where the logged-in chess.com session lives.

    Override with ``CHESS_AGENT_STATE``. Returns the historical project-root
    location when the variable is unset, so an existing setup keeps working.
    """
    override = (os.getenv(STATE_PATH_ENV) or "").strip().strip('"').strip("'")
    return Path(override) if override else DEFAULT_STATE_PATH


# How long to keep stepping a colliding stamp before giving up. One step per
# second, so this is a minute of frames written inside a single second, which is
# far past anything a real turn produces -- it exists only so a directory full
# of past screenshots cannot turn this into an unbounded loop.
SHOT_NAME_ATTEMPTS = 60


def unique_shot_path(directory: Path, stem: str, stamp: str) -> Path:
    """A screenshot path in `directory` that no existing file occupies.

    The second-resolution stamp is what made this necessary. A single turn asks
    for several pictures in a row -- "screenshots of move 12, 14 and 16" -- and
    every one of them was written in the same second to
    ``shot_auto_<stamp>.png``. The second call therefore OVERWROTE the first
    file, and because `agent.collect_artifacts` de-duplicates by path the chat
    front end then had one picture to attach for three requests: the frames
    were taken, on disk, and never sent. The tool has no way to tell the user
    that, because from its point of view it succeeded twice.

    The stamp is advanced by whole seconds rather than a counter appended to the
    name, because the filename shape is load-bearing:
    `discord_bot.describe_artifact` reads `shot_<region>[_<label>]_<stamp>.png`
    with a regex anchored on `\\d{8}_\\d{6}$` to work out what a frame shows. A
    free-standing suffix would leave every interactive screenshot captioned
    "screenshot".

    A stamp that is not a timestamp (a test, an old caller) cannot be advanced,
    so it gets a numeric suffix instead. The path is returned, not written: the
    caller writes it, exactly as before, and should do so before the next call.
    That is not a lock, and does not need to be one -- the Discord front end
    gates every screenshot tool behind the browser lock, and the CLI is a
    single-threaded loop.
    """
    directory.mkdir(parents=True, exist_ok=True)
    try:
        # Every caller stamps in UTC (`datetime.now(tz=timezone.utc)`), so the
        # parsed value is made aware rather than left naive.
        moment = datetime.strptime(stamp, "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        moment = None

    for attempt in range(1, SHOT_NAME_ATTEMPTS + 1):
        if moment is not None:
            candidate = directory / f"{stem}_{moment.strftime('%Y%m%d_%H%M%S')}.png"
        elif attempt == 1:
            candidate = directory / f"{stem}_{stamp}.png"
        else:
            candidate = directory / f"{stem}_{stamp}_{attempt}.png"
        if not candidate.exists():
            return candidate
        if moment is not None:
            moment += timedelta(seconds=1)
    # Every candidate is taken; hand back the last one rather than spin. The
    # write will overwrite a frame from an earlier run, which is the behaviour
    # this whole function exists to improve on, and it is better than a hang.
    return candidate


# chess.com requires a User-Agent header, otherwise it may block the request.
HEADERS = {"User-Agent": "chess-agent/1.0 (beginner learning project)"}

# Mapping of chess.com result codes to win/loss/draw from the user's
# perspective. See https://www.chess.com/news/view/published-data-api
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
    # Unknown code: return the raw code so the model can show it.
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

    Controlled by ``CHESS_AGENT_LOG`` (debug/info/warning/error), because the
    review pipeline swallows a lot of recoverable failures on purpose -- the
    logs are the only way to see *why* something degraded.

    Reconfigures the handlers already on the root logger rather than calling
    ``logging.basicConfig``: that is a documented no-op once a handler exists,
    so a second caller would silently inherit the first caller's settings.
    Passing ``fmt`` here is the way to actually mean it.
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
