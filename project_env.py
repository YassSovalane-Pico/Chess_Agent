"""Where configuration comes from, and how to see what is actually set.

This module exists because of a mistake worth naming. `get_api_key()` used to
load the project `.env` and then, *silently*, fall back to one in the parent
directory as a convenience. The result was that the Gemini key lived in
`D:\\Python\\Ai\\.env` while everything else expected
`D:\\Python\\Ai\\chess-agent\\.env`, and nobody -- including the error messages
-- could say where the setting had come from. A "Missing X" message that does
not name the file it looked in is close to useless, and the silent fallback is
what made two sources of truth possible in the first place.

So: one place resolves the env files, everything that needs a setting asks
here, and nothing falls back without saying so.

Rules, in order:

  1. `chess-agent/.env`        -- where it belongs, and the only file `.env.example`
                                  documents
  2. `chess-agent/.env.local`  -- per-machine overrides, git-ignored, loaded second
  3. the process environment  -- wins outright, so CI and one-off runs work
  4. a parent `.env`           -- accepted ONLY if the project has no `.env` at all,
                                  and reported loudly when it is used

Nothing here ever prints a value. These are tokens and API keys; the tooling
reports whether a key is present, not what it is.
"""

import os
import re
from pathlib import Path

# The project root: the directory holding this file.
PROJECT_DIR = Path(__file__).resolve().parent
PROJECT_ENV = PROJECT_DIR / ".env"
LOCAL_ENV = PROJECT_DIR / ".env.local"
EXAMPLE_ENV = PROJECT_DIR / ".env.example"

# What a real setting name looks like. Applied when reading `.env.example` so a
# prose line containing an `=` is not mistaken for a variable.
_ENV_NAME = re.compile(r"[A-Z][A-Z0-9_]*")

# Secrets that must never reach a log, a screenshot, or a chat message.
SECRET_HINTS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "SESSID")


def is_secret(name: str) -> bool:
    """Whether a setting's value must be redacted."""
    upper = name.upper()
    return any(hint in upper for hint in SECRET_HINTS)


def parent_env_path() -> Path:
    """The `.env` one directory above the project, if there is one.

    Named as a function on purpose. Writing `PROJECT_ENV.parent / ".env"`
    instead yields the *project's own* .env, because the parent of a file is its
    directory -- which is a silent no-op that reads as if it worked. That
    mistake made `setup_env.py` report "carried over: nothing" while a real
    Gemini key sat in the parent directory.
    """
    return PROJECT_DIR.parent / ".env"


def candidate_files() -> list[Path]:
    """The files that would be searched, most specific first.

    Returned rather than used directly so callers -- and the doctor -- can name
    them in an error message.
    """
    files = [PROJECT_ENV, LOCAL_ENV]
    if not PROJECT_ENV.exists():
        # The old convenience, kept but made visible. If the project has its own
        # .env, a parent's is ignored entirely: two files setting the same key is
        # how you get a stale value with no idea where it came from.
        files.append(parent_env_path())
    return files


def load_env(override: bool = False) -> list[Path]:
    """Load the env files. Returns which ones were read.

    `override=False` so the process environment always wins, which is what makes
    `DISCORD_BOT_TOKEN=... python discord_bot.py` work for a one-off.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - dotenv is a hard dependency
        return []

    used: list[Path] = []
    for path in candidate_files():
        if path.exists():
            load_dotenv(dotenv_path=path, override=override)
            used.append(path)
    return used


def describe_sources() -> str:
    """A one-line, printable account of where settings are coming from.

    Names the files, never the values. Meant to be pasted into an error message
    or printed on startup, so "which .env is this even reading?" has an answer.
    """
    loaded = [p for p in candidate_files() if p.exists()]
    if not loaded:
        return (
            f"no .env found. Create {PROJECT_ENV} (copy {EXAMPLE_ENV.name} "
            f"and fill it in)."
        )
    listed = ", ".join(str(p) for p in loaded)
    extra = ""
    if parent_env_path() in loaded:
        extra = (
            f"  NOTE: using the PARENT directory's {parent_env_path().name} "
            f"because {PROJECT_ENV.name} does not exist. Run "
            f"'python setup_env.py' to move it into the project."
        )
    return f"read: {listed}{extra}"


def setting_state(name: str) -> str:
    """``"set"``, ``"placeholder"`` or ``"unset"`` for one setting.

    Distinguishing "your_api_key_here" from a real value matters more than it
    sounds: a copied `.env.example` looks configured to every naive check, and
    the failure then surfaces as a confusing 400 from an API instead of a
    sentence about the file.
    """
    value = os.getenv(name)
    if value is None or not value.strip():
        return "unset"
    stripped = value.strip().strip('"').strip("'")
    if not stripped:
        return "unset"
    if stripped.lower().startswith("your_") or stripped.lower().endswith("_here"):
        return "placeholder"
    return "set"


def missing_required() -> list[str]:
    """Which required settings are not actually usable."""
    return [n for n in ("GEMINI_API_KEY",) if setting_state(n) != "set"]


def parse_env_file(path: Path) -> dict[str, str]:
    """Read one env file into a dict, comments and blanks dropped."""
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, _, value = stripped.partition("=")
        out[name.strip()] = value.strip().strip('"').strip("'")
    return out


def example_keys() -> set[str]:
    """Every setting name documented in `.env.example`, commented or not."""
    return set(parse_env_file(EXAMPLE_ENV))


def documented_keys() -> list[str]:
    """`.env.example` keys in file order, with the commented-out ones too.

    The commented entries are the *optional* settings, and they are exactly the
    ones a hand-written `.env` forgets. So they have to be in this list for the
    doctor to be able to say "your .env is missing 9 documented settings".

    Only names that actually look like environment variables count. A prose
    line such as ``#   all = open it to anyone`` also contains an `=`, and
    reading it as a setting named `all` put a nonsense entry in the report.
    """
    if not EXAMPLE_ENV.exists():
        return []
    names: list[str] = []
    for line in EXAMPLE_ENV.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip().lstrip("#").strip()
        if not stripped or "=" not in stripped:
            continue
        name = stripped.partition("=")[0].strip()
        if _ENV_NAME.fullmatch(name) and name not in names:
            names.append(name)
    return names
