"""Tests for configuration handling.

The failure this file exists to prevent: a real token or API key ending up in
`.env.example`, which is the one env file that is *meant to be committed* and
that every fresh clone copies. That is not a hypothetical -- a bot token was
found sitting in it, and `.env` in the same project held a copy of the Gemini
key from the wrong directory.

Every test here inspects files and reports *whether* a value looks real, never
what it is. The whole point is that this output is safe to paste into a chat.
"""

import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import project_env

ROOT = Path(__file__).resolve().parent.parent

# A Discord bot token is three base64 segments. Recognising the shape is what
# makes "is this a real token or a placeholder?" answerable without printing it.
TOKEN_SHAPED = re.compile(
    r"^[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{20,}$"
)


def _is_real_secret(name: str, value: str) -> bool:
    """Whether this looks like a live secret rather than a documented default."""
    if not project_env.is_secret(name):
        return False
    value = value.strip()
    if not value:
        return False
    if value.lower().startswith("your_") or value.lower().endswith("_here"):
        return False
    return True


# ---------------------------------------------------------------------------
# The template must never hold a real secret
# ---------------------------------------------------------------------------


def test_env_example_contains_no_real_secret():
    """.env.example is committed. A token in it is a token in git history."""
    example = project_env.EXAMPLE_ENV
    assert example.exists(), f"{example} is missing"
    leaks = [
        name
        for name, value in project_env.parse_env_file(example).items()
        if _is_real_secret(name, value)
    ]
    assert not leaks, (
        f"{example.name} holds real values for {leaks}. That file is committed; "
        f"move the value into {project_env.PROJECT_ENV.name} and put a "
        f"placeholder back. If the secret was ever committed, reset it."
    )


def test_env_example_has_no_token_shaped_value_anywhere():
    """Catches a token pasted into a comment, where parse_env_file would miss it."""
    example = project_env.EXAMPLE_ENV
    text = example.read_text(encoding="utf-8")
    for line in text.splitlines():
        for candidate in re.findall(
            r"[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{20,}", line
        ):
            raise AssertionError(
                f"{example.name} contains something shaped like a Discord token: "
                f"{candidate[:12]}... Reset it in the Developer Portal."
            )
        if "api_key_here" not in line and re.search(r"AIza[0-9A-Za-z_-]{20,}", line):
            raise AssertionError(
                f"{example.name} contains what looks like a Google API key. "
                f"Revoke it at https://aistudio.google.com/apikey"
            )


def test_the_real_env_is_git_ignored():
    """The file that DOES hold secrets must not be committable."""
    gitignore = ROOT / ".gitignore"
    assert gitignore.exists(), ".gitignore is missing"
    patterns = {
        line.strip() for line in gitignore.read_text(encoding="utf-8").splitlines()
    }
    for name in (".env", ".env.local"):
        assert name in patterns or ".env*" in patterns, f"{name} is not git-ignored"


def test_settings_are_never_printed_as_values():
    """The reason this whole approach is safe to paste into a chat."""
    assert project_env.is_secret("DISCORD_BOT_TOKEN")
    assert project_env.is_secret("GEMINI_API_KEY")
    assert not project_env.is_secret("GEMINI_MODEL")
    assert not project_env.is_secret("REVIEW_SHOT_REGION")
    assert not project_env.is_secret("REVIEW_HEADLESS")


# ---------------------------------------------------------------------------
# Placeholders must be distinguishable from real values
# ---------------------------------------------------------------------------


def test_a_placeholder_is_not_mistaken_for_a_configured_value():
    """A copied `.env.example` looks configured to every naive check, and the
    failure then surfaces as a confusing 400 from an API rather than a sentence
    about the file."""
    original = os.environ.get("SOME_TEST_KEY")
    try:
        os.environ["SOME_TEST_KEY"] = "your_api_key_here"
        assert project_env.setting_state("SOME_TEST_KEY") == "placeholder"
        os.environ["SOME_TEST_KEY"] = "your_bot_token_here"
        assert project_env.setting_state("SOME_TEST_KEY") == "placeholder"
        os.environ["SOME_TEST_KEY"] = '  "quoted_real_value"  '
        assert project_env.setting_state("SOME_TEST_KEY") == "set"
        os.environ["SOME_TEST_KEY"] = "   "
        assert project_env.setting_state("SOME_TEST_KEY") == "unset"
        os.environ["SOME_TEST_KEY"] = ""
        assert project_env.setting_state("SOME_TEST_KEY") == "unset"
        # a real value that happens to contain an underscore and a hex-looking tail
        os.environ["SOME_TEST_KEY"] = "AIzaSyD-9tSrke72PouQMnMX-a7eZSW0jkFMBWY"
        assert project_env.setting_state("SOME_TEST_KEY") == "set"
    finally:
        if original is None:
            os.environ.pop("SOME_TEST_KEY", None)
        else:
            os.environ["SOME_TEST_KEY"] = original


# ---------------------------------------------------------------------------
# File resolution
# ---------------------------------------------------------------------------


def test_the_project_env_wins_and_a_parent_is_only_a_last_resort():
    """Two files setting one key is how you get a stale value with no idea where
    it came from. The old code fell back to a parent silently and unconditionally.
    """
    files = project_env.candidate_files()
    assert project_env.PROJECT_ENV in files
    assert project_env.PROJECT_ENV in files[:2], "the project file comes first"
    if project_env.PROJECT_ENV.exists():
        assert project_env.parent_env_path() not in files, (
            "a parent .env must be ignored entirely once the project has its own"
        )


def test_parent_env_path_is_not_the_project_env():
    """A silent no-op that reads as if it works.

    `PROJECT_ENV.parent` is the project's own *directory*, so
    `PROJECT_ENV.parent / '.env'` is the project .env again. That mistake made
    setup_env.py report "carried over: nothing" while a real key sat one
    directory away.
    """
    assert project_env.parent_env_path() != project_env.PROJECT_ENV
    assert project_env.parent_env_path() == ROOT.parent / ".env"


def test_describe_sources_names_the_files_it_read():
    """The whole point of the message: "which .env is this even reading?" has to
    have an answer that is not a guess."""
    text = project_env.describe_sources()
    assert str(project_env.PROJECT_ENV) in text
    if not project_env.PROJECT_ENV.exists() and project_env.parent_env_path().exists():
        assert "PARENT" in text
        assert "setup_env.py" in text, "it should say how to fix it"


def test_documented_keys_ignores_prose_that_looks_like_a_setting():
    """A line like `#   all = open it to anyone` contains an `=` and was being
    read as a variable called `all`, which put a nonsense row in the report."""
    keys = project_env.documented_keys()
    assert "all" not in keys, keys
    assert "any" not in keys
    for expected in (
        "GEMINI_API_KEY",
        "DISCORD_BOT_TOKEN",
        "DISCORD_ALLOWED_USER_IDS",
        "REVIEW_SHOT_REGION",
    ):
        assert expected in keys, f"{expected} missing from .env.example"
    for name in keys:
        assert re.fullmatch(r"[A-Z][A-Z0-9_]*", name), name


def test_every_documented_key_appears_in_the_real_env():
    """`.env` should be a complete record of what you can change.

    A hand-written `.env` with two lines hides thirteen settings nobody knew
    existed. Commented-out optional settings count as documented-and-present:
    the point is discoverability, not activation.
    """
    if not project_env.PROJECT_ENV.exists():
        print("skip  no .env yet; run python setup_env.py")
        return
    text = project_env.PROJECT_ENV.read_text(encoding="utf-8")
    missing = [name for name in project_env.documented_keys() if f"{name}=" not in text]
    assert not missing, (
        f".env is missing {len(missing)} documented settings: {missing}. "
        f"Regenerate it as a full copy: python setup_env.py --force"
    )


def test_env_never_reports_a_value():
    """Belt and braces on the redaction: run the doctor and assert no value from
    .env appears in its output."""
    import io
    from contextlib import redirect_stdout

    import check_env

    project_env.load_env()
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        try:
            check_env.report()
        except SystemExit:
            pass
    report = buffer.getvalue()
    assert report, "the doctor printed nothing"

    leaked = []
    for name, value in project_env.parse_env_file(project_env.PROJECT_ENV).items():
        if len(value) < 12 or not project_env.is_secret(name):
            continue
        if value in report:
            leaked.append(name)
    assert not leaked, f"check_env printed the value of {leaked}"


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
