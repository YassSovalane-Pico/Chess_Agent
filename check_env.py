"""Tell you exactly what is configured, and what is not -- without leaking it.

    python check_env.py          # human report, exit 1 if required keys missing
    python check_env.py --quiet  # exit code only, for a pre-commit hook or CI

This exists because "Missing DISCORD_BOT_TOKEN" is a bad error message. It does
not say which file was read, whether the file exists, whether the value is a
real token or the `your_bot_token_here` placeholder from the template, or which
other settings are missing. So the fix is always the same guess: open the
project, find the .env, wonder whether the one you edited is the one being read.

Every value is redacted. These are tokens and API keys, and the whole reason
this report is safe to paste into a chat is that it cannot print one.
"""

import sys

import project_env

BANNER = "=" * 68


def _state_icon(state: str) -> str:
    return {"set": "ok  ", "placeholder": "WARN", "unset": "MISS"}[state]


def report() -> int:
    print(BANNER)
    print("  chess-agent configuration")
    print(BANNER)

    # 1. Which files, and which one wins
    print("\nFILES")
    for path in project_env.candidate_files():
        if path.exists():
            keys = project_env.parse_env_file(path)
            real = sum(
                1 for k, v in keys.items() if v and not v.lower().startswith("your_")
            )
            print(f"  found    {path}  ({real} value(s) set)")
        elif path is project_env.PROJECT_ENV:
            print(f"  MISSING  {path}   <-- copy .env.example here")
        else:
            print(f"  absent   {path}")
    print(f"\n  {project_env.describe_sources()}")

    # 2. Where each setting actually came from. This is the part that answers
    #    "did I even save it?" -- the process environment beats every file, so a
    #    stale exported variable silently wins over the .env you just edited.
    print("\nRESOLVED (the value the program will actually use)")
    required = ("GEMINI_API_KEY", "DISCORD_BOT_TOKEN", "DISCORD_ALLOWED_USER_IDS")
    optional = [k for k in project_env.documented_keys() if k not in required]
    for name in list(required) + optional:
        state = project_env.setting_state(name)
        tag = "required" if name in required else "optional"
        print(f"  {_state_icon(state):5} {name:28} {state:11} ({tag})")

    # 3. Settings the template documents but the .env omits. A hand-written .env
    #    normally has two of the fifteen; this is the list you were missing.
    env_keys = set(project_env.parse_env_file(project_env.PROJECT_ENV))
    parent_keys = set(project_env.parse_env_file(project_env.parent_env_path()))
    documented = set(project_env.documented_keys())
    absent = sorted(documented - env_keys - parent_keys)
    print(f"\nDOCUMENTED BUT NOT SET ANYWHERE ({len(absent)})")
    if absent:
        print("  These all have sensible defaults, so nothing is broken. Listed")
        print("  so `.env` is a complete record of what you can change:")
        for name in absent:
            print(f"    {name}")
    else:
        print("  none - .env covers everything .env.example documents")

    # 4. Two files setting the same key is a real hazard, not a style note.
    both = sorted(env_keys & parent_keys)
    if both:
        print("\nDUPLICATED IN TWO FILES")
        print(f"  {', '.join(both)}")
        print("  The project .env wins, silently. Delete the parent one:")
        print(f"    {project_env.parent_env_path()}")

    # 5. The security check that matters most here.
    print("\nSECRETS")
    leaks = [
        p
        for p in (project_env.PROJECT_ENV, project_env.LOCAL_ENV)
        if p.exists() and not _is_git_ignored(p.name)
    ]
    if leaks:
        print("  WARNING: not git-ignored, could be committed:")
        for path in leaks:
            print(f"    {path}")
    else:
        print("  ok  .env is git-ignored")

    missing = project_env.missing_required()
    if missing:
        print(f"\nRESULT: {', '.join(missing)} not set -- the agent will not start.")
        return 1
    print("\nRESULT: core configuration is usable.")
    if not project_env.setting_state("DISCORD_BOT_TOKEN") == "set":
        print("        (Discord not configured; the CLI still works)")
    return 0


def _is_git_ignored(name: str) -> bool:
    gitignore = project_env.PROJECT_DIR / ".gitignore"
    if not gitignore.exists():
        return False
    patterns = {
        line.strip()
        for line in gitignore.read_text(encoding="utf-8", errors="replace").splitlines()
    }
    return name in patterns or ".env*" in patterns


def main(argv: list[str]) -> int:
    project_env.load_env()
    if "--quiet" in argv:
        return 1 if project_env.missing_required() else 0
    return report()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
