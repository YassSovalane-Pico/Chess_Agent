"""Create chess-agent/.env as a full copy of .env.example, preserving real keys.

The rule: `.env` should be a complete record of everything you can configure,
matching `.env.example` line for line, with your values filled in. A hand-typed
`.env` with two lines in it hides thirteen settings the user did not know
existed.

So this starts from the template and copies across any value already present,
in any env file, matched by name. Values are never printed.
"""

import sys

import project_env

example = project_env.EXAMPLE_ENV
target = project_env.PROJECT_ENV

# Every file a value might already live in, project first so it wins.
existing: dict[str, str] = {}
for path in (
    target,
    project_env.LOCAL_ENV,
    project_env.parent_env_path(),
):
    for key, value in project_env.parse_env_file(path).items():
        if value and not value.lower().startswith("your_"):
            existing.setdefault(key, value)

if not example.exists():
    print(f"error: {example} is missing; nothing to copy.", file=sys.stderr)
    raise SystemExit(1)

if target.exists() and "--force" not in sys.argv:
    print(f"{target} already exists. Pass --force to overwrite it.")
    raise SystemExit(1)

lines = example.read_text(encoding="utf-8").splitlines()
out: list[str] = []
carried: list[str] = []
for line in lines:
    stripped = line.strip()
    if not stripped or "=" not in stripped:
        out.append(line)
        continue
    name = stripped.lstrip("#").partition("=")[0].strip()
    if name in existing:
        value = existing[name]
        # keep the example's own indentation for commented-out entries
        if stripped.startswith("#"):
            prefix = line[: len(line) - len(line.lstrip("#").lstrip())]
            out.append(f"{prefix}# {name}={value}")
        else:
            out.append(f"{name}={value}")
        carried.append(name)
    else:
        out.append(line)

header = [
    "",
    "# --- written by setup_env.py, not by hand ---",
    "# Generated as a full copy of .env.example so every setting is visible.",
    f"# Values carried over from an existing env file: {', '.join(carried) or 'none'}",
    "# Anything left as your_..._here is a placeholder and is treated as unset.",
    "",
]
target.write_text("\n".join(header + out) + "\n", encoding="utf-8")

print(f"wrote {target}  ({len(out)} lines, full copy of .env.example)")
print(f"  carried over: {', '.join(carried) if carried else 'nothing'}")
print(f"  still placeholders: {len(out) - len(carried)} documented settings")
print("\nRun: python check_env.py")
