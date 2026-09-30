"""Syntax-check every JavaScript blob embedded in the Python sources.

The DOM readers in tools/move_list.py and tools/review.py are shipped as Python
strings, so a typo or an unbalanced brace only shows up as a runtime failure
inside the browser -- where it is swallowed by a try/except and looks like
"chess.com has no blunders". Parsing them here turns a silent failure into a
red test.

Needs `esprima`, which is a dev extra rather than a runtime dependency:

    pip install -e ".[dev]"      # or just: pip install esprima

Without it these tests SKIP, loudly, rather than raising ModuleNotFoundError at
import time. That is a deliberate trade: a traceback about a missing dev
dependency tells you nothing about whether your code is broken, and this file
running in a plain runtime venv is normal. It does mean a skipped run is not a
passing run -- the skip line says so.

Run:  python tests/test_js_syntax.py
"""

import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import esprima
except ImportError:  # pragma: no cover - depends on the environment
    esprima = None

ROOT = Path(__file__).resolve().parent.parent

SKIP_REASON = (
    "skip  esprima is not installed -- these JS syntax checks did NOT run. "
    'Install the dev extra: pip install -e ".[dev]"'
)

# Module-level names whose value is a JS arrow function / IIFE we care about.
# The blobs that survived the move to a data-node reader. _READ_MOVES_JS,
# _TALLY_JS and _RATINGS_JS are gone: the first because data-node made the
# density heuristic unnecessary, the other two because _SUMMARY_JS reads the
# same panel more directly.
TARGETS = {
    "tools/move_list.py": [
        "_MOVE_TABLE_JS",
        "_COACH_JS",
        "_SUMMARY_JS",
        "_POSITION_JS",
        "_ACTIVE_PLY_JS",
        "_REVIEW_STATE_JS",
        "_REVIEW_PANEL_JS",
        "_SCROLL_INFO_JS",
        "_SCROLL_TO_JS",
        "_INSPECT_JS",
    ],
    "tools/review.py": [
        "_DRAW_BOXES_JS",
    ],
}


def _js_snippets(path: Path, names: list[str]) -> dict[str, str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name) or target.id not in names:
                continue
            value = node.value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                found[target.id] = value.value
    return found


def test_all_js_blobs_parse():
    if esprima is None:
        print(SKIP_REASON)
        return
    checked = 0
    for rel, names in TARGETS.items():
        path = ROOT / rel
        assert path.exists(), f"missing {rel}"
        blobs = _js_snippets(path, names)
        for name in names:
            assert name in blobs, f"{rel}: {name} not found as a module-level string"
            source = blobs[name]
            # The blob is an arrow function expression, so wrap it for the parser.
            try:
                esprima.parseScript(f"({source});")
            except Exception as exc:
                raise AssertionError(f"{rel}:{name} failed to parse: {exc}") from exc
            checked += 1
            print(f"ok    {rel}:{name} ({len(source)} chars)")
    assert checked == 11, checked


def test_the_js_extracts_facts_and_python_owns_the_vocabulary():
    """The classification words must appear in Python and NOT in the JS.

    The vocabulary used to be duplicated in the reader's JS and the tally
    reader's JS, which is how the two drifted apart and the reply started
    disagreeing with the screenshot beside it. Now the browser reports raw
    facts and RULES turns them into a kind, so there is one table, in one
    language, testable without a browser.
    """
    from tools import move_list

    for kind in move_list.KINDS:
        assert kind in {r["kind"] for r in move_list.RULES}, kind
    for blob in (move_list._MOVE_TABLE_JS, move_list._SUMMARY_JS):
        # the JS may not name a classification at all
        for word in move_list.KINDS:
            assert word not in blob, f"{word!r} leaked into the JS: keep it in RULES"


def test_the_reader_cannot_regress_to_substring_classification():
    """The word-boundary rule is the fix for 'dismiss' -> miss."""
    from tools import move_list

    for rule in move_list.RULES:
        for frag in rule["frags"]:
            assert " " not in frag, "frags are single words; multi-word needs care"
    # classify_class works on word SETS, so 'dismiss' cannot contain 'miss'
    assert move_list.classify_class("dismiss") is None
    assert move_list.classify_class("icon-submission") is None
    # and the JS never matches a kind name itself
    assert "brilliant" not in move_list._MOVE_TABLE_JS


def test_position_probe_has_a_fallback():
    from tools import move_list

    assert "position" in move_list._POSITION_JS
    assert "document" in move_list._POSITION_JS


def main() -> int:
    tests = [(n, o) for n, o in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"ERROR {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
