"""Run the move-table extractor against a fake DOM built from real markup.

`tools/move_list._MOVE_TABLE_JS` is the one piece of logic that cannot be
unit-tested from Python: it walks the DOM. So it is executed in Node against a
minimal stub of the elements it touches, wired up with the exact attribute
shapes found in `dumps/review_started.html`:

    <div class="main-line-row move-list-row" data-whole-move-number="1">
      <div data-node="0-0" class="node white-move main-line-ply">
        <span class="node-annotation-icon"><svg><g id="book"></g></svg></span>
        <span class="node-highlight-content" style="color: var(--color-classification-book)">
          <span class="icon-font-chess bishop-white" data-figurine="B"></span> g4
        </span>
      </div>
      <div data-move-list-el="timestamp" data-ply="1" data-time="8"></div>

The point is the *edges*, not the happy path: a hidden cell must be dropped, a
malformed data-node must be skipped rather than crashing, a duplicate ply must
not produce a second row, a figurine must not leak into the SAN, and centi-
seconds must come back as seconds. Each of those was a real bug or a real trap.

Skips silently when Node is not installed. The Python side of the reader --
the classification vocabulary -- is tested without Node in
test_review_logic.py.

Run:  python -m pytest tests/test_js_detection.py -q
"""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools import move_list

NODE = shutil.which("node")

# The extractor, verbatim, plus a stub `document`. Written out here rather than
# eval'd so the test fails loudly if the blob's shape ever changes.
_HARNESS = r"""
const readMoves = __PAYLOAD__;

// --- the minimum DOM the blob uses -----------------------------------------
class El {
  constructor(attrs, children, text) {
    this._attrs = attrs || {};
    this.children = children || [];
    this._text = text || "";
    this.parentElement = null;
    for (const c of this.children) c.parentElement = this;
  }
  // The real DOM always hands back a string for an attribute, even if it was
  // written as a number. The stub stores numbers deliberately so a test that
  // forgets to stringify cannot pass by luck.
  getAttribute(name) {
    return name in this._attrs ? String(this._attrs[name]) : null;
  }
  setAttribute(name, value) { this._attrs[name] = String(value); }
  removeAttribute(name) { delete this._attrs[name]; }
  get className() { return this._attrs["class"] || ""; }
  get style() { return this._attrs["style"] || ""; }
  get innerText() {
    if (this.children.length) return this.children.map((c) => c.innerText).join("");
    return this._text;
  }
  get textContent() { return this.innerText; }
  get offsetWidth() { return 700; }
  get offsetHeight() { return 700; }
  get offsetParent() { return this; }
  getClientRects() { return this._hidden ? [] : [{}]; }
  querySelector(sel) { return this.querySelectorAll(sel)[0] || null; }
  querySelectorAll(sel) {
    const out = [];
    const parts = sel.split(",").map((s) => s.trim()).filter(Boolean);
    const walk = (el) => {
      for (const c of el.children) {
        if (parts.some((p) => matches(c, p))) out.push(c);
        walk(c);
      }
    };
    walk(this);
    return out;
  }
  closest(sel) {
    let n = this;
    while (n) { if (matches(n, sel)) return n; n = n.parentElement; }
    return null;
  }
}

// Supports the selector shapes the blob uses: "tag", ".cls", "[attr]" and
// '[attr="value"]', any number of them concatenated, plus a single descendant
// step ("a b").
function matchesOne(el, sel) {
  // Attributes first, then the leftover bare segments are classes. Doing it in
  // this order matters: splitting the whole selector on "." would treat
  // "[data-node]" as a class name.
  for (const attr of sel.match(/\[[^\]]+\]/g) || []) {
    const m = /^\[([^\]=]+)(?:="([^"]*)")?\]$/.exec(attr);
    if (!m) return false;
    const value = el.getAttribute(m[1]);
    if (value === null) return false;
    if (m[2] !== undefined && value !== m[2]) return false;
  }
  const tag = /^([a-zA-Z-]+)/.exec(sel.replace(/\[[^\]]+\]/g, ""));
  if (tag && el._attrs["__tag"] && el._attrs["__tag"] !== tag[1]) return false;
  const rest = sel.replace(/\[[^\]]+\]/g, "").slice(tag ? tag[1].length : 0);
  for (const group of rest.split(".").filter(Boolean)) {
    if (!String(el.className).split(/\s+/).includes(group)) return false;
  }
  return true;
}

function matches(el, sel) {
  const bits = sel.split(/\s+/).filter(Boolean);
  if (!matchesOne(el, bits[bits.length - 1])) return false;
  for (let i = bits.length - 2; i >= 0; i--) {
    let n = el.parentElement;
    let ok = false;
    while (n) {
      if (matchesOne(n, bits[i])) { ok = true; break; }
      n = n.parentElement;
    }
    if (!ok) return false;
  }
  return true;
}

function icon(id) {
  const g = new El({ "__tag": "g", id }, [], "");
  const svg = new El({ "__tag": "svg" }, [g], "");
  return new El({ "__tag": "span", "class": "node-annotation-icon" }, [svg], "");
}

function moveCell(node, colour, san, markerId, cssKind, figurine, hidden) {
  // A text node with no element of its own -- innerText contributes the SAN
  // while the figurine span beside it contributes nothing, which is exactly
  // what chess.com renders.
  const text = new El({}, [], san);
  const kids = figurine
    ? [new El({ "__tag": "span", "class": "icon-font-chess " + colour + "-white",
                "data-figurine": figurine }, [], ""), text]
    : [text];
  const content = new El(
    { "__tag": "span", "class": "node-highlight-content",
      style: cssKind ? "color: var(--color-classification-" + cssKind + ")" : "" },
    kids, "",
  );
  const cellKids = [];
  if (markerId) cellKids.push(icon(markerId));
  cellKids.push(content);
  const cell = new El(
    { "__tag": "div", "data-node": node, "class": "node " + colour + "-move main-line-ply" },
    cellKids, "",
  );
  cell._hidden = !!hidden;
  return cell;
}

function row(number, cells, stamps) {
  const kids = [];
  for (const c of cells) kids.push(c);
  for (const s of stamps) {
    kids.push(new El({ "__tag": "div", "class": "node time-" + s.colour,
      "data-move-list-el": "timestamp", "data-ply": s.ply, "data-time": s.cs }, [], ""));
  }
  return new El({ "__tag": "div",
    "class": "main-line-row move-list-row", "data-whole-move-number": number }, kids, "");
}

const cells = [
  moveCell("0-0", "white", " g4", "book", "book", "B", false),
  moveCell("0-1", "black", " e6", null, null, null, false),
  moveCell("1-0", "white", " Nf3", null, null, "N", false),
  moveCell("1-1", "black", " Nc6", null, null, null, false),
  // a hidden cell must be dropped, not reported
  moveCell("2-0", "white", " Bb5", "blunder", "blunder", "B", true),
  // a malformed data-node must be skipped, not crash the blob
  moveCell("junk", "black", " ??", null, null, null, false),
  // a duplicate ply must not become a second row
  moveCell("1-1", "black", " Nc6", "inaccuracy", "inaccuracy", null, false),
  // PascalCase id, as chess.com actually writes it
  moveCell("2-1", "black", " a6", "Brilliant", null, null, false),
];
const rowEl = row(1, cells, [
  { ply: 1, cs: 8, colour: "white" },
  { ply: 2, cs: 30, colour: "black" },
  { ply: 3, cs: 150, colour: "white" },
  { ply: 4, cs: 420, colour: "black" },
]);
const body = new El({ "__tag": "body" }, [rowEl], "");

global.document = { body, querySelectorAll: (sel) => body.querySelectorAll(sel),
                    querySelector: (sel) => body.querySelector(sel) };

const out = readMoves();
process.stdout.write(JSON.stringify(out));
"""


def _run(harness: str = _HARNESS) -> dict:
    """Run the real `_MOVE_TABLE_JS` in Node over the harness's stub DOM."""
    js = harness.replace("__PAYLOAD__", move_list._MOVE_TABLE_JS)
    script = _tmp_js("chess_agent_move_table_test.js", js)
    # Safe: NODE comes from shutil.which and the script is a file this function
    # just wrote into a temp dir. No shell, no user input.
    proc = subprocess.run(  # noqa: S603
        [NODE, str(script)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, f"node failed: {proc.stderr[:800]}"
    return json.loads(proc.stdout)


def _tmp_js(name: str, source: str) -> Path:
    path = Path(tempfile.gettempdir()) / name
    path.write_text(source, encoding="utf-8")
    return path


def test_the_extractor_survives_the_edges_of_the_real_markup():
    if not NODE:
        print("skip  node not installed")
        return

    out = _run()
    assert out["supported"] is True
    rows = out["rows"]
    by_ply = {r["ply"]: r for r in rows}

    # 8 cells: one hidden, one malformed, one duplicate of ply 4
    assert sorted(by_ply) == [1, 2, 3, 4, 6], sorted(by_ply)
    assert out["diagnostics"]["hiddenDropped"] == 1, out["diagnostics"]
    assert out["diagnostics"]["skipped"] == 2, out["diagnostics"]
    # the duplicate ply must not appear twice
    assert len([r for r in rows if r["ply"] == 4]) == 1
    # ...and the first one wins, so a later unlabelled pass cannot erase a
    # classification. That preference is the whole point of the merge.
    assert by_ply[4]["icon_id"] is None
    assert out["diagnostics"]["cells"] == 8, out["diagnostics"]

    # the figurine contributes no text, so the SAN is clean
    assert by_ply[1]["san"] == "g4", by_ply[1]
    assert by_ply[3]["san"] == "Nf3"
    assert by_ply[1]["figurine"] == "B"
    assert by_ply[2]["figurine"] is None

    # colour comes from the cell's own class
    assert by_ply[1]["color"] == "white"
    assert by_ply[2]["color"] == "black"

    # the move number comes from the ROW
    assert by_ply[1]["number"] == 1

    # centiseconds in the DOM, seconds for a person
    assert by_ply[1]["clock_cs"] == 8
    assert by_ply[4]["clock_cs"] == 420

    # the two label sources, and the PascalCase id
    assert by_ply[1]["icon_id"] == "book"
    assert by_ply[6]["icon_id"] == "Brilliant"
    assert by_ply[6]["css_classification"] is None

    # a cell with no label is reported as unlabelled, not guessed at
    assert by_ply[2]["icon_id"] is None
    assert by_ply[2]["css_classification"] is None


def test_the_extractor_says_so_on_a_page_with_no_move_list():
    if not NODE:
        print("skip  node not installed")
        return

    empty = _HARNESS.replace(
        "const cells = [", "const cells = [].slice(0, 0); const unused = ["
    )
    out = _run(empty)
    assert out["supported"] is False
    assert out["rows"] == []
    assert out["diagnostics"]["cells"] == 0


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
        else:
            print(f"ok    {name}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
