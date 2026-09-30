"""Move-list intelligence: read chess.com's move list, classify, navigate.

STATUS: built against a captured `dump_pages.py` run (dumps/), not against
guesses. Every selector here was read out of `review_started.html`, and the
tests assert against that markup's actual shape.

What the dumps established:

  * **The move list is not virtualised.** 29 `[data-node]` cells were in the
    DOM at every one of five scroll offsets, so no scroll sweep is needed.
  * **`data-node="<variation>-<index>"` is an authoritative ply index.**
    `ply = variation * 2 + index + 1`. 29 cells, zero gaps, no counting and no
    density heuristic needed. The move number it could have been keyed on lives
    on the row, not in the cell, so it must not be used as a dedup key.
  * **Every classification is explicitly labelled**, in one of two places on
    the cell: `<span class="node-annotation-icon"><svg><g id="blunder">`, or
    `style="color: var(--color-classification-book)"`. The `g` id casing is
    inconsistent -- `id="Brilliant"` alongside `id="book"` -- so matching is
    case-insensitive. There is no need for a learned colour palette, and none
    is used: a move with no explicit label is reported as unclassified, which
    is the honest answer.
  * **The move number is on the row** as `data-whole-move-number`, and the
    clock is on a sibling as `data-move-list-el="timestamp" data-ply="N"
    data-time="<centiseconds>"`.
  * **The coach panel is rendered for the current move only** -- one
    `move-feedback-box-component` in the whole document. Commentary therefore
    requires navigating to the move first.
  * **Accuracy and the tally exist only BEFORE the review is started.** Both
    are absent from `review_started.html` and present in `review_loaded.html`.
    So the order is fixed and is not a defensive choice: read the summary,
    *then* start, *then* read moves.
  * **`wc-chess-board` is the 8x8 grid itself** (it contains the
    `viewBox="0 0 100 100"` coordinates SVG), so it is the board selector and
    the frame can hug it.
  * The page mounts in **three stages**: layout shell, then the review data
    (a few seconds, computed server-side), then the move list (only after
    `tab-review-start-review-button`). Both "no loading placeholders" and "the
    board is on screen" are true before any of that, so neither can be the
    readiness signal on its own; `wait_for_review_loaded` takes an explicit
    `require="layout" | "panel" | "movelist"`.

Public surface:
  - wait_for_review_loaded(page, require=...)   -> bool
  - read_review_state(page)                    -> dict
  - review_panel_present(page)                 -> bool
  - expand_review_summary(page)                -> bool
  - open_review_mode(page)                     -> bool  (clicks Start Review)
  - open_moves_tab(page)                       -> bool
  - read_moves(page) / read_move(page, ply)    -> (rows, diag) / (row, diag)
  - read_coach_commentary(page)                -> dict
  - read_moves_when_ready(page)                -> (rows, diag)
  - find_classified(page, kinds)               -> (hits, diag)
  - read_tally(page) / read_ratings(page)      -> dict / (white, black)
  - goto_ply(page, ply)                        -> bool
  - step_moves(page, n)                        -> StepResult
  - board_signature(page) / board_is_stable(..)-> verification helpers
  - inspect_dom(page)                          -> dict  (calibration dump)
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from dataclasses import dataclass

_LOG = logging.getLogger(__name__)

try:  # pragma: no cover - depends on whether playwright is installed
    from playwright.async_api import Error as _PlaywrightError
except ImportError:  # pragma: no cover - playwright is an optional dependency
    _PlaywrightError = Exception

# ---------------------------------------------------------------------------
# Selectors. Every one of these was read out of dumps/review_started.html.
# ---------------------------------------------------------------------------

# The move list container. The grid is `wc-chess-board` itself; this is the
# scroll wrapper around the moves.
CONTAINER_HINTS = [
    "wc-move-list",
    ".move-list",
    "[class*='move-list' i]",
    "#board-layout-sidebar",
]

# A board: the 8x8 grid. It contains the `viewBox="0 0 100 100"` coordinates
# SVG, so the element IS the grid, not a wrapper around a column of nameplate,
# clock and controls.
BOARD_HINTS = [
    "wc-chess-board",
    "[class*='chessboard' i]",
    "[class*='board-board' i]",
    ".board",
]

POSITION_HINTS = [
    "wc-chess-board[position]",
    "[class*='chessboard' i][position]",
    "wc-chess-board",
    "#board[position]",
    "[data-position]",
    "[data-fen]",
]

# The review sidebar: coach chat, accuracy, tally, move list. NOT the left
# navigation, which is `.sidebar-container`.
REVIEW_SIDEBAR_HINTS = [
    "#board-layout-sidebar",
    "[class*='board-layout-sidebar' i]",
]

# Placeholders shown while the review is still fetching.
LOADING_HINTS = [
    "[class*='skeleton' i]",
    "[class*='loading' i]",
    "[class*='loader' i]",
]

# Stage 1: the app shell has mounted.
REVIEW_LAYOUT_HINTS = [
    "wc-chess-board",
    "[class*='chessboard' i]",
    *REVIEW_SIDEBAR_HINTS,
]

# Stage 2: the review data -- accuracy pair, or the classification tally.
REVIEW_PANEL_HINTS = [
    ".review-rating-white",
    ".review-rating-black",
    "[class*='tally-row-value' i]",
    "[class*='game-review' i]",
]

# Stage 3: the move list, which only exists after the review is started.
MOVE_LIST_HINTS = [
    "[data-node]",
    "[class*='move-list-row' i]",
    "[class*='main-line-ply' i]",
    "wc-move-list",
]

# The panel itself, for `review_panel_present`. Deliberately excludes the
# move list: "is the review running?" and "is the summary up?" are different
# questions and only one of them is true at a time.
REVIEW_PANEL_STRICT = [
    ".review-rating-white",
    ".review-rating-black",
    "[class*='tally-row-value' i]",
]

# The control that starts the review. The class is stable and far safer than
# matching on the label text; "Game Review" is the sidebar <h1>, never a button.
REVIEW_ENTRY_SELECTORS = [
    ".tab-review-start-review-button",
    "button:has-text('Start Review')",
    "[class*='start-review' i]",
    "text=Start Review",
]
REVIEW_ENTRY_LABELS = ("Start Review",)

# Sidebar sections that can be switched to.
TAB_HINTS = [
    "[role='tab']",
    "[class*='tab-button' i]",
    "[class*='sidebar-tab' i]",
    "[class*='nav-tab' i]",
    "button",
]

# ---------------------------------------------------------------------------
# The classification vocabulary. Python, not JavaScript: the browser extracts
# raw *facts* (an icon id, a CSS custom property) and this table turns them into
# a kind, so there is exactly one place to add a classification and it is
# unit-testable without a browser.
#
# `frags` are matched against the separator- and camelCase-aware form of a
# class name with word boundaries. Stripping punctuation and doing a substring
# search is what turns "dismiss" into a miss and "Facebook" into a book.
# ---------------------------------------------------------------------------

RULES: tuple[dict, ...] = (
    {"kind": "brilliant", "frags": ("brilliant",)},
    {"kind": "blunder", "frags": ("blunder",)},
    {"kind": "inaccuracy", "frags": ("inaccuracy",)},
    {"kind": "mistake", "frags": ("mistake",)},
    {"kind": "miss", "frags": ("miss", "missed")},
    {"kind": "great", "frags": ("greatfind", "greatmove", "great")},
    {"kind": "best", "frags": ("bestmove", "best")},
    {"kind": "book", "frags": ("book",)},
    {"kind": "excellent", "frags": ("excellent",)},
    {"kind": "good", "frags": ("good",)},
)

# chess.com's own scale, in the order it presents them.
KINDS: tuple[str, ...] = (
    "brilliant",
    "blunder",
    "inaccuracy",
    "mistake",
    "miss",
    "great",
    "best",
    "book",
)

# Present in the legend but not worth reporting: they are chess.com approving
# a move, not finding a problem. The tally still counts them.
AUX_KINDS: tuple[str, ...] = ("excellent", "good")

HIGH, MEDIUM, LOW = "high", "medium", "low"

_CAMEL = re.compile(r"([a-z0-9])([A-Z])")


def class_text(value: str | None) -> str:
    """Normalise a class or id for matching.

    ``analysis-bestMove`` -> ``analysis best move``; ``GreatFind`` -> ``great
    find``; ``chess-board_pkg`` -> ``chess board pkg``. Word boundaries then do
    the work.
    """
    if not value:
        return ""
    return _CAMEL.sub(r"\1 \2", str(value)).lower().replace("_", " ").replace("-", " ")


def _normalise_tokens(value: str | None) -> set[str]:
    """Every word in a class/id, as a set, for containment tests."""
    return set(class_text(value).split())


def classify_text(value: str | None) -> str | None:
    """Match a whole string, e.g. a description like "is a blunder"."""
    if not value:
        return None
    words = _normalise_tokens(value)
    if not words:
        return None
    for rule in RULES:
        if any(frag in words for frag in rule["frags"]):
            return rule["kind"]
    return None


def classify_class(value: str | None) -> str | None:
    """Match a class or an element id, e.g. ``analysis-blunder`` or ``Brilliant``.

    Case-insensitive on purpose: the captured markup has ``id="Brilliant"``
    sitting next to ``id="book"`` and ``id="mistake"``.
    """
    if not value:
        return None
    words = _normalise_tokens(value)
    if not words:
        return None
    for rule in RULES:
        for frag in rule["frags"]:
            if frag in words:
                return rule["kind"]
    return None


# The prefix chess.com uses in the inline custom property on a move's SAN span.
CSS_CLASSIFICATION = re.compile(r"--color-classification-([a-zA-Z-]+)")


def classify_facts(icon_id: str | None, css_kind: str | None, classes: str | None):
    """Turn the browser's raw facts into ``(kind, confidence, evidence)``.

    Both sources are explicit labels chess.com itself wrote, so both are
    ``high`` confidence. There is no colour-inference fallback: a move with
    neither is unclassified, and saying so is more useful than guessing from a
    pixel.
    """
    if icon_id:
        kind = classify_class(icon_id) or classify_text(icon_id)
        if kind:
            return kind, HIGH, f"svg marker id={icon_id}"
    if css_kind:
        kind = classify_class(css_kind) or classify_text(css_kind)
        if kind:
            return kind, HIGH, f"style --color-classification-{css_kind}"
    kind = classify_class(classes)
    if kind:
        return kind, MEDIUM, f"class={str(classes)[:60]}"
    return None, None, None


# ---------------------------------------------------------------------------
# Waiting for the review to actually finish loading
# ---------------------------------------------------------------------------

_REVIEW_STATE_JS = r"""
(opts) => {
  const rendered = (el) => {
    if (!el) return false;
    // A hidden placeholder is not a reason to keep waiting.
    if (el.getClientRects && el.getClientRects().length === 0) return false;
    return !!(el.offsetWidth || el.offsetHeight || el.offsetParent);
  };
  const cls = (el) => {
    try { return String(el.className || '').slice(0, 90); } catch (e) { return ''; }
  };
  const probe = (hints) => {
    const found = [];
    for (const sel of (hints || [])) {
      let el = null;
      try { el = document.querySelector(sel); } catch (e) { continue; }
      if (rendered(el) && found.indexOf(sel) === -1) found.push(sel);
    }
    return found;
  };

  const placeholders = [];
  for (const sel of (opts.loading || [])) {
    let els = [];
    try { els = document.querySelectorAll(sel); } catch (e) { continue; }
    for (const el of els) {
      if (!rendered(el)) continue;
      const c = cls(el);
      if (c && placeholders.indexOf(c) === -1) placeholders.push(c);
      if (placeholders.length >= 12) break;
    }
  }

  // How much is really on screen? A mounted app has thousands of elements and
  // a body text of many thousands of characters. 122 characters means the
  // layout shell rendered and nothing else did.
  let sidebarChars = 0;
  try {
    const sb = document.querySelector('#board-layout-sidebar');
    if (sb) sidebarChars = ((sb.innerText || '') + '').trim().length;
  } catch (e) { /* diagnostics only */ }

  const panel = probe(opts.panel);
  const movelist = probe(opts.movelist);
  let cells = 0;
  try { cells = document.querySelectorAll('[data-node]').length; } catch (e) {}
  return {
    placeholders: placeholders,
    layout: probe(opts.layout),
    panel: panel,
    movelist: movelist,
    loaded: placeholders.length === 0,
    layoutFound: probe(opts.layout).length > 0,
    panelFound: panel.length > 0,
    moveListFound: movelist.length > 0,
    moveCells: cells,
    bodyChars: (document.body.innerText || '').length,
    sidebarChars: sidebarChars,
    elements: document.querySelectorAll('*').length,
  };
}
"""

_REVIEW_PANEL_JS = r"""
(hints) => {
  const rendered = (el) => {
    if (!el) return false;
    if (el.getClientRects && el.getClientRects().length === 0) return false;
    return !!(el.offsetWidth || el.offsetHeight || el.offsetParent);
  };
  const found = [];
  for (const sel of hints) {
    let el = null;
    try { el = document.querySelector(sel); } catch (e) { continue; }
    if (rendered(el) && found.indexOf(sel) === -1) found.push(sel);
  }
  return found;
}
"""


async def read_review_state(page) -> dict:
    """One snapshot of how far the review page has got.

    Returns ``{placeholders, layout, panel, movelist, loaded, layoutFound,
    panelFound, moveListFound, moveCells, bodyChars, sidebarChars, elements}``.
    ``loaded`` alone is NOT enough to read from -- see wait_for_review_loaded.
    """
    try:
        return await page.evaluate(
            _REVIEW_STATE_JS,
            {
                "loading": LOADING_HINTS,
                "layout": REVIEW_LAYOUT_HINTS,
                "panel": REVIEW_PANEL_HINTS,
                "movelist": MOVE_LIST_HINTS,
            },
        )
    except _PlaywrightError as exc:
        _LOG.warning("review-state probe failed: %s", exc)
        return {
            "placeholders": [],
            "layout": [],
            "panel": [],
            "movelist": [],
            "loaded": True,
            "layoutFound": False,
            "panelFound": False,
            "moveListFound": False,
            "moveCells": 0,
            "bodyChars": 0,
            "sidebarChars": 0,
            "elements": 0,
            "error": str(exc),
        }


async def review_panel_present(page) -> bool:
    """True when the summary panel -- accuracy or tally -- is on screen.

    Note this is true only BEFORE the review is started. Pressing Start Review
    replaces the summary with the move-by-move view, which is why the read
    order is: summary first, then start, then moves.
    """
    try:
        found = await page.evaluate(_REVIEW_PANEL_JS, REVIEW_PANEL_STRICT)
    except _PlaywrightError as exc:
        _LOG.debug("review-panel probe failed: %s", exc)
        return False
    return bool(found)


def describe_state(state: dict, require: str) -> str:
    """A one-line reason the wait is still going, for logs and for the CLI."""
    return (
        f"placeholders={(state.get('placeholders') or [])[:2]} "
        f"layout={'yes' if state.get('layoutFound') else 'no'} "
        f"panel={(state.get('panel') or [])[:2] or 'none'} "
        f"movelist={'yes' if state.get('moveListFound') else 'no'} "
        f"cells={state.get('moveCells')} "
        f"elements={state.get('elements')} bodyChars={state.get('bodyChars')} "
        f"sidebarChars={state.get('sidebarChars')}"
    )


async def wait_for_review_loaded(
    page,
    require: str = "panel",
    timeout_ms: int | None = None,
    poll_ms: int = 500,
) -> bool:
    """Block until the review page has got far enough to read from.

    Readiness needs BOTH "no placeholders" AND a stage-specific positive
    signal. Absence of loaders alone is true before chess.com's Vue app has
    mounted the sidebar at all, and the board renders with the layout shell, so
    neither is usable as the data signal on its own.

    * ``require="layout"``  the shell: no placeholders, and the board exists
    * ``require="panel"``   the review data: ``.review-rating-*`` or a tally
    * ``require="movelist"`` the moves: ``[data-node]`` exists

    Returns True once the stage is confirmed. A False return means the timeout
    elapsed, the state is reported in the log, and the caller should treat
    whatever it reads as incomplete rather than as "there is nothing here".
    """
    if require not in ("layout", "panel", "movelist"):
        raise ValueError(
            f"require must be 'layout', 'panel' or 'movelist', not {require!r}"
        )
    if timeout_ms is None:
        timeout_ms = {
            "panel": DEFAULT_PANEL_TIMEOUT_MS,
            "layout": int(os.getenv("REVIEW_LAYOUT_TIMEOUT_MS", "30000")),
            "movelist": int(os.getenv("REVIEW_MOVELIST_TIMEOUT_MS", "30000")),
        }[require]
    key = {"panel": "panelFound", "layout": "layoutFound", "movelist": "moveListFound"}[
        require
    ]

    waited = 0
    state: dict = {}
    while waited <= timeout_ms:
        state = await read_review_state(page)
        if state.get("loaded") and state.get(key):
            _LOG.info(
                "review %s ready after %dms (%s)",
                require,
                waited,
                (state.get("panel") if require == "panel" else state.get("layout"))
                if require != "movelist"
                else (state.get("movelist") or []),
            )
            return True
        if waited == 0:
            _LOG.info(
                "waiting for review %s: %s", require, describe_state(state, require)
            )
        try:
            await page.wait_for_timeout(poll_ms)
        except _PlaywrightError:
            return False
        waited += poll_ms
    _LOG.warning(
        "review %s NOT ready after %dms: %s. Reading anyway; results will be incomplete.",
        require,
        timeout_ms,
        describe_state(state, require),
    )
    return False


# How long to wait for the review data. chess.com computes it server-side.
DEFAULT_PANEL_TIMEOUT_MS = 90000


async def expand_review_summary(page) -> bool:
    """Click the chevron that expands the collapsed classification tally.

    The sidebar shows a truncated tally with a chevron underneath that reveals
    the rest, including Brilliant and Inaccuracy. Without this the summary
    screenshot is missing two of the classifications the user most wants to see,
    and the reply can disagree with the picture beside it.

    Uses the button's accessible name rather than a class, because the chevron
    is a generic icon button.
    """
    for sel in (
        "button[aria-label='Expand']",
        "[aria-label='Expand']",
        "button[aria-label*='expand' i]",
    ):
        try:
            loc = page.locator(sel).first
            if await loc.count() == 0 or not await loc.is_visible(timeout=1500):
                continue
            await loc.click(timeout=5000)
            _LOG.info("expanded the collapsed review summary")
            # The panel reflows and may scroll; let it settle.
            try:
                await page.wait_for_timeout(700)
            except _PlaywrightError:
                pass
            return True
        except _PlaywrightError as exc:
            _LOG.debug("expand control %r unusable: %s", sel, exc)
    _LOG.info("no expand control found; the summary may already be full")
    return False


async def scroll_sidebar_to_top(page) -> None:
    """Scroll the review sidebar back to the top before shooting it.

    The sidebar scrolls internally, and it is usually left part-way down after
    the move list was read. An element screenshot of a scrolling container
    captures only what is rendered, so this has to happen first.
    """
    for sel in (*REVIEW_SIDEBAR_HINTS, *CONTAINER_HINTS):
        try:
            loc = page.locator(sel).first
            if await loc.count() == 0:
                continue
            await loc.evaluate(
                "el => { el.scrollTop = 0;"
                " for (let n = el; n; n = n.parentElement) n.scrollTop = 0; }"
            )
            return
        except _PlaywrightError as exc:
            _LOG.debug("could not reset sidebar scroll for %r: %s", sel, exc)


# ---------------------------------------------------------------------------
# The move table
# ---------------------------------------------------------------------------


def ply_from_node(node: str | None) -> int | None:
    """The ply a ``data-node`` attribute refers to, or None if malformed.

    chess.com writes ``data-node="<variation>-<index>"`` on every move cell, so
    the ply is something it told us rather than something we counted. The
    captured game had 29 cells covering plies 1..29 with no gaps and no
    duplicates.

    The same formula appears in ``_MOVE_TABLE_JS``. A test asserts the two stay
    in step, because if they drift the reader silently renumbers the game.
    """
    match = re.match(r"^(\d+)\s*-\s*(\d+)$", (node or "").strip())
    if not match:
        return None
    return int(match.group(1)) * 2 + int(match.group(2)) + 1


# This blob only *extracts facts*. The classification vocabulary lives in
# Python (see RULES), so there is one place to add a kind and it is testable
# without a browser. Everything it reads is an attribute chess.com wrote on
# purpose:
#   data-node="<variation>-<index>"  the authoritative ply
#   data-whole-move-number           on the ROW, not the cell
#   data-ply / data-time             the clock, in centiseconds
#   g[id=...] in the cell's icon     the classification
#   --color-classification-*         the same thing, in the SAN span's style
_MOVE_TABLE_JS = r"""
() => {
  const rendered = (el) => {
    if (!el) return false;
    if (el.getClientRects && el.getClientRects().length === 0) return false;
    return !!(el.offsetWidth || el.offsetHeight || el.offsetParent);
  };
  const clsOf = (el) => {
    try {
      const c = el.className;
      if (c && c.baseVal !== undefined) return String(c.baseVal);
      return typeof c === 'string' ? c : '';
    } catch (e) { return ''; }
  };
  const textOf = (el) => {
    if (!el) return '';
    const t = (el.innerText || '').trim();
    return t || (el.textContent || '').trim();
  };
  const CSS = /--color-classification-([a-zA-Z-]+)/;

  const DIAG = {
    source: 'data-node',
    cells: 0,
    rows: 0,
    skipped: 0,
    hiddenDropped: 0,
    classified: 0,
    unclassified: 0,
    withClock: 0,
    firstNode: null,
    lastNode: null,
    pliesContiguous: true,
  };

  let cells = [];
  try { cells = Array.prototype.slice.call(document.querySelectorAll('[data-node]')); }
  catch (e) { return { rows: [], diagnostics: DIAG, supported: false }; }
  DIAG.cells = cells.length;
  if (!cells.length) {
    return { rows: [], diagnostics: DIAG, supported: false };
  }

  // Tag each cell with its ply so navigation can click exactly this cell.
  // The pairing is by construction: the loop that computes the ply writes the
  // tag. Stale tags from a previous scan are cleared first.
  const PLY_ATTR = 'data-ai-ply';
  for (const stale of Array.prototype.slice.call(
    document.querySelectorAll('[' + PLY_ATTR + ']'))) {
    stale.removeAttribute(PLY_ATTR);
  }

  const out = [];
  const seenPly = {};
  for (const cell of cells) {
    const node = cell.getAttribute('data-node') || '';
    const m = /^(\d+)\s*-\s*(\d+)$/.exec(node);
    if (!m) { DIAG.skipped++; continue; }
    if (!rendered(cell)) { DIAG.hiddenDropped++; continue; }
    const ply = parseInt(m[1], 10) * 2 + parseInt(m[2], 10) + 1;
    if (seenPly[ply]) { DIAG.skipped++; continue; }
    seenPly[ply] = true;

    const row = cell.closest('[data-whole-move-number]');
    const numberRaw = row ? row.getAttribute('data-whole-move-number') : null;
    const number = numberRaw && /^\d{1,3}$/.test(numberRaw.trim())
      ? parseInt(numberRaw, 10) : null;
    if (row) DIAG.rows++;

    const cls = clsOf(cell);
    const color = /\bwhite-move\b/.test(cls) ? 'white'
      : (/\bblack-move\b/.test(cls) ? 'black' : null);

    // The SAN lives in node-highlight-content. A white move also carries a
    // piece figurine as a child element with no text, so innerText is already
    // just the SAN -- no stripping needed.
    const content = cell.querySelector('.node-highlight-content') || cell;
    const san = textOf(content).replace(/\s+/g, ' ').trim();
    const fig = content.querySelector('[data-figurine]');
    const figurine = fig ? fig.getAttribute('data-figurine') : null;

    // The clock is a sibling of the cells, inside the row, keyed by ply.
    let clockCs = null;
    if (row) {
      const ts = row.querySelector(
        '[data-move-list-el="timestamp"][data-ply="' + ply + '"]');
      if (ts) {
        const raw = ts.getAttribute('data-time');
        if (raw && /^\d{1,6}$/.test(raw)) clockCs = parseInt(raw, 10);
      }
    }
    if (clockCs !== null) DIAG.withClock++;

    // The two explicit classification labels.
    let iconId = null;
    const icon = cell.querySelector('.node-annotation-icon svg g[id], svg g[id]');
    if (icon) iconId = icon.getAttribute('id');
    let cssKind = null;
    for (const el of [content, cell].concat(
      Array.prototype.slice.call(cell.querySelectorAll('[style]')))) {
      const m2 = CSS.exec(el.getAttribute && el.getAttribute('style') || '');
      if (m2) { cssKind = m2[1]; break; }
    }
    if (iconId || cssKind) DIAG.classified++; else DIAG.unclassified++;

    cell.setAttribute(PLY_ATTR, String(ply));
    out.push({
      ply: ply,
      node: node,
      number: number,
      color: color,
      san: san,
      figurine: figurine,
      clock_cs: clockCs,
      icon_id: iconId,
      css_classification: cssKind,
      classes: cls.slice(0, 90),
    });
  }

  out.sort((a, b) => a.ply - b.ply);
  if (out.length) {
    DIAG.firstNode = out[0].node;
    DIAG.lastNode = out[out.length - 1].node;
    for (let i = 0; i < out.length; i++) {
      if (out[i].ply !== i + 1) { DIAG.pliesContiguous = false; break; }
    }
  }
  return { rows: out, diagnostics: DIAG, supported: true, plies: out.length };
}
"""

# The coach panel. Rendered for the CURRENT move only -- one
# `move-feedback-box-component` exists in the whole document, so commentary for
# a specific move means navigating to it first.
_COACH_JS = r"""
() => {
  const text = (el) => el ? ((el.innerText || el.textContent || '').trim()) : null;
  // The coach's prose is revealed word by word: revealed words sit in plain
  // spans, unrevealed ones in `.move-feedback-speech-text-hidden`. innerText
  // honours their hidden styling, so it returns only what has been revealed so
  // far; textContent sees the whole sentence.
  const speechOf = (el) => el
    ? (el.textContent || '').replace(/\s+/g, ' ').trim() : null;
  const box = document.querySelector('.move-feedback-box-component');
  if (!box) return { present: false };
  const score = box.querySelector('.move-feedback-box-score');
  const scoreCls = score ? String(score.className || '') : '';
  const scoreText = text(score);
  const icon = box.querySelector('.move-feedback-box-icon svg g[id]');
  const speech = box.querySelector('.move-feedback-speech-text-component');
  // chess.com has two class families for the same thing -- `score-text-negative`
  // and `move-feedback-box-negative` -- so both are checked, and the leading
  // + or - is the last resort. A score reported as "+2.90" with sign null is
  // worse than no score at all.
  let sign = null;
  if (/negative/.test(scoreCls)) sign = -1;
  else if (/positive/.test(scoreCls)) sign = 1;
  else if (scoreText && /^\s*-/.test(scoreText)) sign = -1;
  else if (scoreText && /^\s*\+/.test(scoreText)) sign = 1;
  // Book moves get an opening name ("Grob Opening") instead of a verdict, on a
  // sibling of the feedback box, tagged with the move number it belongs to.
  const book = document.querySelector('.move-feedback-last-book-move-component');
  const bookNum = book ? book.getAttribute('whole-move-number') : null;
  return {
    present: true,
    icon_id: icon ? icon.getAttribute('id') : null,
    // e.g. "Book", "Blunder" -- the human-readable verdict
    verdict: text(box.querySelector('.move-feedback-box-description')),
    san: text(box.querySelector('.move-san-san')),
    score: scoreText,
    score_sign: sign,
    commentary: speechOf(speech),
    opening: text(document.querySelector('.move-feedback-last-book-move-title')),
    last_book_move_number: bookNum && /^\d{1,3}$/.test(bookNum)
      ? parseInt(bookNum, 10) : null,
  };
}
"""

# The summary panel: the accuracy pair, the classification tally, and the
# player names as they appear. Only present BEFORE the review is started.
_SUMMARY_JS = r"""
() => {
  const num = (el) => {
    if (!el) return null;
    const m = /(\d{1,3}(?:\.\d+)?)/.exec((el.innerText || el.textContent || '').trim());
    return m ? parseFloat(m[1]) : null;
  };
  // The tally is a TWO-COLUMN table: one tally-row-value per side, in the same
  // order as the accuracy row above it (`review-rating-white` then
  // `review-rating-black`). The captured game had 20 of them -- 10 kinds x 2
  // sides -- so collecting into a single key and keeping the last value reports
  // ONE player's tally and calls it the game's. Collect every column instead.
  const pairs = {};
  for (const el of Array.prototype.slice.call(
    document.querySelectorAll('[class*="tally-row-value"]'))) {
    const cls = String(el.className || '');
    const m = /analysis-([a-zA-Z]+)/.exec(cls);
    const t = (el.textContent || '').trim();
    if (!m || !/^\d{1,3}$/.test(t)) continue;
    const key = m[1].toLowerCase();
    if (!pairs[key]) pairs[key] = [];
    pairs[key].push(parseInt(t, 10));
  }
  const names = Array.prototype.slice.call(
    document.querySelectorAll('.cc-user-username-component')).map(
    (e) => ((e.innerText || e.textContent || '').trim()) || null);
  const ratings = Array.prototype.slice.call(
    document.querySelectorAll('.cc-user-rating-boldest')).map((e) => {
      const m = /\((\d{2,5})\)/.exec((e.innerText || e.textContent || '').trim());
      return m ? parseInt(m[1], 10) : null;
    });
  return {
    white: num(document.querySelector('.review-rating-white')),
    black: num(document.querySelector('.review-rating-black')),
    // {kind: [white_column, black_column]}
    tally_pairs: pairs,
    tally_columns: names.length === 2 ? 2 : (pairs[Object.keys(pairs)[0]] || []).length,
    // Reported in DOM order, NOT paired to a side: the dump shows the names in
    // a separate block from the accuracies and nothing in the markup says which
    // name belongs to which rating. Say that rather than guess.
    players_in_dom_order: names,
    ratings_in_dom_order: ratings,
  };
}
"""


_ROW_FIELDS = (
    "ply",
    "number",
    "color",
    "san",
    "figurine",
    "clock_seconds",
    "classification",
    "confidence",
    "evidence",
)


def _parse_move_table(payload: dict) -> tuple[list[dict], dict]:
    """Apply the classification vocabulary to the browser's raw facts.

    ``supported`` is echoed into the diagnostics on both paths. "The page had no
    move table" and "the move table was empty" are different failures, and a
    caller that only sees ``rows == []`` cannot tell them apart.
    """
    diag = dict(payload.get("diagnostics", {}) or {})
    diag["supported"] = bool(payload.get("supported"))
    if not payload["supported"]:
        return [], diag
    rows = []
    for raw in payload.get("rows", []):
        kind, confidence, evidence = classify_facts(
            raw.get("icon_id"),
            raw.get("css_classification"),
            raw.get("classes"),
        )
        clock_cs = raw.get("clock_cs")
        rows.append(
            {
                "ply": raw.get("ply"),
                "number": raw.get("number"),
                "color": raw.get("color"),
                "san": raw.get("san", ""),
                "figurine": raw.get("figurine"),
                # centiseconds in the DOM; seconds is what a person wants
                "clock_seconds": round(clock_cs / 10.0, 1)
                if clock_cs is not None
                else None,
                "classification": kind,
                "confidence": confidence,
                "evidence": evidence,
            }
        )
    diag["pliesKnown"] = payload.get("plies")
    diag["pliesRead"] = len(rows)
    diag["coverage"] = f"{len(rows)}/{payload.get('plies')}"
    return rows, diag


async def read_moves(page) -> tuple[list[dict], dict]:
    """Read every move from the sidebar.

    The ply comes from ``data-node``, which chess.com writes on every move
    cell, so the numbering is authoritative: the captured markup had 29 cells
    with plies 1..29 and no gaps. There is no ambiguity left to guard against.

    Returns ``(rows, diagnostics)`` where each row is
    ``{ply, number, color, san, figurine, clock_seconds, classification,
    confidence, evidence}``. ``classification`` is None when chess.com did not
    label the move -- there is no colour inference, so an unlabelled move is
    reported as unlabelled.

    Diagnostics carry ``source`` and ``pliesContiguous``, so a future
    chess.com change that breaks the assumption is visible rather than silent.
    """
    try:
        payload = await page.evaluate(_MOVE_TABLE_JS)
    except _PlaywrightError as exc:
        _LOG.warning("move table read failed: %s", exc)
        return [], {"source": "data-node", "error": str(exc), "supported": False}
    return _parse_move_table(payload)


async def read_move(page, ply: int) -> tuple[dict | None, dict]:
    """Read a single ply out of the move table."""
    rows, diag = await read_moves(page)
    for row in rows:
        if row.get("ply") == ply:
            return row, diag
    return None, diag


async def read_coach_commentary(page) -> dict:
    """Read the coach's panel for the move currently displayed.

    Returns ``{}`` when there is no coach panel, which is the case before the
    review is started and for a move the coach has nothing to say about. The
    panel is rendered for one move at a time -- exactly one
    ``.move-feedback-box-component`` existed in the whole captured document --
    so to comment on a specific move, navigate to it first.
    """
    try:
        return await page.evaluate(_COACH_JS)
    except _PlaywrightError as exc:
        _LOG.debug("coach panel read failed: %s", exc)
        return {}


def split_tally(pairs: dict, columns: int = 2) -> tuple[dict, dict[str, dict]]:
    """Split the two-column tally into a per-side pair and a per-side total.

    chess.com writes ``analysis-greatFind`` and ``analysis-bestMove`` -- the same
    camelCase the move cells use -- so the keys are normalised through RULES too.

    Returns ``(totals, by_side)``. Column 0 is White and column 1 is Black,
    because that is the order of the accuracy row directly above the tally in
    the same ``game-overview-row`` structure, where the classes are literally
    ``review-rating-white`` and ``review-rating-black``.

    The distinction matters: collapsing the columns into one key reports a
    single player's tally as the whole game's, which disagrees with the move
    list it is supposed to be summarising.
    """
    totals: dict[str, int] = {}
    by_side: dict[str, dict[str, int]] = {"white": {}, "black": {}}
    for raw_kind, values in (pairs or {}).items():
        kind = classify_class(raw_kind) or str(raw_kind).lower()
        numbers = [v for v in (values or []) if isinstance(v, int)]
        if not numbers:
            continue
        totals[kind] = sum(numbers)
        for side, value in zip(("white", "black"), numbers[:columns]):
            by_side[side][kind] = value
    return totals, by_side


async def read_summary_panel(page) -> dict:
    """Read the accuracy pair, the classification tally and the player names.

    Only valid BEFORE the review is started: pressing Start Review replaces the
    summary with the move-by-move view, which is why the read order is fixed
    rather than defensive.

    Returns the raw pair-per-kind tally as ``tally_pairs``, plus the two derived
    views a caller actually wants: ``move_counts`` (the whole game) and
    ``move_counts_by_side``.
    """
    try:
        summary = await page.evaluate(_SUMMARY_JS)
    except _PlaywrightError as exc:
        _LOG.debug("summary panel read failed: %s", exc)
        return {}
    if not isinstance(summary, dict):
        return {}
    totals, by_side = split_tally(
        summary.get("tally_pairs"), summary.get("tally_columns") or 2
    )
    summary["move_counts"] = totals
    summary["move_counts_by_side"] = by_side
    return summary


async def read_tally(page) -> dict[str, int]:
    """The classification counts for the WHOLE game, keyed by kind.

    One reader on the real class (``tally-row-value analysis-<kind>``) is
    enough, so that is all there is. Both columns are summed; use
    ``read_tally_by_side`` to see them apart.
    """
    summary = await read_summary_panel(page)
    return summary.get("move_counts") or {}


async def read_tally_by_side(page) -> dict[str, dict[str, int]]:
    """The classification counts split by side.

    This is the view that makes the two numbers add up. The summary's tally row
    is a two-column table, and collapsing the columns is how a review comes to
    claim three mistakes where the move list had one.
    """
    summary = await read_summary_panel(page)
    return summary.get("move_counts_by_side") or {}


async def read_ratings(page) -> tuple[float | None, float | None]:
    """The two accuracy numbers, or ``(None, None)`` if they are not loaded."""
    summary = await read_summary_panel(page)
    return summary.get("white"), summary.get("black")


async def find_classified(
    page, kinds: tuple[str, ...] = ("brilliant", "blunder")
) -> tuple[list[dict], dict]:
    """Every move whose classification is one of ``kinds``, in ply order."""
    rows, diag = await read_moves(page)
    wanted = {k.lower() for k in kinds}
    hits = [r for r in rows if (r.get("classification") or "").lower() in wanted]
    if not hits:
        _LOG.info(
            "no %s among %d moves read from the move table (%s)",
            "/".join(kinds),
            len(rows),
            diag.get("coverage") or "coverage unknown",
        )
    return hits, diag


# Which sidebar tab plausibly holds the move list. Kept deliberately narrow:
# the default tab is the "Game Overview", so a looser pattern could click the
# very tab the moves are not behind.
_MOVE_TAB = re.compile(r"\b(move|moves|movelist|notation|variation)s?\b", re.I)
_TAB_BLOCKLIST = re.compile(r"overview|summary|settings|home|share|report|learn", re.I)


async def open_moves_tab(page) -> bool:
    """Click a sidebar tab that looks like it holds the move list.

    Kept as a fallback for older layouts. On the current page the moves are
    already present once the review is started, so this normally does nothing.
    """
    for selector in TAB_HINTS:
        try:
            locators = await page.locator(selector).all()
        except _PlaywrightError as exc:
            _LOG.debug("tab scan %r failed: %s", selector, exc)
            continue
        for loc in locators:
            try:
                if not await loc.is_visible(timeout=1000):
                    continue
                label = (await loc.inner_text()).strip() or (
                    await loc.get_attribute("aria-label") or ""
                )
            except _PlaywrightError:
                continue
            if not label or len(label) > 30:
                continue
            if _TAB_BLOCKLIST.search(label) or not _MOVE_TAB.search(label):
                continue
            _LOG.info("clicking sidebar tab %r", label)
            try:
                await loc.click(timeout=4000)
                try:
                    await page.wait_for_timeout(1200)
                except _PlaywrightError:
                    pass
                return True
            except _PlaywrightError as exc:
                _LOG.debug("tab %r not clickable: %s", label, exc)
    return False


async def read_moves_when_ready(page) -> tuple[list[dict], dict]:
    """Drive all three stages and return the move table.

    1. The layout shell mounts.
    2. The review data arrives seconds later, server-side computed.
    3. The move list does not exist until ``Start Review`` is pressed, so the
       page can show a complete tally and zero moves at the same time.

    The caller must already have read anything it needs from the summary,
    because step 3 destroys it.
    """
    await wait_for_review_loaded(page, require="panel")
    if not await open_review_mode(page):
        _LOG.info("no review-start control; reading whatever is present")
    await wait_for_review_loaded(page, require="movelist", timeout_ms=15000)

    rows, diag = await read_moves(page)
    if not rows and await open_moves_tab(page):
        await wait_for_review_loaded(page, require="movelist", timeout_ms=15000)
        rows, diag = await read_moves(page)
    return rows, diag


# ---------------------------------------------------------------------------
# Scrolling. The list turned out NOT to be virtualised -- 29 cells at every
# one of five scroll offsets -- so this is belt-and-braces for a future
# layout that is.
# ---------------------------------------------------------------------------

_SCROLL_INFO_JS = r"""
(hints) => {
  for (const sel of hints) {
    let el = null;
    try { el = document.querySelector(sel); } catch (e) { continue; }
    if (!el) continue;
    let n = el;
    for (let i = 0; i < 10 && n; i++) {
      if (n.scrollHeight > n.clientHeight + 20) {
        return {
          found: true,
          scrollHeight: n.scrollHeight,
          clientHeight: n.clientHeight,
          max: Math.max(0, n.scrollHeight - n.clientHeight),
        };
      }
      n = n.parentElement;
    }
  }
  return { found: false, max: 0 };
}
"""

_SCROLL_TO_JS = r"""
(opts) => {
  for (const sel of opts.hints) {
    let el = null;
    try { el = document.querySelector(sel); } catch (e) { continue; }
    if (!el) continue;
    let n = el;
    for (let i = 0; i < 10 && n; i++) {
      if (n.scrollHeight > n.clientHeight + 20) {
        n.scrollTop = Math.max(0, Math.min(opts.top, n.scrollHeight - n.clientHeight));
        return { set: n.scrollTop, max: n.scrollHeight - n.clientHeight };
      }
      n = n.parentElement;
    }
  }
  return null;
}
"""


async def _scroll_to_fraction(page, fraction: float) -> None:
    try:
        info = await page.evaluate(_SCROLL_INFO_JS, CONTAINER_HINTS)
    except _PlaywrightError:
        return
    if not info.get("found"):
        return
    top = int(info.get("max", 0) * max(0.0, min(1.0, fraction)))
    try:
        await page.evaluate(_SCROLL_TO_JS, {"hints": CONTAINER_HINTS, "top": top})
    except _PlaywrightError as exc:
        _LOG.debug("scroll to %d failed: %s", top, exc)


async def scroll_to_ply(page, ply: int, total: int) -> None:
    """Scroll the move list so ``ply`` is likely rendered, then re-read.

    Navigation clicks the cell the reader tagged, and a tagged cell that has
    been scrolled out of a virtualised window no longer exists. Harmless on the
    current layout, where every cell is always in the DOM.
    """
    fraction = 0.0 if total <= 1 else (max(1, ply) - 1) / (total - 1)
    await _scroll_to_fraction(page, fraction)
    try:
        await page.wait_for_timeout(250)
    except _PlaywrightError:
        pass


async def read_moves_full(page, settle_ms: int = 250) -> tuple[list[dict], dict]:
    """Read the move table, sweeping the list if it turns out to be virtualised.

    Merges by ply, preferring a classified row over an unclassified one so
    classification is never lost to a later, blanker pass. On the captured
    layout this is a single read -- the sweep is a no-op -- and that is
    reported rather than hidden.
    """
    merged: dict[int, dict] = {}
    diag: dict = {}
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        await _scroll_to_fraction(page, fraction)
        try:
            await page.wait_for_timeout(settle_ms)
        except _PlaywrightError:
            pass
        rows, diag = await read_moves(page)
        for row in rows:
            ply = row.get("ply")
            if not ply:
                continue
            previous = merged.get(ply)
            if previous is None or (
                not previous.get("classification") and row.get("classification")
            ):
                merged[ply] = row
    await _scroll_to_fraction(page, 0.0)
    rows = [merged[p] for p in sorted(merged)]
    known = diag.get("pliesKnown") or len(rows)
    diag = dict(diag)
    diag["pliesKnown"] = known
    diag["pliesRead"] = len(rows)
    diag["coverage"] = f"{len(rows)}/{known}"
    if len(rows) < known:
        _LOG.info("read %d of %d plies; the list may be virtualised", len(rows), known)
    return rows, diag


# ---------------------------------------------------------------------------
# Board-state verification
# ---------------------------------------------------------------------------

_POSITION_JS = r"""
(hints) => {
  for (const sel of hints) {
    let el = null;
    try { el = document.querySelector(sel); } catch (e) { continue; }
    if (!el) continue;
    const attr = el.getAttribute && el.getAttribute('position');
    if (attr) return 'pos:' + String(attr);
    const fen = el.getAttribute &&
      (el.getAttribute('data-fen') || el.getAttribute('fen'));
    if (fen) return 'fen:' + String(fen);
    if (typeof el.position === 'string' && el.position) {
      return 'prop:' + el.position;
    }
  }
  return null;
}
"""

# A chessboard is square. This is the tolerance for *verification*, where being
# lenient is safe because a false negative only costs a fallback. The crop in
# tools/review.py is much stricter, because a wrong box there is a wrong photo.
MIN_BOARD_PX = 100
MIN_BOARD_ASPECT = 0.7
MAX_BOARD_ASPECT = 1.43


async def position_signature(page) -> str | None:
    """A board-position fingerprint from the DOM, or None if unavailable."""
    for sel in POSITION_HINTS:
        try:
            sig = await page.evaluate(_POSITION_JS, [sel])
        except _PlaywrightError as exc:
            _LOG.debug("position probe %r failed: %s", sel, exc)
            continue
        if sig:
            return sig
    return None


def _looks_like_a_board(clip: dict | None) -> bool:
    """True when a bounding box is plausibly the chessboard.

    Without this, any element whose class contains "board" qualifies -- and the
    sidebar header gear (``board-layout-icon``, 24x24) qualifies perfectly well,
    and animates when you press an arrow key.
    """
    if not clip:
        return False
    w, h = clip.get("width") or 0, clip.get("height") or 0
    if w < MIN_BOARD_PX or h < MIN_BOARD_PX:
        return False
    return MIN_BOARD_ASPECT <= w / h <= MAX_BOARD_ASPECT


async def board_signature(page) -> str | None:
    """A signature that changes whenever the board does.

    Prefers the board's own position attribute, then a hash of the board's
    pixels. If no board-sized square element exists the answer is None and the
    caller reports "cannot verify" rather than inventing a pass.
    """
    sig = await position_signature(page)
    if sig:
        return sig
    for sel in BOARD_HINTS:
        try:
            locs = page.locator(sel)
            for i in range(min(await locs.count(), 3)):
                box = await locs.nth(i).bounding_box()
                if not _looks_like_a_board(box):
                    continue
                region = {
                    "x": max(0, box["x"]),
                    "y": max(0, box["y"]),
                    "width": min(box["width"], 800),
                    "height": min(box["height"], 800),
                }
                shot = await page.screenshot(
                    clip=region, type="png", animations="disabled"
                )
                # Change detection only, not a security primitive.
                return "img:" + hashlib.sha256(shot).hexdigest()
        except _PlaywrightError as exc:
            _LOG.debug("board probe %r failed: %s", sel, exc)
    _LOG.info("no square board-sized element found; cannot verify movement")
    return None


async def board_is_stable(
    page, checks: int = 3, gap_ms: int = 400
) -> tuple[bool, str | None]:
    """Does the board stay put when nothing is pressed?

    The coach walks the game by itself once the review is running, so the board
    can drift between two of our own steps and a self-inflicted move becomes
    indistinguishable from one our button caused. Returns ``(stable, sig)``.
    """
    first = await board_signature(page)
    if first is None:
        return False, None
    for _ in range(max(1, checks - 1)):
        try:
            await page.wait_for_timeout(gap_ms)
        except _PlaywrightError:
            return False, first
        if await board_signature(page) != first:
            return False, await board_signature(page)
    return True, first


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

NEXT_MOVE_SELECTORS = [
    "[class*='game-controls' i] [class*='next' i]",
    "[class*='next-move' i]",
    "[class*='move-next' i]",
    "button[aria-label*='Next' i]",
    "[data-testid*='next' i]",
]
PREV_MOVE_SELECTORS = [
    "[class*='game-controls' i] [class*='prev' i]",
    "[class*='prev-move' i]",
    "[class*='move-prev' i]",
    "button[aria-label*='Previous' i]",
    "button[aria-label*='Back' i]",
    "[data-testid*='prev' i]",
]


@dataclass
class StepResult:
    """Outcome of a navigation request.

    ``verified`` is False when no board signal was readable. Treating "could
    not check" as "it moved" is how a run reports moves on a page it cannot
    actually read.
    """

    requested: int
    moved: int = 0
    verified: bool = False
    method: str = ""
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.verified and self.moved == abs(self.requested)

    def __bool__(self) -> bool:
        return self.ok


async def _click_first(page, selectors: list[str], timeout: int = 3000):
    for sel in selectors:
        try:
            loc = page.locator(sel).first
            if await loc.count() > 0 and await loc.is_visible(timeout=timeout):
                await loc.click(timeout=timeout)
                return sel
        except _PlaywrightError as exc:
            _LOG.debug("nav control %r unusable: %s", sel, exc)
    return None


async def step_moves(page, count: int) -> StepResult:
    """Step ``count`` plies forward (positive) or back (negative).

    Every step is checked against a board signature; a click that changes
    nothing counts as a failure, and an unreadable board yields
    ``verified=False`` rather than an optimistic success.
    """
    result = StepResult(requested=count)
    if count == 0:
        result.verified = True
        return result

    for _ in range(abs(count)):
        before = await board_signature(page)
        if before is None:
            result.reason = "board position is not readable; cannot verify"
            return result

        hit = await _click_first(
            page, NEXT_MOVE_SELECTORS if count > 0 else PREV_MOVE_SELECTORS
        )
        if not hit:
            key = "ArrowRight" if count > 0 else "ArrowLeft"
            try:
                await page.keyboard.press(key)
                hit = f"keyboard:{key}"
            except _PlaywrightError as exc:
                _LOG.debug("keyboard %r failed: %s", key, exc)
                result.reason = "no step control and no keyboard support"
                return result
        result.method = hit

        try:
            await page.wait_for_timeout(150)
        except _PlaywrightError as exc:
            _LOG.debug("settle wait failed: %s", exc)
            result.reason = "settle wait failed"
            return result

        after = await board_signature(page)
        if after is not None and after != before:
            result.moved += 1
        else:
            # An unreadable "after" is NOT a move. ``verified`` must never
            # default to optimistic, which is what StepResult promises.
            result.reason = (
                "board became unreadable"
                if after is None
                else f"{hit} did not change the board"
            )
            return result

    result.verified = True
    return result


async def next_move(page, count: int = 1) -> StepResult:
    """Step forward ``count`` plies."""
    return await step_moves(page, count)


async def previous_move(page, count: int = 1) -> StepResult:
    """Step back ``count`` plies."""
    return await step_moves(page, -count)


async def goto_ply(page, ply: int) -> bool:
    """Navigate to ``ply`` **in-page**, verifying the board moved.

    Clicking the sidebar cell beats a ``?move=N`` reload: it keeps the review
    session, the board tint, the coach bubble and the eval marker, and costs
    no Cloudflare round trip.
    """
    if ply < 1:
        return False
    current = await active_ply(page)
    if current and current != ply and abs(current - ply) <= 8:
        moved = await step_moves(page, ply - current)
        if moved.ok:
            return True
    # Re-read the signature *after* any partial stepping. Capturing it before
    # let a half-completed step satisfy "did the board change?" on its own.
    before = await board_signature(page)
    if before is None:
        _LOG.info("board signature unreadable; cannot verify navigation to ply %d", ply)
        return False
    if await _click_row(page, ply) and await board_signature(page) != before:
        return True
    return False


# Where the board is, read two independent ways and cross-checked.
#
# Counting elements with move-like text up to the active one is wrong in the
# worst direction: on the live review page it reports a ply far ahead of the
# board. `goto_ply` uses `active_ply` to decide whether a short step is
# enough, so a wrong answer there steers navigation onto the wrong position
# while reporting that it verified.
#
# Two real signals exist, both confirmed by `spikes/_diag_active.py`:
#
#   * exactly ONE move cell carries `class="... selected"`, on its
#     `.node-highlight-content` span, inside the cell that has `data-node`
#   * the URL carries `?move=<index>`, and index is the ply-1
#
# Both are read and, when they disagree, the cell wins -- it is the thing the
# click actually targets. Disagreement is logged rather than hidden, so a
# mismatch shows up instead of silently steering navigation.
_ACTIVE_PLY_JS = r"""
() => {
  const m = /[?&]move=(-?\d+)/.exec(location.href);
  const fromUrl = m ? parseInt(m[1], 10) + 1 : null;
  let fromCell = null;
  const sel = document.querySelector(
    '[data-node] .node-highlight-content.selected, [data-node].selected'
  );
  if (sel) {
    const cell = sel.closest('[data-node]') || sel;
    const node = cell.getAttribute('data-node') || '';
    const pm = /^(\d+)\s*-\s*(\d+)$/.exec(node.trim());
    if (pm) fromCell = parseInt(pm[1], 10) * 2 + parseInt(pm[2], 10) + 1;
  }
  return {url: fromUrl, cell: fromCell, agree: fromUrl === fromCell};
}
"""


async def active_ply(page, container_hints: list[str] | None = None) -> int | None:
    """The ply the board is currently showing, or None if it cannot be read.

    Reads the selected move cell and the URL's `?move=` parameter and returns
    the one they agree on, preferring the cell. Pass `container_hints` to
    restrict the search (kept for callers that have a narrower container; the
    `selected` class is specific enough that it is not needed in practice).
    """
    try:
        probe = await page.evaluate(_ACTIVE_PLY_JS)
    except _PlaywrightError as exc:
        _LOG.debug("active-ply probe failed: %s", exc)
        return None
    if not isinstance(probe, dict):
        return None
    cell, url_ply = probe.get("cell"), probe.get("url")
    if cell and url_ply and cell != url_ply:
        # Logged, not raised: the cell is the click target and is trusted, but
        # a mismatch means chess.com changed something and the next failure may
        # be downstream of it.
        _LOG.info(
            "selected cell says ply %s but the URL says ply %s; trusting the cell",
            cell,
            url_ply,
        )
    return cell or url_ply


async def _click_row(page, ply: int) -> bool:
    """Click the move cell the reader counted as ``ply``.

    The cell tags itself while it is being parsed, so the index here is by
    construction the same one the classification came from. Indexing
    positionally into a class-based element union is not usable: on the live
    page that resolves to clock, aria and icon elements as well as moves, so
    ``ply - 1`` would point at an arbitrary element.
    """
    if ply < 1:
        return False
    try:
        loc = page.locator(f'[data-ai-ply="{ply}"]').first
        if await loc.count() == 0:
            _LOG.info(
                "no move cell is tagged as ply %d; read_moves() must run first", ply
            )
            return False
        await loc.scroll_into_view_if_needed(timeout=2000)
        await loc.click(timeout=3000)
        return True
    except _PlaywrightError as exc:
        _LOG.debug("clicking ply %d failed: %s", ply, exc)
        return False


async def open_review_mode(page) -> bool:
    """Click the control that *starts* the review, if it is on screen.

    Not optional decoration: the Game Review page serves the summary first and
    renders no moves at all until this is pressed, so a game containing a
    blunder reads as a game containing no errors without it.

    Returns True when a control was clicked. It deliberately does not wait:
    the caller owns the wait, and folding the two together makes a successful
    click followed by a slow load one indistinguishable False.
    """
    for sel in REVIEW_ENTRY_SELECTORS:
        try:
            loc = page.locator(sel).first
            if await loc.count() == 0 or not await loc.is_visible(timeout=1500):
                continue
            label = ((await loc.inner_text()) or "").strip()[:40]
        except _PlaywrightError as exc:
            _LOG.debug("review entry %r unusable: %s", sel, exc)
            continue
        # A heading is a title, not a control: the sidebar title reads
        # "Game Review". Also, Playwright's locator.evaluate binds the element
        # as the arrow function's *parameter*, so the string must be `el => ...`.
        try:
            is_heading = await loc.evaluate("el => !!el.closest('h1, h2')")
        except _PlaywrightError as exc:
            _LOG.debug("could not classify review entry %r: %s", label, exc)
            continue
        if is_heading:
            _LOG.debug("review entry %r is a heading, not a control", label)
            continue
        _LOG.info("clicking review entry %r (%r)", sel, label)
        try:
            await loc.click(timeout=8000)
        except _PlaywrightError as exc:
            _LOG.debug("review entry %r not clickable: %s", sel, exc)
            continue
        return True
    _LOG.info("no 'Start Review' control found; the review may already be running")
    return False


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

_INSPECT_JS = r"""
(hints) => {
  const out = {containerCandidates: [], signals: [], sidebarHTML: '',
               pageClasses: [], bodyText: '', moveCells: 0, shadowRootsEntered: 0};
  const norm = (s) => String(s || '').toLowerCase();
  // At least as thorough as the reader, or calibration reasons about a page
  // state the reader never saw. The reader pierces shadow roots; this does too.
  const deepAll = (root, outEls, depth) => {
    if (depth > 8 || !root || !root.querySelectorAll) return outEls;
    let list;
    try { list = root.querySelectorAll('*'); } catch (e) { return outEls; }
    for (const el of list) {
      outEls.push(el);
      if (el.shadowRoot) { out.shadowRootsEntered++; deepAll(el.shadowRoot, outEls, depth + 1); }
    }
    return outEls;
  };
  for (const sel of hints.container) {
    let el = null;
    try { el = document.querySelector(sel); } catch (e) { continue; }
    out.containerCandidates.push({
      selector: sel,
      found: !!el,
      childCount: el ? el.children.length : 0,
      text: el ? (el.innerText || '').slice(0, 400) : '',
    });
  }
  const kw = /brilliant|blunder|inaccuracy|mistake|miss|great|best|book|excellent|good/i;
  const seen = {};
  const all = deepAll(document.body, [], 0);
  try { out.moveCells = document.querySelectorAll('[data-node]').length; } catch (e) {}
  for (let i = 0; i < all.length && out.signals.length < 120; i++) {
    const el = all[i];
    let c = '';
    try { c = norm(el.className || ''); } catch (e) {}
    const bits = [c, el.getAttribute && el.getAttribute('alt'),
                  el.getAttribute && el.getAttribute('title'),
                  el.getAttribute && el.getAttribute('aria-label'),
                  el.getAttribute && el.getAttribute('id'),
                  el.getAttribute && el.getAttribute('style'),
                  el.getAttribute && el.getAttribute('data-classification')]
                 .filter(Boolean).join(' | ');
    if (!kw.test(bits)) continue;
    const key = bits.slice(0, 200);
    if (seen[key]) continue;
    seen[key] = true;
    out.signals.push({
      tag: el.tagName.toLowerCase(),
      class: (el.getAttribute('class') || '').slice(0, 140),
      id: ((el.getAttribute && el.getAttribute('id')) || '').slice(0, 60),
      style: ((el.getAttribute && el.getAttribute('style')) || '').slice(0, 90),
      text: (el.innerText || el.textContent || '').trim().slice(0, 60),
    });
  }
  const classes = {};
  for (let i = 0; i < all.length; i++) {
    let c = '';
    try {
      const raw = all[i].className;
      c = (raw && raw.baseVal !== undefined) ? String(raw.baseVal)
        : (typeof raw === 'string' ? raw : '');
    } catch (e) { continue; }
    if (!c) continue;
    for (const part of c.trim().split(/\s+/)) {
      if (part) classes[part] = (classes[part] || 0) + 1;
    }
  }
  out.pageClasses = Object.keys(classes).sort();
  let sb = null;
  for (const sel of hints.container) {
    try { sb = document.querySelector(sel); } catch (e) { continue; }
    if (sb) break;
  }
  if (sb) out.sidebarHTML = sb.outerHTML;
  out.bodyText = (document.body.innerText || '').slice(0, 4000);
  return out;
}
"""


async def inspect_dom(page) -> dict:
    """Dump everything needed to (re)calibrate the classification detection."""
    return await page.evaluate(_INSPECT_JS, {"container": CONTAINER_HINTS})


# ---------------------------------------------------------------------------
# Pure helpers (unit-testable without a browser)
# ---------------------------------------------------------------------------


def summarise(rows: list[dict]) -> str:
    """One-line-per-classification summary for logs and the calibrator."""
    counts: dict[str, int] = {}
    for r in rows:
        k = r.get("classification")
        if k:
            counts[k] = counts.get(k, 0) + 1
    if not counts:
        return "no classified moves"
    return ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))


def validate_against(
    rows: list[dict], expected: dict[str, int], strict: bool = True
) -> list[str]:
    """Compare parsed classifications with ground truth for a known game.

    With ``strict=True`` any classification not in ``expected`` is a false
    positive. That is right for a full ground-truth table and for the unit
    tests, but it makes ``--expect blunder=2`` useless in practice: every real
    game also contains book, great and best moves. Callers that only know a few
    counts should pass ``strict=False``.
    """
    problems: list[str] = []
    counts: dict[str, int] = {}
    for r in rows:
        k = r.get("classification")
        if k:
            counts[k] = counts.get(k, 0) + 1
    for kind, want in expected.items():
        got = counts.get(kind, 0)
        if got != want:
            problems.append(f"{kind}: expected {want}, parsed {got}")
    for kind, got in sorted(counts.items()):
        if strict and kind not in expected:
            problems.append(f"{kind}: parsed {got} but not in ground truth")
    return problems


_SAN_RE = re.compile(
    r"^(?:[NBRQK]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?|O-O(?:-O)?|"
    r"1-0|0-1|1/2-1/2)[+#]*$",
    re.IGNORECASE,
)


def looks_like_san(token: str) -> bool:
    """True when ``token`` is a plausible SAN move (used by tests)."""
    return bool(_SAN_RE.match((token or "").strip()))


def pick_highest_confidence(
    hits: list[dict], kinds: tuple[str, ...] = ()
) -> dict | None:
    """Choose the most trustworthy hit: high confidence first, then earliest."""
    order = {k: i for i, k in enumerate(kinds or KINDS)}
    ranked = sorted(
        hits,
        key=lambda h: (
            {HIGH: 0, MEDIUM: 1, LOW: 2}.get(h.get("confidence"), 3),
            order.get(h.get("classification"), 99),
            h.get("ply", 0),
        ),
    )
    return ranked[0] if ranked else None


def merge_counts(tally: dict[str, dict], fallback: dict[str, int]) -> dict[str, int]:
    """Prefer the class-name tally; fall back to the text parse for gaps.

    Kept for callers that still have a text parse to offer: one reader on the
    real class is enough for ``read_tally``.
    """
    out: dict[str, int] = {}
    for kind, values in (tally or {}).items():
        total = values.get("total") if isinstance(values, dict) else values
        if total is not None:
            out[kind] = total
    for kind, value in (fallback or {}).items():
        out.setdefault(kind, value)
    return out
