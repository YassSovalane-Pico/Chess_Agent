"""Account-based tools (browser automation, logged-in session).

These drive your paid chess.com account in a real browser to capture Game
Review visuals. They need storage_state.json created once by login_chess.py.

Workflow implemented here:
  1. Fixed desktop viewport (1920x1080) + persisted session (storage_state).
  2. DOM-first locators with explicit waits (no raw coordinates).
  3. Screenshots clipped to the board and the review panel, with chess.com's
     left navigation hidden -- no site menu, no search box, no player profile.
  4. Targeted element screenshots (review panel / single move node).

Navigation is in tools/move_list.py, which reads the sidebar structurally
rather than trusting chess.com's class names, and *verifies* that a step
actually changed the board.

Public surface (kept sync so the Gemini tool loop in main.py is untouched):
  - review_game_screenshots(game_url) -> dict
Pipeline utilities (import these in your own agent code):
  - get_annotated_review(game_url, queries=None)  (async)
  - annotate_page_elements(page, queries)         (async)
  - clear_annotations(page)                       (async)
  - screenshot_element(page, selector, path)      (async)
  - screenshot_review_area(page, path, clip)      (async)
  - content_clip(page) / hide_left_nav(page)      (async)
  - capture_review_panel(page, path)              (async)
  - get_classified_moves(page)                    (async)  -> list[dict]
  - find_and_open_classified_move(page, kinds)    (async)  -> dict
  - next_move(page) / previous_move(page)         (async)  -> StepResult
  - goto_move(page, ply)                          (async)  -> bool
"""

import asyncio
import logging
import os
import re
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from . import move_list
from .common import load_browser_profile, session_state_path

_LOG = logging.getLogger(__name__)

# Every Playwright call in this module is best-effort: chess.com renames
# selectors often and the page can close mid-run, so each call degrades
# gracefully instead of aborting the review. We catch the Playwright error
# hierarchy (which covers bad selectors, closed targets and timeouts) rather
# than a bare `Exception`, so real bugs -- AttributeError, TypeError -- stay
# visible instead of being silently swallowed.
try:  # pragma: no cover - depends on whether playwright is installed
    from playwright.async_api import Error as _PlaywrightError
except ImportError:  # pragma: no cover - playwright is an optional dependency
    # Nothing below can reach a browser without playwright anyway; keep the
    # historical broad catch so behaviour is unchanged in that environment.
    _PlaywrightError = Exception

# ---------------------------------------------------------------------------
# Constants: fixed geometry + DOM-first selector tables
# ---------------------------------------------------------------------------

# The layout chess.com lays out at, and the fallback when the page does not
# report its own size.
VIEWPORT = {"width": 1920, "height": 1080}

# The viewport the review actually runs at.
#
# This is the lever on frame width. The board is height-constrained -- it is
# sized to the available height, not the width -- so at 1920x1080 it rendered
# ~980px wide and a board+panel frame came out at 1480px. That is a lot of
# picture for one move, and the user asked for something tighter. At 1400x820
# the board renders ~700px and the frame lands near 1150px wide: same
# composition, proportionate.
#
# Floors are deliberate. Below ~1200 wide chess.com starts reflowing the
# analysis page and the review sidebar stops being beside the board; below
# ~700 tall the board becomes unusable.
VIEWPORT_MIN = {"width": 1200, "height": 700}
VIEWPORT_MAX = {"width": 2560, "height": 1440}


def _configured_viewport() -> dict:
    """The review viewport, clamped, overridable with REVIEW_VIEWPORT_W/H."""

    def size(name: str, fallback: int) -> int:
        raw = os.getenv(name)
        if not raw:
            return fallback
        try:
            return int(raw)
        except ValueError:
            _LOG.warning("%s=%r is not a number; using %d", name, raw, fallback)
            return fallback

    width = min(
        VIEWPORT_MAX["width"],
        max(VIEWPORT_MIN["width"], size("REVIEW_VIEWPORT_W", 1400)),
    )
    height = min(
        VIEWPORT_MAX["height"],
        max(VIEWPORT_MIN["height"], size("REVIEW_VIEWPORT_H", 820)),
    )
    return {"width": width, "height": height}


# No hardcoded UA any more. If login_chess.py recorded the real browser's UA we
# replay that exact string; otherwise we do not override the UA at all, which
# is always better than claiming to be a two-year-old Chrome build.
_PROFILE = load_browser_profile()
USER_AGENT = _PROFILE.get("user_agent") or None

LAUNCH_ARGS = [
    "--disable-blink-features=AutomationControlled",
    "--disable-dev-shm-usage",
    "--no-sandbox",
]

# The review sidebar (coach chat, accuracy, tally, move list). Its class is
# `board-layout-sidebar`, which is deliberately NOT matched by
# `.sidebar-container` -- that one is the left navigation.
SIDEBAR_SELECTORS = [
    "#board-layout-sidebar",
    "[class*='board-layout-sidebar' i]",
]

# The region a screenshot should cover: the board plus that sidebar. Taken as
# the union of the two elements rather than a hardcoded size, so the left
# navigation is excluded by construction.
CONTENT_HINTS = [
    "board-layout-analysis",
    "[class*='board-layout-main' i]",
    *SIDEBAR_SELECTORS,
]

# chess.com's left navigation, hidden so it cannot appear in any frame.
LEFT_NAV_HINTS = [
    ".sidebar-container",
    "[class*='sidebar-container' i]",
    "aside",
    "nav",
]

# The one screenshot always taken, plus the per-move ones.
SHOT_KIND = "summary"
MOVE_SHOT_KINDS = ("brilliant", "blunder")

# A game can contain a dozen of these. Cap the frames so a pathological run
# cannot write hundreds of PNGs; say so plainly when it truncates.
MAX_MOVE_SHOTS = 10

# Slack around the measured region. Kept small on purpose: the union of the
# board and the sidebar is already tight, and a generous margin just brings
# back the page background and the player nameplates the crop was meant to drop.
SHOT_PADDING = int(os.getenv("REVIEW_SHOT_PADDING", "12"))

# A chessboard is square. This is the tolerance for deciding what counts as
# "the board" when cropping, and it is deliberately tight: at 0.85..1.18 an
# 800x1080 layout column (nameplate, clock, controls) passed as the board and
# the frame carried the player's nameplate. A real board is square to within
# about 1%. Stricter than the signature check in move_list, which only needs a
# hint and so can afford to be lenient.
MIN_BOARD_ASPECT = 0.99
MAX_BOARD_ASPECT = 1.01

# The minimum edge length for an element to be considered the board at all.
# Below this a stray icon or a spinner could qualify once squareness is
# required.
MIN_BOARD_PX = 120

# DOM-first selector tables (first hit wins). These are *hints* used for
# screenshots and layout, never for correctness -- see tools/move_list.py.
SELECTORS = {
    # The review summary: accuracy, the classification tally, the game rating.
    "review_summary": [
        ".game-review-view",
        "[class*='game-review' i]",
        "[class*='review-summary' i]",
        "[data-cy*='review' i]",
    ],
    # Move notation list (sidebar).
    "move_list": [
        ".move-list",
        "[class*='move-list' i]",
        "[class*='moveList' i]",
        "#board-layout-sidebar",
    ],
    # Individual move nodes inside the list.
    "move_node": [
        ".move-node",
        "[class*='move-node' i]",
        "[class*='moveNode' i]",
        ".move-list-move",
        "[class*='move-list-move' i]",
    ],
    # Engine evaluation bar next to the board.
    "eval_bar": [
        "#evaluation",
        ".evaluation",
        "[class*='evaluation' i]",
    ],
    # Sidebar navigation tabs (Moves / Info / Openings on game page).
    "nav_tabs": [
        "[class*='board-tab' i]",
        "[role='tablist']",
    ],
    # The board, used to build the screenshot region. Most specific first.
    "board": [
        "wc-chess-board",
        "[class*='chessboard' i]",
        "[class*='board-board' i]",
    ],
    # The move-by-move review column: coach chat, accuracy, tally, move list.
    # NOT the left navigation -- that is `.sidebar-container`.
    "review_sidebar": [
        "#board-layout-sidebar",
        "[class*='board-layout-sidebar' i]",
    ],
    # Step controls on the review panel. Used by goto_move()/step helpers.
    "next_move": move_list.NEXT_MOVE_SELECTORS,
    "prev_move": move_list.PREV_MOVE_SELECTORS,
}

# Classification -> the marker chess.com paints on the move. Purely
# presentation for the reply text; detection never depends on it.
CLASS_COLORS = {
    "brilliant": "teal !!",
    "blunder": "red ??",
    "best": "green star",
    "great": "blue !",
    "mistake": "orange ?",
    "miss": "red X",
    "book": "gray book icon",
}

# Classifications worth telling a player about. `best`, `excellent` and
# `good` are chess.com approving a move, not a finding, so they stay out of
# notable_moves even though the tally counts them.
NOTABLE = frozenset({"brilliant", "great", "blunder", "inaccuracy", "mistake", "miss"})

# Kept for backwards compatibility. move_list.py does the real detection;
# these are only the legacy fallback selectors.
CLASSIFICATION_SELECTORS = {
    "brilliant": [
        "[class*='brilliant' i]",
        "[data-classification*='brilliant' i]",
        "[title*='Brilliant' i]",
        "[aria-label*='Brilliant' i]",
    ],
    "blunder": [
        "[class*='blunder' i]",
        "[data-classification*='blunder' i]",
        "[title*='Blunder' i]",
        "[aria-label*='Blunder' i]",
    ],
    "book": [
        "[class*='book-move' i]",
        "[data-classification*='book' i]",
        "[title*='Book' i]",
        "[aria-label*='Book' i]",
    ],
}

# Default elements to annotate for the vision LLM. NOTE: these are Playwright
# selectors throughout -- `text=` and `:has-text()` are engine features, not
# CSS, and annotation is resolved exclusively through Playwright now.
DEFAULT_ANNOTATION_QUERIES = [
    {
        "key": "board",
        "selector": "#board, wc-chess-board, .board",
        "description": "Main chess board",
    },
    {
        "key": "move_list",
        "selector": ".move-list, [class*='move-list' i], #board-layout-sidebar",
        "description": "Move notation list sidebar (classifications are marked here)",
    },
    {
        "key": "review_summary",
        "selector": ".game-review-view, [class*='review-summary' i], [class*='game-review' i]",
        "description": "Accuracy summary panel",
    },
    {
        "key": "start_review",
        "selector": "button:has-text('Start Review')",
        "description": "Green button entering the move-by-move coach review",
    },
    {
        "key": "eval_bar",
        "selector": "#evaluation, [class*='evaluation' i]",
        "description": "Engine evaluation bar next to the board",
    },
]


def _project_root() -> Path:
    # tools/review.py -> tools/ -> project root
    return Path(__file__).resolve().parent.parent


def _build_review_url(game_url: str) -> tuple[str, str, str]:
    """Validate a /game/ link and derive the direct review URL.

    Returns (kind, game_id, review_url), e.g.
    /game/live/165334333716 -> /analysis/game/live/165334333716/review
    Raises ValueError on bad input.
    """
    url = (game_url or "").strip().strip('"').strip("'")
    if not url.startswith("http"):
        raise ValueError("Please provide a full game link starting with http.")
    if "chess.com/game/" not in url:
        raise ValueError(
            "This tool only works with chess.com game links "
            "(/game/live/... or /game/daily/...)."
        )
    m = re.search(r"/game/(live|daily)/(\d+)", url)
    if not m:
        raise ValueError(
            "Could not find a game id in that link. Use a link like "
            "/game/live/123... or /game/daily/123..."
        )
    kind, game_id = m.group(1), m.group(2)
    return kind, game_id, f"https://www.chess.com/analysis/game/{kind}/{game_id}/review"


def _shot_path(kind: str, game_id: str, stamp: str, label: str = "") -> Path:
    """Path for one screenshot.

    ``label`` disambiguates several shots of the same kind -- which is the
    normal case now, because a game with three brilliants gets three frames:
    ``brilliant_m14_ply27_<id>_<stamp>.png``.
    """
    shots_dir = _project_root() / "screenshots"
    shots_dir.mkdir(exist_ok=True)
    stem = f"{kind}_{label}" if label else kind
    return shots_dir / f"{stem}_{game_id}_{stamp}.png"


# ---------------------------------------------------------------------------
# Browser bootstrap (async, fixed viewport + persisted session)
# ---------------------------------------------------------------------------


async def _launch_browser(p):
    """Launch real Chrome headed/headless with anti-detection flags."""
    headed = os.getenv("REVIEW_HEADLESS", "1") == "0"
    try:
        return await p.chromium.launch(
            channel="chrome",
            headless=not headed,
            args=LAUNCH_ARGS,
            ignore_default_args=["--enable-automation"],
        )
    except _PlaywrightError as exc:
        _LOG.debug("Real Chrome unavailable, using bundled Chromium: %s", exc)
        return await p.chromium.launch(
            headless=not headed,
            args=LAUNCH_ARGS,
            ignore_default_args=["--enable-automation"],
        )


async def _new_logged_in_page(p, browser, state_path: Path):
    """New 1920x1080 page with stored session + webdriver flag hidden.

    Returns (context, page). Callers must close both; use _Session.
    """
    # Sized from the environment so the frame width is tunable: the board is
    # height-constrained, so this is what actually controls how big the board
    # renders in a screenshot.
    kwargs = {"storage_state": str(state_path), "viewport": _configured_viewport()}
    if USER_AGENT:
        kwargs["user_agent"] = USER_AGENT
    else:
        _LOG.info("no recorded user agent; using the browser's real one")
    context = await browser.new_context(**kwargs)
    page = await context.new_page()
    try:
        await page.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        )
    except _PlaywrightError as exc:
        # Cosmetic anti-detection only; never abort the run over it.
        _LOG.warning("Could not hide the webdriver flag: %s", exc)
    return context, page


class _Session:
    """Owns the browser + context so both are closed exactly once.

    The old flow closed the browser on early-return paths *and* again in a
    `finally`, and threw the context away without closing it.
    """

    def __init__(self, p, browser, context, page):
        self.p = p
        self.browser = browser
        self.context = context
        self.page = page
        self._closed = False

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for closer, label in ((self.context, "context"), (self.browser, "browser")):
            try:
                await closer.close()
            except _PlaywrightError as exc:
                _LOG.debug("closing %s failed: %s", label, exc)


async def _open_session(p, state_path: Path) -> _Session:
    browser = await _launch_browser(p)
    try:
        context, page = await _new_logged_in_page(p, browser, state_path)
    except Exception:
        await browser.close()
        raise
    return _Session(p, browser, context, page)


async def _check_cloudflare(page, headed: bool) -> str | None:
    """Return an error string if blocked, else None (waits out a manual solve)."""
    try:
        cf = page.get_by_text("Verify you are human", exact=False).first
        if await cf.count() > 0 and await cf.is_visible(timeout=5000):
            if headed:
                print("[review] Cloudflare check detected. Solve it in the window...")
                await page.get_by_text(
                    "Verify you are human", exact=False
                ).first.wait_for(state="hidden", timeout=90000)
                await page.wait_for_timeout(4000)
                return None
            return (
                "Blocked by Cloudflare check in background mode. Re-run once with "
                "REVIEW_HEADLESS=0 in .env to solve 'Verify you are human' manually, "
                "then switch back to background mode."
            )
    except _PlaywrightError as exc:
        _LOG.warning("Cloudflare probe failed; treating the page as unblocked: %s", exc)
    return None


# ---------------------------------------------------------------------------
# DOM-first waits + data extraction
# ---------------------------------------------------------------------------


async def _wait_for_review_ready(page) -> bool:
    """Kept for backwards compatibility.

    Superseded by ``move_list.wait_for_review_loaded``, which waits for the
    review data to actually appear instead of for the word "Accuracy" -- that
    word is in the sidebar header from the first paint, so the old check
    returned immediately on a still-skeleton page.
    """
    return await move_list.wait_for_review_loaded(page, require="panel")


# Strict, adjacency-only patterns. The old code used
# ``rf"{label}\\D*(\\d+)\\D+(\\d+)"``, whose ``\\D+`` happily spanned other labels
# and reported e.g. the *mistake* count as the black-side *blunder* count.
_COUNT_RE = re.compile(
    r"\b(Brilliant|Great(?:\s*move|\s*find)?|Best(?:\s*move)?|Book|Mistake|"
    r"Miss(?:ed|es)?|Blunder)\b"
    r"\s*[:•–→-]?\s*(\d{1,3})(?!\d)",  # noqa: RUF001 - separators must match chess.com
    re.IGNORECASE,
)
_ACCURACY_ANCHORS = ("Accuracy", "Game Review", "Review Summary", "Score")
_WHITE_RE = re.compile(r"\b(?:white|you|top)\b[^\d]{0,15}?(\d{1,3}(?:\.\d+)?)", re.I)
_BLACK_RE = re.compile(r"\bblack\b[^\d]{0,15}?(\d{1,3}(?:\.\d+)?)", re.I)
# One OR two decimals. The old `\d{2,3}\.\d` required a word boundary right
# after a single decimal digit, so "85.35 79.15" matched nothing at all and the
# accuracy silently came back as None.
_DECIMAL_RE = re.compile(r"\b(\d{1,3}\.\d{1,2})\b")

# How much panel text to consider once the anchor is found. Wide enough to hold
# the two accuracies plus the whole classification legend.
_PANEL_WINDOW = 1200

# chess.com is inconsistent about singular vs plural in the legend, so every
# spelling collapses onto the canonical kind used everywhere else.
_COUNT_CANONICAL = {
    "great": "great",
    "great move": "great",
    "great find": "great",
    "best": "best",
    "best move": "best",
    "book": "book",
    "mistake": "mistake",
    "mistakes": "mistake",
    "miss": "miss",
    "missed": "miss",
    "misses": "miss",
    "blunder": "blunder",
    "blunders": "blunder",
    "brilliant": "brilliant",
    "brilliants": "brilliant",
}


def _canon_count_label(raw: str) -> str:
    key = re.sub(r"\s+", " ", raw).strip().lower()
    key = re.sub(r"\s*move$", "", key)
    key = re.sub(r"\s*find$", "", key)
    return _COUNT_CANONICAL.get(key, key)


def parse_review_panel(text: str) -> tuple[float | None, float | None, dict[str, int]]:
    """Parse the summary panel text. Pure function, so it is unit-testable.

    Returns (white_accuracy, black_accuracy, counts) where counts maps a
    classification name to a single integer. It deliberately returns None
    rather than guessing: the previous version could invent a number, and the
    system prompt then stated that number to the user as fact.

    Accepts "White 71.5 Black 68.8", "85/100", and a legend line like
    "Blunder 2". Anything it cannot read confidently comes back as None / {}.
    """
    if not text:
        return None, None, {}

    anchor = next((a for a in _ACCURACY_ANCHORS if a in text), None)
    if anchor is None:
        return None, None, {}
    start = text.index(anchor)
    section = text[start : start + _PANEL_WINDOW]

    white = black = None
    mw, mb = _WHITE_RE.search(section), _BLACK_RE.search(section)
    if mw and mb:
        white, black = float(mw.group(1)), float(mb.group(1))
    else:
        # No colour labels: only trust two adjacent decimals near the anchor.
        near = _DECIMAL_RE.findall(section[:200])
        if len(near) >= 2:
            white, black = float(near[0]), float(near[1])

    counts: dict[str, int] = {}
    for match in _COUNT_RE.finditer(section):
        counts[_canon_count_label(match.group(1))] = int(match.group(2))
    return white, black, counts


async def _extract_review_data(page) -> tuple[float | None, float | None, dict]:
    """Read accuracy pair + per-kind counts from the summary panel.

    Accuracy comes from ``.review-rating-white`` / ``-black`` (read by class);
    counts from the ``.tally-row-value.analysis-<kind>`` spans. The text parse
    is only a fallback for when chess.com renames those classes.
    """
    white, black = await move_list.read_ratings(page)
    fallback_counts: dict[str, int] = {}
    if white is None or black is None:
        try:
            text = await page.evaluate("() => document.body.innerText || ''")
        except _PlaywrightError as exc:
            _LOG.warning("Could not read the accuracy panel text: %s", exc)
            text = ""
        text_white, text_black, fallback_counts = parse_review_panel(text)
        white = white if white is not None else text_white
        black = black if black is not None else text_black
    tally = await move_list.read_tally(page)
    counts = move_list.merge_counts(tally, fallback_counts)
    if counts:
        _LOG.info("summary tally: %s", counts)
    return white, black, counts


async def _first_visible(page, selectors: list[str]):
    """Return the first visible locator across fallback selectors, else None."""
    for sel in selectors:
        try:
            loc = page.locator(sel).first
            if await loc.count() > 0 and await loc.is_visible(timeout=2000):
                return loc
        except _PlaywrightError as exc:
            _LOG.debug("Selector %r unusable: %s", sel, exc)
    return None


# ---------------------------------------------------------------------------
# Annotation workflow (bounding boxes + JSON map for the vision LLM)
# ---------------------------------------------------------------------------

# Draws boxes from *measured* Playwright bounding boxes. It no longer runs a
# CSS query at all: the old version passed Playwright selectors like
# `text=Accuracy` into `document.querySelector`, which throws, and the JS
# swallowed the error -- so those targets were silently never annotated.
_DRAW_BOXES_JS = r"""
(boxes) => {
  document.querySelectorAll('[data-ai-anno]').forEach(e => e.remove());
  for (const b of boxes) {
    const el = document.createElement('div');
    el.setAttribute('data-ai-anno', String(b.index));
    el.style.cssText =
      'position:fixed;left:' + b.box[0] + 'px;top:' + b.box[1] + 'px;' +
      'width:' + b.box[2] + 'px;height:' + b.box[3] + 'px;' +
      'border:3px solid #FF00FF;box-shadow:0 0 0 2px #000;' +
      'z-index:2147483647;pointer-events:none;box-sizing:border-box;';
    const tag = document.createElement('div');
    tag.textContent = b.index + ': ' + b.key;
    tag.style.cssText =
      'position:absolute;top:-26px;left:-3px;background:#FF00FF;color:#fff;' +
      'font:bold 14px/20px monospace;padding:2px 8px;white-space:nowrap;';
    el.appendChild(tag);
    document.body.appendChild(el);
  }
  return boxes.length;
}
"""


async def annotate_page_elements(page, queries: list[dict] | None = None) -> dict:
    """Draw indexed magenta boxes over key elements.

    Args:
        queries: [{key, selector, description}]. Selectors are *Playwright*
            selectors (so `text=` and `:has-text()` work) and may be comma
            lists; the first matching sub-selector wins. Defaults to
            DEFAULT_ANNOTATION_QUERIES.

    Returns:
        {index: {key, selector, description, box: [x, y, w, h]}} for the
        elements that were actually found and measured. Indices are contiguous
        in call order and the number drawn on screen always matches the key,
        so a vision model can trust the pairing.
    """
    queries = queries if queries is not None else DEFAULT_ANNOTATION_QUERIES

    mapping: dict[int, dict] = {}
    to_draw: list[dict] = []
    for q in queries:
        loc = selector = None
        for sub in [s.strip() for s in q["selector"].split(",")]:
            try:
                candidate = page.locator(sub).first
                if await candidate.count() > 0 and await candidate.is_visible(
                    timeout=1500
                ):
                    loc, selector = candidate, sub
                    break
            except _PlaywrightError as exc:
                _LOG.debug("Annotation selector %r unusable: %s", sub, exc)
        if loc is None:
            _LOG.info(
                "annotation target %r not found (%s)", q.get("key"), q["selector"]
            )
            continue
        try:
            box = await loc.bounding_box()
        except _PlaywrightError as exc:
            _LOG.debug("could not measure %r: %s", q.get("key"), exc)
            continue
        if not box or box["width"] < 2 or box["height"] < 2:
            _LOG.debug("annotation target %r has no usable box", q.get("key"))
            continue
        # The overlay is position:fixed and the screenshot is viewport-only, so
        # a target below the fold cannot be annotated at all. Scrolling it into
        # view was the previous answer, and it was wrong: each scroll moved
        # every box already measured, so the boxes drawn on the screenshot
        # pointed at the wrong elements. Refusing to draw is honest; drawing a
        # box that lands on the wrong element is not.
        # Measured against the page's own viewport, not a constant, so a
        # narrower configured window does not start skipping everything.
        view_h = (getattr(page, "viewport_size", None) or _configured_viewport())[
            "height"
        ]
        if box["y"] < 0 or box["y"] + box["height"] > view_h:
            _LOG.info(
                "annotation target %r is outside the %dpx viewport; skipping",
                q.get("key"),
                view_h,
            )
            continue
        index = len(mapping)
        rounded = [
            round(box["x"], 1),
            round(box["y"], 1),
            round(box["width"], 1),
            round(box["height"], 1),
        ]
        mapping[index] = {
            "key": q["key"],
            "selector": selector,
            "description": q.get("description", ""),
            "box": rounded,
        }
        to_draw.append({"index": index, "key": q["key"], "box": rounded})

    if to_draw:
        try:
            await page.evaluate(_DRAW_BOXES_JS, to_draw)
        except _PlaywrightError as exc:
            _LOG.warning("Could not draw annotation boxes: %s", exc)
    _LOG.info("annotated %d/%d targets", len(mapping), len(queries))
    return mapping


async def clear_annotations(page) -> None:
    """Remove overlay boxes drawn by annotate_page_elements."""
    try:
        await page.evaluate(
            "() => document.querySelectorAll('[data-ai-anno]').forEach(e => e.remove())"
        )
    except _PlaywrightError as exc:
        _LOG.debug("Could not clear the annotation overlay: %s", exc)


# ---------------------------------------------------------------------------
# Screenshot region: the review, without chess.com's navigation
# ---------------------------------------------------------------------------


async def hide_left_nav(page) -> bool:
    """Hide chess.com's left navigation so it cannot appear in a screenshot.

    Clipping is the primary mechanism, but it depends on measuring the right
    two elements; hiding the nav means a fallback viewport shot is still clean.
    Returns True if something was hidden.
    """
    hidden = False
    for sel in LEFT_NAV_HINTS:
        try:
            locs = page.locator(sel)
            for i in range(await locs.count()):
                loc = locs.nth(i)
                if await loc.evaluate("el => !!el.closest('#board-layout-sidebar')"):
                    # Never hide the review sidebar itself.
                    continue
                await loc.evaluate(
                    "el => { el.style.setProperty('visibility', 'hidden', "
                    "'important'); }"
                )
                hidden = True
        except _PlaywrightError as exc:
            _LOG.debug("could not hide %r: %s", sel, exc)
    return hidden


async def _first_box(page, selectors: list[str], min_px: float = 120.0):
    """First usable bounding box across `selectors`, or None."""
    for sel in selectors:
        try:
            loc = page.locator(sel).first
            if await loc.count() == 0:
                continue
            box = await loc.bounding_box()
        except _PlaywrightError as exc:
            _LOG.debug("box probe %r failed: %s", sel, exc)
            continue
        if box and box["width"] >= min_px and box["height"] >= min_px:
            return box
    return None


def _is_square(box: dict | None, min_px: float = 120.0) -> bool:
    """True when a bounding box is a usable square.

    Strict on purpose. An earlier tolerance of 0.85..1.18 let an 800x1080
    layout column through as "the board" -- so the crop carried the player
    nameplate and the clock. A real board is square to within about 1%.
    """
    if not box:
        return False
    w, h = box.get("width") or 0, box.get("height") or 0
    if w < min_px or h < min_px:
        return False
    return MIN_BOARD_ASPECT <= w / h <= MAX_BOARD_ASPECT


async def _square_box(page, selectors: list[str], min_px: float = 120.0):
    """First *square* box across `selectors`, or None.

    Squareness is what separates the board from the layout column around it.
    The user asked for a tighter frame, so the board must be identified as a
    square rather than as anything with "board" in its class.
    """
    for sel in selectors:
        try:
            locs = page.locator(sel)
            for i in range(min(await locs.count(), 3)):
                box = await locs.nth(i).bounding_box()
                if _is_square(box, min_px):
                    return box
        except _PlaywrightError as exc:
            _LOG.debug("square probe %r failed: %s", sel, exc)
    return None


async def content_clip(page) -> dict | None:
    """Bounding box of the board plus the review sidebar, or None.

    This is the region the user actually wants: the chessboard itself and the
    move-by-move review with the coach's commentary. Two things keep it tight:

      * the board is found as a *square*, so the nameplate, clock and control
        column around it are excluded;
      * only a small padding is added, because the union of board and sidebar
        is already snug and a generous margin reintroduces the page background.

    The left navigation is excluded by construction: it is part of neither box.

    What to include is tunable via ``REVIEW_SHOT_REGION``:
      ``board_sidebar`` (default) board and review panel -- the composition in
                          the reference screenshot
      ``board``            just the chessboard, for a tight look at a position
      ``sidebar``          just the review panel
    And the margin via ``REVIEW_SHOT_PADDING`` (default 12px, 0 for flush).
    """
    region = (os.getenv("REVIEW_SHOT_REGION", "board_sidebar") or "").strip().lower()
    want_board = region in ("board_sidebar", "board", "")
    want_sidebar = region in ("board_sidebar", "sidebar")

    boxes = []
    if want_board:
        # Note what is NOT here: the player nameplate and the clock bar. The
        # board is identified as a square, and the frame is the union of that
        # square with the panel -- so it is the union that decides the height,
        # because the panel is full-height. That is why a narrower viewport
        # shrinks the board and the frame together.
        board = await _square_box(page, SELECTORS["board"])
        if board:
            boxes.append(board)
    if want_sidebar:
        sidebar = await _first_box(page, SIDEBAR_SELECTORS, min_px=80.0)
        if sidebar:
            boxes.append(sidebar)
    if not boxes:
        box = await _first_box(page, CONTENT_HINTS, min_px=200.0)
        if box:
            boxes.append(box)
    if not boxes:
        return None
    pad = max(0, SHOT_PADDING)
    left = min(b["x"] for b in boxes) - pad
    top = min(b["y"] for b in boxes) - pad
    right = max(b["x"] + b["width"] for b in boxes) + pad
    bottom = max(b["y"] + b["height"] for b in boxes) + pad
    clip = {
        "x": max(0.0, left),
        "y": max(0.0, top),
        "width": right - left,
        "height": bottom - top,
    }
    # A clip outside the viewport is a Playwright error, not a small crop.
    # Prefer the page's own size: a caller may have opened it at a different
    # viewport, and guessing would silently crop the wrong region.
    view = getattr(page, "viewport_size", None) or _configured_viewport()
    clip["width"] = min(clip["width"], view["width"] - clip["x"])
    clip["height"] = min(clip["height"], view["height"] - clip["y"])
    if clip["width"] < 120 or clip["height"] < 120:
        return None
    return clip


async def screenshot_review_area(page, path: Path, clip: dict | None = None) -> str:
    """Screenshot the board + review sidebar. Never the left navigation.

    `clip=None` measures the region first. If the region cannot be measured it
    falls back to the viewport, which is still clean because `hide_left_nav`
    has run.
    """
    region = clip if clip is not None else await content_clip(page)
    try:
        if region:
            await page.screenshot(path=str(path), clip=region, full_page=False)
            _LOG.info("shot %s clipped to %s", path.name, region)
        else:
            _LOG.warning(
                "no content region found for %s; using the viewport", path.name
            )
            await page.screenshot(path=str(path), full_page=False)
    except _PlaywrightError as exc:
        _LOG.warning(
            "clipped shot of %s failed (%s); using the viewport", path.name, exc
        )
        try:
            await page.screenshot(path=str(path), full_page=False)
        except _PlaywrightError as exc2:
            _LOG.error("screenshot of %s failed outright: %s", path.name, exc2)
    return str(path)


async def screenshot_element(page, selector: str, path: Path) -> str:
    """Screenshot ONE element (scrolls it into view first). Returns str(path)."""
    loc = page.locator(selector).first
    try:
        await loc.scroll_into_view_if_needed(timeout=5000)
        await loc.screenshot(path=str(path))
    except _PlaywrightError as exc:
        # A selector that no longer matches must degrade to the review area,
        # not raise out of a tool the model is calling.
        _LOG.warning(
            "element %r could not be shot (%s); using the review area", selector, exc
        )
        return await screenshot_review_area(page, path)
    return str(path)


async def capture_review_panel(page, path: Path) -> str:
    """Screenshot the review panel: accuracy, tally, coach chat.

    An element shot of the sidebar rather than a viewport crop, because a
    1920x1080 crop of a narrow column would be mostly chessboard. The sidebar's
    internal scroll is reset first -- an element screenshot of a scrolling
    container captures only what is currently rendered.
    """
    await move_list.scroll_sidebar_to_top(page)
    target = await _first_visible(page, SELECTORS["review_sidebar"])
    if target is not None:
        try:
            await target.screenshot(path=str(path))
            return str(path)
        except _PlaywrightError as exc:
            _LOG.debug("Sidebar screenshot failed, falling back: %s", exc)
    return await screenshot_review_area(page, path)


# ---------------------------------------------------------------------------
# Move classification + navigation (delegated to tools/move_list.py)
# ---------------------------------------------------------------------------


def _describe_move(target: dict, page_url: str) -> dict:
    """Turn a parsed row into the shape the model and the reply both need.

    ``color`` is the side that played the move ("white"/"black"), which the
    move-list cell carries; it used to hold the legend blurb instead, so the
    system prompt's "(<move>, <color> marker)" line had no real colour to
    print. The blurb moves to ``marker`` where it belongs.
    """
    kind = target.get("classification")
    number = target.get("number")
    return {
        "kind": kind,
        "san": target.get("san", ""),
        "ply": target.get("ply"),
        # Fall back to the ply so the field is never None in a user-facing reply.
        "move_number": number if number is not None else target.get("ply"),
        "color": target.get("color") or "unknown",
        "marker": CLASS_COLORS.get(kind, ""),
        "confidence": target.get("confidence"),
        "evidence": target.get("evidence"),
        "url": page_url,
    }


def _shot_label(target: dict) -> str:
    """A short, sortable label for a per-move screenshot filename."""
    parts = []
    number = target.get("move_number")
    if number is not None:
        parts.append(f"m{number}")
    if target.get("ply") is not None:
        parts.append(f"ply{target['ply']}")
    return "_".join(parts) or "unknown"


async def get_classified_moves(page) -> list[dict]:
    """Every classified move in the sidebar.

    Each item: {"ply": 29, "number": 15, "color": "black", "san": "b1",
    "classification": "brilliant", "confidence": "high",
    "evidence": "g[id=brilliant]"}.

    ``ply`` is positional and always correct. ``confidence`` tells you whether
    chess.com explicitly labelled the move or we inferred it from a marker, so
    you can decide how firmly to state it to the user.
    """
    rows, _ = await move_list.read_moves(page)
    return [r for r in rows if r.get("classification")]


async def find_and_open_classified_move(
    page, kinds: tuple[str, ...] = ("brilliant", "blunder")
) -> dict:
    """Open the first move classified as one of ``kinds``.

    Book moves are never targeted (not in the default kinds). Returns
    ``{"kind", "san", "ply", "move_number", "color", "marker", "confidence",
    "evidence", "url"}`` or ``{}`` when nothing was found or the board would
    not move.
    """
    hits, _ = await move_list.find_classified(page, kinds)
    if not hits:
        return {}
    target = move_list.pick_highest_confidence(hits, kinds)
    if target is None:
        return {}
    if not await move_list.goto_ply(page, target["ply"]):
        _LOG.warning(
            "found a %s at ply %s but the board would not move to it",
            target.get("classification"),
            target.get("ply"),
        )
        return {}
    return _describe_move(target, page.url)


async def next_move(page, count: int = 1):
    """Step forward ``count`` plies.

    Returns a ``move_list.StepResult`` -- check ``.ok`` for a verified success
    and ``.reason`` for why it did not work.
    """
    return await move_list.step_moves(page, count)


async def previous_move(page, count: int = 1):
    """Step back ``count`` plies. See ``next_move`` for the return value."""
    return await move_list.step_moves(page, -count)


async def goto_move(page, ply: int) -> bool:
    """Navigate in-page to ``ply`` (1-based), verifying the board moved."""
    return await move_list.goto_ply(page, ply)


async def walk_moves(page, plies: list[int]) -> dict[int, bool]:
    """Visit several plies in order, e.g. to capture a sequence of positions.

    Returns {ply: moved_ok}. Useful for the "show me what happened around
    move 29" follow-up question, which the old pipeline could not answer.
    """
    out: dict[int, bool] = {}
    for ply in plies:
        out[ply] = await move_list.goto_ply(page, ply)
    return out


async def open_review_mode(page) -> bool:
    """Click 'Start Review' to enter the move-by-move coach.

    Thin wrapper around ``move_list.open_review_mode``, which owns the logic
    and knows both the "Game Review" and "Start Review" labels.

    Opt-in only, and deliberately not part of the default flow. The old code
    clicked this and then immediately did a `page.goto(?move=N)`, which tore
    the review session straight back down -- so the screenshot ended up of a
    cold page with no board tint and no coach bubble, i.e. the click bought
    nothing. In-page navigation (goto_move) replaces it for normal use; call
    this when you specifically want the auto-playing coach.
    """
    return await move_list.open_review_mode(page)


# ---------------------------------------------------------------------------
# Full flows
# ---------------------------------------------------------------------------


def _missing_playwright() -> dict:
    return {
        "error": "Playwright is not installed. Run: pip install -r requirements.txt"
    }


def _not_logged_in() -> dict:
    return {
        "error": "Not logged in. Run 'python login_chess.py' once to save your session."
    }


class _ToolError(Exception):
    """A failure worth reporting to the user verbatim.

    These tools are called by a language model, so a refusal has to be a
    sentence the model can relay, not a traceback it has to guess at.
    """


def _headed() -> bool:
    return os.getenv("REVIEW_HEADLESS", "1") == "0"


def _run(coro_factory, *args) -> dict:
    """Run a coroutine factory on a fresh loop, turning failures into dicts.

    Every tool here is a *synchronous* entry point for the agent loop, which
    JSON-serialises the return value. An ``async def`` tool would hand back a
    coroutine and raise "Object of type coroutine is not JSON serializable",
    aborting the turn -- which is exactly what happened once already.
    """
    try:
        return asyncio.run(coro_factory(*args))
    except _ToolError as exc:
        return {"error": str(exc)}
    except RuntimeError:  # e.g. an already-running loop in a notebook
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro_factory(*args))
        finally:
            loop.close()
    except Exception as exc:  # noqa: BLE001 - tool boundary: never raise
        _LOG.warning("tool failed", exc_info=True)
        return {"error": f"Browser automation failed: {exc}"}


@asynccontextmanager
async def _review_page(game_url: str):
    """Open a logged-in page on a game's Game Review, loaded and sanitised.

    Yields ``(page, game_id, review_url)`` and guarantees the browser is closed.
    The left navigation is hidden here, so every caller inherits a page whose
    screenshots can be cropped without the site chrome.
    """
    from playwright.async_api import async_playwright

    try:
        _, game_id, review_url = _build_review_url(game_url)
    except ValueError as exc:
        raise _ToolError(str(exc)) from exc
    state_path = session_state_path()
    if not state_path.exists():
        raise _ToolError(_not_logged_in()["error"])

    async with async_playwright() as p:
        session = await _open_session(p, state_path)
        try:
            page = session.page
            err = await _load_review_page(session, review_url, _headed())
            if err:
                raise _ToolError(err)
            await hide_left_nav(page)
            yield page, game_id, review_url
        finally:
            await session.close()


async def _load_review_page(session: _Session, review_url: str, headed: bool):
    """goto + settle + Cloudflare + readiness. Returns an error string or None."""
    page = session.page
    await page.goto(review_url, wait_until="domcontentloaded", timeout=30000)
    try:
        await page.wait_for_load_state("networkidle", timeout=20000)
    except _PlaywrightError as exc:
        _LOG.debug("networkidle not reached after goto: %s", exc)

    blocked = await _check_cloudflare(page, headed)
    if blocked:
        return blocked

    # Correct guard: we land on /analysis/... normally, so the fallback only
    # fires when chess.com bounced us to a consent/login/interstitial page.
    if "/analysis/" not in page.url:
        _LOG.info("bounced off the review page (%s), trying the tab", page.url)
        btn = await _first_visible(
            page,
            [
                "button:has-text('Game Review')",
                "a[href*='tab=review']",
                "a:has-text('Game Review')",
            ],
        )
        if btn is not None:
            try:
                await btn.click(timeout=8000)
                await page.wait_for_url("**/analysis**", timeout=20000)
            except _PlaywrightError as exc:
                _LOG.debug("Game Review tab click failed: %s", exc)

    # Two-stage wait. The layout shell mounts in well under a second; the
    # review data (accuracy pair, tally, move list) is computed server-side and
    # takes much longer. Two mistakes are being guarded against here: waiting
    # only for the absence of loaders (true before the app has mounted
    # anything) and using the board as the data signal (the board renders with
    # the shell, so that reported "ready" at 0ms against an empty sidebar).
    layout_ready = await move_list.wait_for_review_loaded(page, require="layout")
    if not layout_ready:
        _LOG.warning("review layout did not mount; results will be incomplete")
    panel_ready = await move_list.wait_for_review_loaded(page, require="panel")
    if not panel_ready:
        _LOG.warning(
            "review data did not arrive within the timeout; results will be "
            "incomplete. Raise REVIEW_TIMEOUT_MS on a slow connection."
        )
    if "Performing security verification" in await page.content():
        return "Blocked by Cloudflare security check."
    return None


async def _run_review_async(game_url: str) -> dict:
    """End-to-end review, producing a focused set of screenshots.

    Nothing here is a full-viewport shot. The user asked for the review and
    the coach's commentary, not chess.com's chrome, so the left navigation is
    hidden and every frame is clipped to the board plus the review sidebar:

      summary  -- the review panel: accuracy, the FULL classification tally
                  (the collapsed chevron is clicked first, so Brilliant and
                  Inaccuracy are visible rather than hidden behind it), the
                  game rating and the coach's opening line. Taken BEFORE the
                  review is started, because that is the only state the
                  accuracy pair exists in.
      <kind>_m<N>_ply<P> -- one frame per brilliant and per blunder, the board
                  plus the panel at that move, so you can see the position AND
                  the move highlighted in the list. There is no separate "all
                  moves" frame: the user asked for it to be dropped, and the
                  move list is legible in the per-move frames.

    A kind with no moves in this game gets no frames rather than a duplicate.
    """
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        return _missing_playwright()

    try:
        _, game_id, review_url = _build_review_url(game_url)
    except ValueError as e:
        return {"error": str(e)}

    state_path = session_state_path()
    if not state_path.exists():
        return _not_logged_in()
    stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    headed = os.getenv("REVIEW_HEADLESS", "1") == "0"

    session = None
    try:
        async with async_playwright() as p:
            session = await _open_session(p, state_path)
            page = session.page

            err = await _load_review_page(session, review_url, headed)
            if err:
                return {"error": err}

            # Belt and braces: the clip excludes the navigation, and hiding it
            # means even a fallback viewport shot stays clean.
            await hide_left_nav(page)

            # The tally starts collapsed, showing six of the nine rows. Expand
            # it BEFORE reading, so the numbers and the picture agree.
            await move_list.expand_review_summary(page)

            # ORDER MATTERS. The accuracy pair and the classification tally
            # only exist BEFORE the review is started; read_moves_when_ready
            # presses Start Review, which swaps the summary for the
            # move-by-move view. Reading them afterwards returns None/{} and
            # looks exactly like a page that had no summary.
            white_acc, black_acc, counts = await _extract_review_data(page)

            # The one always-taken shot: the review panel, fully expanded.
            summary_path = _shot_path(SHOT_KIND, game_id, stamp)
            await capture_review_panel(page, summary_path)

            # This presses Start Review, waits for the move list, and returns
            # the rows it parsed. A second read would re-walk the whole
            # document and could disagree with the rows used to pick a target.
            all_rows, read_diag = await move_list.read_moves_when_ready(page)
            rows = [r for r in all_rows if r.get("classification")]
            total_plies = read_diag.get("pliesKnown") or len(all_rows)
            _LOG.info(
                "parsed %d classified moves: %s (%s)",
                len(rows),
                move_list.summarise(rows),
                read_diag.get("coverage") or "coverage unknown",
            )

            # One frame per brilliant and per blunder, in move order.
            screenshots: dict[str, str] = {SHOT_KIND: str(summary_path)}
            highlights: dict[str, dict] = {}
            truncated = 0
            for kind in MOVE_SHOT_KINDS:
                candidates = [r for r in rows if r.get("classification") == kind]
                if not candidates:
                    _LOG.info("no %s in this game; no %s screenshot", kind, kind)
                    continue
                candidates.sort(key=lambda r: r.get("ply") or 0)
                for target in candidates:
                    if len(highlights) >= MAX_MOVE_SHOTS:
                        truncated += 1
                        continue
                    # The list is virtualised, so scroll the target into the
                    # rendered window and re-read: the re-read re-stamps the
                    # per-ply tags that navigation clicks.
                    await move_list.scroll_to_ply(
                        page, target.get("ply") or 1, total_plies
                    )
                    await move_list.read_moves(page)
                    if not await goto_move(page, target["ply"]):
                        _LOG.warning(
                            "found a %s at ply %s but navigation failed",
                            kind,
                            target.get("ply"),
                        )
                        continue
                    described = _describe_move(target, page.url)
                    path = _shot_path(kind, game_id, stamp, _shot_label(described))
                    await screenshot_review_area(page, path)
                    name = path.stem
                    screenshots[name] = str(path)
                    highlights[name] = described
            if truncated:
                _LOG.warning(
                    "%d further %s move(s) had no screenshot: the cap is %d. "
                    "Raise MAX_MOVE_SHOTS in tools/review.py to see them.",
                    truncated,
                    "/".join(MOVE_SHOT_KINDS),
                    MAX_MOVE_SHOTS,
                )
    except Exception as e:  # noqa: BLE001 - tool boundary: never raise, return an error
        _LOG.warning("Browser automation failed", exc_info=True)
        return {"error": f"Browser automation failed: {e}"}
    finally:
        if session is not None:
            await session.close()

    # The full classified list, so the model can answer "list every mistake"
    # instead of only ever seeing one highlighted move. `best`, `excellent`
    # and `good` are excluded: they are chess.com's approval of a move, not
    # something worth reporting.
    notable = [
        {
            "kind": r.get("classification"),
            "san": r.get("san"),
            "move_number": r.get("number"),
            "color": r.get("color"),
            "ply": r.get("ply"),
            "confidence": r.get("confidence"),
        }
        for r in rows
        if r.get("classification") in NOTABLE
    ]
    first = next(iter(highlights.values()), {})
    return {
        "game_url": (game_url or "").strip(),
        "review_url": review_url,
        "move_url": first.get("url"),
        "white_accuracy": white_acc,
        "black_accuracy": black_acc,
        "move_counts": counts,
        "notable_moves": notable,
        "moves_read": read_diag.get("coverage") or len(all_rows),
        "highlight": first,
        "highlights": highlights,
        "screenshots": screenshots,
        "note": "Screenshots are clipped to the board and the review panel; "
        "chess.com's left navigation is not included. The summary frame shows "
        "the fully expanded classification tally. Reply with the accuracy "
        "numbers, the notable moves, and these file paths.",
    }


async def get_annotated_review(
    game_url: str, queries: list[dict] | None = None
) -> dict:
    """Vision-oriented entry point: annotated screenshot + element map.

    Goes to the direct review URL, waits for the analysis to render,
    draws indexed boxes over `queries` (defaults to DEFAULT_ANNOTATION_QUERIES)
    and returns {"annotated_shot": path, "element_map": {...},
    "white_accuracy": ..., "black_accuracy": ..., "move_counts": ...,
    "moves": [...], "review_url": ...}.
    Feed annotated_shot + element_map to Gemini Vision to pick a target, then
    use goto_move() / screenshot_element().

    This entry point keeps the whole annotated frame; review_game_screenshots
    is the one with the curated, cropped shots.
    """
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        return _missing_playwright()
    try:
        _, game_id, review_url = _build_review_url(game_url)
    except ValueError as e:
        return {"error": str(e)}

    state_path = session_state_path()
    if not state_path.exists():
        return _not_logged_in()
    stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    annotated_shot = _shot_path("annotated", game_id, stamp)
    headed = os.getenv("REVIEW_HEADLESS", "1") == "0"

    session = None
    try:
        async with async_playwright() as p:
            session = await _open_session(p, state_path)
            page = session.page
            err = await _load_review_page(session, review_url, headed)
            if err:
                return {"error": err}
            await hide_left_nav(page)
            white_acc, black_acc, counts = await _extract_review_data(page)
            element_map = await annotate_page_elements(page, queries)
            await page.screenshot(path=str(annotated_shot), full_page=False)
            await clear_annotations(page)
            all_rows, _ = await move_list.read_moves_when_ready(page)
            moves = [r for r in all_rows if r.get("classification")]
    except Exception as e:  # noqa: BLE001 - tool boundary: never raise, return an error
        _LOG.warning("Browser automation failed", exc_info=True)
        return {"error": f"Browser automation failed: {e}"}
    finally:
        if session is not None:
            await session.close()

    return {
        "review_url": review_url,
        "white_accuracy": white_acc,
        "black_accuracy": black_acc,
        "move_counts": counts,
        "annotated_shot": str(annotated_shot),
        "element_map": element_map,
        "classified_moves": moves,
    }


async def _goto_and_read(page, rows, diag, ply: int) -> dict:
    """Navigate to one move, re-reading so the ply tags are fresh.

    Split out because three tools need exactly this: scroll the target into the
    rendered window, re-read so the tags the click depends on are current, then
    click and verify the board actually moved.

    Returns the reader row with the description's page URL merged in, so the
    caller gets the clock and the evidence alongside the reply-shaped fields.
    """
    total = diag.get("pliesKnown") or len(rows)
    await move_list.scroll_to_ply(page, ply, total)
    await move_list.read_moves(page)
    if not await goto_move(page, ply):
        return {}
    row = next((r for r in rows if r.get("ply") == ply), None)
    if row is None:
        return {}
    return {**row, "url": page.url}


def _move_summary(m: dict) -> dict:
    """The one shape the model sees for a move.

    Accepts either a reader row (which calls the sidebar's move number
    ``number`` and the verdict ``classification``) or an ``_describe_move``
    result (``move_number`` and ``kind``). Both are read here rather than at
    each call site, because guessing wrong is silent: a live run reported
    ``move_number: null`` and ``classification: null`` for a blunder it had
    otherwise found perfectly, purely because this function looked for the
    wrong key.
    """
    return {
        "ply": m.get("ply"),
        "move_number": m.get("move_number", m.get("number")),
        "color": m.get("color"),
        "san": m.get("san", ""),
        "classification": m.get("classification") or m.get("kind"),
        "confidence": m.get("confidence"),
        "clock_seconds": m.get("clock_seconds"),
    }


async def _get_moves_async(game_url: str) -> dict:
    """The full move table, one entry per ply."""
    async with _review_page(game_url) as (page, _game_id, review_url):
        # Starting the review destroys the summary panel, so nothing may be
        # read from it here. Call get_review_state for that.
        await move_list.expand_review_summary(page)
        # Read the summary BEFORE starting the review: Start Review replaces it.
        # Doing so also lets this tool answer "what went wrong in this game"
        # completely -- the per-move positions chess.com annotates, plus the
        # totals for the kinds it does not annotate at all.
        summary = await move_list.read_summary_panel(page)
        rows, diag = await move_list.read_moves_when_ready(page)
        if not rows:
            raise _ToolError(
                "The move list could not be read. chess.com may have changed its "
                "markup -- run: python dump_pages.py " + game_url
            )
        moves = [_move_summary(r) for r in rows]
        counts = _tally_of(moves)
        totals = summary.get("move_counts") or {}
        result = {
            "game_url": (game_url or "").strip(),
            "review_url": review_url,
            "moves": moves,
            "count": len(moves),
            "notable_moves": [m for m in moves if m.get("classification") in NOTABLE],
            "moves_read": diag.get("coverage") or len(moves),
            "plies_contiguous": diag.get("pliesContiguous"),
            # what the sidebar actually annotates, per move
            "classifications": counts,
            # what the summary counts, per game -- the only source for the kinds
            # chess.com does not annotate move by move
            "summary_counts": totals,
            "white_accuracy": summary.get("white"),
            "black_accuracy": summary.get("black"),
        }
        result.update(_annotation_report(moves, counts, totals))
        if not diag.get("pliesContiguous", True):
            result["warning"] = (
                "The plies returned are not 1..N with no gaps, so chess.com's "
                "markup has probably changed. Treat the move numbers with care."
            )
        return result


def _annotation_report(
    moves: list[dict], counts: dict[str, int], totals: dict[str, int]
) -> dict:
    """Say out loud which classifications could not be pinned to a move.

    chess.com's sidebar only annotates some moves: in the captured game, 7 of
    29 cells carried a label (brilliant, blunder, two mistakes, book and two
    best moves) while the summary counted 7 inaccuracies, 1 miss and 13 best
    moves that no cell mentions. The words appear in the legend and the tally
    only. The analysis board was checked and has no per-move classification
    markup at all, so there is no second source to read them from.

    So `classification: null` means "chess.com put no annotation on this move",
    which is NOT the same as "this move was fine" -- a null may be an
    inaccuracy. Returning the counts and saying so is the honest answer; letting
    the model infer "null means good" is how a review tells the player they had
    no inaccuracies when they had seven.
    """
    annotated = sum(counts.values())
    unannotated = len(moves) - annotated
    report = {
        "annotated_moves": annotated,
        "unannotated_moves": unannotated,
    }
    # The shortfall per kind: how many of the summary's count no move accounts
    # for. NOT the whole count -- "best" is annotated on 2 of 13 moves, so
    # reporting 13 as unlocated would claim 15 best moves between two readers
    # that only ever had 13, and the reconciliation would never balance.
    not_located = {
        k: v - counts.get(k, 0)
        for k, v in sorted(totals.items())
        if v and counts.get(k, 0) < v
    }
    if not_located:
        report["counts_without_a_move"] = not_located
        report["note"] = (
            "chess.com counts these in the summary but does not annotate the "
            "moves in the sidebar, so their positions are not available: "
            + ", ".join(f"{k} x{v}" for k, v in not_located.items())
            + ". A move with classification null is one of them or an ordinary "
            "best/good move -- it does not mean the move was sound."
        )
    else:
        report["note"] = (
            "Every classification the summary counts is also annotated on a "
            "move. A null classification means chess.com put no annotation there."
        )
    return report


def _tally_of(moves: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for m in moves:
        k = m.get("classification")
        if k:
            counts[k] = counts.get(k, 0) + 1
    return counts


def get_moves(game_url: str) -> dict:
    """Read the full move list of a game, with each move's classification.

    Use this for questions about what was played, not just what went wrong:
    "what did I play on move 22", "how many book moves were there", "list every
    blunder", "what was the move before the mistake". For accuracy numbers and
    the per-move engine evaluation use get_review_state and
    get_coach_commentary; for a whole-game review with screenshots use
    review_game_screenshots.

    This is the Gemini tool entry point, so it is deliberately synchronous even
    though the pipeline underneath is async.

    Args:
        game_url: A chess.com game link, e.g.
            "https://www.chess.com/game/live/184577982780". Must contain
            "/game/live/" or "/game/daily/" followed by a numeric id.

    Returns:
        {"moves": [{ply, move_number, color, san, classification, confidence,
        clock_seconds}, ...], "count": n, "notable_moves": [...],
        "moves_read": "29/29", "classifications": {"blunder": 1, ...},
        "summary_counts": {"blunder": 1, "inaccuracy": 7, ...},
        "white_accuracy": 94.1, "black_accuracy": 75.5,
        "annotated_moves": 7, "unannotated_moves": 22,
        "counts_without_a_move": {"inaccuracy": 7, ...}, "note": "..."}

        `color` is the side that played the move ("white"/"black").
        `clock_seconds` is the time spent on the move, or null when the sidebar
        is showing clocks off.

        READ `classifications` AND `summary_counts` TOGETHER, and never treat a
        null `classification` as "that move was fine". chess.com annotates only
        SOME moves in the sidebar -- brilliant, blunder, mistake, book and the
        occasional best move -- and the kinds it leaves off (inaccuracy and miss
        in particular) exist only as totals in the summary tally. Their positions
        are not in the page; the analysis board was checked and does not have
        them either. So:

          * to say HOW MANY blunders there were, use summary_counts
          * to say WHERE a blunder was, use the moves' `classification`
          * do not say "the rest were fine". Say "the summary counts N
            inaccuracies whose moves chess.com does not mark".

        `note` spells this out for the current game, and `counts_without_a_move`
        is machine-readable.
        On failure returns {"error": "..."}.
    """
    return _run(_get_moves_async, game_url)


async def _get_review_state_async(game_url: str) -> dict:
    """The summary panel: accuracy, classification tally, player names."""
    async with _review_page(game_url) as (page, _game_id, review_url):
        await move_list.expand_review_summary(page)
        summary = await move_list.read_summary_panel(page)
        state = await move_list.read_review_state(page)
        players = summary.get("players_in_dom_order") or []
        ratings = summary.get("ratings_in_dom_order") or []
        counts = summary.get("move_counts") or {}
        by_side = summary.get("move_counts_by_side") or {}
        result = {
            "game_url": (game_url or "").strip(),
            "review_url": review_url,
            "white_accuracy": summary.get("white"),
            "black_accuracy": summary.get("black"),
            # the whole game: both players' moves summed
            "move_counts": counts,
            "move_counts_by_side": by_side,
            "players": [
                {"name": n, "rating": ratings[i] if i < len(ratings) else None}
                for i, n in enumerate(players)
            ],
            "review_started": bool(state.get("moveListFound")),
            "moves_available": state.get("moveCells") or 0,
            "page": {
                "elements": state.get("elements"),
                "body_text_length": state.get("bodyChars"),
                "still_loading": state.get("placeholders") or [],
            },
        }
        notes = []
        # The per-side split is derived from the tally's column order, which is
        # the same white-then-black order the accuracy row uses. Say so, and say
        # what to fall back on, rather than presenting it as labelled fact.
        if by_side:
            notes.append(
                "move_counts_by_side follows the tally's column order, which is "
                "the same White-then-Black order as the accuracy row."
            )
        if len(players) == 2:
            notes.append(
                "The two players are listed in the order chess.com renders "
                "them; the markup does not say which one is White, so do not "
                "attach a name to white_accuracy/black_accuracy."
            )
        if notes:
            result["notes"] = notes
        return result


def get_review_state(game_url: str) -> dict:
    """Read the summary panel: accuracy, the classification tally, the players.

    Use this when the user asks "how accurate was I", "what was my rating",
    "what did the review say about the game" -- questions about the game as a
    whole rather than about one move. For the moves themselves use get_moves.

    The accuracy pair and the tally exist only BEFORE the review is started;
    this tool reads them at that point and does not start the review. That
    ordering is not defensive: the captured markup has zero `review-rating-*`
    and zero `tally-row-value` elements once Start Review has been pressed.

    This is the Gemini tool entry point, so it is deliberately synchronous even
    though the pipeline underneath is async.

    Args:
        game_url: A chess.com game link, e.g.
            "https://www.chess.com/game/live/184577982780". Must contain
            "/game/live/" or "/game/daily/" followed by a numeric id.

    Returns:
        {"white_accuracy": float|None, "black_accuracy": float|None,
        "move_counts": {"blunder": 1, "inaccuracy": 5, ...},
        "move_counts_by_side": {"white": {...}, "black": {...}},
        "players": [{"name": ..., "rating": ...}, ...], "review_started": bool,
        "moves_available": n, "page": {...}, "notes": [...]}

        `move_counts` is the whole game, both players summed -- that is what
        "how many mistakes were in this game" means, and it is the number that
        agrees with get_moves. The per-side split is derived from the tally's
        column order; the `notes` field says so. Accuracies are null when
        chess.com did not load them; they are never invented. The players are in
        chess.com's render order, which the markup does not tie to a colour, so
        do not attach a name to an accuracy.
        On failure returns {"error": "..."}.
    """
    return _run(_get_review_state_async, game_url)


async def _get_coach_commentary_async(
    game_url: str, move_number: int = 0, colour: str = "", ply: int = 0
) -> dict:
    """The coach's verdict, engine number and prose for one move."""
    async with _review_page(game_url) as (page, _game_id, review_url):
        await move_list.expand_review_summary(page)
        rows, diag = await move_list.read_moves_when_ready(page)
        if not rows:
            raise _ToolError(
                "The move list could not be read, so no commentary is available. "
                "Run: python dump_pages.py " + game_url
            )
        matches, err = resolve_move(rows, ply, move_number, colour)
        if err:
            raise _ToolError(err)

        described = await _goto_and_read(page, rows, diag, matches[0]["ply"])
        if not described:
            raise _ToolError(
                f"Found move {matches[0].get('number')} but the board would not "
                f"move to it (ply {matches[0]['ply']})."
            )
        # The coach panel is rendered for the current move only -- one
        # .move-feedback-box-component existed in the whole captured document --
        # so it has to be read after navigating, not before.
        coach = await move_list.read_coach_commentary(page)
        move = _move_summary(described)
        # When the coach disagrees with the sidebar's own label, the coach is
        # looking at this move right now and the label was read earlier from a
        # list, so prefer the coach. Reported separately either way.
        icon_kind = move_list.classify_class(coach.get("icon_id"))
        if icon_kind and icon_kind != move.get("classification"):
            move["classification_from_list"] = move["classification"]
            move["classification"] = icon_kind
        return {
            "game_url": (game_url or "").strip(),
            "review_url": review_url,
            "move": move,
            "verdict": coach.get("verdict"),
            "engine_score": coach.get("score"),
            "engine_score_sign": coach.get("score_sign"),
            "commentary": coach.get("commentary"),
            "coach_icon": coach.get("icon_id"),
            "coach_present": bool(coach.get("present")),
            # Book moves get the opening's name instead of a verdict, e.g.
            # "Grob Opening" on a game that opened 1.g4.
            "opening": coach.get("opening"),
            "last_book_move_number": coach.get("last_book_move_number"),
            "moves_read": diag.get("coverage") or len(rows),
        }


def get_coach_commentary(
    game_url: str, move_number: int = 0, colour: str = "", ply: int = 0
) -> dict:
    """The coach's own words about one move, plus that move's engine number.

    This is the teaching material, and it is the part a plain screenshot cannot
    give the model: chess.com's coach says things like "is a blunder" and "you
    are preparing to unleash the bishop", and prints the engine evaluation that
    move produced. Use it when the user wants to understand a move -- "why was
    that a mistake", "what did the coach say about move 12" -- rather than
    merely to see it.

    Number the move the way the user did:

    * "move 12"      -> move_number=12, and add colour, because both sides
                        played a 12th move
    * "the 24th move" -> ply=24, unambiguous on its own
    * "12...Nf6"     -> move_number=12, colour="black"

    If the move is ambiguous and no colour is given, this returns an error
    naming both sides rather than guessing -- guessing would attribute the
    coach's words to the wrong move. Read the error, then call again with the
    colour.

    This is the Gemini tool entry point, so it is deliberately synchronous even
    though the pipeline underneath is async.

    Args:
        game_url: A chess.com game link, e.g.
            "https://www.chess.com/game/live/184577982780". Must contain
            "/game/live/" or "/game/daily/" followed by a numeric id.
        move_number: The move number chess.com prints in the sidebar, e.g. 12
            for "12. exd5". Zero means "not specified".
        colour: Which side played it: "white" or "black". Needed whenever both
            sides played a move with the same number, which is most moves.
        ply: The half-move index counting every move in the game, e.g. 24 for
            the 24th move played. An alternative to move_number, not an
            addition to it.

    Returns:
        {"move": {ply, move_number, color, san, classification, ...},
        "verdict": "is a blunder", "engine_score": "-1.07",
        "engine_score_sign": -1, "commentary": "the coach's prose in prose",
        "coach_icon": "blunder", "coach_present": true}

        `commentary` is null when the coach has nothing to say about that move,
        which is normal for quiet moves. `engine_score` is the evaluation after
        the move, with `engine_score_sign` carrying its direction. On failure
        returns {"error": "..."}, including when the move does not exist or is
        ambiguous.
    """
    return _run(_get_coach_commentary_async, game_url, move_number, colour, ply)


def resolve_move(
    rows: list[dict],
    ply: int = 0,
    move_number: int = 0,
    color: str = "",
) -> tuple[list[dict], str]:
    """Find the move a user meant, from either numbering chess.com uses.

    chess.com labels moves two ways and the user may use either: the sidebar
    prints the *full move number* (1., 2., 3. ...) and a half-move count runs
    alongside it, so "move 14" and "the 27th half-move" are different ways of
    pointing at the same position. Pure function, so it is unit-testable
    without a browser.

    Returns ``(matches, error)``. ``error`` is non-empty when the request could
    not be resolved unambiguously -- and it says what *is* available rather
    than silently photographing the wrong position.
    """
    if ply:
        hits = [r for r in rows if r.get("ply") == ply]
        if not hits:
            return [], (
                f"No ply {ply}. This game has {len(rows)} moves "
                f"(1 to {rows[-1]['ply'] if rows else 0})."
            )
        return hits, ""

    if not move_number:
        return [], "Provide either move_number (e.g. 14) or ply (e.g. 27)."

    want = (color or "").strip().lower()
    hits = [r for r in rows if r.get("number") == move_number]
    if not want:
        if len(hits) == 1:
            return hits, ""
        if not hits:
            return [], (
                f"No move {move_number} in this game. It has moves 1 to "
                f"{max((r.get('number') or 0) for r in rows) if rows else 0}."
            )
        # Ambiguous: white and black both played a move N. Report both rather
        # than guessing, because guessing photographs the wrong position.
        return [], (
            f"Move {move_number} was played by both "
            f"{', '.join(sorted({r.get('color') or 'unknown' for r in hits}))}. "
            "Say which one, e.g. colour='black'."
        )

    exact = [r for r in hits if (r.get("color") or "").lower() == want]
    if exact:
        return exact, ""
    return [], (
        f"No {want} move {move_number} in this game. "
        f"Move {move_number} was "
        f"{', '.join(sorted({r.get('color') or 'unknown' for r in hits}))}."
        if hits
        else f"No move {move_number} in this game."
    )


async def _screenshot_move_async(
    game_url: str, ply: int = 0, move_number: int = 0, color: str = ""
) -> dict:
    """Open one specific move and photograph the position."""
    stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    async with _review_page(game_url) as (page, game_id, review_url):
        await move_list.expand_review_summary(page)
        # read_moves_when_ready drives all three load stages and returns the
        # rows it parsed, so the target is chosen from the same data that
        # navigation uses rather than a second, possibly different, read.
        rows, diag = await move_list.read_moves_when_ready(page)
        if not rows:
            raise _ToolError(
                "The move list could not be read, so no move could be "
                "photographed. Run: python dump_pages.py " + game_url
            )
        matches, err = resolve_move(rows, ply, move_number, color)
        if err:
            raise _ToolError(err)
        described = await _goto_and_read(page, rows, diag, matches[0]["ply"])
        if not described:
            raise _ToolError(
                f"Found move {matches[0].get('number')} but the board would not "
                f"move to it (ply {matches[0]['ply']})."
            )
        summary = _move_summary(described)
        path = _shot_path("move", game_id, stamp, _shot_label(summary))
        await screenshot_review_area(page, path)
        summary["evidence"] = described.get("evidence")
        summary["screenshot"] = str(path)
        return {
            "game_url": (game_url or "").strip(),
            "review_url": review_url,
            "move": described,
            "moves_read": diag.get("coverage") or len(rows),
            "note": "The frame is clipped to the board and the review panel. "
            "The navigation is not included.",
        }


def screenshot_game_move(
    game_url: str, move_number: int = 0, colour: str = "", ply: int = 0
) -> dict:
    """Photograph ONE specific move of a game, at any move, not just the errors.

    Use this when the user names a move -- "screenshot move 14", "show me the
    position after 22...Nf6", "what did it look like on move 8?" -- rather than
    asking for a whole review. For a full review with accuracy and the
    brilliant/blunder frames use review_game_screenshots.

    This is the Gemini tool entry point, so it is deliberately synchronous even
    though the pipeline underneath is async.

    Args:
        game_url: A chess.com game link, e.g.
            "https://www.chess.com/game/live/184577982780". Must contain
            "/game/live/" or "/game/daily/" followed by a numeric id.
        move_number: The move number chess.com prints in the sidebar, e.g. 14
            for "14. O-O". Use this for "move 14". Zero means "not specified".
        colour: Which side played it: "white" or "black". Optional, but
            needed when both sides played a move with the same number (which is
            most moves). Leave it out and the tool will tell you it is
            ambiguous rather than guessing.
        ply: The half-move index, counting every move in the game, e.g. 27 for
            the 27th move played. Use this for "the 27th move". An alternative
            to move_number, not an addition to it.

    Returns:
        On success: {"move": {san, move_number, color, ply, classification,
        confidence, evidence, url, screenshot}, "moves_read": "29/29", ...}
        where "screenshot" is the absolute path of the PNG.
        On failure: {"error": "..."} -- including when the move does not exist
        or is ambiguous, in which case the message says what is available.
    """
    return _run(_screenshot_move_async, game_url, ply, move_number, colour)


def review_game_screenshots(game_url: str) -> dict:
    """Review a chess.com game and save annotated screenshots.

    This is the Gemini tool entry point, so it is deliberately **synchronous**
    even though the whole pipeline underneath is async. An earlier version was
    accidentally declared `async def`, which handed the agent loop a coroutine
    object; `from_function_response` then failed with "Object of type coroutine
    is not JSON serializable" and the review silently never ran.

    Use this when the user gives a chess.com /game/ link and asks for a game
    review, brilliant/blunder moves, accuracy, or screenshots. It needs a
    saved session: run `python login_chess.py` once first.

    Args:
        game_url: A chess.com game link, e.g.
            "https://www.chess.com/game/live/184577982780". Must contain
            "/game/live/" or "/game/daily/" followed by a numeric id.

    Returns a dict with:
      - white_accuracy, black_accuracy: floats, or None if chess.com did not
        load them. Never invented.
      - move_counts: {"brilliant": 1, "blunder": 1, "inaccuracy": 5, ...}
        keyed by classification, taken from the summary tally.
      - notable_moves: [{kind, san, move_number, color, ply, confidence}]
        every move worth reporting -- brilliant, great, blunder, inaccuracy,
        mistake, miss -- with the move number and the side that played it.
      - highlight: the most trustworthy brilliant-or-blunder, as
        {kind, san, move_number, color, marker, ply, confidence, evidence, url}.
      - highlights: one entry per screenshot actually taken, keyed by the
        filename stem, e.g. "brilliant_m14_ply27" -> {...}.
      - moves_read: "29/29" style coverage, or None if unknown. The move list
        is virtualised, so a short read means some moves were never rendered.
      - screenshots: {summary, <per-move stems>} -> absolute PNG paths,
        containing only the shots actually taken. Every frame is clipped to
        the board and the review sidebar, and chess.com's left navigation is
        hidden, so no frame contains the site menu or the player's profile.
          summary        - the review panel: accuracy, the FULL classification
                           tally (the collapsed chevron is expanded first), the
                           game rating and the coach's commentary.
          brilliant_m..  - the board and the panel at that brilliant move.
          blunder_m..    - the board and the panel at that blunder move.
        There is at most MAX_MOVE_SHOTS per-move frames.

    On failure returns {"error": "human readable message"}.
    """
    return _run(_run_review_async, game_url)
