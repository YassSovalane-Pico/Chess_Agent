"""A long-lived, logged-in chess.com browser session.

Why this exists
---------------
Every existing browser tool opens Chrome, does one thing, and closes it
(`tools/review.py::_run`). That is fine for "review this game" and useless for
"go forward two moves, then screenshot, then step back". An interactive session
has to outlive a single tool call, which the current shape cannot do for two
independent reasons -- both measured in `spikes/s0_1_loop_affinity.py`:

  1. **Loop affinity.** `_run` uses `asyncio.run`, so a new event loop per call.
     A Playwright `Page` is bound to the loop it was created on; once that loop
     dies the transport is gone and `page.evaluate` raises
     `AttributeError: 'NoneType' object has no attribute 'send'`. So the session
     owns ONE private event loop, on ONE dedicated thread, and submits every
     browser operation to it with `asyncio.run_coroutine_threadsafe`.

  2. **Operations must be serialised.** Two operations on one page at the same
     instant destroy each other: a `page.goto` tears down the execution context
     a concurrent `page.evaluate` was about to use ("Execution context was
     destroyed, most likely because of a navigation"). So every submit takes a
     lock. This is not optional -- it is what the R half of the spike measured.

Design
------
* One Chromium, one context, one page, for the life of the session.
* `submit()` is the only way in. It ensures the session is open, takes the
  operation lock, runs a coroutine on the private loop, and updates the idle
  clock. Callers never touch the loop or the lock.
* An idle TTL closes the browser when nobody has asked for it for a while, so a
  forgotten session does not hold a Chromium and ~300MB for ever. Call it with
  `close_browser_session()` and it closes immediately.
* A navigation allow-list keeps the agent on review/analysis/lesson/puzzle
  pages and off anything that could change the account.

The allow-list
--------------
The Discord gate exists because these tools drive the owner's logged-in
chess.com account. Giving the agent free navigation widens that from "reads a
game review" to "browses my account", so it is fenced: only the board/review/
lesson/puzzle surface is reachable. Settings, logout, friends and messages are
refused by name. The URL is checked *before* a navigation and *again* on the
page's final URL, because chess.com is a single-page app that can redirect
client-side after `goto` returns.

Concurrency note
----------------
The Discord front end already gates every browser tool with an `asyncio.Lock`
(`agent.BROWSER_TOOLS`), so an interactive call never runs concurrently with a
review tool. This session's own `threading.RLock` handles the finer-grained case
of several interactive operations racing on the one page. An idle session
Chromium and a cold review Chromium may briefly coexist in memory; that is a
resource overlap, not a correctness one (each browser reads the session cookies
from disk; neither writes them). Phase 4 unifies the two so the overlap goes
away.
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
import time
from collections.abc import Callable
from typing import Any

from .common import load_browser_profile, session_state_path

_LOG = logging.getLogger(__name__)

# How long the browser may sit unused before it closes itself. Long enough to
# carry a short conversation ("go forward", "screenshot that", "now back two"),
# short enough that a forgotten session does not hold a Chromium for ever.
DEFAULT_IDLE_SECONDS = 90.0

# Browsers are opened headed or hidden by the same env var as the review tools,
# so `REVIEW_HEADLESS=0` is the one switch that shows the window and lets the
# owner solve a Cloudflare check by hand.
LAUNCH_ARGS = [
    "--disable-blink-features=AutomationControlled",
    "--disable-dev-shm-usage",
    "--no-sandbox",
]

# The page must be on chess.com and inside this fence. Prefixes are matched
# against the path, so `/analysis/game/...` and `/analysis/` both pass while
# `/analysisfoo` does not (a trailing boundary check follows each prefix).
ALLOWED_PATH_PREFIXES = (
    "/",
    "/home",
    "/analysis",
    "/game/live/",
    "/game/daily/",
    "/lessons/",
    "/puzzles/",
    "/openings/",
)

# Named so the refusal message can be specific rather than a blanket "no".
REFUSED_EXAMPLES = {
    "settings": "changing account settings",
    "logout": "logging out of your account",
    "friends": "your friends list",
    "messages": "messages and notifications",
}

_NAV_TIMEOUT = 30000


def _normalise_url(url: str) -> str:
    """Absolute, lowercase host, no query/fragment -- the value we fence on."""
    url = (url or "").strip()
    if not re.match(r"^https?://", url, re.I):
        url = "https://" + url.lstrip("/")
    if not re.match(r"^https?://", url, re.I):
        url = "https://" + url.lstrip("/")
    url = url.split("#", 1)[0].split("?", 1)[0]
    # A trailing slash is dropped so a prefix boundary check is unambiguous --
    # except on the site root, where "/" IS the path.
    if url.endswith("/") and url.count("/") > 2:
        url = url[:-1]
    return url


def url_allowed(url: str) -> tuple[bool, str]:
    """Is this URL reachable by the session? -> (allowed, reason).

    A pure function so it is testable without a browser, and so the refusal
    message the model reads is produced in one place.
    """
    norm = _normalise_url(url)
    if not norm:
        return False, "That is not a URL."
    m = re.match(r"^https?://([^/]+)(/.*)?$", norm, re.I)
    if not m:
        return False, f"I could not make sense of that URL: {url!r}."
    host = m.group(1).lower()
    path = m.group(2) or "/"
    if host not in ("www.chess.com", "chess.com"):
        return False, (
            f"The session only browses chess.com, and that link is on {host}."
        )
    if path == "/":
        return True, ""
    # Each prefix is matched at a path boundary, so `/analysis` and
    # `/analysis/game/...` pass while `/analysisfoo` does not. Prefixes are
    # compared with any trailing slash stripped, because the normaliser drops
    # it from the URL too -- otherwise `/puzzles/` would never match the
    # prefix `/puzzles/`.
    for prefix in ALLOWED_PATH_PREFIXES:
        if prefix == "/":
            continue
        base = prefix.rstrip("/")
        if path == base or path.startswith(base + "/"):
            return True, ""
    hint = ""
    for frag, what in REFUSED_EXAMPLES.items():
        if frag in path.lower():
            hint = f" ({what} is off-limits on purpose)"
            break
    return False, (
        f"I will only open the chess.com board, review, lesson, puzzle and "
        f"opening pages -- not {path}{hint}."
    )


# ---------------------------------------------------------------------------
# The private event loop
# ---------------------------------------------------------------------------


class _LoopThread:
    """One event loop, on one thread, alive for the session.

    The loop is created here rather than by `asyncio.run` so browser objects
    stay on the loop that made them. `run_forever` on a daemon thread keeps the
    process able to exit even if the loop is never stopped cleanly.
    """

    def __init__(self, name: str = "chess-agent-browser-loop") -> None:
        self.loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self.thread = threading.Thread(target=self._run, name=name, daemon=True)
        self.thread.start()
        self._ready.wait(10)

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self._ready.set()
        self.loop.run_forever()

    def run(self, factory: Callable[[], Any], timeout: float) -> Any:
        """Run `factory()` (a zero-arg coroutine factory) on the private loop."""
        return asyncio.run_coroutine_threadsafe(factory(), self.loop).result(timeout)

    def close(self) -> None:
        try:
            self.loop.call_soon_threadsafe(self.loop.stop)
        except RuntimeError:  # already closed
            return
        self.thread.join(timeout=10)


async def _drain() -> None:
    """Cancel and gather pending tasks so nothing is 'destroyed but pending'."""
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


# ---------------------------------------------------------------------------
# The session
# ---------------------------------------------------------------------------


class SessionClosed(Exception):
    """Raised inside the session when an operation needs a closed browser."""


class BrowserSession:
    """One logged-in chess.com page, outliving individual tool calls.

    Not thread-safe on its own: callers go through `submit`, which serialises.
    """

    def __init__(self, idle_seconds: float | None = None) -> None:
        raw = idle_seconds
        if raw is None:
            import os

            raw = float(os.getenv("SESSION_IDLE_SECONDS", DEFAULT_IDLE_SECONDS))
        self.idle_seconds = max(5.0, float(raw))
        # Guards `_runner`/open bookkeeping and every submit against each
        # other. An RLock, not a Lock: `close` is legitimately re-entered from
        # the TTL timer while a caller already holds it.
        self._op_lock = threading.RLock()
        self._runner: _LoopThread | None = None
        self._page = None
        self._context = None
        self._browser = None
        self._playwright = None
        self._url: str = ""
        self._last_used = 0.0
        self._ttl_thread: threading.Thread | None = None
        self._ttl_stop = threading.Event()
        # Per-game snapshot of the review summary, which the review page only
        # shows before "Start Review" wipes it. See `_snapshot_review_summary`.
        self._summary_cache: dict[str, dict] = {}

    # -- state -------------------------------------------------------------

    @property
    def is_open(self) -> bool:
        return self._page is not None and self._runner is not None

    @property
    def current_url(self) -> str:
        return self._url

    def describe(self) -> dict:
        """A small snapshot of the session for tool replies and `!status`."""
        return {
            "open": self.is_open,
            "url": self._url,
            "idle_seconds_ago": (
                round(time.monotonic() - self._last_used, 1)
                if self._last_used
                else None
            ),
            "idle_ttl": self.idle_seconds,
        }

    # -- lifecycle ---------------------------------------------------------

    async def _open_async(self) -> Any:
        from playwright.async_api import async_playwright

        state = session_state_path()
        if not state.exists():
            raise SessionClosed(
                "Not logged in. Run 'python login_chess.py' once to save your "
                "session, then try again."
            )
        profile = load_browser_profile()
        headed = _headed()
        playwright = await async_playwright().start()
        browser = None
        try:
            try:
                browser = await playwright.chromium.launch(
                    channel="chrome",
                    headless=not headed,
                    args=LAUNCH_ARGS,
                    ignore_default_args=["--enable-automation"],
                )
            except Exception as exc:  # noqa: BLE001 - fall back to bundled Chromium
                _LOG.debug("real Chrome unavailable (%s); using Chromium", exc)
                browser = await playwright.chromium.launch(
                    headless=not headed,
                    args=LAUNCH_ARGS,
                    ignore_default_args=["--enable-automation"],
                )
            from .review import _configured_viewport

            kwargs: dict[str, Any] = {
                "storage_state": str(state),
                "viewport": _configured_viewport(),
            }
            if profile.get("user_agent"):
                kwargs["user_agent"] = profile["user_agent"]
            context = await browser.new_context(**kwargs)
            page = await context.new_page()
            try:
                await page.add_init_script(
                    "Object.defineProperty(navigator,'webdriver',{get:()=>undefined})"
                )
            except Exception as exc:  # noqa: BLE001 - cosmetic only
                _LOG.debug("could not hide the webdriver flag: %s", exc)
            # Close any popups chess.com opens: they would otherwise pile up
            # unseen tabs against the same session.
            context.on(
                "page", lambda p: p if p is page else asyncio.ensure_future(p.close())
            )
            self._playwright = playwright
            self._browser = browser
            self._context = context
            self._page = page
            # Published so a coroutine submitted to the loop later can reach the
            # live page without being handed it across the thread boundary.
            publish_session(self)
            _LOG.info("session browser opened (headless=%s)", not headed)
            return page
        except Exception:
            # Roll back a partial open so a retry starts clean rather than
            # leaving an orphaned Chromium holding the profile.
            if browser is not None:
                try:
                    await browser.close()
                except Exception as exc:  # noqa: BLE001
                    _LOG.debug("rollback: close browser: %s", exc)
            try:
                await playwright.stop()
            except Exception as exc:  # noqa: BLE001
                _LOG.debug("rollback: stop playwright: %s", exc)
            raise

    async def _close_async(self) -> None:
        for obj, label in ((self._context, "context"), (self._browser, "browser")):
            if obj is not None:
                try:
                    await obj.close()
                except Exception as exc:  # noqa: BLE001
                    _LOG.debug("close %s: %s", label, exc)
        if self._playwright is not None:
            try:
                await self._playwright.stop()
            except Exception as exc:  # noqa: BLE001
                _LOG.debug("stop playwright: %s", exc)
        try:
            await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            pass
        await _drain()
        publish_session(None)
        self._page = self._context = self._browser = self._playwright = None
        self._url = ""
        _LOG.info("session browser closed")

    def open(self, url: str = "https://www.chess.com/home") -> Any:
        """Open the browser, landing on `url` (the home page by default).

        The navigation is part of opening on purpose. Launching a context gives
        a blank page, and a blank page is not a session a tool can do anything
        with -- the first live run reported `open=True` with an empty title and
        an empty URL, then reported "did not open on chess.com" while every
        later step worked. Opening *onto* the site makes `is_open` and
        `current_url` meaningful the moment they are read.
        """
        with self._op_lock:
            if self._runner is None or not self._runner.thread.is_alive():
                self._runner = _LoopThread()
            if self.is_open:
                return self._page
            self._runner.run(self._open_async, 180)
            self._last_used = time.monotonic()
            self._start_ttl()
            # The landing navigation happens HERE, once, on the page that was
            # just created. It must not happen on an already-open session: an
            # earlier version re-ran this on every call, so `navigate(review)`
            # -> `open()` -> `navigate(home)` and the requested URL was silently
            # replaced by the home page. The page stayed on /home while the
            # caller was told it had landed on the review.
            self._url = ""
            self._navigate(url)
            return self._page

    def close(self) -> bool:
        """Close now. Returns True if a browser was actually open."""
        with self._op_lock:
            self._ttl_stop.set()
            self._ttl_thread = None
            if not self.is_open:
                if self._runner:
                    self._runner.close()
                    self._runner = None
                return False
            try:
                self._runner.run(self._close_async, 90)
            except Exception as exc:  # noqa: BLE001
                _LOG.warning("closing the session browser failed: %s", exc)
            if self._runner:
                self._runner.close()
                self._runner = None
            self._summary_cache.clear()
            return True

    # -- the one way in ----------------------------------------------------

    def submit(
        self,
        factory: Callable[[], Any],
        timeout: float = 120.0,
        ensure_open: bool = True,
    ) -> Any:
        """Run a coroutine factory against the open page, under the lock.

        Ensures the session is open first, unless `ensure_open=False` -- which
        is what a read-only probe wants. Asking "where are we" must not launch a
        browser, or `where_am_i` becomes an expensive way to learn nothing. That
        was the first live run's first finding: `where_am_i` reported the home
        page and an open session when nothing had been asked for yet.

        The lock is what the S0.1 R half of the spike measured as mandatory:
        without it a `goto` and an `evaluate` issued together destroy each
        other's execution context.
        """
        with self._op_lock:
            if ensure_open:
                # `open` re-enters this lock (an RLock, so that is legal) but
                # it must NOT re-enter `submit`: navigate -> submit -> open was
                # the recursion that blew the stack. `open` only does the goto.
                self.open()
            if self._runner is None:  # pragma: no cover - defensive
                raise SessionClosed("The session browser could not be opened.")
            try:
                result = self._runner.run(factory, timeout)
            finally:
                self._last_used = time.monotonic()
                self._start_ttl()
            return result

    # -- idle TTL ----------------------------------------------------------

    def _start_ttl(self) -> None:
        if self._ttl_thread and self._ttl_thread.is_alive():
            return
        self._ttl_stop.clear()

        def reap() -> None:
            # Wake often enough to close promptly after the last use, without
            # spinning. Stops on close(), so a live session never races it.
            while not self._ttl_stop.wait(max(5.0, self.idle_seconds / 3)):
                with self._op_lock:
                    if not self.is_open:
                        return
                    idle = time.monotonic() - self._last_used
                    if idle < self.idle_seconds:
                        continue
                    _LOG.info("closing the idle session (%.0fs unused)", idle)
                    self.close()
                    return

        self._ttl_thread = threading.Thread(
            target=reap, name="chess-agent-session-ttl", daemon=True
        )
        self._ttl_thread.start()

    # -- navigation --------------------------------------------------------

    def _navigate_async_factory(self, url: str, wait: str):
        async def go():
            from .review import hide_left_nav

            page = current_page()
            await page.goto(url, wait_until="domcontentloaded", timeout=_NAV_TIMEOUT)
            try:
                await page.wait_for_load_state("networkidle", timeout=20000)
            except Exception as exc:  # noqa: BLE001 - a busy page is not an error
                _LOG.debug("networkidle not reached: %s", exc)
            # Fence the FINAL url too: chess.com can redirect client-side after
            # goto returns, so the pre-check alone is not enough.
            allowed, reason = url_allowed(page.url)
            if not allowed:
                raise SessionClosed(reason)
            if wait in ("load", "networkidle", "ready"):
                try:
                    await page.wait_for_timeout(1500)
                except Exception as exc:  # noqa: BLE001
                    _LOG.debug("settle wait after navigation: %s", exc)
            try:
                await hide_left_nav(page)
            except Exception as exc:  # noqa: BLE001 - hiding nav is best-effort
                _LOG.debug("hide_left_nav: %s", exc)
            self._url = page.url
            return page

        return go

    def _navigate(self, url: str, wait: str = "load") -> str:
        """Goto `url` on the already-open page, after the allow-list check.

        The caller must already hold `_op_lock` and have an open browser. That
        is the whole difference from `navigate`, which takes the lock and
        opens the browser for you.
        """
        allowed, reason = url_allowed(url)
        if not allowed:
            raise SessionClosed(reason)
        self._runner.run(self._navigate_async_factory(url, wait), 180)
        return self._url

    def navigate(self, url: str, wait: str = "load") -> str:
        """Go to `url`, opening the browser first if it is not already open."""
        with self._op_lock:
            self.open()
            return self._navigate(url, wait)

    # -- the review-summary snapshot --------------------------------------

    async def _snapshot_summary_async(self):
        """Capture the review summary BEFORE anything starts the review.

        Accuracy and the classification tally exist on the review page only
        before "Start Review" is pressed -- `tools/move_list.py` documents this
        as a load-bearing ordering, not a defensive habit. In an interactive
        session the user can start the review at any time, so we read and cache
        the summary the moment we land on a review URL.
        """
        from . import move_list

        page = current_page()
        if "/review" not in page.url:
            return None
        try:
            await move_list.expand_review_summary(page)
            summary = await move_list.read_summary_panel(page)
        except Exception as exc:  # noqa: BLE001 - a missing panel is not fatal
            _LOG.debug("summary snapshot failed: %s", exc)
            return None
        if summary:
            self._summary_cache[self._game_id_from_url(page.url)] = summary
        return summary

    @staticmethod
    def _game_id_from_url(url: str) -> str:
        m = re.search(r"/game/(?:live|daily)/(\d+)", url or "")
        return m.group(1) if m else (url or "")

    def snapshot_review_summary(self) -> Any:
        """Read+cache the review summary for the current page, if it is one."""
        return self.submit(self._snapshot_summary_async, 120)

    def cached_summary(self, game_id: str = "") -> dict:
        return self._summary_cache.get(game_id) or {}


def _headed() -> bool:
    import os

    return os.getenv("REVIEW_HEADLESS", "1") == "0"


# The process-wide session the tools share, and the session the private-loop
# coroutines currently belong to.
#
# These are two things, not one. `_SESSION` is the object the tools call; a
# coroutine running ON the private loop reaches the live page through
# `_ACTIVE`, because the caller that built the coroutine factory is a different
# thread from the loop that runs it and cannot hand it over.
#
# `_ACTIVE` is a plain global, not a `contextvars.ContextVar`. A ContextVar set
# inside one task does not survive into the next: `run_coroutine_threadsafe`
# gives every task its own copy of the submitting context, so the value set
# while opening was gone by the time the next coroutine ran -- which is exactly
# the `SessionClosed: page is not available on this loop` the first live run
# produced. A global is correct here precisely because there is only ever one
# browser loop in the process.
_SESSION: BrowserSession | None = None
_SESSION_LOCK = threading.Lock()
_ACTIVE: "BrowserSession | None" = None


def get_session() -> BrowserSession:
    """The process-wide browser session, created on first use."""
    global _SESSION
    if _SESSION is None:
        with _SESSION_LOCK:
            if _SESSION is None:
                _SESSION = BrowserSession()
    return _SESSION


def publish_session(session: "BrowserSession | None") -> None:
    """Record `session` as the one owning the private loop's page.

    Called from inside `_open_async` / `_close_async`, i.e. on the private
    loop, and cleared at close so a coroutine can never pick up a page whose
    transport has just died -- the S0.1-A failure exactly.
    """
    global _ACTIVE
    _ACTIVE = session


def current_page() -> Any:
    """The live page, from inside a coroutine running on the private loop.

    Raises rather than returning None, because every caller here is a browser
    tool that cannot do anything without a page, and a silent None surfaces as
    `AttributeError: 'NoneType'` several frames away from the cause.
    """
    session = _ACTIVE
    if session is None or session._page is None:
        raise SessionClosed("The session browser is not open.")
    return session._page


def close_session() -> bool:
    """Close the shared session if one is open. True if it was."""
    global _SESSION
    with _SESSION_LOCK:
        if _SESSION is None:
            return False
        was = _SESSION.close()
        _SESSION = None
        return was
