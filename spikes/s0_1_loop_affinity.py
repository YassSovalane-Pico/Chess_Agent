"""S0.1 -- is a Playwright page bound to the event loop it was created on?

Three parts, deliberately:

  A. NEGATIVE CONTROL. Create the page on loop A, let loop A die (which is what
     `asyncio.run` does), then try to drive it from loop B. This is the current
     `tools/review.py::_run` pattern stretched past its own lifetime. If this
     half SUCCEEDS the hazard is not real and the dedicated-loop design is
     optional. If it FAILS the hazard is real.

  B. THE PROPOSED DESIGN. One long-lived loop on one dedicated thread, with
     `async_playwright()` started inside it and never exited. Several callers,
     on several threads, drive the same page via `run_coroutine_threadsafe`,
     with operations SERIALISED -- which is what the session will do.

  R. THE RACE. Same loop, but a `goto` and an `evaluate` issued at the same
     moment on the same page. Expected to fail: `page.goto` destroys the
     execution context the `evaluate` was about to use. This is not a defect
     in the design; it is the measurement that makes a lock MANDATORY rather
     than merely tidy.

Answers wanted: does A fail, does B hold, and does R fail?

    python spikes/s0_1_loop_affinity.py

Nothing here is imported by the project. It prints VERDICT: lines and exits
0 only if A fails, B holds and R fails -- the combination the Phase 1 design
is built on.

First run of B failed, and it was worth knowing: the naive version issued a
`goto` and an `evaluate` concurrently, and both probes died with
"Execution context was destroyed". That is why B serialises and R exists.
"""

import asyncio
import logging
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.common import (  # noqa: E402
    configure_logging,
    load_browser_profile,
    session_state_path,
)

LOG = logging.getLogger("s0_1")

TARGET = "https://www.chess.com/"
PROBE_JS = "() => document.title"
NAV_TIMEOUT = 30000
STEP_TIMEOUT = 45

LAUNCH_ARGS = [
    "--disable-blink-features=AutomationControlled",
    "--disable-dev-shm-usage",
    "--no-sandbox",
]


def _box(label: str) -> None:
    print(f"\n{'=' * 68}\n  {label}\n{'=' * 68}")


def _verdict(ok: bool, label: str) -> bool:
    print(f"\nVERDICT {label}: {'HOLDS' if ok else 'FAILED'}")
    return ok


# ---------------------------------------------------------------------------
# A. Negative control: a page outlives the loop it was made on
# ---------------------------------------------------------------------------


async def _open_then_die(box: dict) -> None:
    """Build a live page, then return -- which destroys the loop on exit."""
    from playwright.async_api import async_playwright

    playwright = await async_playwright().start()
    box["playwright"] = playwright

    profile = load_browser_profile()
    kwargs = {
        "storage_state": str(session_state_path()),
        "viewport": {"width": 1400, "height": 820},
    }
    if profile.get("user_agent"):
        kwargs["user_agent"] = profile["user_agent"]

    headed = os.getenv("REVIEW_HEADLESS", "1") == "0"
    try:
        browser = await playwright.chromium.launch(
            channel="chrome",
            headless=not headed,
            args=LAUNCH_ARGS,
            ignore_default_args=["--enable-automation"],
        )
    except Exception as exc:  # noqa: BLE001
        LOG.debug("no real Chrome (%s); using bundled Chromium", exc)
        browser = await playwright.chromium.launch(
            headless=not headed,
            args=LAUNCH_ARGS,
            ignore_default_args=["--enable-automation"],
        )
    context = await browser.new_context(**kwargs)
    page = await context.new_page()
    await page.goto(TARGET, wait_until="domcontentloaded", timeout=NAV_TIMEOUT)
    box.update(browser=browser, context=context, page=page)
    print(f"  loop A: opened {page.url}")
    print(f"  loop A: title = {await page.evaluate(PROBE_JS)!r}")


async def _drive_from_a_new_loop(box: dict) -> None:
    """Loop B. The page was built on loop A, which no longer exists."""
    page = box["page"]
    title = await asyncio.wait_for(page.evaluate(PROBE_JS), STEP_TIMEOUT)
    print(f"  loop B: title = {title!r}")
    await asyncio.wait_for(
        page.goto(
            TARGET + "analysis/", wait_until="domcontentloaded", timeout=NAV_TIMEOUT
        ),
        STEP_TIMEOUT,
    )
    print(f"  loop B: navigated to {page.url}")


async def _teardown(box: dict) -> None:
    for key, call in (
        ("context", lambda o: o.context.close()),
        ("browser", lambda o: o.browser.close()),
        ("playwright", lambda o: o.playwright.stop()),
    ):
        obj = box.get(key)
        if obj is None:
            continue
        try:
            await call(box)
        except Exception as exc:  # noqa: BLE001
            LOG.debug("teardown %s: %s", key, exc)


def negative_control() -> bool:
    _box("S0.1-A  NEGATIVE CONTROL -- page survives the loop that made it")
    box: dict = {}
    try:
        asyncio.run(asyncio.wait_for(_open_then_die(box), 120))
    except Exception as exc:  # noqa: BLE001
        print(f"  loop A itself failed: {type(exc).__name__}: {exc}")
        return _verdict(True, "A (hazard confirmed before the loop even died)")

    try:
        asyncio.run(asyncio.wait_for(_drive_from_a_new_loop(box), STEP_TIMEOUT + 30))
        worked = True
        reason = ""
    except Exception as exc:  # noqa: BLE001
        worked = False
        reason = f"{type(exc).__name__}: {exc}"

    if worked:
        print("  UNEXPECTED: the page worked from a second loop.")
        ok = False
    else:
        print(f"  as expected, it did not: {reason[:400]}")
        ok = True
    try:
        asyncio.run(_teardown(box))
    except Exception:  # noqa: BLE001
        pass
    return _verdict(ok, "A -- hazard is real, as assumed")


# ---------------------------------------------------------------------------
# B. The proposed design: one loop, one thread, many callers
# ---------------------------------------------------------------------------


class LoopThread:
    """A private event loop on a private thread, alive for the session."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.ready = threading.Event()
        self.thread = threading.Thread(
            target=self._run, name="chess-agent-browser-loop", daemon=True
        )
        self.thread.start()
        self.ready.wait(10)

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.ready.set()
        self.loop.run_forever()

    def call(self, factory, timeout: float = STEP_TIMEOUT):
        """Run `factory()` on the private loop. Raises whatever it raises."""
        return asyncio.run_coroutine_threadsafe(factory(), self.loop).result(timeout)

    def close(self) -> None:
        try:
            self.loop.call_soon_threadsafe(self.loop.stop)
        except RuntimeError:
            return
        self.thread.join(timeout=10)


async def _open_session(box: dict) -> str:
    """Start playwright, chrome, context and page -- all on the private loop."""
    from playwright.async_api import async_playwright

    playwright = await async_playwright().start()
    box["playwright"] = playwright

    profile = load_browser_profile()
    kwargs = {
        "storage_state": str(session_state_path()),
        "viewport": {"width": 1400, "height": 820},
    }
    if profile.get("user_agent"):
        kwargs["user_agent"] = profile["user_agent"]

    headed = os.getenv("REVIEW_HEADLESS", "1") == "0"
    try:
        browser = await playwright.chromium.launch(
            channel="chrome",
            headless=not headed,
            args=LAUNCH_ARGS,
            ignore_default_args=["--enable-automation"],
        )
    except Exception as exc:  # noqa: BLE001
        LOG.debug("no real Chrome (%s); using bundled Chromium", exc)
        browser = await playwright.chromium.launch(
            headless=not headed,
            args=LAUNCH_ARGS,
            ignore_default_args=["--enable-automation"],
        )
    context = await browser.new_context(**kwargs)
    page = await context.new_page()
    await page.goto(TARGET, wait_until="domcontentloaded", timeout=NAV_TIMEOUT)
    box.update(browser=browser, context=context, page=page)
    return page.url


async def _probe(box: dict, note: str) -> str:
    """A separate coroutine, submitted later, driving the same page."""
    page = box["page"]
    title = await page.evaluate(PROBE_JS)
    print(f"  {note:<28} title={title[:34]!r}")
    return title


async def _navigate(box: dict, url: str, note: str) -> str:
    page = box["page"]
    await page.goto(url, wait_until="domcontentloaded", timeout=NAV_TIMEOUT)
    print(f"  {note:<28} url={page.url[:44]}")
    return page.url


async def _close_session(box: dict) -> None:
    for key in ("context", "browser"):
        obj = box.get(key)
        if obj is None:
            continue
        try:
            await obj.close()
        except Exception as exc:  # noqa: BLE001
            LOG.debug("close %s: %s", key, exc)
    pw = box.get("playwright")
    if pw is not None:
        try:
            await pw.stop()
        except Exception as exc:  # noqa: BLE001
            LOG.debug("stop playwright: %s", exc)
    # Let the driver's own tasks finish, then cancel anything still pending.
    # Without this the loop is torn down under a live transport and the
    # interpreter prints a ResourceWarning plus an EPIPE from the node side on
    # exit -- noise that says nothing about the spike and makes a real failure
    # hard to spot.
    try:
        await asyncio.sleep(0.5)
    except asyncio.CancelledError:
        pass
    await _drain()


async def _drain() -> None:
    """Cancel and gather every task except this one, so nothing is 'pending'."""
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


def proposed_design() -> tuple[bool, dict, LoopThread | None]:
    """B: serialised operations from many threads, on one private loop.

    Returns (holds, box, runner) so R can reuse the SAME loop and the same
    page -- otherwise R would confound two variables (locking and loop
    affinity) and prove nothing.
    """
    _box("S0.1-B  PROPOSED DESIGN -- one loop, one thread, SERIALISED callers")
    state = session_state_path()
    if not state.exists():
        print(f"  no session at {state}; run python login_chess.py first")
        return _verdict(False, "B"), {}, None
    box: dict = {}
    runner = LoopThread()
    print(f"  private loop on thread {runner.thread.name!r}, id={id(runner.loop)}")

    url = runner.call(lambda: _open_session(box), 180)
    print(f"  opened {url}")

    # Six separate coroutines, each submitted from a different thread, each one
    # its own `run_coroutine_threadsafe` round trip -- which is what a Discord
    # bot serving several users actually does. The lock is what makes this B
    # rather than R.
    plan = [
        ("probe #1", lambda: _probe(box, "probe #1 (thread 1)")),
        (
            "nav  ->analysis",
            lambda: _navigate(box, TARGET + "analysis/", "nav -> analysis (thread 2)"),
        ),
        ("probe #2", lambda: _probe(box, "probe #2 (thread 3)")),
        ("nav  ->", lambda: _navigate(box, TARGET, "nav -> home (thread 4)")),
        ("probe #3", lambda: _probe(box, "probe #3 (thread 5)")),
        ("probe #4", lambda: _probe(box, "probe #4 (thread 6)")),
    ]
    results: dict[str, object] = {}
    lock = threading.Lock()

    def worker(name, factory):
        with lock:  # the serialisation the session will do
            try:
                results[name] = runner.call(factory, STEP_TIMEOUT)
            except Exception as exc:  # noqa: BLE001
                results[name] = exc

    for name, factory in plan:
        t = threading.Thread(target=worker, args=(name, factory))
        t.start()
        t.join(timeout=STEP_TIMEOUT + 40)

    bad = {k: v for k, v in results.items() if isinstance(v, BaseException)}
    if bad:
        for name, exc in bad.items():
            print(f"  FAILED {name}: {type(exc).__name__}: {exc}")
        return _verdict(False, "B -- the private loop does NOT hold up"), box, runner

    print(
        f"\n  all {len(results)} operations succeeded, each from its own "
        f"thread, on loop id={id(runner.loop)}"
    )
    return _verdict(True, "B -- the private-loop design works"), box, runner


def the_race(box: dict, runner: LoopThread | None) -> bool:
    """R: a goto and an evaluate issued together on one page, unlocked.

    Expected to FAIL. This is the measurement that makes the session's lock a
    requirement rather than a nicety: without it, two turns asking the bot to
    "step forward" at the same moment destroy each other's execution context
    and report a browser error nobody can explain.
    """
    _box("S0.1-R  THE RACE -- goto and evaluate at the same instant, unlocked")
    if not box.get("page") or runner is None:
        print("  no open session from B; skipping")
        return _verdict(False, "R -- not measured")

    # The SAME loop and the SAME page as B. The only variable changed is the
    # lock.
    print(f"  reusing loop id={id(runner.loop)} -- only the lock differs from B")

    results: dict[str, object] = {}

    def go_nav():
        try:
            results["nav"] = runner.call(
                lambda: _navigate(box, TARGET + "analysis/", "nav (raced)"),
                STEP_TIMEOUT + 30,
            )
        except Exception as exc:  # noqa: BLE001
            results["nav"] = exc

    def go_probe():
        try:
            results["probe"] = runner.call(
                lambda: _probe(box, "probe (raced)"), STEP_TIMEOUT
            )
        except Exception as exc:  # noqa: BLE001
            results["probe"] = exc

    a, b = threading.Thread(target=go_nav), threading.Thread(target=go_probe)
    a.start()
    b.start()
    a.join(timeout=STEP_TIMEOUT + 40)
    b.join(timeout=STEP_TIMEOUT + 40)

    for key, val in results.items():
        if isinstance(val, BaseException):
            print(f"  {key:<6} raised {type(val).__name__}: {str(val)[:130]}")
        else:
            print(f"  {key:<6} ok: {str(val)[:80]}")

    try:
        runner.call(_close_session, 90)
    except Exception as exc:  # noqa: BLE001
        LOG.debug("close session: %s", exc)
    runner.close()

    # The race is proven if any operation lost. Usually the evaluate does,
    # because goto replaces its execution context mid-flight.
    raced = any(isinstance(v, BaseException) for v in results.values())
    if raced:
        print("\n  a concurrent operation was destroyed, so a SERIALISING LOCK")
        print("  IS MANDATORY in the session. Design confirmed, with that")
        print("  requirement attached.")
        return _verdict(True, "R -- the race is real, so lock the session")
    print("\n  the race did not bite this time. Do not read that as safety: it")
    print("  is a timing accident. Still lock it.")
    return _verdict(False, "R -- inconclusive, still lock the session")


def main() -> int:
    configure_logging("info")
    print(f"chess-agent S0.1 loop-affinity spike   t={time.strftime('%H:%M:%S')}")
    if not session_state_path().exists():
        print(f"\nNo session at {session_state_path()}.")
        print("Run: python login_chess.py")
        return 2

    a_ok = negative_control()
    b_ok, box, runner = proposed_design()
    r_ok = the_race(box, runner) if b_ok else False

    _box("SUMMARY")
    print(
        f"  A  page survives its own loop ....... {'no  (hazard real)' if a_ok else 'yes (UNEXPECTED)'}"
    )
    print(f"  B  private loop + serialised calls .... {'yes' if b_ok else 'NO'}")
    print(
        f"  R  unlocked concurrent calls ......... {'race is real' if r_ok else 'no bite this run'}"
    )

    if b_ok and r_ok:
        print("\n  CONCLUSION: the design holds, with one hard requirement attached.")
        print("  The session must own one long-lived event loop on a dedicated")
        print("  thread and submit every browser call to it with")
        print("  asyncio.run_coroutine_threadsafe, AND it must serialise those")
        print("  calls under a lock. Confirmed by measurement, not assumption.")
        return 0
    if b_ok:
        print("\n  CONCLUSION: the design holds. R did not bite this run, but a")
        print("  timing accident is not safety -- lock the session regardless.")
        return 0
    print("\n  CONCLUSION: the design does not hold. Stop and rethink before")
    print("  building anything on it.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
