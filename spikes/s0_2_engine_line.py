"""S0.2 -- find chess.com's engine line / eval in the DOM.

Gate for Phase 3 (`read_engine_line`). Per this project's own rule -- "every
selector here was read out of dumps/review_started.html" -- we do not guess the
markup. We open the analysis board, let the engine compute, and hunt for the
evaluation and the principal variation, then report exactly what exists and
where.

Two boards are probed, because the game board (a finished review) and the
practice board (`/analysis/`) may expose the engine differently:

  * /analysis/game/live/<id>   the finished game on the analysis board
  * /analysis/                 the blank practice board

For each board we dump:
  * candidate eval elements (by a wide selector sweep + class/text keywords)
  * candidate engine/line/PV panels
  * the sidebar's own HTML, and the raw text, so nothing is lost
  * the best-move arrow / highlight, if any

    python spikes/s0_2_engine_line.py [game_url] [--out spikes/s0_2] [--headed]

Writes a readable report per board to <out>/<name>.json plus <name>.txt, and
prints a SUMMARY of what was and was not found. Exit 0 if the engine line is
found (Phase 3 is unblocked), 1 if not (Phase 3 needs a different approach).
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.common import (  # noqa: E402
    configure_logging,
    load_browser_profile,
    session_state_path,
)
from tools.review import _build_review_url  # noqa: E402

NAV_TIMEOUT = 45000

# Wide nets for the evaluation and the principal variation. chess.com renames
# classes often, so these are candidates to be CONFIRMED by the report, not
# trusted.
EVAL_SELECTORS = [
    "#evaluation",
    "[class*='evaluation' i]",
    "[class*='eval-bar' i]",
    "[class*='evalBar' i]",
    "[data-testid*='eval' i]",
    "[aria-label*='evaluation' i]",
]
ENGINE_SELECTORS = [
    "[class*='engine' i]",
    "[class*='best-move' i]",
    "[class*='bestMove' i]",
    "[class*='principal' i]",
    "[class*='variation' i]",
    "[class*='analysis-line' i]",
    "[class*='pv' i]",
    "[class*='computer' i]",
    "[class*='stockfish' i]",
    "[data-testid*='engine' i]",
    "[aria-label*='engine' i]",
    "[aria-label*='best' i]",
    "[class*='depth' i]",
    "[class*='knps' i]",
]

# A keyword sweep over the whole document, mirroring move_list._INSPECT_JS.
_SWEEP_JS = r"""
() => {
  const norm = (s) => String(s || '').toLowerCase();
  const kw = /eval|engine|stockfish|principal|variation|depth|knps|best move|nodes\/sec|book|mistake|blunder|brilliant|inaccuracy|pv\b/;
  const seen = new Set();
  const hits = [];
  const all = Array.prototype.slice.call(document.querySelectorAll('*'));
  for (const el of all) {
    let c = '';
    try { const raw = el.className; c = (raw && raw.baseVal !== undefined) ? String(raw.baseVal) : (typeof raw === 'string' ? raw : ''); } catch (e) {}
    const bits = [c,
      el.getAttribute && el.getAttribute('id'),
      el.getAttribute && el.getAttribute('data-testid'),
      el.getAttribute && el.getAttribute('aria-label'),
      el.getAttribute && el.getAttribute('title'),
    ].filter(Boolean).join(' | ');
    if (!kw.test(norm(bits))) continue;
    const key = bits.slice(0, 160);
    if (seen.has(key)) continue;
    seen.add(key);
    hits.push({
      tag: el.tagName.toLowerCase(),
      class: c.slice(0, 120),
      id: (el.getAttribute && el.getAttribute('id') || '').slice(0, 60),
      text: (el.textContent || '').trim().slice(0, 80),
    });
    if (hits.length >= 120) break;
  }
  return hits;
}
"""

_PROBE_JS = r"""
(selectors) => {
  const out = {};
  for (const sel of selectors) {
    let els = [];
    try { els = Array.prototype.slice.call(document.querySelectorAll(sel)); } catch (e) { continue; }
    out[sel] = {
      count: els.length,
      items: els.slice(0, 6).map((el) => ({
        class: (el.className && el.className.baseVal !== undefined ? String(el.className.baseVal) : String(el.className || '')).slice(0, 120),
        text: (el.textContent || '').trim().slice(0, 120),
        aria: el.getAttribute && el.getAttribute('aria-label'),
        style: (el.getAttribute && el.getAttribute('style') || '').slice(0, 80),
      })),
    };
  }
  return out;
}
"""

_SIDEBAR_JS = r"""
() => {
  const sb = document.querySelector('#board-layout-sidebar');
  return sb ? sb.outerHTML : '(no #board-layout-sidebar)';
}
"""


async def _open(p, state_path, profile, headed):
    kwargs = {
        "storage_state": str(state_path),
        "viewport": {"width": 1400, "height": 820},
    }
    if profile.get("user_agent"):
        kwargs["user_agent"] = profile["user_agent"]
    try:
        browser = await p.chromium.launch(
            channel="chrome",
            headless=not headed,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
                "--no-sandbox",
            ],
            ignore_default_args=["--enable-automation"],
        )
    except Exception:  # noqa: BLE001
        browser = await p.chromium.launch(
            headless=not headed,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
                "--no-sandbox",
            ],
            ignore_default_args=["--enable-automation"],
        )
    context = await browser.new_context(**kwargs)
    page = await context.new_page()
    try:
        await page.add_init_script(
            "Object.defineProperty(navigator,'webdriver',{get:()=>undefined})"
        )
    except Exception:  # noqa: BLE001
        pass
    return browser, context, page


async def probe_board(page, url: str, name: str, out_dir: Path, settle_ms: int) -> dict:
    print(f"\n--- {name}: {url}")
    await page.goto(url, wait_until="domcontentloaded", timeout=NAV_TIMEOUT)
    try:
        await page.wait_for_load_state("networkidle", timeout=25000)
    except Exception:  # noqa: BLE001
        pass
    # Let the engine actually compute.
    try:
        await page.wait_for_timeout(settle_ms)
    except Exception:  # noqa: BLE001
        pass
    title = await page.evaluate("() => document.title")
    print(f"    title: {title[:70]!r}")

    sweep = await page.evaluate(_SWEEP_JS)
    eval_probe = await page.evaluate(_PROBE_JS, EVAL_SELECTORS)
    engine_probe = await page.evaluate(_PROBE_JS, ENGINE_SELECTORS)
    sidebar = await page.evaluate(_SIDEBAR_JS)
    body_text = await page.evaluate(
        "() => (document.body.innerText||'').slice(0, 4000)"
    )

    report = {
        "url": page.url,
        "title": title,
        "sweep": sweep,
        "eval_selectors": eval_probe,
        "engine_selectors": engine_probe,
        "sidebar_html": sidebar,
        "body_text": body_text,
    }
    (out_dir / f"{name}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (out_dir / f"{name}.txt").write_text(body_text, encoding="utf-8")

    # Quick verdict for the console.
    eval_found = {k: v["count"] for k, v in eval_probe.items() if v["count"] > 0}
    engine_found = {k: v["count"] for k, v in engine_probe.items() if v["count"] > 0}
    print(f"    sweep hits: {len(sweep)}")
    print(f"    eval selectors with matches: {eval_found or 'none'}")
    print(f"    engine selectors with matches: {engine_found or 'none'}")
    print(f"    wrote {name}.json / {name}.txt")
    return report


def score(report: dict) -> tuple[bool, str]:
    """Did we find an evaluation and/or a line/PV? A coarse but honest signal."""
    eval_hits = {k for k, v in report["eval_selectors"].items() if v["count"] > 0}
    engine_hits = {k for k, v in report["engine_selectors"].items() if v["count"] > 0}
    text = (report.get("body_text") or "").lower()
    # A numeric eval like "+0.35" or "-1.07", or engine words, in the text.
    has_eval_text = any(ch.isdigit() for ch in text) and (
        "eval" in text or "+" in text or "-" in text
    )
    line_words = any(
        w in text
        for w in (
            "variation",
            "principal",
            "line",
            "stockfish",
            "depth",
            "knps",
            "nodes",
        )
    )
    found_eval = bool(eval_hits) or has_eval_text
    found_line = bool(engine_hits) or line_words
    verdict = (
        f"eval_selectors={sorted(eval_hits)} engine_selectors={sorted(engine_hits)} "
        f"eval_text={has_eval_text} line_words={line_words}"
    )
    return (found_eval or found_line), verdict


async def run(game_url: str, out_dir: Path, headed: bool, settle_ms: int) -> int:
    from playwright.async_api import async_playwright

    state_path = session_state_path()
    if not state_path.exists():
        print(f"No session at {state_path}. Run: python login_chess.py")
        return 2
    profile = load_browser_profile()
    if not profile.get("user_agent"):
        print(
            "NOTE: no recorded user agent; re-run login_chess.py for a matched fingerprint.\n"
        )

    _, game_id, _ = _build_review_url(game_url)
    kind = "live" if "/live/" in game_url else "daily"
    analysis_game = f"https://www.chess.com/analysis/game/{kind}/{game_id}"
    analysis_blank = "https://www.chess.com/analysis/"

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"chess-agent S0.2 engine-line spike   -> {out_dir.resolve()}")

    results: dict[str, dict] = {}
    async with async_playwright() as p:
        browser = context = page = None
        try:
            browser, context, page = await _open(p, state_path, profile, headed)
            results["analysis_game"] = await probe_board(
                page, analysis_game, "analysis_game", out_dir, settle_ms
            )
            # Let the page settle before the second board.
            results["analysis_blank"] = await probe_board(
                page, analysis_blank, "analysis_blank", out_dir, settle_ms
            )
        finally:
            for obj, name in ((context, "context"), (browser, "browser")):
                if obj is not None:
                    try:
                        await obj.close()
                    except Exception:  # noqa: BLE001
                        pass

    print("\n" + "=" * 68)
    print("  SUMMARY")
    print("=" * 68)
    any_found = False
    for name, report in results.items():
        ok, verdict = score(report)
        any_found = any_found or ok
        print(f"\n  [{name}] {'FOUND' if ok else 'not found'}")
        print(f"    {verdict}")
    print()
    if any_found:
        print("  The engine line (or at least an evaluation) EXISTS in the DOM.")
        print("  Phase 3 is unblocked -- build read_engine_line against these files:")
        for name in results:
            print(f"    {out_dir / (name + '.json')}")
        return 0
    print("  No engine line or evaluation found in the DOM on either board.")
    print("  Phase 3 needs a different source (chess.com may only compute the line")
    print("  for live moves, or only after a move is played). Inspect the dumped")
    print("  sidebar_html/body_text by hand before deciding.")
    return 1


def main() -> int:
    configure_logging("info")
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "game_url", nargs="?", default="https://www.chess.com/game/live/184571437220"
    )
    ap.add_argument("--out", default=str(ROOT / "spikes" / "s0_2"))
    ap.add_argument("--headed", action="store_true")
    ap.add_argument(
        "--settle", type=int, default=9000, help="ms to let the engine compute"
    )
    args = ap.parse_args()
    if args.headed:
        os.environ["REVIEW_HEADLESS"] = "0"
    return asyncio.run(run(args.game_url, Path(args.out), args.headed, args.settle))


if __name__ == "__main__":
    sys.exit(main())
