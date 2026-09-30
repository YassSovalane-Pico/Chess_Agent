"""Pure-logic tests for tools/session.py. No browser, no network.

The point is the navigation allow-list and the URL normaliser: they decide
whether the agent can reach settings/logout/friends, so they get tested like
any other rule rather than trusted.

    python tests/test_session.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.session import _normalise_url, url_allowed  # noqa: E402

OK = "https://www.chess.com/analysis/game/live/123456789"


def _allowed(url: str) -> bool:
    return url_allowed(url)[0]


def test_review_and_analysis_pages_are_allowed():
    for url in (
        OK,
        "https://www.chess.com/analysis/",
        "https://www.chess.com/analysis/game/daily/999/review",
        "https://www.chess.com/game/live/184571437220",
        "https://www.chess.com/game/daily/184571437220",
        "https://www.chess.com/lessons/abc",
        "https://www.chess.com/puzzles",
        "https://www.chess.com/openings/Sicilian-Defense",
        "https://www.chess.com/",
    ):
        assert _allowed(url), url


def test_account_and_social_pages_are_refused():
    for url in (
        "https://www.chess.com/settings",
        "https://www.chess.com/logout",
        "https://www.chess.com/friends",
        "https://www.chess.com/messages",
        "https://www.chess.com/settings/account",
    ):
        assert not _allowed(url), url


def test_other_hosts_are_refused():
    for url in (
        "https://evil.example.com/analysis/",
        "https://google.com",
        "http://chess.com.evil.example/analysis/",
    ):
        assert not _allowed(url), url


def test_a_prefix_without_a_boundary_is_refused():
    # /analysisfoo must not sneak through on a startswith('/analysis').
    assert not _allowed("https://www.chess.com/analysisfoo")
    assert not _allowed("https://www.chess.com/gamelive/1")


def test_bare_host_and_relative_urls_normalise():
    # No scheme is filled in, query/fragment are dropped, and a trailing slash
    # is removed. The bare host is NOT rewritten to www -- `url_allowed`
    # accepts chess.com and www.chess.com equally, so rewriting here would be
    # one more thing that can disagree with the host check.
    assert _normalise_url("chess.com/analysis/") == "https://chess.com/analysis"
    assert _normalise_url("https://www.chess.com/analysis/?x=1#y") == (
        "https://www.chess.com/analysis"
    )
    assert _allowed("chess.com/analysis/")
    assert _allowed("https://chess.com/analysis/")
    assert _allowed("www.chess.com/analysis/")


def test_refusal_messages_name_the_reason():
    _, settings = url_allowed("https://www.chess.com/settings")
    assert "settings" in settings
    _, host = url_allowed("https://example.com/analysis/")
    assert "chess.com" in host


def test_empty_url_is_refused_not_crashed():
    assert not _allowed("")
    assert not _allowed(None)


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for test in tests:
        try:
            test()
            print(f"ok    {test.__name__}")
            passed += 1
        except AssertionError as exc:
            print(f"FAIL  {test.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR {test.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
