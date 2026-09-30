"""One-time login to chess.com to save your session.

Run once:
    python login_chess.py

It opens a real Chrome window on the login page. Log in manually with your paid
account (username + password + 2FA), then go back to the terminal and press
Enter. Your session (cookies) is saved to storage_state.json.

The real browser's User-Agent is recorded alongside it so the review tools
replay the exact same fingerprint. A session minted under one Chrome build and
replayed under a different User-Agent header is a large part of why Cloudflare
challenges a perfectly valid login.

After that, review_game_screenshots() reuses the session without needing your
password again.

IMPORTANT: storage_state.json is like a password. It is git-ignored -- never
share it or commit it.
"""

from tools.common import BROWSER_PROFILE_PATH, save_browser_profile, session_state_path

# Cookies that only exist on a real chess.com login. Saving the state without
# one of these mints a session the review tool later reports as "Not logged in"
# with no clue why.
_AUTH_COOKIES = ("PHPSESSID", "CHESSCOM_REMEMBERME", "me", "ACCESS_TOKEN")

# Same anti-detection posture as the review run, so the session is not created
# under a different fingerprint than the one it is replayed in.
LAUNCH_ARGS = [
    "--disable-blink-features=AutomationControlled",
    "--disable-dev-shm-usage",
    "--no-sandbox",
]
HIDE_WEBDRIVER = "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"


def main() -> None:
    try:
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Playwright is not installed. Run: pip install -r requirements.txt")
        return

    state_path = session_state_path()
    state_path.parent.mkdir(parents=True, exist_ok=True)

    print("Opening chess.com login in Chrome...")
    print("IMPORTANT: type your chess.com email/username + password")
    print("directly in the chess.com form. Do NOT click 'Login with Google'.")
    print("Google blocks automated browsers (the error in your screenshot).")
    print("If your account was created via Google and has no password,")
    print("first set one at chess.com > Settings > Password, or use")
    print("'Forgot password' to create one.\n")

    with sync_playwright() as p:
        # Real installed Chrome where available; Google is less likely to flag it
        # than Playwright's bundled Chromium.
        browser = None
        try:
            browser = p.chromium.launch(
                channel="chrome", headless=False, args=LAUNCH_ARGS
            )
        except PlaywrightError as exc:
            print(f"(Real Chrome not found, falling back to bundled Chromium: {exc})")
            browser = p.chromium.launch(headless=False, args=LAUNCH_ARGS)

        try:
            context = browser.new_context()
            context.add_init_script(HIDE_WEBDRIVER)
            page = context.new_page()

            # Record the fingerprint this session is created under, so the
            # review tool can send the identical User-Agent later.
            user_agent = page.evaluate("() => navigator.userAgent") or None
            if user_agent:
                save_browser_profile(
                    user_agent=user_agent,
                    channel="chrome",
                    launch_args=LAUNCH_ARGS,
                )
                print(f"Recorded browser fingerprint -> {BROWSER_PROFILE_PATH.name}")

            page.goto("https://www.chess.com/login", wait_until="domcontentloaded")
            try:
                input("Press Enter AFTER you are logged in... ")
            except (KeyboardInterrupt, EOFError):
                print("\nCancelled.")
                return
            cookies = context.cookies()
            found = [
                n for n in _AUTH_COOKIES if any(c.get("name") == n for c in cookies)
            ]
            if not found:
                print("\nNo chess.com login cookies found -- so you are not logged in.")
                print("Session NOT saved. Log in for real, then run this again.")
                return
            context.storage_state(path=str(state_path))
        finally:
            browser.close()

    print(f"Session saved to {state_path}.")
    print("Next: python calibrate_review.py <game link>   # verify detection")
    print("      python main.py                          # start the agent")


if __name__ == "__main__":
    main()
