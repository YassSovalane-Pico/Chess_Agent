# Chess Agent ♟️ — Setup

A chess.com assistant: you ask a question in plain language, it calls chess.com
and answers. Runs in the terminal, or in Discord.

```
you: "how accurate was I in my last game?"
it:  reads chess.com, replies with the numbers, the coach's notes, and a
     picture of the board at the move that went wrong
```

This file is the setup and nothing else.

---

## 1. What you need

| | |
|---|---|
| **Python** | 3.11 or newer |
| **A Gemini API key** | free — [Google AI Studio](https://aistudio.google.com/apikey) |
| **Google Chrome** | only for the game-review half. `playwright install chromium` works if you don't have it |
| **A chess.com account** | only for the game-review half — any account you can sign into |
| **A Discord application** | only for the Discord bot |

Player lookups and game history need **none** of the last three. Only Game
Review needs a login, because chess.com has no API for it — it exists purely as
a web page you can see while signed in.

---

## 2. Install

```bash
git clone <this-repo-url> && cd chess-agent

python -m venv .venv
# Windows:       .venv\Scripts\activate
# macOS/Linux:   source .venv/bin/activate

pip install -r requirements.txt
playwright install chrome
```

`pip install playwright` installs the Python package only. **Skipping the
`playwright install` step makes every review fail** with
*"Executable doesn't exist at …ms-playwright…"*. Use
`playwright install chromium` instead of `chrome` if you have no Chrome.

---

## 3. Configure

```bash
python setup_env.py
```

This writes `.env` as a full copy of `.env.example`, so every setting the
project has is visible in the file you are about to edit. Put your key in it:

```ini
GEMINI_API_KEY=your_key_here
```

Then check it took:

```bash
python check_env.py
```

It prints which file each setting came from and which are still placeholders,
and **never prints a value** — safe to paste into a chat. `--quiet` gives an
exit code only, for CI.

If you ever add a setting to `.env.example` by hand and `.env` falls behind,
`python setup_env.py --force` regenerates it as a full copy and carries your
real values across.

---

## 4. Run it

```bash
python main.py
```

You are in. Ask something like *"show me the last 5 games of hikaru"* or
*"how accurate was I?"*.

---

## 5. Optional: game review

This half drives your logged-in chess.com account in a real browser, so it
needs one login first:

```bash
python login_chess.py
```

A real Chrome window opens. **Log in by hand**, then press Enter in the
terminal. It also records your browser's User-Agent into
`.browser_profile.json` so the review replays the same fingerprint — chess.com
ties the session cookie to it.

> **Keep `storage_state.json` private.** It holds a live chess.com
> `ACCESS_TOKEN` + `PHPSESSID`; anyone holding the file has your account. It is
> git-ignored, which is thin protection for a folder that also gets zipped and
> backed up. Point it outside the project:
> ```ini
> CHESS_AGENT_STATE=%LOCALAPPDATA%\chess-agent\storage_state.json
> ```

Then check the review actually works, on a real game:

```bash
python calibrate_review.py https://www.chess.com/game/live/184571437220
```

It prints every move it read and how confident it is, and exits non-zero on
every condition it checks — including *"no board could be located"* and *"every
hit was inferred"*, which is exactly the shape of a run that looked like it
passed while measuring the wrong element. It is the difference between "the bot
says you had no blunders" and "the bot actually read the page".

```bash
python calibrate_review.py <link> --expect blunder=2,brilliant=1   # against your own ground truth
```

---

## 6. Optional: the Discord bot

Same assistant, in Discord. `python discord_bot.py`, and it prints an invite
link.

**1. Create the application**

Go to [discord.com/developers/applications](https://discord.com/developers/applications)
→ **New Application** → name it → **Bot** in the sidebar → **Reset Token**.

The token is shown **once**. Copy it now.

**2. Turn on Message Content Intent**

Still on the Bot page, switch on **Message Content Intent** under **Privileged
Gateway Intents**. It is off by default.

> Without it Discord delivers every message with an **empty body** — no text, not
> even a mention of its name. The bot will look online, sit in your channel, and
> answer nothing. This is the single most common reason the setup appears
> broken, and nothing is wrong with the code.

**3. Add the settings**

In the project's `.env`:

```ini
DISCORD_BOT_TOKEN=the_token_you_just_copied
DISCORD_ALLOWED_USER_IDS=your_numeric_user_id
```

Your ID: **User Settings → Advanced → enable Developer Mode**, then right-click
yourself → **Copy User ID**.

`DISCORD_ALLOWED_USER_IDS` is the only thing standing between a stranger and
your logged-in chess.com account driving a real browser on your machine. Leave
it blank and the bot refuses **everyone**, including in DMs — that is the
default on purpose, and it says so on startup. Put `all` there to open it up on
purpose. It is read once at startup, so changing it means a restart.

**4. Run it**

```bash
python check_env.py     # confirm both values registered
python discord_bot.py
```

Open the invite link it prints, pick your servers, Authorize. Then mention it or
DM it — no prefix needed. `!reset`, `!status` and `!help` work in any channel.

---

## 7. Settings worth knowing

Everything has a working default. Override in `.env`; `.env.example` documents
all of them, with the comment explaining each.

| Setting | Default | |
|---|---|---|
| `GEMINI_API_KEY` | — | **required** |
| `GEMINI_MODEL` | `gemini-3.5-flash-lite` | `gemini-3.8-flash` is stronger and slower |
| `CHESS_AGENT_STATE` | `./storage_state.json` | where the chess.com login lives |
| `REVIEW_HEADLESS` | `1` | `0` shows the browser, to solve a Cloudflare check by hand |
| `REVIEW_VIEWPORT_W`, `REVIEW_VIEWPORT_H` | `1400`, `820` | the main lever on screenshot size |
| `SESSION_IDLE_SECONDS` | `90` | seconds a browser page may sit idle |
| `AGENT_MAX_TOOL_ROUNDS` | `12` | tool round-trips per turn |
| `AGENT_MAX_TURNS` | `12` | turns before a conversation starts fresh |
| `AGENT_MEMORY_TURNS` | `8` | exchanges remembered. `0` turns it off |
| `AGENT_MEMORY_ITEM_CHARS` | `400` | how much of one message is kept |
| `AGENT_MEMORY_CHARS` | `2000` | ceiling on the whole remembered block |
| `DISCORD_ALLOWED_USER_IDS` | *nobody* | who may run a tool. `all` for everyone |
| `DISCORD_TURN_TIMEOUT` | `600` | seconds one answer may take |
| `DISCORD_USER_COOLDOWN` | `5` | seconds between one person's questions |
| `DISCORD_BROWSER_GRACE` | `120` | before the browser is handed back after a timeout |
| `DISCORD_MAX_SESSIONS` | `200` | conversations kept alive at once |
| `DISCORD_SESSION_IDLE` | `3600` | minutes before an unused conversation is dropped |
| `DISCORD_SHOW_TOOLS` | off | `1` shows tool names in the progress line |
| `DISCORD_MAX_UPLOAD_BYTES` | `7 MB` | byte budget per message of screenshots |
| `CHESS_AGENT_LOG` | `warning` | `debug` shows why something degraded |

---

## 8. When the setup doesn't work

| Symptom | |
|---|---|
| Discord bot answers nothing | Message Content Intent is off — section 6, step 2 |
| *"Missing GEMINI_API_KEY"* | `python check_env.py` names the file it read |
| *"Executable doesn't exist at …ms-playwright…"* | `playwright install chrome` |
| *"Not logged in. Run python login_chess.py"* | the review needs the one-time login — section 5 |
| *"404 … no longer available"* on the model | set `GEMINI_MODEL=gemini-3.8-flash` in `.env` |
| A Cloudflare check appears | `set REVIEW_HEADLESS=0` (Windows) or `export REVIEW_HEADLESS=0`, solve it once by hand |
| A review finds nothing in a game that had blunders | `python calibrate_review.py <link>` — it says which step failed |
| The bot says "no picture came through" | almost always the expired chess.com sign-in — `python login_chess.py` again, once. `!status` shows how old the saved session is |
| A selector stopped matching chess.com | `python dump_pages.py <link>` saves the real page to `dumps/` |
| Want to see why something degraded | `set CHESS_AGENT_LOG=debug` |
| A setting "isn't taking effect" | `python check_env.py` |

---

## 9. Confirm the install

```bash
pip install -e ".[dev]"

python tests/test_review_logic.py     # and the rest in tests/
```

171 tests, about a second, no browser and no network. Each file has its own
runner, so `python tests/test_interact.py` works on its own.

```bash
python tests/test_review_logic.py     # 60  review logic, tool contract
python tests/test_discord_bot.py      # 74  chunking, access, concurrency, memory
python tests/test_session.py          #  7  the navigation allow-list
python tests/test_interact.py         # 13  interactive arguments and framing
python tests/test_js_syntax.py        #  4  every embedded JS blob parses
python tests/test_js_detection.py     #  2  the extractor, executed in Node
python tests/test_project_env.py      # 11  no real secret in .env.example
```

Two of them need something extra: `test_js_syntax.py` needs the `esprima`
package (it comes with `pip install -e ".[dev]"`), and `test_js_detection.py`
needs Node on your PATH — it skips itself, with a note, when Node is missing.

---

