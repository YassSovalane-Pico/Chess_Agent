# Chess Agent ♟️

A beginner-friendly command-line AI agent powered by Google Gemini.
You ask in natural language (e.g. *"show me the last 5 games of hikaru"*),
the agent calls tools to fetch real data from the public chess.com API,
then replies with a readable summary.

## How it works (the core idea)

```
user message → model decides to call a tool → tool runs (HTTP to chess.com)
→ result goes back to the model → final answer
```

- `tools/` — plain Python functions grouped by domain.
  Gemini reads their docstrings to know when to call them.
  - `games.py` — `get_recent_games` (+ opening parsing)
  - `profiles.py` — `get_player_profile`
  - `move_list.py` — the move table (`data-node` plies), the classification
    vocabulary, the summary/coach panels, navigation
  - `review.py` — the seven review tools: `review_game_screenshots`,
    `screenshot_game_move`, `get_review_state`, `get_moves`,
    `get_coach_commentary` (all via one logged-in browser session)
  - `common.py` — shared constants/helpers + logging + browser fingerprint
- `agent.py` — the agent itself: the system instruction, the tool list, the
  loop, screenshot collection, and one conversation's worth of Gemini chat.
- `main.py` — terminal front end over `agent.py`.
- `discord_bot.py` — Discord front end over the same `agent.py`. Mention or DM.
- `calibrate_review.py` — proves blunder/brilliant detection and navigation
  work on a real game, and dumps the raw DOM chess.com emitted.
- `make_fixture.py` — distils a dump into the 18 KB fixture the tests read.

## Setup

1. Create a virtual environment and activate it:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

2. Install requirements **and the browser**:

```bash
pip install -r requirements.txt
playwright install chrome          # or: playwright install chromium
```

> `pip install playwright` installs the Python package only. Skipping this
> step means the first review run dies with
> *"Executable doesn't exist at …ms-playwright…"*. The review tools prefer a
> real Google Chrome (`channel="chrome"`) and fall back to bundled Chromium.

3. Get a Gemini API key from Google AI Studio:
   https://aistudio.google.com/apikey

4. Create your `.env` — in the **project folder**, as a full copy of the
   template:

```bash
python setup_env.py
```

That writes `.env` line-for-line from `.env.example`, carrying over anything
already set in another env file, so every available setting is visible rather
than hidden behind defaults. Then set `GEMINI_API_KEY` in it.

Check it any time:

```bash
python check_env.py          # what is set, what is a placeholder, and where from
python check_env.py --quiet  # exit code only, for a hook or CI
```

> **`.env` holds secrets and `.env.example` does not.** The template is the file
> that gets committed and that every fresh clone copies, so a real value in it is
> a real value in git history. `tests/test_project_env.py` fails if one appears,
> including a token-shaped string pasted into a comment. If a secret does land
> there, deleting the line is not enough — reset it.

5. Run the agent:

```bash
python main.py
```

> **Model:** the default is `gemini-3.5-flash-lite`. If you came from an older
> guide, `gemini-2.5-flash` no longer works on new API keys — Google restricts
> the 2.5 models to keys that have used them before, and new keys get
> `404 … no longer available`. Override in `.env`, e.g.
> `GEMINI_MODEL=gemini-3.8-flash` for the current flagship.

## Discord bot

The same agent, in Discord. `agent.py` holds the loop; `main.py` and
`discord_bot.py` are two front ends over it, so they cannot drift apart.

```bash
python discord_bot.py
```

### 1. Create the application

[discord.com/developers/applications](https://discord.com/developers/applications)
→ **New Application** → name it → open **Bot** on the left → **Reset Token** and
copy it (it is shown once).

> ### Turn on the Message Content Intent — or the bot will never answer
>
> Still on the **Bot** page, under **Privileged Gateway Intents**, switch on
> **Message Content Intent**. It is off by default.
>
> With it off, Discord delivers the bot every message with an **empty body** —
> no text, and no mentions either. The bot looks online, sits in your channel,
> and responds to nothing. It is by far the most common way this setup fails,
> and the cause is one toggle in a web page, not anything in the code.

### 2. Configure `.env`

Edit `chess-agent/.env` — the project one, not a copy somewhere else:

```ini
DISCORD_BOT_TOKEN=the_token_you_just_copied
DISCORD_ALLOWED_USER_IDS=your_numeric_user_id
```

Get your user ID: **User Settings → Advanced → enable Developer Mode**, then
right-click yourself → **Copy User ID**.

Run `python check_env.py` afterwards. It prints what is set, what is still a
placeholder, and which file each value came from — without printing any value.

### 3. Run it, then open the invite link it prints

```
  Discord bot ready: ChessAgent#1234
  Client ID:        1234567890123456789

Add it to your servers with this link:

  https://discord.com/oauth2/authorize?client_id=1234567890123456789&scope=bot&permissions=117760
```

Open that in a browser, pick your servers, Authorize. The bot is in them.

The link is built from the token's own client ID and a minimal permission set —
`117760` is view channel, send messages, embed links, attach files, read message
history. Nothing administrative.

### 4. Talk to it

Mention it, or DM it. No prefix needed.

```
@ChessAgent how was my last game?
@ChessAgent review https://www.chess.com/game/live/184571437220
@ChessAgent what did the coach say about move 12 as black?
@ChessAgent show me the last 5 games of hikaru
!reset      # start a fresh conversation
!status     # what the model is, and whether the browser is busy
```

### Who is allowed, and why that setting matters

`DISCORD_ALLOWED_USER_IDS` is the only gate between a stranger and your logged-in
chess.com account driving a real browser on your PC. Left blank, the bot refuses
everyone — that is deliberate, and it is why the bot starts with a note saying
so. Set `all` to open it up on purpose.

You can also gate by hand in Discord (a role, a private channel) and leave the
env var listing your own ID.

### How the three hard parts work

| Problem | What was done |
|---|---|
| A review takes 30s–2min and would freeze every channel the bot is in | the agent runs in a worker thread via `asyncio.to_thread`; the bot replies immediately, shows a typing indicator, and **edits** that message with each tool as it starts |
| Two reviews at once would fight over one Chromium profile and one `storage_state.json` | a lock held around each **browser tool call** (`agent.BROWSER_TOOLS`), and the second person is told their place in the queue instead of both silently crawling. A lock around the whole *turn* was the first attempt and it was wrong twice: it made a two-second `get_recent_games` wait out a two-minute review, and it told that user they were queued behind a browser they never touched |
| A tool that never returns would hold that lock for ever and silently block everyone behind it | each turn has a deadline (`DISCORD_TURN_TIMEOUT`), the user is told, and a watchdog takes the lock back after a grace period so the queue drains |
| The model cannot see images, so it writes prose about screenshots — useless in Discord | screenshots are collected at the **tool boundary** into `AgentResult.artifacts` and attached as real files; the model is additionally told never to print a filesystem path |
| Every reply is text a language model wrote, and `discord.py` defaults to parsing `@everyone`, roles and users | `AllowedMentions.none()` is set on the bot and passed at every send and edit, so nothing this bot posts can ping anybody |
| Ten screenshots in one message is ~6MB against Discord's 8MB ceiling | uploads are budgeted by **bytes as well as file count**, so an oversized frame is skipped with a message rather than failing its own batch and every batch after it |

That last-but-one row is worth dwelling on. Asking the model to list the paths
it produced would be the obvious implementation and it is wrong twice over: the
model can misremember a path, and a path is useless to someone reading a message
on a phone. Collecting the paths where the tool actually returned them cannot
misreport them.

### In a channel, the bot

- **edits one message** rather than posting a "working on it" above every answer,
  coalescing rapid updates so a burst of tool calls cannot outrun Discord's
  ~5 edits per 5s or leave the line showing a stale step
- **splits long answers** on paragraph and line boundaries, never mid-table and
  never mid-emoji, so a nine-game summary arrives whole
- **attaches frames** with a caption naming the move: `brilliant — move 14
  (ply 27)`, `review summary — accuracy, full tally, coach commentary`
- **keeps one conversation per person per channel**, so two people in one room
  do not inherit each other's *"what about the opponent in game 2?"*; idle and
  surplus conversations are dropped (`DISCORD_SESSION_IDLE`,
  `DISCORD_MAX_SESSIONS`) so the map cannot grow for ever
- **takes one question at a time per person** (`DISCORD_USER_COOLDOWN`,
  default 5s), because each one is a real browser against *your* chess.com
  session
- **starts a fresh conversation** after `AGENT_MAX_TURNS` (default 12) and says
  so, because Gemini chat history cannot be truncated and a silent reset looks
  like the bot forgetting mid-sentence

### Discord settings worth knowing

| Setting | Default | What it does |
|---|---|---|
| `DISCORD_ALLOWED_USER_IDS` | *(unset → nobody)* | the allow-list. `all` opens it to everyone on purpose |
| `DISCORD_TURN_TIMEOUT` | `600` | seconds one turn may take, queue included. `0` disables |
| `DISCORD_BROWSER_GRACE` | `120` | seconds after an abandoned turn before the browser lock is taken back |
| `DISCORD_USER_COOLDOWN` | `5` | seconds between one person's turns. `0` disables |
| `DISCORD_SESSION_IDLE` | `3600` | minutes before an unused conversation is dropped |
| `DISCORD_MAX_SESSIONS` | `200` | most live conversations kept |
| `DISCORD_MAX_UPLOAD_BYTES` | `7340032` | byte budget per message of screenshots |
| `DISCORD_MAX_ATTACHMENT_BYTES` | `6291456` | a single frame larger than this is skipped, not posted |

`DISCORD_ALLOWED_USER_IDS` is read once at startup, so changing it means
restarting the bot. `python check_env.py` shows every setting and where it came
from.

## Game Review (needs a login) — ⚠️ unverified

The review tool drives your paid account, so it needs a one-time login:

```bash
python login_chess.py
```

That also records your real browser User-Agent into `.browser_profile.json`
(git-ignored) so the review run replays the *same* fingerprint. A session
minted under one Chrome build and replayed under a hardcoded "Chrome/126"
header is a large part of why Cloudflare challenges a valid login.

> `storage_state.json` holds a live chess.com `ACCESS_TOKEN` + `PHPSESSID`:
> **anyone who has that file has your account.** It is git-ignored, but that is
> thin protection for a folder that also gets zipped and backed up. Set
> `CHESS_AGENT_STATE` in `.env` to a path outside the project (for example
> `%LOCALAPPDATA%\chess-agent\storage_state.json`).

Then verify detection before trusting it:

```bash
python calibrate_review.py https://www.chess.com/game/live/123456789
python calibrate_review.py <link> --expect blunder=2,brilliant=1
```

It prints every parsed move with its classification, the confidence and the
evidence behind it, checks counts against your ground truth, and proves
forward/backward stepping actually changed the board. `confidence: low` means
the marker was inferred from a glyph rather than read from a chess.com label.

### Honest status of the review half

**The API half works. The review half now works end to end**, verified against
a real game (white 95.0 / black 76.6, 29 plies, `board idle on its own: True`).
Getting there meant fixing seven distinct bugs, each of which made the tool
report "no blunders" for a game that had them.

**How the page actually behaves:**

- `/analysis/game/live/<id>/review` is the right URL. The sidebar header reads
  `Game Review` and offers a `Go to Analysis` button.
- Accuracy and the full classification tally read correctly from the class
  names: `.review-rating-white/-black` and `tally-row-value analysis-<kind>`.
- **Three stages**, and conflating them is what broke everything:

| # | Stage | How long | Signalling |
|---|-------|----------|-----------|
| 1 | Layout shell — board + sidebar | sub-second | `wc-chess-board`, `#board-layout-sidebar` |
| 2 | Review data — accuracy + tally | a few seconds | `.review-rating-*`, `[class*='tally' i]` |
| 3 | **Move list — the moves themselves** | only after a click | `main-line-ply`, `move-list-row` |

Stage 3 only exists after `tab-review-start-review-button` is pressed.
Pressing it also **replaces** the summary, so the order is fixed: read the
summary, *then* start, *then* read moves.

- A move is classified in the DOM as:

```html
<div class="node white-move main-line-ply">
  <span class="node-annotation-icon"><svg><g id="blunder">…</g></svg></span>
  <span style="color: var(--color-classification-book)">g4</span>
</div>
```

  Neither label is a class, an `alt`, a `title` or an `aria-label` — and the
  icon is a *sibling* of the span the reader keys on. Both are read now, at
  `confidence: high`.

- The move list is **virtualised**: a real run had 29 `main-line-ply` elements
  but only 16 `move-list-row`s in the DOM. The reader scrolls the list, merges
  by ply, and reports coverage as `read/known`.

- The move **number** lives on the row (`data-whole-move-number`), never in
  the cell. Reading it from the cell gave `0` for every move and made the
  dedup key collide on any repeated SAN — one real game lost 3 of 29 plies
  (`d6`, `e4` and `b6` each appear twice).

- The board must be **identified, not guessed**. `[class*='board' i]` matches
  the 24px `board-layout-icon` gear, and because that icon animates on its own
  a run once "verified" ArrowLeft on a board it never found. Board boxes must
  now be square and ≥100px.

`calibrate_review.py` prints the page state before anything else, writes its
DOM dump immediately after the load wait *and* again after Start Review, and
exits 1 when the shell does not mount, when the data never arrives, when the
move list never appears, when every hit was merely inferred, or when no board
can be located. It prints `v0.3.1` on every run, so it is always clear which
build produced an output.

To re-verify:

```bash
python calibrate_review.py https://www.chess.com/game/live/184571437220
```

Expect `ply coverage: 29/29` and classifications matching the pre-start tally.
If coverage is short, raise the sweep (`read_moves_full(page, steps=...)`) or
the timeouts (`REVIEW_TIMEOUT_MS`, `REVIEW_MOVELIST_TIMEOUT_MS`).

Troubleshooting:

```bash
# see why something degraded (the pipeline swallows failures on purpose)
set CHESS_AGENT_LOG=debug        # Windows PowerShell
# export CHESS_AGENT_LOG=debug    # macOS/Linux

# solve a Cloudflare challenge by hand, once
set REVIEW_HEADLESS=0
```

## Development

```bash
pip install -e ".[dev]"           # installs ruff + esprima
ruff check . && ruff format --check .
python tests/test_review_logic.py   # 55 — pure logic, no browser
python tests/test_js_syntax.py      #  4 — parses every embedded JS blob
python tests/test_js_detection.py   #  2 — runs the extractor in Node (skips if absent)
python tests/test_discord_bot.py    # 26 — the bot: chunking, access, artifacts
python tests/test_project_env.py    # 11 — no real secret in .env.example
```

`test_review_logic.py` includes an agent-contract test: every tool in
`agent.FUNCTION_MAP` must be a *synchronous* function with a parameterised
docstring, and must return something `json.dumps` accepts. That test exists
because `review_game_screenshots` was once declared `async def` — the loop
received a coroutine, `from_function_response` raised *"Object of type
coroutine is not JSON serializable"*, and the user saw only "Agent error" with
no hint that the review had never started. Every other test imported the
function without ever calling it from a sync context, so nothing failed.

`esprima` is required by the JS syntax test: the DOM readers ship as Python
strings, so a JavaScript typo would otherwise show up only as a swallowed
in-browser failure that looks like "chess.com has no blunders".

`test_js_detection.py` goes further and *executes* the classifier against the
exact move-cell markup captured in `screenshots/calibration.json`. It exists
because string-level assertions could not have caught the bug that mattered
most: chess.com labels a move with an SVG `<g id="blunder">` and a
`var(--color-classification-book)` custom property, neither of which is a
class, an alt, a title or an aria-label. Parse checks passed while the feature
silently found 2 of 11 moves. It needs Node on `PATH` and skips itself if Node
is not installed.

## Example prompts

- `show me the last 5 games of hikaru`
- `what are gothamchess's recent results?`
- `profile of hikaru`
- `what about the opponent in game 2?` (follow-up, history is kept)
- `muéstrame las últimas partidas de hikaru` (answers in your language)

## Project structure

```
chess-agent/
├── .env                # YOUR settings + secrets. Git-ignored. Never commit.
├── .env.example        # the template. Git-ignored is WRONG here; placeholders only
├── agent.py            # the agent: instruction, tools, loop, artifact collection
├── main.py             # terminal front end
├── discord_bot.py      # Discord front end
├── project_env.py      # where .env is read from, and what is set (never prints values)
├── setup_env.py        # writes .env as a full copy of .env.example
├── check_env.py        # the doctor: what is set, what is a placeholder, from where
├── tools/          # tool functions (package)
│   ├── __init__.py # re-exports all tools
│   ├── games.py    # get_recent_games + openings
│   ├── profiles.py # get_player_profile
│   ├── move_list.py # the move table, classification vocabulary, navigation
│   ├── review.py   # the five review/screenshot tools (login needed)
│   └── common.py   # constants, logging, browser fingerprint
├── login_chess.py  # one-time chess.com login -> storage_state.json
├── calibrate_review.py  # verify detection + navigation on a real game
├── dump_pages.py  # full HTML of every page state -> dumps/
├── make_fixture.py # distil a dump -> the 18 KB fixture the tests read
├── tests/          # logic tests, no browser required
│   ├── test_review_logic.py  # pure logic + the agent tool contract
│   ├── test_js_syntax.py     # every embedded JS blob parses
│   ├── test_js_detection.py  # the move-table extractor, executed in Node
│   ├── test_discord_bot.py   # the Discord front end and the agent core
│   └── test_project_env.py   # no real secret ever in .env.example
├── pyproject.toml   # deps + ruff config
├── requirements.txt
├── .gitignore
└── README.md
```

## Tools

- `get_recent_games(username, limit=5)` — walks monthly archives
  newest → oldest until `limit` games are collected (with opening/ECO).
- `get_player_profile(username)` — name, title, country, join date,
  followers, rapid/blitz/bullet ratings.
- `review_game_screenshots(game_url)` — reviews one game and saves focused
  screenshots. Every frame is clipped to the board and the review panel, and
  chess.com's left navigation is hidden, so no frame contains the site menu,
  the search box or your profile:

  | Shot | Shows |
  |------|-------|
  | `summary` | the review panel: accuracy, the **full** classification tally, the game rating, the coach's commentary |
  | `brilliant_m<N>_ply<P>` | the board at **that** brilliant move — one frame for **each** brilliant in the game |
  | `blunder_m<N>_ply<P>` | the board at **that** blunder move — one frame for **each** blunder |

  The tally starts collapsed showing six of nine rows, so the
  `button[aria-label="Expand"]` chevron is clicked *before* the panel is read
  and photographed — otherwise the picture is missing Brilliant and Inaccuracy
  and can disagree with the numbers beside it. A kind with no moves gets no
  frames, and the result only contains paths that were actually written. Move
  frames are capped at `MAX_MOVE_SHOTS` (10) so a pathological game cannot
  write hundreds of PNGs.

  ### Frame size

  The frame is the union of the board and the review panel, and the **board is
  height-constrained** — chess.com sizes it to the available height, not the
  width. So the browser window is the main lever on how wide a screenshot
  ends up. It used to run at 1920×1080, where the board rendered ~980px and a
  board+panel frame came out at **1480px**. It now runs at **1400×820**, so the
  board renders ~700px and the frame lands near **1150px** — the same
  composition, proportionate.

  Two more knobs if that still isn't right:

  ```bash
  # just the board, no review panel  (or: sidebar for just the panel)
  set REVIEW_SHOT_REGION=board
  # flush against the edges
  set REVIEW_SHOT_PADDING=0
  # go narrower still (floor 1200x700, below which chess.com reflows)
  set REVIEW_VIEWPORT_W=1300
  set REVIEW_VIEWPORT_H=780
  ```

  The board is identified as a **square within 1%**, which is what excludes the
  column of player nameplate and clock around it. An earlier tolerance of
  0.85–1.18 let an 800×1080 column through as "the board", which is why the
  frame carried the nameplates. The left navigation is excluded by
  construction — it is part of neither box.

  Returns `white_accuracy`, `black_accuracy`, `move_counts`, `notable_moves`,
  `moves_read` coverage, `highlight`, `highlights` and a `screenshots` object.

- `screenshot_game_move(game_url, move_number, colour, ply)` — photographs
  **any** single move, not only the errors, for when the user names one:
  *"screenshot move 14"*, *"show me move 22 as black"*, *"what did it look
  like on my 8th move?"*.

  chess.com numbers moves two ways and the user may use either, so both are
  accepted:

  | User says | Pass |
  |-----------|------|
  | "move 14" | `move_number=14` — almost always needs `colour` too, because both sides play a move 14 |
  | "the 27th move" | `ply=27` — unambiguous on its own |
  | "22 as black" | `move_number=22, colour="black"` |

  If you omit the colour and the number is ambiguous, the tool **replies with
  an error naming both sides** rather than guessing — guessing would
  photograph the wrong position. The system prompt tells the model to read
  that error and call again with the colour, not to retry with a different
  number.

- `get_review_state(game_url)` — the summary panel: the accuracy pair, the
  classification tally for the whole game and per side, and the two players
  with their ratings. Use it for *"how accurate was I"*, *"how many mistakes
  did I make"*, *"what was my rating"*.

  The tally is a **two-column table** — one number per player per kind — and
  both columns are read and summed. Reading only one is how a review came to
  claim 2 mistakes where the move list had found 3, and 0 brilliant moves where
  the move list had found 1, with no error anywhere. The per-side split follows
  the column order, which is the same White-then-Black order the accuracy row
  uses; the tool says so in its `notes`.

- `get_moves(game_url)` — the full move table, one entry per ply: ply, move
  number, side, SAN, chess.com's verdict, the clock used, and the piece
  figurine. Plies come from `data-node="<variation>-<index>"`, which chess.com
  writes on every move cell, so the numbering is authoritative — 29 cells
  covering plies 1..29 with no gaps and no dedup.

  It also reads the summary tally **before** starting the review (starting it
  destroys the summary) and returns both, plus a reconciliation.

  > ### A `classification` of `null` does not mean the move was sound
  >
  > chess.com annotates only *some* moves. In the captured game, 7 of 29 cells
  > carried a label — brilliant, blunder, two mistakes, book and two best moves.
  > The summary counted **7 inaccuracies, 1 miss and 13 best moves** that no
  > cell mentions; those words exist in the legend and the tally and nowhere
  > else. The analysis board was checked too and has no per-move classification
  > markup at all, so there is no second source.
  >
  > So: use `summary_counts` for *how many*, `classification` for *where*, and
  > never tell a player their other moves were fine when the summary says they
  > had seven inaccuracies. `get_moves` returns `annotated_moves`,
  > `unannotated_moves`, `counts_without_a_move` and a `note` spelling this out
  > for the game in hand, and the system prompt tells the model to read it.

- `get_coach_commentary(game_url, move_number, colour, ply)` — chess.com's own
  words about one move: the verdict (*"Blunder"*), the engine evaluation that
  move produced with its sign, and the coach's coaching prose. Uses the same
  move-argument rules as `screenshot_game_move`, and refuses the same way when
  a move number is ambiguous.

  The coach panel is rendered for the **current move only** — exactly one
  `.move-feedback-box-component` exists in the whole document — so the tool
  navigates to the move and *then* reads. Reading first returns the previous
  move's words.

  Two details the naive read gets wrong, both fixed here:

  | Trap | Symptom | Fix |
  |------|----------|-----|
  | the prose is revealed word by word, with unrevealed words in `.move-feedback-speech-text-hidden` | a live read returned `"You get"` | read `textContent`, not `innerText`, so the whole sentence comes back |
  | the sign is in `score-text-negative` / `move-feedback-box-negative` — two class families | `+2.90` with `engine_score_sign: null` | check both families, then the leading `+`/`-` as a last resort |

  Book moves get the opening's name (`opening`, e.g. *"Grob Opening"*) instead
  of a verdict, plus `last_book_move_number`.

```json
{
  "move": {
    "san": "O-O", "move_number": 14, "color": "white", "ply": 27,
    "classification": "brilliant", "confidence": "high",
    "evidence": "g[id=brilliant]", "screenshot": "D:\\...\\move_m14_ply27_....png"
  },
  "moves_read": "29/29"
}
```

  Because the move list is virtualised, the tool scrolls the target into the
  rendered window and re-reads (which re-stamps the per-ply tags navigation
  depends on) before clicking it.

### Example review result

```json
{
  "white_accuracy": 69.3,
  "black_accuracy": 79.9,
  "move_counts": {"brilliant": 1, "blunder": 1, "inaccuracy": 3, "mistake": 5},
  "notable_moves": [
    {"kind": "blunder",   "san": "d5",  "move_number": 3,  "color": "black", "confidence": "high"},
    {"kind": "brilliant", "san": "O-O", "move_number": 14, "color": "black", "confidence": "high"}
  ],
  "moves_read": "29/29",
  "highlights": {
    "blilliant_m14_ply27": {"kind": "brilliant", "san": "O-O", "move_number": 14,
                            "color": "black", "confidence": "high",
                            "evidence": "g[id=brilliant]"},
    "blunder_m3_ply6":     {"kind": "blunder", "san": "d5", "move_number": 3,
                            "color": "black", "confidence": "high",
                            "evidence": "g[id=blunder]"}
  },
  "screenshots": {
    "summary":              "D:\\...\\summary_184577982780_20260930_101500.png",
    "brilliant_m14_ply27":  "D:\\...\\brilliant_m14_ply27_184577982780_...png",
    "blunder_m3_ply6":      "D:\\...\\blunder_m3_ply6_184577982780_...png"
  }
}
```

`confidence: high` means chess.com explicitly labelled the move — the reader
found an SVG marker id or a `--color-classification-*` custom property.
`medium`/`low` means it was inferred from a colour, and the system prompt
tells the model to hedge accordingly rather than state it as fact.

## Dumping the page HTML

```bash
python dump_pages.py https://www.chess.com/game/live/184571437220
python dump_pages.py <link> --out dumps --headed
```

Writes the **full document** for every state into `dumps/`, plus a
`manifest.json` describing each file:

| State | What it is |
|-------|-----------|
| `game` | the `/game/...` page |
| `analysis` | the `/analysis/game/...` board |
| `review_loaded` | the review page with the summary panel loaded |
| `review_expanded` | the same, after clicking the tally's expand chevron |
| `review_started` | after **Start Review** — the moves only exist here |
| `moves_scrolled_000…100` | the move list at 5 scroll offsets — a check on whether it is virtualised. It is **not**: all 29 cells were present at every offset, so the sweep is a no-op and is kept only as insurance |

The review states also get a companion `*.sidebar.html` with the review
sidebar's own markup, which is the part the reader actually works on.

**Run this whenever a selector stops working**, and send the manifest plus the
state that failed. Most of the bugs in this project were invisible because only
a *fragment* of the page ever reached the code: a `class` lookup failed, the
reader fell through to a weaker signal, and the tool reported "no blunders" for
a game that had one — while the real markup sat right there. Everything below
was found by finally looking at the whole document, not by guessing:

- the move list lives in `wc-move-list` / `chessboard-pkg-move-list-component`
- **`data-node="<variation>-<index>"` is an authoritative ply index** —
  `ply = variation * 2 + index + 1`. 29 cells, plies 1..29, no gaps. This
  replaced a density heuristic that was silently dropping 3 of 29 plies.
- a move is classified by `<g id="blunder">` and a
  `var(--color-classification-book)` custom property — not by any class.
  Casing is inconsistent: `id="Brilliant"` sits beside `id="book"`.
- the move number is on the row, as `data-whole-move-number`
- the clock is a sibling: `data-move-list-el="timestamp" data-ply="N"
  data-time="<centiseconds>"`
- the summary tally is a **two-column table** — 20 values, 10 kinds × 2 sides
- the sidebar title is an `<h1>Game Review</h1>`, not a button
- `tab-review-start-review-button` is what actually starts the review
- the tally is collapsed behind a `button[aria-label="Expand"]` chevron
- the coach's prose is revealed word by word, with the unrevealed words in
  `.move-feedback-speech-text-hidden` — so `textContent`, never `innerText`

### You can delete `dumps/`

It is 2.7 MB of gitignored HTML and **nothing reads it at runtime**. It is a
capture, not a dependency.

To keep the markup-pinned tests alive without it, `make_fixture.py` distils a
dump into `tests/fixtures/chess_com_review_markup.json` — **18 KB, a 151×
reduction** — keeping every fact the tests assert on and collapsing each `<svg>`
to `<svg/>`. The tests read the fixture, not the dumps, so with `dumps/` deleted
all 61 still pass with **no skips**. Regenerate after a fresh capture:

```bash
python dump_pages.py https://www.chess.com/game/live/184571437220
python make_fixture.py
```

### Do not delete `tests/`

They are not on the runtime path, and they run in about a second:

```bash
python tests/test_review_logic.py     # 55 tests, no browser
python tests/test_js_syntax.py        #  4 — every JS blob parses
python tests/test_js_detection.py     #  2 — the extractor in real Node
python tests/test_discord_bot.py      # 26 — the Discord front end
python tests/test_project_env.py      # 11 — .env hygiene
```

They are the only thing standing between you and a silent regression. Nearly
every bug in this project's history was a *plausible* wrong value rather than a
crash, and each one is now a named test.


