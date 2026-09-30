# Chess Agent — Design Notes

Everything here was learned by running the review pipeline against real
chess.com pages and watching it fail. It is the reasoning behind the code;
the code itself carries only one-line "why" notes.

Read this before changing a selector, a wait condition, or a tool contract.

---

## 1. What the project actually does

Two halves, with very different risk profiles:

| Half | Mechanism | Risk |
|---|---|---|
| **API** | `requests` → chess.com public PubAPI | none (read-only, public) |
| **Review** | Playwright drives your **logged-in** account in a real browser | credentials, Cloudflare, DOM churn |

The review half is ~90% of the code, because chess.com changes its markup
constantly and every selector is a potential silent failure.

### Layering

```
agent.py                 the agent loop, tool registry, prompt
  │
  ├── discord_bot.py     front end: access control, concurrency, message shaping
  ├── main.py            front end: terminal
  │
  └── tools/             the tool implementations
        ├── games.py / profiles.py    plain HTTP
        ├── move_list.py              DOM reading + navigation  (the core)
        ├── review.py                 browser lifecycle + screenshots + 5 tools
        ├── session.py                the long-lived browser + its allow-list
        └── interact.py               6 interactive tools over the session
```

`move_list.py` reads and navigates. `review.py` presents and transports.
Nothing in `move_list.py` knows about screenshots.

---

## 2. chess.com DOM facts

These are not guesses. Each one was read out of a `dump_pages.py` capture
(`review_started.html`, `review_loaded.html`, `analysis.html`). When a
selector breaks, re-capture and re-read rather than reasoning about it.

### The page loads in three stages

| # | Stage | How long | Positive signal |
|---|---|---|---|
| 1 | Layout shell — board + sidebar | < 1s | `wc-chess-board`, `#board-layout-sidebar` |
| 2 | Review data — accuracy + tally | seconds (server-side) | `.review-rating-white/-black`, `tally-row-value` |
| 3 | **Move list** | only after a click | `[data-node]`, `move-list-row` |

Stage 3 exists only after `tab-review-start-review-button` is pressed.

### Read order is fixed

```
expand collapsed tally  →  read summary  →  press Start Review  →  read moves
```

Pressing Start Review **replaces** the summary panel. Reading the summary
afterwards returns `None`/`{}` — indistinguishable from "the page had no
summary". So the order is a hard contract, not politeness:

- `get_review_state` never starts the review (that would destroy its own input)
- `get_moves` and `screenshot_game_move` read the summary *then* start
- `session._snapshot_summary_async` snapshots the moment it lands on a review URL

### The move cell

```html
<div data-node="0-13" class="node white-move main-line-ply">
  <span class="node-annotation-icon"><svg><g id="blunder">…</g></svg></span>
  <span class="node-highlight-content" style="color: var(--color-classification-book)">g4</span>
</div>
```

Facts and where they live:

| Fact | Where | Note |
|---|---|---|
| **ply** | `data-node="<variation>-<index>"` on the cell | authoritative: `ply = variation*2 + index + 1` |
| **move number** | `data-whole-move-number` on the **ROW**, not the cell | reading it from the cell gives `0` for every move |
| **clock** | sibling `[data-move-list-el="timestamp"][data-ply="N"][data-time="cs"]` | centiseconds, not seconds |
| **classification** | `<g id="...">` in the SVG, or `var(--color-classification-*)` | neither is a class / alt / title / aria-label |
| **side** | `white-move` / `black-move` in the cell's class | |

The annotation icon is a **sibling** of the text span, not an ancestor.

### Classification is not a class

The legend words appear as:
- an SVG marker id: `<g id="blunder">`, `<g id="Brilliant">` (casing inconsistent)
- a CSS custom property: `style="color: var(--color-classification-book)"`

`class`, `alt`, `title`, `aria-label` are all empty. An earlier reader checked
all four, found nothing, and fell back to a learned colour palette — it
reported **2 of the game's 11** notable moves, with no error anywhere.

Both real signals are `high` confidence. A class-name-only match is `medium`.
**There is no colour-inference fallback** — an unlabelled move is reported
unlabelled.

### The summary tally is a two-column table

20 `tally-row-value` elements = 10 kinds × 2 players. Collapsing them into one
key per kind and keeping the last value reports **one player's tally as the
whole game's**.

Column 0 is White and column 1 is Black, because that is the order of the
accuracy row directly above in the same `game-overview-row`, where the classes
are literally `review-rating-white` then `review-rating-black`.

### Other markup facts

- sidebar title is `<h1>Game Review</h1>` — a heading, not a button
- the tally starts collapsed behind `button[aria-label="Expand"]`
- exactly **one** `.move-feedback-box-component` exists per document — the coach
  panel is rendered for the **current move only**
- the coach's prose is revealed word by word; unrevealed words live in
  `.move-feedback-speech-text-hidden` → **use `textContent`, never `innerText`**
- the score's sign is in `score-text-negative` **or** `move-feedback-box-negative`
  (two families for the same thing)
- the username class is `cc-user-username-component` — **not** `cc-user-username`
- the move list is **not** virtualised (29 cells at all 5 scroll offsets). The
  sweep in `read_moves_full` is insurance, and the `read/known` coverage string
  exists so a future change to that would be a number in a log, not a silent
  short read
- `/analysis/` has **no** per-move classification markup, so there is no second
  source for the summary-only kinds (inaccuracy, miss)

---

## 3. Design decisions and why

### The session needs its own event loop

The one-shot tools call `asyncio.run`, so each gets a fresh loop. A Playwright
`Page` is bound to the loop that created it, so a page that outlives one call
raises `Page.evaluate: 'NoneType' object has no attribute 'send'`.

`session.py` therefore owns **one private loop on one dedicated thread**, and
every operation goes through `run_coroutine_threadsafe`.

The loop lives on a thread (not just a loop) because the tools are synchronous
functions called from the agent's worker thread; `run_coroutine_threadsafe` is
the only safe way across that boundary.

`_ACTIVE` is a plain global, **not** a `contextvars.ContextVar`:
`run_coroutine_threadsafe` gives each task a copy of the submitting context, so
a value set while opening would be gone by the time the next coroutine ran.

### Every session operation is serialised

Two concurrent operations on one page destroy each other: a `page.goto` tears
down the execution context a concurrent `page.evaluate` was about to use.

`BrowserSession.submit` holds an `RLock` (re-entrant because the idle-reap
thread legitimately calls `close()` while a caller holds it) across the whole
round trip. `open()` may re-enter `submit`'s lock but must never re-enter
`submit` itself — `navigate → submit → open` was a stack overflow.

### Navigation is fenced to the board

`storage_state.json` holds a live `ACCESS_TOKEN` + `PHPSESSID`. "Go wherever
you like" has to mean "go wherever you like *on the board*".

Allowed prefixes: `/`, `/home`, `/analysis`, `/game/live/`,
`/game/daily/`, `/lessons/`, `/puzzles/`, `/openings/`.

- each prefix matches at a **path boundary**, so `/analysis` passes but
  `/analysisfoo` does not
- the host is compared for **equality**, so `chess.com.evil.example` is refused
- the check runs **before** `goto` and again on `page.url` afterwards, because
  chess.com is an SPA that can redirect client-side after `goto` returns

`REFUSED_EXAMPLES` (settings / logout / friends / messages) is wording only. It
never changes the decision; the allow-list already refused by then.

### The board is identified, never guessed

`[class*='board' i]` matches the 24px `board-layout-icon` gear, and that icon
animates on its own when you press an arrow key. Signing it produces a
changing hash, so the verifier "confirmed" moves on an icon.

A board is square and ≥100px. The tolerance differs by purpose:

| Where | Bounds | Rationale |
|---|---|---|
| `move_list.py` (verification) | 0.7 – 1.43, ≥100px | a false negative only costs a fallback |
| `review.py` (screenshot crop) | 0.99 – 0.01, ≥120px | a wrong box is a wrong photo |

### Never report a pass without a positive signal

`StepResult.verified` is False when no board signal was readable. `goto_ply`
returns False when the signature cannot be read. `read_review_state`'s error
path reports `loaded: True` **only because** `wait_for_review_loaded` also
requires a stage-specific key, which is False on that path.

Two consequences to preserve:

- an unreadable board is **not** a move (`after is not None and after != before`)
- `goto_ply` captures its "before" signature **after** any partial stepping —
  otherwise a half-completed step alone satisfies "did the board change?"

### Navigation clicks tagged cells

`_MOVE_TABLE_JS` writes `data-ai-ply` in the same loop that computes the
classification, and `_click_row` clicks `[data-ai-ply="N"]`. The index used by
the click is therefore **the same one** the classification came from, by
construction.

An earlier version indexed into the union of
`[class*='move' i] *, [class*='ply' i] *, [class*='turn' i] *`, which resolves
to clock, aria and icon elements as well as moves — so `ply - 1` pointed at an
arbitrary element and navigation "succeeded" onto the wrong position.

### Clicking beats reloading

A `?move=N` reload tears down the review session: no board tint, no coach
bubble, no eval marker, and a Cloudflare round trip. In-page clicking keeps all
of it.

### The vocabulary lives in Python, not in JS

The browser extracts **raw facts** (`icon_id`, `css_classification`, `classes`);
`move_list.RULES` turns them into a kind. One place to add a classification,
and unit-testable without a browser.

Matching is **whole-token set membership**, never substring search —
substring matching turns `"dismiss"` into a `miss` and `"Facebook"` into a
`book`.

Two details that look redundant but are load-bearing:

- `classify_class("analysis-bestMove")` splits to `{analysis, best, move}` and
  matches on the generic fragment `"best"`
- `classify_class("bestmove")` (the already-lowercased tally key) matches only
  on the compound fragment `"bestmove"`

So removing the compound fragments breaks the summary tally while leaving the
move table working.

### Splitting long answers for Discord

`MAX_MESSAGE_CHARS = 1900`. Split order: paragraphs → lines → hard cut.

`_orphaned(text, i)` answers one question: *would `text[i]` be the first
character of the remainder, separated from the character it modifies?* True for
ZWJ, variation selectors and combining marks. Only that character is checked —
a combining mark at the **end** of a chunk is correct, and treating it as a
reason to back off is what produces chunks that begin with a bare accent.

### Collecting screenshots at the tool boundary

The model cannot see images, so it writes prose about them. Asking it to list
the paths it produced is the obvious implementation and it is wrong twice: the
model can misremember a path, and a path is useless to someone reading a
message on a phone.

`collect_artifacts` walks the tool's return value generically, keeps strings
with an image suffix, and **verifies the file exists**. A tool that names a
file it never wrote loses the picture at the collector instead of 404-ing in a
channel.

De-duplication is by path, per turn, which is right -- the same frame reached
two ways should be posted once. It makes a filename a *promise*: two different
shots must never produce one path, or the second is silently swallowed here
with nothing anywhere reporting it. The names are built from a
second-resolution stamp, so `common.unique_shot_path` advances the stamp until
the name is free. Advancing the stamp rather than appending a counter is
deliberate: `describe_artifact` reads the name with a regex anchored on
`\d{8}_\d{6}$` to work out what the frame shows.

### A screenshot that was never produced is announced, with the reason

A model handed `{"error": ...}` still writes "here is the screenshot". The
user then finds no picture, and the most natural reading is that the bot
ignored the request. So `_post` appends a line when a screenshot tool ran and
the turn carries no file.

The *reason* has to come with it, and it cannot come from the model: the tools
report failure as an `error` key, the model is handed that key and writes prose
about it, and the front end never sees a tool's return value. So `run_agent`
collects `(tool, message)` for every `error` it gets back -- through an
optional `errors` list, because the three-tuple return is the contract
everything else uses -- and `AgentResult.tool_errors` carries it to the front
end. `picture_failure_note` then says what actually happened, with two
special cases worth naming:

- **the sign-in expired.** Every review tool reports it the same way, it is the
  most common reason a picture does not arrive, and it is the one the user can
  act on: the owner re-runs `login_chess.py` once and the same request works.
  A generic "no picture" line here reads as a broken bot; this one reads as
  "ask the owner to sign in again".
- **the fix belongs to the owner.** The tools name a fix ("Run: python
  dump_pages.py <link>") because that is right for the terminal. A Discord
  reader cannot run anything on the owner's machine, so the command is stripped
  and the reason kept.

`!status` also reports the age of the saved session file. It cannot know
whether chess.com still accepts the cookie without loading a page, so it states
the age and nothing more -- claiming "expired" from a timestamp would be a
plausible wrong value, which is the one thing this project keeps designing
against.

The same applies one stage later, where it is quieter still: a frame over
`MAX_ATTACHMENT_BYTES` is dropped *before* Discord is asked, so there is no
HTTP error, nothing in the channel, and only a log line. `_attach` counts what
`_upload_batches` refused and says how many pictures were left out. Both paths
lose a picture the user asked for; neither may be silent.

### "My brilliant move" is a class of move, not a number

`screenshot_game_move` photographs ONE numbered move, so it needs a move
number. "Screenshot the brilliant move for me" has no number in it, and a model
that reaches for the single-move tool has to invent one -- photographing a
position the user never asked about, or failing outright.

`review_game_screenshots` is the tool that answers it: it takes a frame per
brilliant **and** per blunder and returns them, and they are attached
automatically. So the prompt names that class of request explicitly, and pairs
it with where the game comes from: a link already in the conversation
(including the remembered history), else `get_recent_games` for the username,
else ask. Never guess a game id.

The general shape is worth stating because it is not specific to this bug: a
tool that needs an argument the user's phrasing does not contain is a prompt
problem before it is a code problem.

### Frame what was asked for, and say when you could not

`content_clip` falls back to the whole window when the board cannot be
located. That is the right degradation for a tool -- a full-page shot beats no
shot -- and the wrong answer for a *reply*, because the model describes "the
board after 12...e4" over a photograph of an entire browser window. So the
screenshot tools report `framed`, and the system prompt tells the model to
mention it when the frame is not the board.

`take_screenshot(region="auto")` had the same problem in a quieter form: it set
`REVIEW_SHOT_REGION` to `sidebar` for every region that was not `board`, so the
DEFAULT shot contained the commentary column and no board at all -- filed under
a caption claiming "the board and the review panel". `auto` now means what it
says and does not touch the setting.

### The progress line is written for the reader

The message a turn edits while it runs is the user's, not the log's. It said
`` _running `screenshot_game_move` gameurl=https://www.chess.com/...`_ ``:
code, in the middle of an answer, with half a URL pasted into the channel, and
no answer to the question the user actually had ("is it working?").
`TOOL_PROGRESS` maps each tool to a phrase; an unknown tool falls back to a
generic phrase rather than to its name, and arguments are never shown.
`DISCORD_SHOW_TOOLS=1` restores the technical line for the owner watching the
bot, which is the only audience it was ever for.

### Memory lives outside the model's chat

"I asked you that two messages ago and you forgot" has three causes that look
identical from the outside: the turn cap replaces the Gemini chat and the old
one is unrecoverable, a long chat fills with tool output (`get_moves` alone is
a few hundred plies) and the messages at the front stop being attended to, and
a restart or an idle sweep drops the chat entirely.

None of them is fixed by asking the model to remember better, and none is fixed
by raising `AGENT_MAX_TURNS` -- that only moves the cliff. So
`agent.ConversationMemory` keeps the last few exchanges on the *session* and
prepends them to every message. The reset that actually bites is bridged,
because the memory does not live in the chat that gets replaced. It is bounded
by count, per-item and total characters, because it is prepended to every turn,
and the budget drops whole turns from the old end: a remembered question with
its answer cut off is worse than no memory. `!reset` clears it -- that one is
the user asking for a clean slate.

### Upload budgets by bytes, not just count

Ten 1400×820 board crops measure ~620KB each: ~6MB against Discord's 8MB
ceiling for a server with no boost. A wider `REVIEW_VIEWPORT` tips it over and
the whole batch is refused with a 413.

So batches are cut on **count and bytes**. A single file over
`MAX_ATTACHMENT_BYTES` is skipped on its own rather than failing its batch and
every batch after it.

---

## 4. Honest limits

These are real and are stated in the tool docstrings so the model reports them
rather than papering over them:

- **chess.com does not publish a multi-move engine line.** `where_am_i` and the
  navigation tools report a single `engine_eval` (e.g. `-0.61`). The system
  prompt tells the model never to invent a PV.
- **`classification: null` does not mean the move was sound.** In a captured
  game, 7 of 29 cells carried a label while the summary counted 7
  inaccuracies, 1 miss and 13 best moves that no cell mentions. Those words
  exist in the legend and the tally only. Use `summary_counts` for *how many*,
  `classification` for *where*, and `counts_without_a_move` for the gap.
- **Player names are in DOM order, not tied to a colour.** Nothing in the
  markup says which name belongs to which accuracy, so `get_review_state` says
  that rather than guessing.
- **`moves_read` is coverage, not a guarantee.** A short read means the page
  rendered only part of the list.

---

## 5. Gotchas worth remembering

Short, because each one has a named test guarding it.

| Trap | What happens if you forget |
|---|---|
| Tool declared `async def` | the loop gets a coroutine → "not JSON serializable" → turn aborts |
| `IMAGE_SUFFIXES` as a `frozenset` | `str.endswith` raises → **every screenshot fails to attach** |
| A screenshot name built from a timestamp alone | two shots in the same second overwrite each other, and the collector de-dupes by path → N requests, one picture, no error anywhere |
| Gate released in `else` not `finally` | any tool exception deadlocks every later request |
| `except TypeError` before calling the tool | a crash inside a tool is reported as bad arguments |
| Waiting only for loaders to vanish | reports "loaded" before the app has mounted anything |
| Using the board as the data-readiness signal | reports "ready" at 0ms against a 119-char body |
| `locator.evaluate("() => ...el...")` | Playwright binds the element as the arrow function's **parameter** |
| Reading the summary after Start Review | `None`/`{}`, which looks exactly like "no summary" |
| Confidence defaulting to optimistic | "moved 3" on a page that could not be read |
| A status line built from the tool name and args | the log gets posted into the user's channel |
| Trusting the model's chat history to be the memory | the turn cap replaces the chat and the conversation is gone |
| `Intents.default()` | ~27 flags, some privileged → gateway closed with 4014 |
| `AllowedMentions` left at the default | `@everyone` in model-written text actually pings |
| Allow-list checked only before `goto` | an SPA redirect lands you off the board |

---

## 6. Where to look when something breaks

| Symptom | Tool |
|---|---|
| "no blunders" for a game that had them | `python calibrate_review.py <link>` |
| a selector stopped matching | `python dump_pages.py <link>` then re-read the markup |
| a tool returns a plausible wrong value | `set CHESS_AGENT_LOG=debug` |
| a screenshot came out as the whole page, or with no board | `framed` in the tool's reply says so; `set REVIEW_SHOT_REGION=board` frames the board alone |
| "no picture came through" in a channel | the turn's `tool_errors` say why; in practice it is the chess.com sign-in needing `login_chess.py` again, and `!status` shows the session file's age |
| Cloudflare challenge | `set REVIEW_HEADLESS=0` |
| a setting "isn't taking effect" | `python check_env.py` |
| want to see which step of a turn is slow | `set DISCORD_SHOW_TOOLS=1` |
| want to re-verify the whole review path | `python calibrate_review.py <link> --expect blunder=2,brilliant=1` |

`calibrate_review.py` prints its version on every run and exits non-zero on every
condition it checks — including "no board could be located" and "every hit was
inferred", which is exactly the shape of a run that *looked* like it passed
while measuring the wrong element.

The move-list and classification selectors are pinned to
`tests/fixtures/chess_com_review_markup.json` (18KB, distilled from a real
capture). If chess.com changes its markup, regenerate with
`dump_pages.py` → `make_fixture.py` and re-pin.
