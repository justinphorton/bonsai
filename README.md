# bonsai — steerable context management

bonsai splits a long conversation into labelled topics and compacts the transcript through that
index: you choose which topics stay verbatim, the rest collapse to one line each, and any of
them expands again by id. The point is to decide what the model keeps while you still can,
rather than discovering what it lost.

A Claude Code session is a JSONL file of a few thousand turns. `/compact` replaces its visible
history with a generated summary. That summary records no mapping back to the turns it came
from, so once it runs there is no way to ask what was dropped, no way to recover a specific part
of it, and no way to have said in advance which parts should survive.

bonsai keeps that mapping. One model call segments the transcript into units (a direction the
work took) and threads (the messages inside one), each with an id, a label, a synopsis and the
message ids it covers. Compaction then operates on the index rather than on prose: a thread is
left verbatim, or replaced by one line carrying its label, turn span, token count and id.
Because the id persists, the reduction is reversible (`recall`), addressable before it happens
(`keep`, `drop`, `focus`), and annotatable (`note`, `relabel`).

The same index serves three purposes, which is why it is worth maintaining by hand: it is what
the model sees after compaction, the set of handles you steer with, and a readable account of
what the session contains — what was tried, what was rejected, what is unfinished.

`bonsai threads -v` prints the index — here, 28 threads in 6 units over 412,886 tokens. A unit
is a direction the work took:

```
g2u4           21,506 tok full       turns 492-602        Retrieval tuning
```

A thread inside it, marked `keep`, which holds it verbatim at any budget. The indented line is
its synopsis, shown by `-v`:

```
    g2u4.1      8,104 tok full  keep turns 492-539        BM25 sweep
                          k1=1.2, b=0.75 chosen. 85.94% on the 20k set.
```

And a thread that is currently `trace`, meaning collapsed — so these two lines are the whole of
what the model sees of it:

```
    g2u4.2     13,402 tok trace      turns 540-602        Stemming: rejected
                          Both stemmers: 85.46% vs 85.94%, ~11x slower.
```

13,402 tokens rendered as about 25, and enough to separate a measured rejection from something
never tried. A summary that dropped the result cannot express that difference, and it is the one
that most often has an assistant re-propose work already done. `bonsai recall g2u4.2` returns
the original turns.

## Install

```bash
git clone https://github.com/justinphorton/bonsai && cd bonsai
python -m venv .venv && .venv/Scripts/activate      # or source .venv/bin/activate
pip install -e ".[dev]"
```

Python 3.11+. The only dependency is `anthropic`. For the commands that call the API, put your
key in a `.env` in the directory you run from (a key already exported in your shell wins):

```
ANTHROPIC_API_KEY=sk-ant-...
```

## Quick start

`bonsai` goes on your PATH. Run it from any directory you've used Claude Code in and it picks up
the newest session for that directory.

```bash
bonsai segment                  # split the conversation into topics
bonsai status                   # size, thread count, what you've marked
bonsai threads -v               # the topics, with ids and synopses

bonsai keep "the auth refactor" # mark these to stay verbatim
bonsai plan --budget 40000      # preview the result; changes nothing
bonsai compact --budget 40000   # rewrite the transcript

bonsai recall g3u7.2            # the original text of a collapsed thread
```

`segment` is the only command that costs money or takes real time — one model call over the whole
transcript, so a few minutes on a long session. `plan` and `compact` make no API calls.

A compaction takes effect on reload: `Ctrl+Shift+P` → *Developer: Reload Window*. The original
transcript is copied to a `.bak.jsonl` beside it first.

[`.claude/commands/bonsai.md`](.claude/commands/bonsai.md) is a slash command that passes
arguments through, so you can run any of this from inside the chat. Copy it to
`~/.claude/commands/` for every project.

## Commands

Every command takes an optional session path as its last argument; omit it to use the current
directory's newest session. `--model` goes before the subcommand.

| Command | What it does |
| --- | --- |
| `status` | Size, thread count, your marks, whether the index is stale |
| `threads [-v] [--recent]` | The index. `-v` adds synopses, `--recent` orders newest first |
| `segment [--by "..."] [--units N] [--subunits M]` | Build or rebuild the index. One API call |
| `plan --budget N` | What a compaction would keep, and why |
| `compact --budget N` | Rewrite the transcript |
| `keep` / `drop` `<id or description>` | Mark threads to stay verbatim, or to collapse |
| `focus <description>` | Keep what matches, collapse the rest |
| `steering [--clear]` | What you've marked so far |
| `recall <id>` | A collapsed thread's original messages |
| `note <id> "..."` | Attach a note; it appears in the collapsed line |
| `relabel <id> "..."` | Replace a label with your own wording |
| `merge <id> <id> [...]` | Join threads that should have been one |
| `split <id> <turn>` | Cut a thread in two at a turn |

`keep`, `drop` and `focus` take either an id or a plain-language description — a description
costs one API call to match against the labels, because finding three ids among 188 by hand
isn't something anyone will do. If a match fails, nothing changes and the error says so.

## How it works

Messages go into an append-only SQLite log under `~/.bonsai/`, so the full history is always
there to restore from. One model call groups them into **units** (a direction the work took) and
**threads** inside them (the actual messages), each with a label and a short synopsis. Ids look
like `g3u7.2` — generation 3, unit 7, thread 2. The prompt asks for concrete nouns and numbers,
since once a thread is collapsed its label is all the model sees.

**Detail levels.** A thread is either `full` or `trace` — one line with label, turn span, size,
id and any note. There's no middle level. There used to be: a per-thread summary meant one API
call per thread, each able to fail, each a different length every run, and overshooting often
enough that the finished context had to be measured and trimmed afterwards. Without it, every
piece has a known size before anything is rendered, so `plan` and `compact` make no API calls
and return the same result every run.

**Allocation** is three rules: what you marked, then the thread holding the newest message, then
whatever else fits, newest first. Your marks are the only thing that can make a budget
impossible, and bonsai refuses rather than guessing; everything it decides on its own degrades to
a trace instead of failing. When every thread in a unit collapses, the unit prints as one line
listing them by id — an abandoned direction costs one line, not twelve.

**Surviving a rebuild.** Re-segmenting regroups everything, so marks attach to *messages* rather
than threads and reattach to whichever thread now holds them. Groupings are versioned and never
deleted, so an id you wrote down earlier still resolves.

**Editing.** The segmenter is a model, so it produces vague labels, splits one topic in two, runs
two together. `relabel`, `merge` and `split` fix that in place; before they existed the only fix
was paying for a rebuild. A note is the one part of a collapsed line a model didn't write, so
it's prefixed `NOTE FROM THE USER`.

## Configuration

| Variable | Effect |
| --- | --- |
| `ANTHROPIC_API_KEY` | Required for `segment` and description matching |
| `BONSAI_MODEL` | Which model segments. Sonnet by default: mechanical work, a fifth of Opus's price |
| `BONSAI_READONLY` | Disables the transcript rewrite |
| `BONSAI_QUIET` | Silences progress output during a long segmentation |

Everything bonsai owns lives under `~/.bonsai/`. The only time it writes elsewhere is `compact`,
which makes that the one place it could conflict with another tool managing the same session —
and the one place to switch off. With `BONSAI_READONLY=1`, indexing, planning, marking, notes and
recall all still work; only the rewrite refuses, before doing the work rather than after.

## Benchmarks

Sixty public questions from [LongMemEval](https://github.com/xiaowu0162/LongMemEval) (ICLR 2025),
four categories, three disjoint groups of twenty. Four arms at equal token budgets:

| | truncation | flat summary | bonsai |
| --- | --- | --- | --- |
| accuracy | 26.7% | 60.0% | **83.3%** |

Where exactly one arm answered correctly, bonsai won 17 and lost 3 (exact binomial p = 0.003),
on fewer tokens than the flat summary in five of six groups. Keeping the budgets equal was most
of the work — an early version gave bonsai six times the context and produced a flattering
number that meant nothing.

The margin depends on how tight the budget is: it is largest when the context is small, and
closes as the budget approaches the point where a flat summary can hold nearly everything.

**Ceiling and floor.** Across nine measurements on two systems, including Claude Code's own
`/compact`, a flat summary tops out between 5,000 and 12,000 tokens regardless of the budget it
is given: ask for 80,000 and you get 9,000. Generation has an upper bound that the budget does
not control. bonsai composes existing text rather than generating new text, so it has no
ceiling, but it has a floor — the thread list itself costs tokens, around 12K for 188 threads.
On a 3,000-turn conversation the two ranges do not overlap, so above the ceiling the comparison
is not against a worse summary but against truncation.

The harness and raw results aren't in this repo — they contain transcript data I can't publish,
so these numbers are reported rather than reproducible from a clone.

## Limitations

- **Below ~12K tokens on a many-threaded conversation** the thread list alone eats the budget.
  bonsai refuses a budget it can't meet rather than producing something useless.
- **Segmenting past 800 turns works in chunks** and merges, because one call runs out of output
  tokens before it runs out of conversation. A thread can't span a chunk boundary, so a topic
  that recurs much later can come back as two — which is what `merge` is for.
- **Rebuilding re-reads everything.** Incremental segmentation would be cheaper; it was the most
  complicated code in the project and never paid for itself.
- **The transcript format was worked out by reading real session files**, so a change there could
  break the adapter. The library underneath doesn't depend on it.
- `llm.py` is 633 lines doing four jobs and should be split.

## Future plans

**Continuous indexing** — per-turn maintenance cheap enough to run constantly, using embeddings
rather than a model call, so the index stays current as you work and `segment` becomes a
one-time cost.

**Intermediate detail levels** — a thread held at something between its full text and a single
line. The design constraint is that every level needs a size known before rendering, which is
what keeps `plan` free and deterministic.

**Steering by intent** — "I'm done with the parser" collapses the right threads without naming
them.

**Retrieval across sessions** — the same topics recur across conversations, and the index is
already the right shape to search.

Each of these extends the current structure rather than replacing it. A thread is deliberately
untyped — one object serving as steering handle, compaction unit and index entry — and ids
resolve across generations, so anything built on top applies to indexes that already exist.

## Development

```bash
pytest          # 109 tests, all offline, no API key
ruff check .
```

`Session` takes its store, segmenter and token counter as constructor arguments, which is how the
tests run with no key — the fakes in [`tests/conftest.py`](tests/conftest.py) stand in for the
model. The Claude Code integration is one file; underneath, bonsai is a normal library.

```
bonsai/
  models.py      the data types. Dataclasses, no logic.
  store.py       SQLite. Events append-only, groupings versioned and never deleted.
  llm.py         every API call: token counting, segmentation, description matching
  compact.py     the allocation policy and rendering. Pure functions.
  session.py     the main API
  transcript.py  Claude Code's session file to events, and the rewrite back
  cli.py         the commands
```
