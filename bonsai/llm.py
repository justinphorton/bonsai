"""Every call to Claude lives here: counting tokens and splitting a transcript into topics.

No other module talks to the API. That's deliberate — it means you can read any other file
without wondering whether a line of it costs money, and it's what lets the tests run offline by
substituting fakes for the two things this module provides.

Token counts come from the API's own counter, never from an estimate like characters ÷ 4. A
token budget is only worth having if the number behind it is exact.

The API key is read from the ANTHROPIC_API_KEY environment variable.
"""

import json
import os
import re
import sys
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import anthropic

from bonsai.models import Event, Unit

# Which model does the segmenting. Override it with the BONSAI_MODEL
# environment variable; the CLI and the library both read it from here.
#
# Sonnet by default, deliberately. Splitting is the one thing bonsai pays for, and the job is
# mechanical enough that Sonnet does it well at about a fifth of Opus's price -- every real
# segmentation in this project was Sonnet and the labels were good. Defaulting to the expensive
# model would mean a fresh clone quietly paying 5x for its first run. Set BONSAI_MODEL to
# claude-opus-5 if you want it.
MODEL = os.environ.get("BONSAI_MODEL", "claude-sonnet-5")

# Built once and reused. Creating a client per call would re-do the connection setup every time.
_client: anthropic.Anthropic | None = None


class EmptyReply(RuntimeError):
    """The call succeeded but carried no text — only thinking, or nothing at all.

    Its own type because it is worth retrying, unlike a refusal (a decision) or a truncation
    (which needs a bigger max_tokens, not another identical attempt).
    """


def client() -> anthropic.Anthropic:
    """The shared Anthropic client, built on first use."""
    global _client
    if "ANTHROPIC_API_KEY" not in os.environ:
        raise RuntimeError("Set ANTHROPIC_API_KEY. See README > Setup.")
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


def count(text: str) -> int:
    """Exactly how many tokens `text` costs, measured by the API.

    We measure it as a single user message, so the number includes a small constant for the
    message framing. That's fine: it's the same constant every time, and it's what the text
    will actually cost when we send it.
    """
    if not text:
        return 0
    return (
        client()
        .messages.count_tokens(model=MODEL, messages=[{"role": "user", "content": text}])
        .input_tokens
    )


def _text(
    model: str,
    system: str | None,
    user: str,
    max_tokens: int,
    label: str = "",
    schema: dict | None = None,
) -> str:
    """Make one model call and return its text.

    The call is streamed, and if `label` is given we report progress to stderr as characters
    arrive. That's not decoration: splitting a long transcript takes minutes of generation, and
    without any output a long wait is indistinguishable from a hang. Set BONSAI_QUIET to silence
    it.
    """
    quiet = os.environ.get("BONSAI_QUIET") or not label
    started, chars = time.time(), 0
    # A schema makes the shape a constraint instead of a request: the model cannot return prose
    # where an object is required. Asking nicely was not enough -- on a long transcript the model
    # ignored the instruction entirely and carried on writing the conversation.
    extra = (
        {"output_config": {"format": {"type": "json_schema", "schema": schema}}} if schema else {}
    )
    with client().messages.stream(
        model=model,
        max_tokens=max_tokens,
        system=system or "",
        messages=[{"role": "user", "content": user}],
        **extra,
    ) as stream:
        for event in stream:
            if quiet or getattr(event, "type", "") != "content_block_delta":
                continue
            chars += len(getattr(event.delta, "text", "") or "")
            minutes, seconds = divmod(int(time.time() - started), 60)
            # \r returns to the start of the line so the next write overwrites it. Padding to
            # 78 characters matters: \r only moves the cursor, it doesn't erase, so a shorter
            # line would leave the tail of the previous one behind and look corrupted.
            line = f"  {label}: {chars:,} chars, {minutes}m{seconds:02d}s"
            print("\r" + line.ljust(78), end="", file=sys.stderr)
        if not quiet:
            print(file=sys.stderr)
        return _answer(stream.get_final_message())


def _answer(final) -> str:
    """The text of a finished reply, or a clear error saying why there isn't one.

    Worth the lines: a reply cut off at max_tokens, a safety refusal, and a reply with no text
    block at all used to be indistinguishable here — the caller just got a fragment or a bare
    StopIteration, and a failed segmentation looked like a parse bug. Each one needs a different
    response from whoever is reading the error, so each one says what it is.
    """
    reason = getattr(final, "stop_reason", None)
    if reason == "max_tokens":
        raise RuntimeError("reply hit max_tokens and is truncated; raise max_tokens and retry")
    if reason == "refusal":
        details = getattr(final, "stop_details", None)
        category = getattr(details, "category", None)
        raise RuntimeError(f"the model refused this request (category {category!r})")
    text = next((b.text for b in final.content if b.type == "text"), None)
    if text is None:
        kinds = [getattr(b, "type", "?") for b in final.content]
        raise EmptyReply(f"no text block in the reply; stop_reason={reason!r}, blocks={kinds}")
    return text


# --- splitting a transcript into topics --------------------------------------

SEGMENT_PROMPT = """You segment a long conversation into units for a context-management system.

Group the turns into about {n_units} UNITS — the conversation's big directions or bubbles of
meaning — each holding {n_sub} SUBUNITS: the threads inside it (one task, one decision and its
reasoning, one tangent). A subunit may include non-adjacent turns if a thread is picked up
again later. Every turn index must appear in exactly one subunit.{by}

Labels matter more than anything else: a label is what the model sees when the thread has been
collapsed, and it must let the model tell whether it needs the thread. Use the actual nouns:
names, numbers, files, decisions. "Membership fee: sliding scale $40/$80/$120" is a good label;
"Discussion about fees" is useless. The synopsis is 2-4 sentences: what was decided, the key
values, and why — exact numbers and names, never paraphrased. No forward references to OTHER
threads: describe a thread as of its own turns, and if a later thread changes something, the
earlier one must not mention that it was later changed.

But a thread must be chronological WITHIN itself. A thread is topical, so it may pick a subject
up again many turns later; if one of its own values, counts or decisions changes across its own
turns, give every version in order with the turn it came from and the most recent last — "four
short stories (turn 44), seven by turn 299" — never just the latest and never just the first.

Give a thread's turns as RUNS, not one index at a time. "40-96" means every turn from 40 to 96
inclusive; a bare number means that one turn; mix the two freely. ALWAYS a list, even for one
run: ["40-96"], never "40-96". ["40-96", 150, "298-300"] is a thread that ran for a while, came
back once, and again at the end — use that mixed form whenever a thread is picked up again, and
do not force a thread to be one tidy block just because runs are cheap to write.

Cover every turn. Sorted by first turn, your runs should tile 0 to {last}: each run starting
where the previous one ended plus one. The easiest mistake to make with runs is writing a tidy
"619-700" and then starting the next thread at 1355, leaving 701 to 1354 in no thread at all.
Check the joins before you return, and account for all {n_turns} turns.

Return ONLY a JSON array:
[{{"label": "...", "synopsis": "...",
   "subunits": [{{"label": "...", "synopsis": "...", "turns": ["0-45", 97, "210-230"]}},
                ...]}}, ...]"""


# The shape the reply must take. This is enforced by the API, not requested in the prompt, which
# is the difference between "please return JSON" and being unable to return anything else.
# `turns` is a list of strings so "40-96" and "97" are the same kind of thing; _expand reads both.
_THREAD = {
    "type": "object",
    "properties": {
        "label": {"type": "string"},
        "synopsis": {"type": "string"},
        "turns": {"type": "array", "items": {"type": "string"}, "minItems": 1},
    },
    "required": ["label", "synopsis", "turns"],
    "additionalProperties": False,
}

SEGMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "units": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "synopsis": {"type": "string"},
                    "subunits": {"type": "array", "items": _THREAD, "minItems": 1},
                },
                "required": ["label", "synopsis", "subunits"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["units"],
    "additionalProperties": False,
}


class LLMSegmenter:
    """Splits a whole transcript into units and threads with a single model call.

    The Units it returns use temporary ids ("u0", "u0.1"); the store renumbers them on the way
    in, so this class never has to know about ids already in the database.

    `existing` is accepted and then ignored — the whole transcript is always re-read. There used
    to be code that segmented only the new tail of a growing conversation, which is cheaper in
    principle, but it was by far the most intricate code in the project and never earned its
    keep. `Session.units()` solves the same problem by gathering new turns into one verbatim
    stand-in thread instead.
    """

    def __init__(
        self,
        model: str = MODEL,
        by: str | None = None,
        n_units: int = 6,
        n_sub: str = "3-10",
    ) -> None:
        self.model = model
        self.by = by  # the user's own instruction for how to group it, if they gave one
        self.n_units = n_units
        self.n_sub = n_sub

    # Above this many turns the transcript is segmented in pieces. The model does good work and
    # then simply stops: on a 2,943-turn transcript it placed turns 0-1308 in 85 well-labelled
    # threads, then ended its turn with 40,000 output tokens still unused. It was not cut off and
    # it was not confused about the format -- it believed it had finished. Nothing in a prompt
    # fixes that, because there is no signal left telling it there is more to do. A smaller input
    # has no far end to lose track of. 800 sits comfortably inside the 1,308 it managed.
    CHUNK_TURNS = 800

    def segment(self, events: list[Event], existing: list[Unit] | None = None) -> list[Unit]:
        """Split these events into units and threads."""
        if not events:
            raise ValueError("no events to segment")
        ordered = sorted(events, key=lambda e: e.seq)
        if len(ordered) <= self.CHUNK_TURNS:
            return self._whole(ordered)
        return self._chunked(ordered)

    def _chunked(self, events: list[Event]) -> list[Unit]:
        """Segment a long transcript in pieces and merge the results.

        Each piece is segmented on its own, so its turn indices are local to it and map straight
        back to its own events. The only real work here is renumbering: every chunk numbers its
        units from u0, so without this the second chunk's u0 would collide with the first's.

        The cost is that a thread cannot span two chunks. A topic raised early and returned to
        much later becomes two threads rather than one, which is a genuine loss -- threads being
        topical rather than positional is the point of them.
        """
        size = self.CHUNK_TURNS
        chunks = [events[i : i + size] for i in range(0, len(events), size)]
        if not os.environ.get("BONSAI_QUIET"):
            print(
                f"  {len(events):,} turns: segmenting in {len(chunks)} chunks of up to {size}",
                file=sys.stderr,
            )
        out: list[Unit] = []
        units_so_far = 0
        for number, chunk in enumerate(chunks, 1):
            try:
                units = self._whole(chunk)
            except Exception as e:  # noqa: BLE001 - any failure, and the other chunks survive
                # Each chunk is its own paid call. Losing the ones that worked because a later
                # one came back malformed is exactly the waste this design exists to avoid, so
                # a bad chunk becomes unsorted turns and the rest of the run carries on.
                if not os.environ.get("BONSAI_QUIET"):
                    print(
                        f"  chunk {number}/{len(chunks)} failed ({type(e).__name__}: {e}); "
                        "keeping its turns as unsorted",
                        file=sys.stderr,
                    )
                units = _sweep_unclaimed(chunk, set(), 0, chunk[0].session_id)
            parents = [u for u in units if u.parent_id is None]
            renamed = {u.id: f"u{units_so_far + k}" for k, u in enumerate(parents)}
            for unit in units:
                if unit.parent_id is None:
                    out.append(replace(unit, id=renamed[unit.id]))
                else:
                    parent = renamed[unit.parent_id]
                    suffix = unit.id.split(".", 1)[1]
                    out.append(replace(unit, id=f"{parent}.{suffix}", parent_id=parent))
            units_so_far += len(parents)
        return out

    # Roughly how many turns one thread should cover. A 2,927-turn transcript asked for the
    # default 6 units came back with 28 threads -- about 105 turns each -- and the model stopped
    # accounting for turns long before the end, leaving 83% of the conversation unplaced. Threads
    # that size are not useful anyway: the point of a thread is to be one topic.
    TURNS_PER_THREAD = 25
    THREADS_PER_UNIT = 8

    def _prompt(self, turn_count: int) -> str:
        """The system prompt. Needs the turn count to size the ask and to name real turn numbers.

        `n_units` acts as a floor rather than a target, so an explicit setting is never silently
        lowered, but a long transcript still gets asked for enough threads to cover it.
        """
        wanted_threads = max(12, turn_count // self.TURNS_PER_THREAD)
        units = max(self.n_units, -(-wanted_threads // self.THREADS_PER_UNIT))
        by = f"\n\nSegment along these lines, as the user asked: {self.by}" if self.by else ""
        return SEGMENT_PROMPT.format(
            n_units=units,
            n_sub=self.n_sub,
            by=by,
            last=turn_count - 1,
            n_turns=turn_count,
        )

    def _whole(self, events: list[Event]) -> list[Unit]:
        turn_count = len(events)

        # The instruction is repeated *after* the transcript as well as before it. With a very
        # long transcript in front of it, the model otherwise tends to carry on writing the
        # conversation instead of segmenting it.
        user = (
            f"<transcript turns={turn_count}>\n{_numbered(events)}\n</transcript>\n\n"
            f"That is the whole transcript, indices 0 to {turn_count - 1}. Do NOT continue it, "
            f"and do not write anything after the JSON array. Every one of the {turn_count} turns "
            "must fall inside exactly one subunit's runs: sorted by first turn, your runs must "
            f"join end to end from 0 to {turn_count - 1} with no gap. Segment it as instructed "
            "and return ONLY the JSON array."
        )
        reply = _text(
            self.model,
            self._prompt(turn_count),
            user,
            48_000,
            f"segmenting {turn_count} turns",
            schema=SEGMENT_SCHEMA,
        )

        # Save the reply before parsing it. Segmenting a long transcript costs real money, and a
        # reply the parser chokes on would otherwise be lost along with what we paid for it.
        # This exists because it happened: an unparseable reply threw away $0.88.
        _keep(f"segment-{events[0].session_id}", reply)

        session = events[0].session_id
        units: list[Unit] = []
        claimed: set[int] = set()
        raw = _json(reply)
        for unit_index, unit in enumerate(raw):
            parent = Unit(f"u{unit_index}", session, unit["label"], [], unit.get("synopsis", ""))
            children = []
            for thread_index, thread in enumerate(_subunits(unit, unit_index)):
                # First claim wins. A turn named by two threads would otherwise be rendered
                # twice, which wastes budget and makes the token accounting wrong.
                turns = [t for t in _expand(thread["turns"], turn_count) if t not in claimed]
                if not turns:
                    continue  # nothing left after deduping; an empty thread would break rendering
                claimed.update(turns)
                children.append(
                    Unit(
                        f"u{unit_index}.{thread_index}",
                        session,
                        thread["label"],
                        [events[turn].id for turn in turns],
                        thread.get("synopsis", ""),
                        f"u{unit_index}",
                    )
                )
            if children:
                units.append(parent)
                units += children

        units += _sweep_unclaimed(events, claimed, len(raw), session)
        # Always passes now that leftovers are swept up; kept as a guard against a bug here.
        _check_coverage(sorted(claimed), turn_count)
        return units


UNSORTED_SYNOPSIS = (
    "Turns the segmenter did not place in any thread. They are kept here so nothing is lost; "
    "recall this thread to read them."
)


def _sweep_unclaimed(
    events: list[Event], claimed: set[int], next_index: int, session: str
) -> list[Unit]:
    """Gather turns no thread claimed into their own unit, and add them to `claimed`.

    The segmenter does miss turns, and on a long transcript it can miss a lot of them — one real
    run left 1,691 of 2,753 unplaced, including an unbroken stretch of 736. Refusing the whole
    segmentation threw away a call we had paid for; dropping the turns quietly would be worse,
    because a turn in no thread is never rendered at all, and that silent loss is the exact
    thing this project exists to prevent.

    So they become ordinary threads with ordinary ids: they render, they cost budget, they can
    be recalled. The segmentation is visibly worse rather than secretly wrong, and the label
    says which turns these are so the model can ask for them.
    """
    missing = [i for i in range(len(events)) if i not in claimed]
    if not missing:
        return []

    runs: list[list[int]] = []
    for turn in missing:
        if runs and turn == runs[-1][1] + 1:
            runs[-1][1] = turn
        else:
            runs.append([turn, turn])

    parent = Unit(f"u{next_index}", session, "unsorted turns", [], UNSORTED_SYNOPSIS)
    out = [parent]
    for j, (first, last) in enumerate(runs):
        span = f"{first}-{last}" if first != last else f"{first}"
        out.append(
            Unit(
                f"u{next_index}.{j}",
                session,
                f"unsorted: turns {span}",
                [events[t].id for t in range(first, last + 1)],
                UNSORTED_SYNOPSIS,
                f"u{next_index}",
            )
        )
    claimed.update(missing)
    if not os.environ.get("BONSAI_QUIET"):
        print(
            f"  swept {len(missing):,} unplaced turn(s) into {len(runs)} thread(s) "
            f"({len(claimed) - len(missing):,}/{len(events):,} were placed by the segmenter)",
            file=sys.stderr,
        )
    return out


def _keep(name: str, text: str) -> Path:
    """Save a model reply to ~/.bonsai/replies/ so a parse failure doesn't lose what we paid for.

    The filename carries a timestamp because these are receipts for money already spent. An
    earlier version wrote to one name per session, and a second attempt overwrote a usable first
    reply that we then had to pay for again.
    """
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = Path.home() / ".bonsai" / "replies" / f"{name}.{stamp}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return out


def _subunits(unit: dict, unit_index: int) -> list[dict]:
    """The threads inside one unit of the model's reply.

    On a long transcript the model sometimes forgets to nest and puts `turns` straight on the
    unit. We accept that shape as a unit holding one thread, rather than throwing away a
    segmentation that covers every turn correctly over a formatting slip.
    """
    if unit.get("subunits"):
        return unit["subunits"]
    if "turns" in unit:
        return [
            {"label": unit["label"], "synopsis": unit.get("synopsis", ""), "turns": unit["turns"]}
        ]
    # Neither shape. Skip this unit rather than throw away a reply we have paid for: whatever
    # turns it should have held are swept into an unsorted thread, so nothing is lost.
    if not os.environ.get("BONSAI_QUIET"):
        print(
            f"  skipped unit {unit_index}: no 'subunits' or 'turns'; keys were {sorted(unit)}",
            file=sys.stderr,
        )
    return []


def _numbered(events: list[Event]) -> str:
    """The transcript as numbered records for the model to sort.

    Deliberately not markdown dialogue. An earlier version wrote each turn as
    `### turn 0 (user)` followed by the text, which is exactly what a conversation looks like --
    and when handed 475,000 tokens of it, the model stopped segmenting and started *continuing*
    the conversation instead. Three runs in a row failed that way, one returning 24 characters.
    Tagged records read as data to process rather than a script to carry on with.
    """
    return "\n".join(
        f'<turn index="{index}" role="{event.role}">{event.content}</turn>'
        for index, event in enumerate(events)
    )


def _json(text: str) -> list[dict]:
    """Parse the model's reply: the first complete JSON value, ignoring anything after it.

    Tolerates a ```json fence, and tolerates trailing text. The trailing case is real: on a
    2,753-turn transcript the model emitted the whole correct array and then kept going, writing
    a hallucinated "### turn 2753" as if continuing the conversation. `json.loads` rejects that
    outright, which would throw away a segmentation we had already paid for. `raw_decode` reads
    one value and reports where it stopped, so the array survives and the noise is dropped.

    An array that never closes still raises, because that is a genuinely incomplete reply.
    """
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    value, _ = json.JSONDecoder(strict=False).raw_decode(text)
    # A JSON schema's root has to be an object, so structured output arrives as {"units": [...]}.
    # A bare array is the older shape and still parses, which keeps saved replies readable.
    if isinstance(value, dict):
        return value["units"]
    return value


def _expand(turns: list, turn_count: int) -> list[int]:
    """Turn one thread's turn list into plain indices.

    `"40-96"` means every turn from 40 to 96 inclusive, and a bare `97` means that one turn.
    Both forms can appear in the same list.

    Runs are what make segmenting a long conversation affordable. A thread covering 60
    consecutive turns is six characters to write instead of sixty numbers — and sixty fewer
    chances for the model to miscount. On a 2,886-turn transcript, writing out every index is
    roughly 60% of the entire reply.

    Bounds are checked here rather than later at `events[index]`, so a bad number explains
    itself instead of surfacing as a bare IndexError somewhere else.

    A thread that is one unbroken run often comes back as `"2719-2752"` rather than
    `["2719-2752"]`, which is a fair reading of the instruction. Iterating that string would
    hand this loop single characters, so a bare run or number is wrapped instead of rejected.
    """
    if turns is None:
        return []
    if isinstance(turns, (str, int)):
        turns = [turns]
    out: list[int] = []
    junk: list = []
    for item in turns:
        text = str(item).strip()
        run = re.fullmatch(r"(\d+)-(\d+)", text)
        if run:
            first, last = int(run[1]), int(run[2])
            if last < first:
                junk.append(item)  # a backwards run says nothing usable; drop it
                continue
            out += range(first, last + 1)
        elif text.isdigit():
            out.append(int(text))
        else:
            junk.append(item)

    inside = [turn for turn in out if 0 <= turn < turn_count]
    junk += [turn for turn in out if turn not in set(inside)]
    if junk and not os.environ.get("BONSAI_QUIET"):
        print(f"  ignored {len(junk)} unreadable turn entr(y/ies): {junk[:5]}", file=sys.stderr)
    return inside


def _check_coverage(turns_seen: list[int], turn_count: int) -> None:
    """Every turn must land in exactly one thread, or we reject the whole segmentation.

    This is the guard that makes "nothing disappears silently" true. If the model dropped a
    turn or filed one twice, we'd rather fail loudly than save a grouping with a hole in it.
    """
    if sorted(turns_seen) == list(range(turn_count)):
        return
    missing = sorted(set(range(turn_count)) - set(turns_seen))
    # With runs, the likeliest mistake is an off-by-one at a boundary, which shows up as a turn
    # landing in two threads rather than going missing. So name both, or the message reads
    # "missing []" and tells you nothing.
    twice = sorted(turn for turn in set(turns_seen) if turns_seen.count(turn) > 1)
    raise ValueError(
        f"segmenter did not cover every turn exactly once; missing {missing}, twice {twice}"
    )


# --- matching threads to a description ----------------------------------------

MATCH_PROMPT = """Pick the threads of a conversation that match what the user described.

THE USER DESCRIBED:
{description}

THE THREADS:
{threads}

Return the ids of every thread that genuinely matches. Judge by what the thread is about, not
by shared words -- "the caching bug" should match a thread about a prompt-cache invalidation
even with no word in common. Include a thread when it clearly belongs and leave it out when it
does not; a wrong id is worse than a missing one, because the user is about to act on this.
If nothing matches, return an empty list."""

MATCH_SCHEMA = {
    "type": "object",
    "properties": {"ids": {"type": "array", "items": {"type": "string"}}},
    "required": ["ids"],
    "additionalProperties": False,
}


def match_threads(description: str, threads: list[Unit], model: str = MODEL) -> list[str]:
    """Thread ids matching a plain-English description, in the order they were given.

    This is what makes steering usable. Pinning by id means reading a couple of hundred labels
    to find three of them, which nobody will do, and steering nobody uses is the same as having
    none. Only labels and synopses are sent, so the call is small and cheap whatever the size of
    the conversation.

    Ids the model invents are discarded rather than trusted -- the caller is about to act on
    this, and acting on a thread that does not exist is worse than matching nothing.
    """
    listing = "\n".join(f"{u.id}: {u.label}\n    {u.synopsis}" for u in threads)
    reply = _text(
        model,
        None,
        MATCH_PROMPT.format(description=description, threads=listing),
        4_000,
        schema=MATCH_SCHEMA,
    )
    wanted = set(_json_object(reply).get("ids", []))
    return [u.id for u in threads if u.id in wanted]


def _json_object(text: str) -> dict:
    """Parse a reply that should be one JSON object."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    value, _ = json.JSONDecoder(strict=False).raw_decode(text)
    return value if isinstance(value, dict) else {}
