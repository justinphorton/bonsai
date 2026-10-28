"""The main thing you use. Holds a conversation together and ties the other modules to it.

    s = Session.open("my-project")
    s.append("user", "Postgres 15, because reconciliation needs transactional joins.")
    plan = s.plan(budget=50_000)           # what would be kept, and why. Generates nothing.
    ctx = s.build_context(budget=50_000)   # messages that fit, ready to send
    s.recall(unit_id)                      # the original text of any topic
    s.pin(event_id) / s.prune(event_id) / s.level(event_id, 0.5)

For a Claude Code session, transcript.py loads the events out of the editor's own log file
instead of you calling append() yourself.
"""

import uuid
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from bonsai import compact, llm
from bonsai.models import Annotation, Context, Event, Mark, Unit
from bonsai.store import Store

# Instructions we give the model for reading a context bonsai built. Without this it sees turn
# numbers on every collapsed thread and has no idea what they mean.
#
# This constant is shared by the Claude Code header and by any program using the library, so
# the two can't drift apart and start telling the model different things.
HOW_TO_READ = (
    "Parts of this conversation are collapsed. A line starting with [thread: <label> ...] stands "
    "for messages replaced by that label and a short synopsis. Everything else is verbatim.\n"
    "Each thread line shows the turns it covers. Threads are grouped by topic, not by time, so "
    "two can describe the same thing at different points — where they disagree, the one whose "
    "turns run later is current, and a synopsis that gives two values with turn numbers ends with "
    "the current one.\n"
    "Every thread is listed, so if nothing relates to what you need it is probably not in this "
    "conversation — but recall the closest thread before concluding that."
)

# The tool definition we hand the model so it can reopen a collapsed thread itself. This is what
# makes a trace a pointer rather than a loss.
RECALL_TOOL = {
    "name": "recall",
    "description": (
        "Reopen a collapsed thread of this conversation and get its original messages back. "
        "Collapsed threads appear as [thread: <label> · <N> tok · id:<id>] followed by a short "
        "synopsis. Call this with that id when the synopsis isn't enough — an exact value, some "
        "code, or the reasoning behind a decision. Don't recall just to browse."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"unit_id": {"type": "string"}},
        "required": ["unit_id"],
    },
}

# A function that measures text, e.g. llm.count.
CountTokens = Callable[[str], int]


# Turns that arrived after the last time we split the conversation get gathered into this
# stand-in thread. It's computed on the fly and never saved, so it has no generation number —
# which means `recall` and `level` don't apply to it. It disappears into the real threads the
# next time you run `segment`.
TAIL_ID = "new"
TAIL_LABEL = "since the last segmentation"
TAIL_SYNOPSIS = "Turns added after the conversation was last split into threads."


def new_id() -> str:
    """A short random id for an event or annotation."""
    return uuid.uuid4().hex[:12]


def context_text(messages: list[dict]) -> str:
    """A built context flattened to one string, so we can measure it against the budget.

    Only the message contents, no role labels, because the budget is counted in the same terms
    the allocator uses and that's message text.
    """
    return "\n\n".join(message["content"] for message in messages)


# How many characters of tool traffic the segmenter is allowed to read.
#
# A tool *call* is always reproducible — for a file write, the call input literally is the file
# on disk — so it's capped hard at 200. A tool *result* sometimes isn't reproducible (a test
# run, a timing, an API response), and 2,000 characters covers about 90% of real results, so
# only genuine file dumps get cut.
TOOL_HEAD = {"tool_call": 200, "tool_result": 2_000}


def elide_tools(events: list[Event], head: dict[str, int] | None = None) -> list[Event]:
    """Copies of the events with tool calls and results truncated.

    This only changes what the segmenter *reads*. The database still holds every
    event whole, so `recall` and any thread kept verbatim are unaffected.

    Why bother: in a coding session, two thirds of the transcript is tool traffic whose bodies
    are file contents that are still sitting on disk. Paying a model to compress those is worse
    than useless — it also crowds out the labels and synopses that make a collapsed thread
    usable in the first place. This roughly halves the cost of splitting a coding session.

    The `tokens` field on the returned copies is deliberately left alone, since nothing that
    allocates a budget reads these shortened copies.
    """
    caps = TOOL_HEAD if head is None else head
    out = []
    for event in events:
        cap = caps.get(event.type)
        if not cap or len(event.content) <= cap:
            out.append(event)
            continue
        if _looks_like_an_error(event.content):
            # Keep both ends. A traceback puts the stack first and the actual error on the LAST
            # line, so cutting the tail throws away the only part worth keeping — and "we tried
            # X and it failed with Y" is exactly what stops an agent retrying X.
            keep_end = cap // 2
            cut = len(event.content) - cap - keep_end
            shortened = (
                f"{event.content[:cap]}\n[... {cut} more characters ...]\n"
                f"{event.content[-keep_end:]}"
            )
        else:
            cut = len(event.content) - cap
            shortened = f"{event.content[:cap]}\n[... {cut} more characters]"
        out.append(replace(event, content=shortened))
    return out


# Markers that say a tool result is a failure rather than output. Deliberately a short, boring
# list: a false positive just means one result keeps its last thousand characters.
ERROR_MARKERS = (
    "Traceback (most recent call last)",
    "Error:",
    "error:",
    "ERROR",
    "Exception",
    "FAILED",
    "failed with exit code",
    "assert",
)


def _looks_like_an_error(content: str) -> bool:
    return any(marker in content for marker in ERROR_MARKERS)


def unit_text(unit: Unit, events: list[Event]) -> str:
    """One topic's messages as `role: content` blocks. This is what `recall` gives back."""
    events_by_id = {event.id: event for event in events}
    members = sorted((events_by_id[eid] for eid in unit.event_ids), key=lambda e: e.seq)
    return "\n\n".join(f"{event.role}: {event.content}" for event in members)


class Session:
    """One conversation: its event log, its topics, and the operations on them."""

    def __init__(
        self,
        session_id: str,
        store: Store,
        segmenter,
        count_tokens: CountTokens,
        model: str = llm.MODEL,
    ) -> None:
        """Build a Session from its parts.

        The store, segmenter and token counter are handed in rather than created here. That one
        choice is why the whole test suite runs offline with no API key: the tests pass a fake
        segmenter that chunks events mechanically, and `len` as the token counter.
        `Session.open()` below wires up the real ones for normal use.

        `segmenter` just needs a `.segment(events, existing)` method returning a list of Units.
        """
        self.session_id = session_id
        self.store = store
        self.segmenter = segmenter
        self.count_tokens = self._caching(count_tokens)
        self.model = model

    def _caching(self, count_tokens: CountTokens) -> CountTokens:
        """Wrap a token counter so each distinct piece of text is only ever measured once."""

        def counted(text: str) -> int:
            hit = self.store.cached_count(text)
            if hit is not None:
                return hit
            tokens = count_tokens(text)
            self.store.put_count(text, tokens)
            return tokens

        return counted

    @classmethod
    def open(
        cls,
        session_id: str,
        db: Path | None = None,
        model: str = llm.MODEL,
        count_tokens: CountTokens | None = None,
    ) -> "Session":
        """The normal way to get a Session: real database, real model calls."""
        count = count_tokens or llm.count
        return cls(session_id, Store(db), llm.LLMSegmenter(model), count, model)

    # --- the event log --------------------------------------------------------

    def events(self) -> list[Event]:
        """Every message in this conversation, oldest first."""
        return self.store.events(self.session_id)

    def append(
        self,
        role: str,
        content: str,
        type: str = "message",
        id: str | None = None,
        tokens: int | None = None,
    ) -> Event:
        """Add one message to the log.

        `id` lets the Claude Code adapter reuse the editor's own uuids instead of minting new
        ones, which is what makes the two logs line up. `tokens` lets a caller that already
        counted the text pass the number in, rather than paying for another API round trip.
        """
        existing = self.events()
        event = Event(
            id=id or new_id(),
            session_id=self.session_id,
            seq=existing[-1].seq + 1 if existing else 0,
            role=role,
            type=type,
            content=content,
            tokens=self.count_tokens(content) if tokens is None else tokens,
            created_at=datetime.now(UTC).isoformat(),
        )
        self.store.add_event(event)
        return event

    # --- topics ---------------------------------------------------------------

    def units(self) -> list[Unit]:
        """The saved topics, plus one stand-in for anything said since they were made.

        New messages deliberately do **not** trigger a re-split. Splitting a transcript is one
        model call over the whole thing — minutes of waiting and about a dollar on a long
        session — so re-running it every time a message arrived would make bonsai unusable.

        Instead, newer turns are gathered into a thread of their own and kept verbatim, which
        is what you want for recent work anyway. Run `segment` when you want them folded into
        the real topics.

        This never segments. It used to, the first time it was called on an unsplit
        conversation, which meant `bonsai status` -- a command that reads as free -- could start
        a model call costing minutes and real money without asking. Splitting is the one
        expensive thing bonsai does, so `segment` is the only thing that does it. Before that,
        every turn simply shows up in the stand-in thread as not yet indexed.
        """
        events = self.events()
        units = self._apply_marks(self.store.units(self.session_id))
        already_grouped = {eid for unit in units for eid in unit.event_ids}
        new_events = [event.id for event in events if event.id not in already_grouped]
        if new_events:
            tail = Unit(TAIL_ID, self.session_id, TAIL_LABEL, new_events, TAIL_SYNOPSIS)
            units = [*units, tail]
        return units

    def _apply_marks(self, units: list[Unit]) -> list[Unit]:
        """Overlay your own labels and notes onto the threads the segmenter produced.

        Done here so the rest of bonsai never has to know marks exist: `compact` renders a Unit,
        and by the time it sees one, your label has already replaced the model's and your notes
        are on it. A thread with several marked messages shows every note, newest last.
        """
        marks = self.store.marks(self.session_id)
        if not marks:
            return units
        by_event: dict[str, list[Mark]] = {}
        for mark in marks:
            by_event.setdefault(mark.event_id, []).append(mark)

        for unit in units:
            notes, label = [], None
            for event_id in unit.event_ids:
                for mark in by_event.get(event_id, []):
                    if mark.kind == "label":
                        label = mark.text  # a later label wins over an earlier one
                    else:
                        notes.append(mark.text)
            if label:
                unit.label = label
            if notes:
                unit.note = "\n".join(notes)
        return units

    def segment(self, by: str | None = None, n_units: int = 6, n_sub: str = "3-10") -> list[Unit]:
        """Split the conversation into topics from scratch.

        `by` is your own instruction for how to group it ("the eval work vs the library work").
        `n_units` is roughly how many top-level units to aim for, and `n_sub` how many threads
        inside each, as a range like "3-10".

        With no arguments this uses the segmenter the Session was built with, so a caller that
        injected one — the tests, or anything embedding bonsai — actually gets it. Only asking
        for a custom grouping builds a configured LLM segmenter, since those settings are what
        it exists to carry.
        """
        asked_for_custom = by is not None or n_units != 6 or n_sub != "3-10"
        segmenter = (
            llm.LLMSegmenter(self.model, by, n_units, n_sub) if asked_for_custom else self.segmenter
        )
        topics = segmenter.segment(elide_tools(self.events()))
        return self.store.replace_units(self.session_id, topics)

    # --- compaction -----------------------------------------------------------

    def plan(self, budget: int) -> compact.Plan:
        """What compacting to `budget` would keep, and why. Costs nothing and changes nothing."""
        units = self.units()
        fixed = compact.overrides(units, self.store.annotations(self.session_id))
        return compact.allocate(units, self.events(), budget, self.count_tokens, fixed)

    def build_context(self, budget: int) -> Context:
        """The messages to send, guaranteed to fit the budget.

        No model calls and no I/O beyond reading the log, so this is free and instant. It is
        also exact rather than approximate: with only full text and traces, every piece has a
        known size, so what `allocate` computed is what gets built. The old version had to
        measure the finished context and demote overlong summaries until it fit.
        """
        plan = self.plan(budget)
        messages = compact.build_context(self.units(), self.events(), plan.detail)
        return Context(messages=messages, detail=plan.detail)

    # --- recall and the knobs -------------------------------------------------

    def recall(self, unit_id: str) -> str:
        """The original text of any topic, from any grouping, word for word."""
        unit = self.store.unit(unit_id)
        if unit is None:
            raise KeyError(f"no unit with id {unit_id!r}")
        return unit_text(unit, self.events())

    def level(self, event_id: str, ratio: float) -> None:
        """Fix how this message's thread is shown: 1 keeps it verbatim, 0 collapses it."""
        self.store.add_annotation(
            Annotation(id=new_id(), session_id=self.session_id, event_id=event_id, level=ratio)
        )

    # --- reshaping the index --------------------------------------------------

    def _structure(self) -> list[tuple[Unit, list[Unit]]]:
        """The current threads as (unit, its threads) pairs, newest grouping only."""
        units = self.store.units(self.session_id)
        parents = [u for u in units if not u.parent_id]
        return [(p, [u for u in units if u.parent_id == p.id]) for p in parents]

    def _write_structure(self, groups: list[tuple[Unit, list[Unit]]]) -> list[Unit]:
        """Save a reshaped grouping as the next generation. No model call, so this is free.

        Units are never edited in place, so merging and splitting write a new generation the
        same way `segment` does. The old ids keep resolving, which is what lets a thread id the
        model is still holding carry on working after you reorganise.
        """
        local: list[Unit] = []
        for k, (parent, kids) in enumerate(g for g in groups if g[1]):
            local.append(Unit(f"u{k}", self.session_id, parent.label, [], parent.synopsis))
            for j, kid in enumerate(kids):
                local.append(
                    Unit(
                        f"u{k}.{j}",
                        self.session_id,
                        kid.label,
                        list(kid.event_ids),
                        kid.synopsis,
                        f"u{k}",
                    )
                )
        return self.store.replace_units(self.session_id, local)

    def _current_thread(self, unit_id: str) -> Unit:
        """A thread from the grouping in force now. Reshaping an older one is meaningless."""
        for _, kids in self._structure():
            for kid in kids:
                if kid.id == unit_id:
                    return kid
        raise KeyError(f"no current thread with id {unit_id!r}")

    def merge(self, unit_ids: list[str], label: str | None = None) -> list[Unit]:
        """Join two or more threads into one. Free, instant, and nothing is lost.

        The segmenter splits things you think of as one topic more often than it lumps separate
        ones together, so this is the common correction. The joined thread sits where the first
        of its parts was, and its messages are put back in the order they happened.
        """
        wanted = list(dict.fromkeys(unit_ids))
        if len(wanted) < 2:
            raise ValueError("merge needs at least two different threads")
        targets = [self._current_thread(uid) for uid in wanted]

        seq = {e.id: e.seq for e in self.events()}
        joined = sorted({eid for u in targets for eid in u.event_ids}, key=lambda e: seq[e])
        new_label = label or " + ".join(u.label for u in targets)
        new_synopsis = " ".join(u.synopsis for u in targets if u.synopsis)

        taken = {u.id for u in targets}
        groups, placed = [], False
        for parent, kids in self._structure():
            keep = []
            for kid in kids:
                if kid.id not in taken:
                    keep.append(kid)
                elif not placed:
                    keep.append(
                        Unit("merged", self.session_id, new_label, joined, new_synopsis, parent.id)
                    )
                    placed = True
            groups.append((parent, keep))
        return self._write_structure(groups)

    def split(self, unit_id: str, at_turn: int) -> list[Unit]:
        """Cut a thread in two at a turn: everything before it, and everything from it on.

        Refuses to leave an empty half, because a thread with no messages has nothing to render
        and nothing to recall.
        """
        target = self._current_thread(unit_id)
        seq = {e.id: e.seq for e in self.events()}
        before = [e for e in target.event_ids if seq[e] < at_turn]
        after = [e for e in target.event_ids if seq[e] >= at_turn]
        if not before or not after:
            covered = sorted(seq[e] for e in target.event_ids)
            raise ValueError(
                f"splitting at turn {at_turn} leaves one half empty; "
                f"this thread covers turns {covered[0]}-{covered[-1]}"
            )

        groups = []
        for parent, kids in self._structure():
            keep = []
            for kid in kids:
                if kid.id != unit_id:
                    keep.append(kid)
                    continue
                keep.append(
                    Unit("a", self.session_id, f"{kid.label} (1)", before, kid.synopsis, parent.id)
                )
                keep.append(
                    Unit("b", self.session_id, f"{kid.label} (2)", after, kid.synopsis, parent.id)
                )
            groups.append((parent, keep))
        return self._write_structure(groups)

    def note(self, unit_id: str, text: str) -> Unit:
        """Attach your own note to a thread. It renders with the thread's trace.

        This is the difference between a hint and a guarantee: `/compact "keep X"` asks the
        summarizer to try, whereas a note is text you wrote that goes in verbatim and stays
        there however tightly the conversation is compacted.
        """
        return self._mark(unit_id, "note", text)

    def relabel(self, unit_id: str, text: str) -> Unit:
        """Replace a thread's label with your own wording.

        Fixing one bad label used to mean paying for a whole new segmentation.
        """
        return self._mark(unit_id, "label", text, replace_existing=True)

    def _mark(self, unit_id: str, kind: str, text: str, replace_existing: bool = False) -> Unit:
        unit = self.store.unit(unit_id)
        if unit is None:
            raise KeyError(f"no thread with id {unit_id!r}")
        if not unit.event_ids:
            raise ValueError(f"thread {unit_id} holds no messages to anchor a {kind} to")
        if replace_existing:
            self.store.clear_marks(self.session_id, unit.event_ids, kind)
        # Anchored on the thread's first message: re-splitting moves it to whichever thread
        # that message lands in, which is the nearest thing to "this topic" that survives.
        self.store.add_mark(
            Mark(
                id=new_id(),
                session_id=self.session_id,
                event_id=unit.event_ids[0],
                kind=kind,
                text=text,
                created_at=datetime.now(UTC).isoformat(),
            )
        )
        return unit

    def steering(self) -> dict[str, str]:
        """Thread id -> "keep" or "drop", for every choice you have made that still applies.

        Resolved against the current threads, so after a re-segmentation this shows where your
        choices actually landed rather than the ids you originally typed.
        """
        fixed = compact.overrides(self.units(), self.store.annotations(self.session_id))
        return {tid: ("keep" if level >= 1 else "drop") for tid, level in fixed.items()}

    def clear_steering(self) -> int:
        """Forget every keep and drop on this conversation. Returns how many were cleared."""
        return self.store.clear_annotations(self.session_id)

    def pin(self, event_id: str) -> None:
        """Always keep this message's thread verbatim."""
        self.level(event_id, 1.0)

    def prune(self, event_id: str) -> None:
        """Always collapse this message's thread to its label and synopsis."""
        self.level(event_id, 0.0)

    # --- letting the model reopen things itself -------------------------------

    def recall_tool(self) -> dict:
        """The tool definition to pass in `tools=[...]` so the model can recall for itself."""
        return RECALL_TOOL

    def tool_result(self, tool_use) -> dict:
        """Answer a `recall` call from an API response with the tool_result block it expects."""
        try:
            content = self.recall(tool_use.input["unit_id"])
        except KeyError as e:
            return {
                "type": "tool_result",
                "tool_use_id": tool_use.id,
                "content": str(e),
                "is_error": True,
            }
        return {"type": "tool_result", "tool_use_id": tool_use.id, "content": content}
