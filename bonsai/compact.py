"""Decide which threads the model sees in full, and render them into messages.

Everything here is a pure function: it reads data, returns data, and never touches the database
or the network. Since dropping per-thread summaries it makes no model calls at all, which is why
`plan` and `compact` are free, instant, and give the same answer every time.

A thread is kept one of two ways:

  **full** — the original messages, word for word.
  **trace** — the label, the turns it covers, and a short synopsis, plus the id to ask for it
  back. Every thread keeps at least this, always. That is the one guarantee: a topic is never
  dropped silently, so the model can see that something exists even when it cannot read it.

The default policy, when you have not pinned anything:

  1. Everything starts as a trace. That is the floor, and it is what any budget must cover.
  2. The thread holding the newest message is kept full if it fits.
  3. Whatever budget is left is spent keeping more threads full, newest first.

Cost is measured exactly the way `blocks` renders it. That matters more than it sounds: an
earlier version counted a collapsed unit as one trace per thread and came up with 14,200 tokens
for something that renders as about 300, which made bonsai refuse budgets it could easily meet.

Settings you made yourself are applied first and are never argued with.
"""

from collections.abc import Callable
from dataclasses import dataclass

from bonsai.models import Annotation, Detail, Event, Unit

# A function that measures text, e.g. llm.count. It's passed in rather than imported so this
# module stays pure and the tests can hand it `len` instead of calling a paid API.
CountTokens = Callable[[str], int]


class BudgetError(ValueError):
    """Raised when what you pinned yourself can't fit in the budget, whatever we do.

    This is the one case bonsai refuses instead of degrading, because the alternative is
    quietly ignoring an instruction you gave it. `held` names the threads you set a level on,
    so you know which ones to loosen.
    """

    def __init__(self, overage: int, held: list[str]) -> None:
        self.overage, self.held = overage, held
        if held:
            msg = f"pinned threads exceed the budget by {overage} tokens; held: {held}"
        else:
            # No pins, so it is the budget itself: even one line per thread does not fit.
            msg = (
                f"budget is too small by {overage} tokens for even a label and synopsis per "
                "thread; raise --budget"
            )
        super().__init__(msg)


@dataclass
class Plan:
    """What a compaction would do, before anything is generated or written."""

    detail: dict[str, Detail]  # thread id -> "full" / "trace"
    reasons: dict[str, str]  # thread id -> why it got that, for the plan printout
    estimate: int  # tokens the finished context should come to


def leaves(units: list[Unit]) -> list[Unit]:
    """The threads — the units that hold events rather than other units.

    A unit is a "leaf" when nothing lists it as a parent. In practice that's the threads
    inside each unit, plus the rare top-level unit that was never split further.
    """
    parents = {unit.parent_id for unit in units if unit.parent_id}
    return [unit for unit in units if unit.id not in parents]


def overrides(units: list[Unit], annotations: list[Annotation]) -> dict[str, float]:
    """Turn per-message settings into per-thread settings: thread id -> level.

    A thread holds many messages and you might have set levels on several of them, so we take
    the highest. Keeping something outranks dropping it — if you pinned one message in a
    thread, you clearly wanted that text, so we don't let a prune elsewhere in the same thread
    throw it away.
    """
    level_by_event = {a.event_id: a.level for a in annotations}
    out = {}
    for thread in leaves(units):
        levels = [level_by_event[eid] for eid in thread.event_ids if eid in level_by_event]
        if levels:
            out[thread.id] = max(levels)
    return out


def span_text(unit: Unit, events_by_id: dict[str, Event]) -> str:
    """Which turns this thread covers, e.g. "turns 40-96, 298-300".

    Threads are grouped by topic, not by time, so one thread can pick a subject up again
    hundreds of turns later. Without this the model has no way to tell which of two threads
    that disagree is the up-to-date one. The turn numbers come from the event order, so this
    is a fact rather than something the model guessed.
    """
    turns = sorted(events_by_id[eid].seq for eid in unit.event_ids if eid in events_by_id)
    if not turns:
        return ""

    # Collapse consecutive turns into [first, last] pairs: 1,2,3,7 becomes [1,3] and [7,7].
    runs: list[list[int]] = []
    for turn in turns:
        if runs and turn == runs[-1][1] + 1:
            runs[-1][1] = turn
        else:
            runs.append([turn, turn])

    if len(runs) > 3:
        # Too scattered to list without burning more tokens than it's worth; the overall
        # extent is the useful part.
        return f"turns {turns[0]}-{turns[-1]} in {len(runs)} spans"
    return "turns " + ", ".join(f"{a}-{b}" if a != b else str(a) for a, b in runs)


def trace_text(unit: Unit, tokens: int, children: list[Unit] = (), span: str = "") -> str:
    """The placeholder that stands in for a collapsed topic.

    This is the single most important string in the project: when a thread is collapsed, this
    is *all* the model sees of it. It carries the label, the turns covered, the size of what
    was replaced, and the id to ask for if it wants the real thing back.

    A note you wrote goes first, before the model's own synopsis, and is marked as yours. It is
    the one part of this string a model did not write, so it should not be mistaken for one.
    """
    head = f"[thread: {unit.label} · {span + ' · ' if span else ''}{tokens} tok · id:{unit.id}]"
    lines = [head]
    if unit.note:
        lines += [f"  NOTE FROM THE USER: {line}" for line in unit.note.splitlines()]
    if unit.synopsis:
        lines.append(unit.synopsis)
    lines += [f"  - {child.id}: {child.label}" for child in children]
    return "\n".join(lines)


def allocate(
    units: list[Unit],
    events: list[Event],
    budget: int,
    count_tokens: CountTokens,
    fixed: dict[str, float],
) -> Plan:
    """Decide which threads stay verbatim and which collapse to their trace.

    Two levels, no middle gear. That is a deliberate simplification: a per-thread summary meant
    a model call per thread, which could fail, came back a different length every run, and
    routinely overshot its target badly enough that the context had to be measured and trimmed
    afterwards. Dropping it makes this function pure arithmetic — `plan` and `compact` now make
    no model calls at all, so once a conversation is segmented they are free, instant and give
    the same answer every time.

    `fixed` is thread id -> what you asked for: >= 1 keep verbatim, <= 0 collapse.
    """
    events_by_id = {event.id: event for event in events}
    threads = leaves(units)
    if not threads:
        return Plan({}, {}, 0)

    full_size = {t.id: sum(events_by_id[eid].tokens for eid in t.event_ids) for t in threads}
    spans = {t.id: span_text(t, events_by_id) for t in threads}
    trace_size = {
        t.id: count_tokens(trace_text(t, full_size[t.id], span=spans[t.id])) for t in threads
    }
    render_groups = _render_groups(units, threads, events_by_id, count_tokens)

    # Everything starts collapsed -- the cheapest legal state -- and we spend upward from there.
    detail: dict[str, Detail] = {t.id: "trace" for t in threads}
    reasons: dict[str, str] = {}

    def spent() -> int:
        return _measure(render_groups, detail, full_size, trace_size)

    def keep_full_if_it_fits(thread: Unit, why: str) -> bool:
        detail[thread.id], reasons[thread.id] = "full", why
        if spent() <= budget:
            return True
        detail[thread.id] = "trace"
        del reasons[thread.id]
        return False

    # --- 1: what you asked for, exactly as you asked for it -------------------
    for thread in threads:
        if thread.id not in fixed:
            continue
        if fixed[thread.id] >= 1:
            detail[thread.id], reasons[thread.id] = "full", "kept"
        else:
            reasons[thread.id] = "dropped"  # already a trace; just record why

    # Your own choices are the only thing that can make a budget impossible. Everything this
    # function decides on its own degrades to traces instead of failing.
    if spent() > budget:
        raise BudgetError(spent() - budget, [tid for tid in fixed if fixed[tid] >= 1])

    # --- 2: the thread holding the newest message -----------------------------
    newest_turn = max(event.seq for event in events_by_id.values())
    current = next(
        t for t in threads if any(events_by_id[eid].seq == newest_turn for eid in t.event_ids)
    )
    if current.id not in fixed:
        keep_full_if_it_fits(current, "current")

    # --- 3: spend what is left, newest first ----------------------------------
    # Recency is the tie-breaker because this runs on a working session: what you touched last
    # is what you are most likely to still need. Anything you disagree with, you pin.
    def last_turn(thread: Unit) -> int:
        return max(events_by_id[eid].seq for eid in thread.event_ids)

    spare = [t for t in threads if t.id not in fixed and detail[t.id] == "trace"]
    for thread in sorted(spare, key=last_turn, reverse=True):
        keep_full_if_it_fits(thread, "fits")

    return Plan(detail, reasons, spent())


def _render_groups(
    units: list[Unit],
    threads: list[Unit],
    events_by_id: dict[str, Event],
    count_tokens: CountTokens,
) -> list[tuple[list[str], int]]:
    """Which threads print as one block when they all collapse, and what that block costs.

    Returns one entry per group: the thread ids in it, and the token cost of the single
    unit-level trace they'd collapse into. That cost is 0 for a top-level unit with no parent,
    since there's nothing for it to collapse into.

    This is what lets `_measure` charge one line for a dead direction instead of one line per
    thread inside it.
    """
    thread_ids = {t.id for t in threads}
    parents = {u.id: u for u in units if u.id not in thread_ids}
    groups, already_grouped = [], set()
    for thread in threads:
        if thread.id in already_grouped:
            continue
        if thread.parent_id is None:
            groups.append(([thread.id], 0))
            already_grouped.add(thread.id)
            continue
        siblings = [t for t in threads if t.parent_id == thread.parent_id]
        tokens = sum(events_by_id[eid].tokens for sibling in siblings for eid in sibling.event_ids)
        parent = parents[thread.parent_id]
        collapsed = trace_text(parent, tokens, siblings, span_text(parent, events_by_id))
        groups.append(([s.id for s in siblings], count_tokens(collapsed)))
        already_grouped.update(s.id for s in siblings)
    return groups


def _measure(
    render_groups: list[tuple[list[str], int]],
    detail: dict[str, Detail],
    full_size: dict[str, int],
    trace_size: dict[str, int],
) -> int:
    """What `blocks` will actually render, in tokens.

    The special case is the whole point: when every thread in a unit is a trace, the renderer
    prints one line for the unit, so that's what we charge — not one line per thread.
    """
    total = 0
    for thread_ids, collapsed_cost in render_groups:
        if collapsed_cost and all(detail[tid] == "trace" for tid in thread_ids):
            total += collapsed_cost
            continue
        for tid in thread_ids:
            total += full_size[tid] if detail[tid] == "full" else trace_size[tid]
    return total


@dataclass
class Block:
    """One piece of the rendered context: either real messages, or one placeholder."""

    unit: Unit
    kind: Detail
    event_ids: list[str]
    text: str = ""  # empty for "full" blocks, which carry the original messages instead


def blocks(units: list[Unit], events: list[Event], detail: dict[str, Detail]) -> list[Block]:
    """Turn a plan into rendered pieces, in the order they happened.

    The collapse rule lives here: when every thread in a unit is a trace, they're replaced by
    one unit-level trace that lists them by id and label. So an abandoned direction costs one
    block, and the model can still see the names of everything inside it.
    """
    events_by_id = {event.id: event for event in events}
    threads = leaves(units)
    thread_ids = {t.id for t in threads}
    parents = {u.id: u for u in units if u.id not in thread_ids}
    out: list[Block] = []
    done: set[str] = set()

    def first_turn(thread: Unit) -> int:
        return min(events_by_id[eid].seq for eid in thread.event_ids)

    for thread in sorted(threads, key=first_turn):
        if thread.id in done:
            continue

        siblings = (
            [t for t in threads if t.parent_id == thread.parent_id] if thread.parent_id else []
        )
        # A thread `detail` never mentioned falls back to its trace. The trace always exists and
        # keeps the label and the id, so the model can still recall it -- degrading to that is
        # always better than failing to build a context.
        if siblings and all(detail.get(s.id, "trace") == "trace" for s in siblings):
            parent = parents[thread.parent_id]
            event_ids = [eid for sibling in siblings for eid in sibling.event_ids]
            tokens = sum(events_by_id[eid].tokens for eid in event_ids)
            text = trace_text(parent, tokens, siblings, span_text(parent, events_by_id))
            out.append(Block(parent, "trace", event_ids, text))
            done.update(s.id for s in siblings)
            continue

        tokens = sum(events_by_id[eid].tokens for eid in thread.event_ids)
        # A thread `detail` never mentioned falls back to its trace, which always exists.
        if detail.get(thread.id, "trace") == "full":
            out.append(Block(thread, "full", list(thread.event_ids)))
        else:
            text = trace_text(thread, tokens, span=span_text(thread, events_by_id))
            out.append(Block(thread, "trace", list(thread.event_ids), text))
        done.add(thread.id)
    return out


def build_context(units: list[Unit], events: list[Event], detail: dict[str, Detail]) -> list[dict]:
    """The finished `messages=[...]` list to hand to the Anthropic API.

    Full blocks keep each message's original role. Placeholders go in as user messages, because
    that's the only role we can insert text under without pretending the assistant said it.
    """
    events_by_id = {event.id: event for event in events}
    messages = []
    for block in blocks(units, events, detail):
        if block.kind == "full":
            originals = sorted((events_by_id[eid] for eid in block.event_ids), key=lambda e: e.seq)
            for event in originals:
                messages.append({"role": event.role, "content": event.content})
        else:
            messages.append({"role": "user", "content": block.text})
    return messages
