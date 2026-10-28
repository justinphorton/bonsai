"""Reading and rewriting Claude Code's own session file.

This is the adapter that makes bonsai work on a real editor session rather than just as a
library. Claude Code keeps each conversation as a JSONL file — one JSON object per line — under
`~/.claude/projects/<folder>/`.

What we worked out about that format (checked against Claude Code 2.1.240):

  - A line is part of the conversation when its `type` is "user" or "assistant" and it isn't
    marked `isSidechain`. Those lines each have a `uuid`.
  - The lines form a linked list: each one's `parentUuid` points at the one before it. On
    resume, Claude Code starts at the last line and follows parentUuid *backwards*.
  - Attachments sit inside the chain just after the user line they belong to.
  - Each content block of an assistant turn is its own line.
  - Lines with no uuid are bookkeeping and never reach the model.

That linked list is the thing to be careful about: a line pointing at a uuid that no longer
exists silently cuts off everything before it. Every byte is still in the file, but the
conversation is gone as far as the editor is concerned. This cost a real conversation once.

A compacted file keeps every surviving original line byte-for-byte (apart from a repaired
parentUuid) and adds user lines carrying a top-level "bonsai" key: one header explaining the
conventions, and one placeholder per collapsed thread. Each placeholder cites a line range in
`~/.bonsai/<session>.md`, a companion file holding the full readable transcript by thread, so
the original text is reachable with ordinary file tools.
"""

import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from bonsai import compact
from bonsai.models import Event, Unit
from bonsai.session import HOW_TO_READ, Session

BONSAI_DIR = Path.home() / ".bonsai"

# What the model reads at the top of a compacted conversation. Shares HOW_TO_READ with the
# library so both can't drift, then adds the bit that's specific to a file on disk.
HEADER = (
    "[bonsai] "
    + HOW_TO_READ
    + "\nA collapsed thread ends with `full text: <file> lines A-B`; read that range with your "
    "own file tools when you need the original. Don't reopen threads to browse."
)


@dataclass
class Line:
    """One line of the session file: the original text, and the parsed JSON.

    We keep `raw` so that any line we don't touch is written back exactly as it came in. No
    reformatting, no key reordering, no risk of changing something we didn't understand.
    """

    raw: str
    obj: dict

    @property
    def uuid(self) -> str | None:
        return self.obj.get("uuid")

    @property
    def is_conversation(self) -> bool:
        """True for real user/assistant turns, false for bookkeeping and sidechains."""
        return self.obj.get("type") in ("user", "assistant") and not self.obj.get("isSidechain")

    @property
    def bonsai(self) -> dict | None:
        """The tag we add to lines we wrote ourselves, so we can recognize them later."""
        return self.obj.get("bonsai")


def read(path: Path) -> list[Line]:
    """Load a session file, skipping blank lines."""
    return [
        Line(raw, json.loads(raw))
        for raw in path.read_text(encoding="utf-8").splitlines()
        if raw.strip()
    ]


def read_only() -> bool:
    """True when BONSAI_READONLY is set to anything non-empty."""
    return bool(os.environ.get("BONSAI_READONLY"))


def write(path: Path, lines: list[Line]) -> None:
    """Write a session file back out, one JSON object per line.

    This is the only function in bonsai that writes a file it does not own — everything else
    lives under ~/.bonsai/. That makes it the single point where bonsai could collide with
    another tool managing the same conversation, so it is also the single point to switch off.

    Set BONSAI_READONLY and this refuses. Nothing else in bonsai changes: you can still index,
    search, plan, steer and recall. Only the rewrite stops, which is the whole of bonsai's
    footprint outside its own directory.
    """
    if read_only():
        raise PermissionError(
            f"BONSAI_READONLY is set, so {path.name} was not written. Everything except "
            "compaction still works; unset it to allow the rewrite."
        )
    path.write_text("\n".join(line.raw for line in lines) + "\n", encoding="utf-8")


def session_id(lines: list[Line]) -> str:
    """The conversation's id, taken from the first line that carries one."""
    return next(line.obj["sessionId"] for line in lines if "sessionId" in line.obj)


def content_text(obj: dict) -> str:
    """Flatten one message's content blocks into plain text.

    Tool calls and results are kept (labelled, so they're recognizable) because they're part of
    what happened. Thinking blocks are dropped: they belong to the conversation that produced
    them and aren't ours to replay.
    """
    content = obj["message"]["content"]
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        kind = block.get("type")
        if kind == "text":
            parts.append(block["text"])
        elif kind == "tool_use":
            arguments = json.dumps(block["input"], ensure_ascii=False)
            parts.append(f"[tool_use {block['name']}] {arguments}")
        elif kind == "tool_result":
            inner = block.get("content", "")
            if isinstance(inner, list):
                inner = "\n".join(
                    part.get("text", "") for part in inner if part.get("type") == "text"
                )
            parts.append(f"[tool_result] {inner}")
        elif kind != "thinking":
            parts.append(f"[{kind}]")
    return "\n".join(part for part in parts if part)


def event_type(obj: dict) -> str:
    """Classify a line as "message", "tool_call" or "tool_result".

    The point of the distinction is cost. In a coding session, a tool call's input holds whole
    file bodies and a result holds whole file reads — together about two thirds of the
    transcript — and all of it can be read back off disk. A summary of a file you still have is
    worth less than a pointer to it, so tool traffic gets truncated before the model reads it
    (see `session.elide_tools`). "message" is anything a person or the model actually said.
    """
    content = obj["message"]["content"]
    if isinstance(content, str):
        return "message"
    kinds = {block.get("type") for block in content}
    if "text" in kinds:
        return "message"
    if "tool_result" in kinds:
        return "tool_result"
    return "tool_call" if "tool_use" in kinds else "message"


def import_events(session: Session, lines: list[Line], workers: int = 16) -> int:
    """Copy every conversation line we don't already have into the event log, in file order.

    Token counts are exact rather than estimated, which means one API call per message. On a
    long coding transcript that's thousands of calls — about twelve minutes of pure waiting for
    requests that are free and completely independent of each other. So we count them all in a
    thread pool first and hand the numbers to `append`, which brought a real 2,886-turn import
    down from 5.2 minutes to 18 seconds.

    This and the summary generation in session.py are the only two places bonsai uses threads.
    """
    known = {event.id for event in session.events()}
    pending = [
        (line, content_text(line.obj))
        for line in lines
        if line.is_conversation and not line.bonsai and line.uuid not in known
    ]
    pending = [(line, text) for line, text in pending if text.strip()]
    if not pending:
        return 0

    def count_or_estimate(text: str) -> int:
        """Exact if the API answers, a rough estimate if it does not.

        This is thousands of parallel calls on a long transcript, so it is the likeliest place
        in bonsai to meet a rate limit. Losing the whole import over one of them is far worse
        than one event's size being approximate: four characters per token is close enough that
        the budget stays meaningful, and the alternative is no budget at all.
        """
        try:
            return session.count_tokens(text)
        except Exception:  # noqa: BLE001 - any API failure falls back to the estimate
            return max(1, len(text) // 4)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        counts = list(pool.map(count_or_estimate, [text for _, text in pending]))

    for (line, text), tokens in zip(pending, counts, strict=True):
        session.append(line.obj["type"], text, event_type(line.obj), id=line.uuid, tokens=tokens)
    return len(pending)


def collapsed(lines: list[Line]) -> dict[str, str]:
    """Which threads are currently collapsed in the file: thread id -> "trace"."""
    return {
        line.bonsai["unit_id"]: line.bonsai["kind"]
        for line in lines
        if line.bonsai and line.bonsai.get("kind") == "trace"
    }


# --- the companion file ---------------------------------------------------------


def write_companion(session: Session, units: list[Unit], events: list[Event]) -> tuple[Path, dict]:
    """Write `~/.bonsai/<session>.md`: the whole transcript, organized by thread.

    This is what makes a collapsed thread recoverable without bonsai being installed — the
    placeholder in the transcript cites a line range in this file, and any file-reading tool
    can fetch it.

    Returns the path and a mapping of thread id -> (first line, last line).
    """
    events_by_id = {event.id: event for event in events}
    lines = [f"# {session.session_id}", ""]
    ranges: dict[str, tuple[int, int]] = {}

    # Threads first: they hold the actual messages.
    for unit in units:
        is_parent = any(child.parent_id == unit.id for child in units)
        if is_parent:
            continue  # a parent's range is worked out from its children, below
        start = len(lines) + 1
        lines += [f"## {unit.id} · {unit.label}", unit.synopsis, ""]
        members = sorted((events_by_id[eid] for eid in unit.event_ids), key=lambda e: e.seq)
        for event in members:
            lines += [f"**{event.role}:** {event.content}", ""]
        ranges[unit.id] = (start, len(lines))

    # Now each parent unit spans all of its children.
    for unit in units:
        children = [child.id for child in units if child.parent_id == unit.id]
        if children:
            ranges[unit.id] = (
                min(ranges[cid][0] for cid in children),
                max(ranges[cid][1] for cid in children),
            )

    BONSAI_DIR.mkdir(parents=True, exist_ok=True)
    path = BONSAI_DIR / f"{session.session_id}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path, ranges


# --- the rewrite ----------------------------------------------------------------


def _user_line(template: dict, parent: str | None, text: str, tag: dict) -> Line:
    """Build a new user line for the session file.

    `template` is a real user line from the same file; we copy its environment keys so our
    inserted lines look native to Claude Code rather than hand-made.
    """
    obj = {
        "parentUuid": parent,
        "isSidechain": False,
        "type": "user",
        "message": {"role": "user", "content": text},
        "uuid": str(uuid.uuid4()),
        "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "bonsai": tag,
    }
    for key in ("userType", "entrypoint", "cwd", "sessionId", "version", "gitBranch"):
        if key in template:
            obj[key] = template[key]
    return Line(json.dumps(obj, ensure_ascii=False), obj)


def rechain(arrived: list[Line], after: list[Line]) -> list[Line]:
    """Attach lines that appeared during the rewrite onto the end of it.

    Generating summaries takes minutes, and if the session is live Claude Code is appending to
    the file the whole time — including the messages in which you asked for the compaction.
    Those lines point at uuids from *before* the rewrite, and some of those uuids are gone now.

    Since Claude Code reads the conversation by following parentUuid backwards from the last
    line, one dangling pointer makes everything before it unreachable. So any arrived line
    whose parent isn't itself one of the arrived lines gets repointed at the last line we wrote.
    """
    if not arrived:
        return []
    parent = next((line.uuid for line in reversed(after) if line.uuid), None)
    seen_in_arrived: set[str] = set()
    out = []
    for line in arrived:
        needs_repair = line.uuid and parent and line.obj.get("parentUuid") not in seen_in_arrived
        if needs_repair:
            obj = {**line.obj, "parentUuid": parent}
            line = Line(json.dumps(obj, ensure_ascii=False), obj)
        if line.uuid:
            seen_in_arrived.add(line.uuid)
            parent = line.uuid
        out.append(line)
    return out


def rewrite(
    lines: list[Line], session: Session, units: list[Unit], plan: compact.Plan
) -> list[Line]:
    """Build the compacted session file.

    We walk the original lines in order. A thread kept in full keeps its own lines untouched. A
    collapsed thread becomes one placeholder line, inserted where its first line used to be.
    A thread that was collapsed before and is coming back gets its text restored.
    """
    events = session.events()
    events_by_id = {event.id: event for event in events}
    rendered = compact.blocks(units, events, plan.detail)
    block_for_event = {eid: block for block in rendered for eid in block.event_ids}
    companion, ranges = write_companion(session, units, events)

    def stands_for(line: Line) -> list[str]:
        """Which events this line represents — usually itself, but a placeholder stands for many."""
        if line.bonsai and line.bonsai.get("unit_id"):
            old_unit = session.store.unit(line.bonsai["unit_id"])
            return old_unit.event_ids if old_unit else []
        if line.is_conversation and line.uuid in events_by_id:
            return [line.uuid]
        return []

    represents = [stands_for(line) for line in lines]
    template = next(
        line.obj for line in lines if line.obj.get("type") == "user" and not line.bonsai
    )

    out: list[Line] = []
    parent: str | None = None
    already_emitted: set[int] = set()
    # True while we're inside a span being kept verbatim. Attachments and empty content blocks
    # have no events of their own, so they ride along with whatever span they sit in.
    inside_full_span = False

    def emit(line: Line) -> None:
        """Append a line, repairing its parentUuid so the chain stays unbroken."""
        nonlocal parent
        if line.uuid:
            if line.obj.get("parentUuid") != parent:
                line.obj["parentUuid"] = parent
                line = Line(json.dumps(line.obj, ensure_ascii=False), line.obj)
            parent = line.uuid
        out.append(line)

    if any(block.kind != "full" for block in rendered):
        emit(_user_line(template, None, HEADER, {"kind": "header"}))

    for line, event_ids in zip(lines, represents, strict=True):
        if line.bonsai and line.bonsai.get("kind") == "header":
            continue  # a fresh one was just written above

        if not event_ids:
            if line.uuid is None or inside_full_span:
                emit(line)  # bookkeeping always; attachments only inside a verbatim span
            continue

        # One line can touch several blocks. Keyed by id() to dedupe while keeping order.
        blocks_here = {id(block_for_event[eid]): block_for_event[eid] for eid in event_ids}
        for block in blocks_here.values():
            if block.kind == "full":
                if not line.bonsai:
                    emit(line)
                else:
                    # A thread that was collapsed before and is now expanding again: write back
                    # the text the old placeholder stood for.
                    ordered = sorted(event_ids, key=lambda eid: events_by_id[eid].seq)
                    text = "\n\n".join(
                        f"{events_by_id[eid].role}: {events_by_id[eid].content}" for eid in ordered
                    )
                    tag = {"kind": "restored", "unit_id": block.unit.id}
                    emit(_user_line(template, parent, text, tag))
                inside_full_span = not line.bonsai
            elif id(block) not in already_emitted:
                first_line, last_line = ranges[block.unit.id]
                text = f"{block.text}\nfull text: {companion} lines {first_line}-{last_line}"
                tag = {"kind": block.kind, "unit_id": block.unit.id}
                emit(_user_line(template, parent, text, tag))
                already_emitted.add(id(block))
                inside_full_span = False
            else:
                inside_full_span = False

    # The last real line of the conversation always survives, whatever the plan said.
    #
    # Claude Code keeps appending after a compaction, and the next thing it writes records
    # whichever uuid was last in the file. Collapse that line into a placeholder and its uuid is
    # gone — so the next message points at nothing and every message before it becomes
    # unreachable. This is exactly how a real conversation got lost once.
    #
    # Usually the plan keeps it anyway, because the thread holding it is the "current" one, but
    # an explicit prune outranks that. Hence the belt and braces.
    final = next(
        (line for line in reversed(lines) if line.is_conversation and not line.bonsai), None
    )
    if final is not None and final.uuid not in {line.uuid for line in out if line.uuid}:
        emit(final)
    return out
