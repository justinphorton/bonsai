"""The data types bonsai passes around.

These are plain dataclasses — just named fields, no methods and no logic. Everything that
actually does something lives in the other modules. This file imports nothing from bonsai, so
nothing here can ever depend on the rest of the package.

Reading order: an Event is one message. A Unit is a group of messages about one topic. An
Annotation is a setting you put on a message, and a Mark is something you wrote yourself.
A Context is what we end up sending to the model.
"""

from dataclasses import dataclass
from typing import Literal

# How much of a topic the model gets to see. Two choices, and deliberately only two:
#   "full"  - the original messages, word for word
#   "trace" - the label, the turns it covers and a short synopsis, plus the id to ask for it
#             back. Every topic keeps at least this, which is the whole idea: the model can
#             always see that a topic exists even when it cannot read the contents.
#
# There used to be a third, "summary", generated per thread to hit a token target. It cost a
# model call per thread that could fail, came back a different length every run, and overshot
# badly enough that the finished context had to be measured and trimmed afterwards. Removing it
# means planning and compacting make no model calls at all.
Detail = Literal["full", "trace"]


@dataclass(frozen=True)
class Event:
    """One message in the conversation.

    `frozen=True` means Python refuses to let you change an Event after you make one. That is
    on purpose: the event log is the record of what actually happened, and if any part of the
    code could quietly edit it we would lose the ability to recover the original text.
    """

    id: str
    session_id: str
    seq: int  # position in the conversation, starting at 0 — this is what "turn 42" means
    role: Literal["user", "assistant", "system"]
    # What kind of message it is. Tool calls and their results get capped before the model
    # summarizes them, because a 50,000-character file dump is not worth summarizing.
    type: Literal["message", "tool_call", "tool_result", "file_read"]
    content: str
    tokens: int  # measured with the API's own counter, never estimated
    created_at: str


@dataclass
class Unit:
    """A group of events that belong together — one topic.

    Units nest exactly two levels deep, and the two levels have different jobs:

      - A unit with no parent is a broad area of the conversation ("the benchmark work").
      - Its children, which we call threads, are the specific topics inside it ("chose
        LongMemEval over LoCoMo"). These are the ones that hold the actual events.

    Two levels is enough to be useful and shallow enough to stay readable. The payoff is that
    when every thread in a unit collapses to a trace, we can print one line for the whole unit
    instead of one line per thread.

    Units are derived from the events, so we can always recompute them. We never delete an old
    one, though — if the model is still holding a thread id from an earlier grouping, that id
    has to keep working when it asks to reopen it.
    """

    id: str
    session_id: str
    label: str  # what the model sees when this topic is collapsed, so it has to be specific
    event_ids: list[str]
    synopsis: str = ""  # 2-4 sentences: what was decided, the numbers, and why
    parent_id: str | None = None  # None means this is a top-level unit, not a thread
    # Anything you wrote about this topic yourself. Rendered with the trace, so when the topic
    # collapses the model still reads your words rather than only the model's summary of events.
    # Filled in by `Session.units()` from the marks you have set; the segmenter never writes it.
    note: str = ""


@dataclass
class Annotation:
    """A setting you put on a message to override what bonsai would pick on its own.

    `level` runs from 0.0 to 1.0: 1.0 keeps the text in full, 0.0 collapses it to a trace, and
    anything in between asks for a summary at roughly that fraction of the original size.

    The important detail is that this attaches to a *message*, not to a topic. Topics get
    regrouped every time you re-split the conversation, so a setting pinned to a topic would
    vanish the moment you did that. Pinned to a message, it survives — you can reshape the
    whole conversation and your pins are still where you put them.
    """

    id: str
    session_id: str
    event_id: str
    level: float


@dataclass
class Context:
    """The finished result: the messages to send to the model, and what each topic got.

    `detail` is here so you can check the decision after the fact — it maps a thread id to the
    "full" / "summary" / "trace" it ended up with.
    """

    messages: list[dict]
    detail: dict[str, Detail]


@dataclass
class Mark:
    """Something you wrote about a topic: a note, or a label of your own.

    Anchored to a message rather than to a thread, for the same reason levels are. Re-splitting
    the conversation regroups the threads, so a note attached to a thread id would quietly
    vanish the next time you ran `segment`. Attached to a message it follows that message into
    whatever thread it ends up in.
    """

    id: str
    session_id: str
    event_id: str
    kind: Literal["note", "label"]
    text: str
    created_at: str
