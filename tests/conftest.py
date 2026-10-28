"""Fakes so the core can be tested without an API key: a segmenter that chunks events into
units of subunits, and a summarizer that returns a fixed-size marker."""

from dataclasses import replace

from bonsai.models import Event, Unit
from bonsai.session import Session
from bonsai.store import Store


class FakeSegmenter:
    """`size` events per subunit, `per_unit` subunits per unit, in seq order."""

    def __init__(self, size: int = 2, per_unit: int = 2) -> None:
        self.size, self.per_unit = size, per_unit
        self.calls = 0  # so a test can prove a model call did not happen

    def segment(self, events: list[Event], existing=None) -> list[Unit]:
        self.calls += 1
        events = sorted(events, key=lambda e: e.seq)
        units: list[Unit] = []
        chunks = [events[i : i + self.size] for i in range(0, len(events), self.size)]
        for k in range(0, len(chunks), self.per_unit):
            group = chunks[k : k + self.per_unit]
            pid = f"u{k // self.per_unit}"
            sid = events[0].session_id
            units.append(Unit(pid, sid, f"unit {group[0][0].seq}–{group[-1][-1].seq}", []))
            for j, chunk in enumerate(group):
                label = f"events {chunk[0].seq}–{chunk[-1].seq}"
                units.append(Unit(f"{pid}.{j}", sid, label, [e.id for e in chunk], "", pid))
        return units


def fake_summarizer(text: str, target: int) -> tuple[str, int]:
    return f"SUMMARY({target})", target


def make_event(i: int, tokens: int = 100, content: str | None = None) -> Event:
    return Event(
        id=f"e{i}",
        session_id="s",
        seq=i,
        role="user" if i % 2 == 0 else "assistant",
        type="message",
        content=content or f"message {i}",
        tokens=tokens,
        created_at="2026-01-01T00:00:00Z",
    )


def typed(event: Event, kind: str) -> Event:
    """A copy of `event` with a different type, for testing the tool-traffic caps."""
    return replace(event, type=kind)


def make_session(tmp_path, size=2, per_unit=2) -> Session:
    return Session("s", Store(tmp_path / "t.db"), FakeSegmenter(size, per_unit), len)


def segmented_session(tmp_path, turns=8, size=2, per_unit=2) -> Session:
    """A session with turns already logged and split, for tests that need threads to exist.

    Splitting is explicit now -- `units()` no longer does it quietly the first time -- so tests
    that want threads have to ask, the same as a user would.
    """
    s = make_session(tmp_path, size, per_unit)
    for i in range(turns):
        s.append("user" if i % 2 == 0 else "assistant", f"message {i}", tokens=100)
    s.segment()
    return s
