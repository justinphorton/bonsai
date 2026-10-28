import random

from bonsai.models import Annotation, Event, Unit
from bonsai.store import Store


def make_event(i: int, session_id: str = "s") -> Event:
    return Event(
        id=f"e{i}",
        session_id=session_id,
        seq=i,
        role="user" if i % 2 == 0 else "assistant",
        type="message",
        content=f"message {i} — with unicode ✓",
        tokens=i + 1,
        created_at=f"2026-01-01T00:00:{i:02d}Z",
    )


def test_events_round_trip_in_seq_order(tmp_path):
    store = Store(tmp_path / "t.db")
    events = [make_event(i) for i in range(100)]
    shuffled = events[:]
    random.Random(0).shuffle(shuffled)
    for e in shuffled:
        store.add_event(e)

    assert store.events("s") == events
    assert store.events("other") == []


def test_replace_units_retires_old_generation_and_keeps_event_order(tmp_path):
    store = Store(tmp_path / "t.db")
    for i in range(6):
        store.add_event(make_event(i))

    first = store.replace_units(
        "s",
        [
            Unit(id="x", session_id="s", label="first", event_ids=[]),
            Unit(id="x.a", session_id="s", label="a", event_ids=["e0", "e1"], parent_id="x"),
            Unit(id="x.b", session_id="s", label="b", event_ids=["e2"], parent_id="x"),
        ],
    )
    assert store.units("s") == first
    assert [u.id for u in first] == ["g1u0", "g1u0.0", "g1u0.1"]  # the store owns ids
    assert first[0].event_ids == ["e0", "e1", "e2"]  # a parent holds its subunits' events
    assert first[1].parent_id == "g1u0"

    # Non-contiguous, deliberately out of seq order: the store must not reorder event_ids.
    second = [
        Unit(id="u2", session_id="s", label="two", event_ids=["e5", "e1"]),
        Unit(id="u3", session_id="s", label="three", event_ids=["e3"]),
    ]
    second = store.replace_units("s", second)
    assert store.units("s") == second
    assert [u.id for u in second] == ["g2u0", "g2u1"]  # flat units work too
    # Retired, not deleted: the old generation's unit is still findable by id (recall needs it).
    assert store.unit("g1u0.0") == first[1]
    assert store.unit("nope") is None


def test_annotations_round_trip(tmp_path):
    store = Store(tmp_path / "t.db")
    store.add_event(make_event(0))
    a = Annotation(id="a1", session_id="s", event_id="e0", level=1.0)
    b = Annotation(id="a2", session_id="s", event_id="e0", level=0.5)
    store.add_annotation(a)
    store.add_annotation(b)
    assert store.annotations("s") == [a, b]


def test_two_sessions_in_one_store_do_not_collide(tmp_path):
    """Every session shares ~/.bonsai/bonsai.db, and unit ids are a globally unique primary key
    that `unit()` resolves without a session, so the generation counter must be global too."""
    store = Store(tmp_path / "t.db")
    for sid in ("alpha", "beta"):
        store.add_event(Event(f"{sid}-e0", sid, 0, "user", "message", "hi", 1, "now"))
        store.replace_units(sid, [Unit("u0", sid, f"{sid} thread", [f"{sid}-e0"])])
    a = store.units("alpha")
    b = store.units("beta")
    assert len(a) == 1 and len(b) == 1
    assert a[0].id != b[0].id  # the collision this test exists for
    assert store.unit(a[0].id).label == "alpha thread"
    assert store.unit(b[0].id).label == "beta thread"

    # Re-segmenting one session must still give that session the newest generation.
    store.replace_units("alpha", [Unit("u0", "alpha", "alpha again", ["alpha-e0"])])
    assert store.units("alpha")[0].label == "alpha again"
    assert store.units("beta")[0].label == "beta thread"  # untouched


def test_token_counts_are_cached_across_runs(tmp_path):
    """Planning measures one trace per thread. Without a cache that is hundreds of sequential
    API round trips every time you look at a plan, which makes steering unusable."""
    s = Store(tmp_path / "t.db")
    assert s.cached_count("hello") is None
    s.put_count("hello", 7)
    assert s.cached_count("hello") == 7
    assert Store(tmp_path / "t.db").cached_count("hello") == 7  # survives a reopen


def test_clearing_annotations_only_touches_one_session(tmp_path):
    s = Store(tmp_path / "t.db")
    s.add_annotation(Annotation("a", "s1", "e0", 1.0))
    s.add_annotation(Annotation("b", "s2", "e9", 0.0))
    assert s.clear_annotations("s1") == 1
    assert s.annotations("s1") == []
    assert len(s.annotations("s2")) == 1
