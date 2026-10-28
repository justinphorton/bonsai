"""Editing the index: notes, your own labels, and reshaping threads.

These four commands are the difference between an index you can only read and one you can
correct. The segmenter is good but not right every time, and before this the only way to fix a
bad grouping was to pay for a whole new segmentation.
"""

import pytest
from conftest import make_session

from bonsai import compact


def session_with_threads(tmp_path, turns=16):
    s = make_session(tmp_path, size=2, per_unit=2)
    for i in range(turns):
        s.append("user" if i % 2 == 0 else "assistant", f"message {i}", tokens=100)
    s.segment()  # splitting is explicit now; nothing does it for you
    return s


def threads(s):
    return [u for u in s.units() if u.parent_id]


# --- notes --------------------------------------------------------------------


def test_a_note_reaches_the_model_when_the_thread_collapses(tmp_path):
    """The whole point: a note survives compaction verbatim, where a summarizer hint might not."""
    s = session_with_threads(tmp_path)
    target = threads(s)[1]
    s.note(target.id, "abandoned - breaks admissibility")

    marked = next(u for u in threads(s) if u.id == target.id)
    assert marked.note == "abandoned - breaks admissibility"
    rendered = compact.trace_text(marked, 500)
    assert "abandoned - breaks admissibility" in rendered
    assert "NOTE FROM THE USER" in rendered, "the model must not mistake it for its own synopsis"


def test_notes_accumulate_in_order(tmp_path):
    s = session_with_threads(tmp_path)
    target = threads(s)[0]
    s.note(target.id, "first thought")
    s.note(target.id, "second thought")
    assert next(u for u in threads(s) if u.id == target.id).note == "first thought\nsecond thought"


def test_a_note_survives_a_resegmentation(tmp_path):
    """Marks anchor to a message, not a thread id, so a re-split carries them along."""
    s = session_with_threads(tmp_path)
    anchor_event = threads(s)[2].event_ids[0]
    s.note(threads(s)[2].id, "keep this in mind")

    s.segment()  # regroup from scratch; every thread id changes
    carried = [u for u in threads(s) if anchor_event in u.event_ids]
    assert len(carried) == 1
    assert "keep this in mind" in carried[0].note


def test_noting_a_thread_that_does_not_exist_is_an_error(tmp_path):
    s = session_with_threads(tmp_path)
    with pytest.raises(KeyError):
        s.note("g9u9.9", "nope")


# --- relabel ------------------------------------------------------------------


def test_relabel_replaces_the_models_wording(tmp_path):
    s = session_with_threads(tmp_path)
    target = threads(s)[1]
    assert target.label != "MST heuristic chosen after swap_medium failed"
    s.relabel(target.id, "MST heuristic chosen after swap_medium failed")
    assert next(u for u in threads(s) if u.id == target.id).label == (
        "MST heuristic chosen after swap_medium failed"
    )


def test_relabelling_twice_keeps_only_the_latest(tmp_path):
    """Otherwise every correction would stack up and the last one would be ambiguous."""
    s = session_with_threads(tmp_path)
    target = threads(s)[0]
    s.relabel(target.id, "first attempt")
    s.relabel(target.id, "second attempt")
    assert next(u for u in threads(s) if u.id == target.id).label == "second attempt"
    assert len([m for m in s.store.marks(s.session_id) if m.kind == "label"]) == 1


def test_a_label_and_a_note_coexist(tmp_path):
    s = session_with_threads(tmp_path)
    target = threads(s)[1]
    s.relabel(target.id, "my label")
    s.note(target.id, "my note")
    got = next(u for u in threads(s) if u.id == target.id)
    assert got.label == "my label" and got.note == "my note"


# --- merge --------------------------------------------------------------------


def test_merge_joins_two_threads_keeping_every_message(tmp_path):
    s = session_with_threads(tmp_path)
    before = threads(s)
    a, b = before[0], before[1]
    expected = set(a.event_ids) | set(b.event_ids)

    s.merge([a.id, b.id], label="the joined topic")
    after = threads(s)
    assert len(after) == len(before) - 1
    joined = next(u for u in after if u.label == "the joined topic")
    assert set(joined.event_ids) == expected
    # nothing anywhere else may have been lost or duplicated
    seen = [eid for u in after for eid in u.event_ids]
    assert sorted(seen) == sorted(e.id for e in s.events())
    assert len(seen) == len(set(seen))


def test_merge_needs_at_least_two_real_threads(tmp_path):
    s = session_with_threads(tmp_path)
    with pytest.raises(ValueError):
        s.merge([threads(s)[0].id])
    with pytest.raises(KeyError):
        s.merge([threads(s)[0].id, "g9u9.9"])


def test_merging_threads_from_different_units_works(tmp_path):
    """Threads the segmenter filed under separate units are exactly the ones you want to join."""
    s = session_with_threads(tmp_path, turns=16)
    a = threads(s)[0]
    b = next(u for u in threads(s) if u.parent_id != a.parent_id)
    expected = set(a.event_ids) | set(b.event_ids)
    s.merge([a.id, b.id])
    joined = [u for u in threads(s) if expected <= set(u.event_ids)]
    assert len(joined) == 1


# --- split --------------------------------------------------------------------


def test_split_divides_a_thread_at_a_turn(tmp_path):
    s = session_with_threads(tmp_path)
    target = next(u for u in threads(s) if len(u.event_ids) > 1)
    events = {e.id: e for e in s.events()}
    turns = sorted(events[eid].seq for eid in target.event_ids)
    at = turns[len(turns) // 2]

    s.split(target.id, at)
    after = threads(s)
    halves = [u for u in after if set(u.event_ids) & set(target.event_ids)]
    assert len(halves) == 2
    low, high = sorted(halves, key=lambda u: min(events[e].seq for e in u.event_ids))
    assert max(events[e].seq for e in low.event_ids) < at
    assert min(events[e].seq for e in high.event_ids) == at
    seen = [eid for u in after for eid in u.event_ids]
    assert sorted(seen) == sorted(e.id for e in s.events())


def test_split_refuses_a_turn_that_would_leave_an_empty_half(tmp_path):
    """An empty thread breaks rendering, so the split has to actually divide something."""
    s = session_with_threads(tmp_path)
    target = next(u for u in threads(s) if len(u.event_ids) > 1)
    events = {e.id: e for e in s.events()}
    first = min(events[eid].seq for eid in target.event_ids)
    with pytest.raises(ValueError):
        s.split(target.id, first)  # everything would land in the second half


def test_split_refuses_a_single_message_thread(tmp_path):
    s = make_session(tmp_path, size=1, per_unit=2)
    for i in range(4):
        s.append("user", f"m{i}", tokens=100)
    s.segment()
    target = next(u for u in s.units() if u.parent_id)
    with pytest.raises(ValueError):
        s.split(target.id, 0)


# --- the index stays usable after editing it ----------------------------------


def test_old_ids_still_recall_after_merging(tmp_path):
    """Threads are versioned, never deleted, so an id the model is still holding keeps working."""
    s = session_with_threads(tmp_path)
    a, b = threads(s)[0], threads(s)[1]
    old_text = s.recall(a.id)
    s.merge([a.id, b.id])
    assert s.recall(a.id) == old_text


def test_marks_survive_merging_and_splitting(tmp_path):
    s = session_with_threads(tmp_path)
    target = threads(s)[1]
    s.note(target.id, "do not lose me")
    s.merge([target.id, threads(s)[0].id])
    assert any("do not lose me" in u.note for u in threads(s))


def test_a_plan_still_builds_after_editing_the_index(tmp_path):
    s = session_with_threads(tmp_path)
    s.note(threads(s)[0].id, "a note")
    s.relabel(threads(s)[1].id, "a label")
    s.merge([threads(s)[2].id, threads(s)[3].id])
    plan = s.plan(100_000)
    assert plan.detail and plan.estimate > 0
    ctx = s.build_context(100_000)
    assert ctx.messages
