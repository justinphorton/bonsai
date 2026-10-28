import pytest
from conftest import FakeSegmenter, make_event

from bonsai.compact import (
    BudgetError,
    allocate,
    blocks,
    build_context,
    leaves,
    overrides,
    trace_text,
)
from bonsai.models import Annotation

EVENTS = [make_event(i) for i in range(8)]  # 8 events × 100 tokens
UNITS = FakeSegmenter(size=2, per_unit=2).segment(EVENTS)  # u0 = [u0.0, u0.1], u1 = [u1.0, u1.1]
LEAVES = leaves(UNITS)
TRACES = sum(len(trace_text(u, 200)) for u in LEAVES)  # trace floor via len


def test_segmenter_shape_and_leaves():
    assert [u.id for u in UNITS] == ["u0", "u0.0", "u0.1", "u1", "u1.0", "u1.1"]
    assert [u.id for u in LEAVES] == ["u0.0", "u0.1", "u1.0", "u1.1"]
    assert UNITS[0].event_ids == [] and UNITS[1].event_ids == ["e0", "e1"]


def test_overrides_take_the_highest_level_in_a_leaf():
    ann = [
        Annotation("a", "s", "e0", 0.0),
        Annotation("b", "s", "e1", 1.0),
        Annotation("c", "s", "e4", 0.5),
    ]
    assert overrides(UNITS, ann) == {"u0.0": 1.0, "u1.0": 0.5}


def test_default_policy_keeps_the_current_thread_then_spends_what_is_left():
    plan = allocate(UNITS, EVENTS, budget=10_000, count_tokens=len, fixed={})
    assert plan.detail == {u.id: "full" for u in LEAVES}  # everything fits

    # Tight: only the thread holding the newest turn survives verbatim.
    plan = allocate(UNITS, EVENTS, TRACES + 250, len, {})
    assert plan.detail["u1.1"] == "full" and plan.reasons["u1.1"] == "current"
    assert {plan.detail[u] for u in ("u0.0", "u0.1", "u1.0")} == {"trace"}
    assert plan.estimate <= TRACES + 250


def test_what_you_asked_for_wins_over_the_default():
    plan = allocate(UNITS, EVENTS, 10_000, len, fixed={"u1.1": 0.0, "u0.0": 1.0})
    # Dropping beats "current": you said so explicitly.
    assert plan.detail["u1.1"] == "trace" and plan.reasons["u1.1"] == "dropped"
    assert plan.detail["u0.0"] == "full" and plan.reasons["u0.0"] == "kept"


def test_the_floor_is_one_line_per_collapsed_unit_not_one_per_thread():
    # Everything pruned: blocks() renders one trace per unit, so that is what the plan costs.
    fixed = dict.fromkeys((u.id for u in LEAVES), 0.0)
    plan = allocate(UNITS, EVENTS, 10_000, len, fixed)
    assert set(plan.detail.values()) == {"trace"}
    per_unit = len(trace_text(UNITS[0], 400, LEAVES[:2])) + len(
        trace_text(UNITS[3], 400, LEAVES[2:])
    )
    assert plan.estimate == per_unit < TRACES


def test_every_budget_down_to_the_floor_is_met_without_refusing():
    """The point of traces: any budget at or above the collapsed floor produces a plan that
    fits, dropping threads to their label instead of refusing."""
    big = [make_event(i, tokens=5_000) for i in range(8)]  # 4 threads x 10,000 tokens
    floor = allocate(UNITS, big, 10**9, len, dict.fromkeys((u.id for u in LEAVES), 0.0)).estimate
    for budget in (floor, floor + 100, 2_000, 20_000, 39_000):
        plan = allocate(UNITS, big, budget, len, {})
        assert plan.estimate <= budget, budget
    assert set(allocate(UNITS, big, floor, len, {}).detail.values()) == {"trace"}
    assert set(allocate(UNITS, big, 45_000, len, {}).detail.values()) == {"full"}


def test_budget_error_when_a_pin_cannot_fit():
    with pytest.raises(BudgetError):
        allocate(UNITS, EVENTS, 250, len, {"u0.0": 1.0})


def test_blocks_collapse_a_dead_unit_into_one_line():
    detail = {"u0.0": "trace", "u0.1": "trace", "u1.0": "trace", "u1.1": "full"}
    bs = blocks(UNITS, EVENTS, detail)
    assert [(b.unit.id, b.kind) for b in bs] == [
        ("u0", "trace"),
        ("u1.0", "trace"),
        ("u1.1", "full"),
    ]
    assert bs[0].text.startswith("[thread: unit 0–3 · 400 tok · id:u0]") and "- u0.1:" in bs[0].text
    msgs = build_context(UNITS, EVENTS, detail)
    assert msgs[1] == {
        "role": "user",
        # u1.0 is collapsed, so it stands in as one trace line with its id.
        "content": "[thread: events 4–5 · turns 4-5 · 200 tok · id:u1.0]",
    }
    assert msgs[2:] == [
        {"role": "user", "content": "message 6"},
        {"role": "assistant", "content": "message 7"},
    ]


def test_allocate_with_no_threads_returns_an_empty_plan():
    """A segmentation that produced nothing should not raise a bare StopIteration."""
    events = [make_event(i) for i in range(3)]
    plan = allocate([], events, 1000, len, {})
    assert plan.detail == {} and plan.estimate == 0


def test_budget_too_small_for_even_the_labels_says_so():
    """With no pins set, the user needs to hear that the budget itself is impossible --
    "held: []" reads as if their own settings were the problem."""
    events = [make_event(i) for i in range(4)]
    units = FakeSegmenter().segment(events)
    with pytest.raises(BudgetError, match="too small"):
        allocate(units, events, 0, len, {})


def test_blocks_treats_an_unknown_thread_as_a_trace():
    """detail may not mention every thread; the safe default is the cheapest one."""
    events = [make_event(i, tokens=500) for i in range(8)]
    units = FakeSegmenter().segment(events)
    out = blocks(units, events, {})
    covered = {eid for b in out for eid in b.event_ids}
    assert covered == {e.id for e in events}
