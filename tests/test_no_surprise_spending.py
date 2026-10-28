"""Nothing may start a paid segmentation unless you asked for one.

Splitting a transcript is the single expensive thing bonsai does: minutes of waiting and real
money on a long conversation. It happened implicitly from `status` and `threads` -- commands
that read as free -- because `units()` segmented whenever the store was empty. Running a
read-only command must never spend money.
"""

import pytest
from conftest import make_session

from bonsai.session import TAIL_ID


def test_units_never_segments_on_its_own(tmp_path):
    s = make_session(tmp_path)
    for i in range(8):
        s.append("user", f"m{i}", tokens=100)
    before = s.segmenter.calls
    s.units()
    assert s.segmenter.calls == before, "reading the threads must not call the segmenter"


def test_an_unsegmented_conversation_still_reports_itself(tmp_path):
    """It should say "none of this is indexed yet", not refuse and not quietly pay to fix it."""
    s = make_session(tmp_path)
    for i in range(8):
        s.append("user", f"m{i}", tokens=100)
    units = s.units()
    assert len(units) == 1 and units[0].id == TAIL_ID
    assert len(units[0].event_ids) == 8, "every turn is accounted for as not-yet-indexed"


def test_segment_is_the_only_thing_that_segments(tmp_path):
    s = make_session(tmp_path)
    for i in range(8):
        s.append("user", f"m{i}", tokens=100)
    assert s.segmenter.calls == 0
    s.segment()
    assert s.segmenter.calls == 1
    s.units()
    s.plan(10_000)
    assert s.segmenter.calls == 1, "nothing after the explicit segment may call it again"


def test_cli_refuses_to_plan_an_unindexed_conversation(tmp_path):
    """A plan over one undifferentiated blob is not a plan, so say what to run instead."""
    from bonsai import cli

    s = make_session(tmp_path)
    for i in range(8):
        s.append("user", f"m{i}", tokens=100)
    with pytest.raises(SystemExit, match="segment"):
        cli.plan(s, 10_000)


def test_cli_plans_normally_once_segmented(tmp_path):
    from bonsai import cli

    s = make_session(tmp_path)
    for i in range(8):
        s.append("user", f"m{i}", tokens=100)
    s.segment()
    the_plan, text = cli.plan(s, 10_000)
    assert the_plan.detail and text


def test_segmented_is_false_before_and_true_after(tmp_path):
    from bonsai import cli

    s = make_session(tmp_path)
    for i in range(6):
        s.append("user", f"m{i}", tokens=100)
    assert cli.segmented(s) is False
    s.segment()
    assert cli.segmented(s) is True


def test_a_session_with_no_messages_at_all_is_not_segmented(tmp_path):
    from bonsai import cli

    assert cli.segmented(make_session(tmp_path)) is False
