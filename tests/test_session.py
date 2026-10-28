from types import SimpleNamespace

import pytest
from conftest import make_event, make_session, typed

from bonsai.session import elide_tools


def test_append_assigns_seq_and_tokens(tmp_path):
    s = make_session(tmp_path)
    a = s.append("user", "hello")
    b = s.append("assistant", "hi there", id="fixed-id")
    assert (a.seq, a.tokens) == (0, 5) and (b.seq, b.tokens, b.id) == (1, 8, "fixed-id")


def test_new_turns_do_not_trigger_a_resegmentation(tmp_path):
    """Splitting a transcript costs a model call over all of it, so one new message must not pay
    for it again. Newer turns get a standing thread of their own until `segment` is run."""
    s = make_session(tmp_path)
    for i in range(4):
        s.append("user", f"m{i}")
    s.segment()
    first = s.units()
    assert [u.id for u in first] == ["g1u0", "g1u0.0", "g1u0.1"]
    assert first[0].event_ids == first[1].event_ids + first[2].event_ids  # parent = union
    assert s.units() == first
    assert s.segmenter.calls == 1

    s.append("user", "m4")
    after = s.units()
    assert s.segmenter.calls == 1  # the point: no second call
    assert [u.id for u in after] == ["g1u0", "g1u0.0", "g1u0.1", "new"]
    assert after[-1].event_ids == [s.events()[-1].id]

    # Re-segmenting is an explicit act. It makes a new generation and retires the old one,
    # which is kept so ids handed out earlier still resolve.
    s.store.replace_units(s.session_id, s.segmenter.segment(s.events()))
    assert [u.id for u in s.units()] == ["g2u0", "g2u0.0", "g2u0.1", "g2u1", "g2u1.0"]
    assert s.store.unit("g1u0.0") is not None


def test_recall_and_tool_result(tmp_path):
    s = make_session(tmp_path)
    s.append("user", "the answer is 42")
    s.append("assistant", "noted")
    s.segment()
    uid = s.units()[0].id  # the parent unit: recall works at either level
    assert s.recall(uid) == "user: the answer is 42\n\nassistant: noted"
    ok = s.tool_result(SimpleNamespace(id="t1", input={"unit_id": uid}))
    assert ok["type"] == "tool_result" and "42" in ok["content"]
    bad = s.tool_result(SimpleNamespace(id="t2", input={"unit_id": "zzz"}))
    assert bad["is_error"]
    with pytest.raises(KeyError):
        s.recall("nope")


def test_elide_keeps_the_end_of_a_failing_tool_result():
    """A traceback puts the actual error on the LAST line, so cutting the head keeps the useless
    part and drops the part that matters.

    This is the case bonsai exists for: "we tried X and it failed with Y" is exactly what stops an
    agent retrying it. Truncating Y away defeats the point.
    """
    tail = "ValueError: segmenter did not cover every turn"
    content = "Traceback (most recent call last):\n" + ("  File x, line 1\n" * 400) + tail
    [out] = elide_tools([typed(make_event(0, content=content), "tool_result")])
    assert tail in out.content, "the actual error must survive"
    assert len(out.content) < len(content)


def test_elide_still_caps_an_ordinary_tool_result_from_the_head():
    body = "line\n" * 2000
    [out] = elide_tools([typed(make_event(0, content=body), "tool_result")])
    assert out.content.startswith("line")
    assert len(out.content) < len(body)


def test_counting_the_same_text_twice_only_asks_once(tmp_path):
    calls = []

    def counting(text):
        calls.append(text)
        return len(text)

    s = make_session(tmp_path)
    s.count_tokens = s._caching(counting)
    assert s.count_tokens("abc") == 3
    assert s.count_tokens("abc") == 3
    assert calls == ["abc"], "the second ask should come from the cache"


def test_steering_reports_what_you_kept_and_dropped(tmp_path):
    """There was no way to ask "what have I pinned?" -- you had to infer it from a column in
    the threads listing. Steering you cannot inspect is steering you stop trusting."""
    s = make_session(tmp_path)
    for i in range(8):
        s.append("user", f"m{i}", tokens=100)
    s.segment()
    threads = [u for u in s.units() if u.parent_id]
    for e in threads[0].event_ids:
        s.pin(e)
    for e in threads[1].event_ids:
        s.prune(e)
    assert s.steering() == {threads[0].id: "keep", threads[1].id: "drop"}

    assert s.clear_steering() == 4  # two threads x two events
    assert s.steering() == {}
