"""The JSONL shape here mirrors real Claude Code 2.1.240 transcripts."""

import json

import pytest
from conftest import make_session

from bonsai import compact, transcript
from bonsai.compact import Plan

COMMON = {"isSidechain": False, "sessionId": "s", "cwd": "C:\\x", "version": "2.1.240"}


def line(kind, uid, parent, **extra):
    return json.dumps({"parentUuid": parent, "type": kind, "uuid": uid, **COMMON, **extra})


def jsonl() -> str:
    """Four exchanges: user -> attachment -> assistant(thinking) -> assistant(text), with
    bookkeeping lines (no uuid) in between, like the real files."""
    rows = [json.dumps({"type": "queue-operation", "sessionId": "s"})]
    parent = None
    for i in range(4):
        rows.append(line("user", f"u{i}", parent, message={"role": "user", "content": f"q{i}"}))
        rows.append(line("attachment", f"a{i}", f"u{i}", attachment={}))
        rows.append(json.dumps({"type": "file-history-snapshot", "n": i}))
        thinking = [{"type": "thinking", "thinking": ""}]
        rows.append(
            line("assistant", f"t{i}", f"a{i}", message={"role": "assistant", "content": thinking})
        )
        text = [{"type": "text", "text": f"a{i}"}]
        rows.append(
            line("assistant", f"s{i}", f"t{i}", message={"role": "assistant", "content": text})
        )
        parent = f"s{i}"
    rows.append(json.dumps({"type": "last-prompt", "lastPrompt": "q3"}))
    return "\n".join(rows) + "\n"


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(transcript, "BONSAI_DIR", tmp_path / "bonsai")
    path = tmp_path / "s.jsonl"
    path.write_text(jsonl(), encoding="utf-8")
    lines = transcript.read(path)
    session = make_session(tmp_path)  # 2 events per subunit, 2 subunits per unit
    transcript.import_events(session, lines)
    session.segment()  # splitting is explicit; nothing does it for you
    return path, lines, session


def plan_for(session, **detail) -> Plan:
    d = {u.id: detail.get(u.id, "full") for u in session.units() if u.parent_id}
    return Plan(d, {}, 0)


def test_import_is_idempotent_and_skips_thinking(setup):
    _, lines, session = setup
    assert [e.id for e in session.events()] == ["u0", "s0", "u1", "s1", "u2", "s2", "u3", "s3"]
    assert transcript.import_events(session, lines) == 0


def test_nothing_collapsed_is_byte_identical(setup):
    _, lines, session = setup
    out = transcript.rewrite(lines, session, session.units(), plan_for(session))
    assert [ln.raw for ln in out] == [ln.raw for ln in lines]


def test_collapse_writes_header_placeholder_companion_and_repairs_chain(setup):
    path, lines, session = setup
    # g1u0 = [g1u0.0 (u0,s0), g1u0.1 (u1,s1)]; collapse the whole first unit and g1u1.0 too.
    plan = plan_for(session, **{"g1u0.0": "trace", "g1u0.1": "trace", "g1u1.0": "trace"})
    out = transcript.rewrite(lines, session, session.units(), plan)
    tags = [ln.bonsai for ln in out if ln.bonsai]
    assert [t["kind"] for t in tags] == ["header", "trace", "trace"]
    assert tags[1]["unit_id"] == "g1u0"  # both subunits traced -> one unit-level placeholder
    body = next(ln for ln in out if ln.bonsai and ln.bonsai["kind"] == "trace").obj["message"][
        "content"
    ]
    assert body.startswith("[thread: unit 0–3 · ") and "- g1u0.1:" in body
    assert "full text: " in body and " lines " in body
    # The companion file exists and the cited range holds the collapsed messages.
    companion = transcript.BONSAI_DIR / "s.md"
    a, b = map(int, body.rsplit(" lines ", 1)[1].split("-"))
    cited = companion.read_text(encoding="utf-8").splitlines()[a - 1 : b]
    assert any("**user:** q0" in x for x in cited) and any("**assistant:** a1" in x for x in cited)
    # Dropped: the collapsed unit's lines, its attachments, its empty thinking lines.
    kept = {ln.uuid for ln in out if ln.uuid}
    assert not {"u0", "s0", "a0", "t0", "u1", "s1", "u2", "s2", "a2", "t2"} & kept
    assert {"u3", "s3", "a3", "t3"} <= kept
    # Chain intact: every parentUuid points at an earlier kept line.
    seen = set()
    for ln in out:
        if ln.uuid:
            assert ln.obj["parentUuid"] is None or ln.obj["parentUuid"] in seen
            seen.add(ln.uuid)
    # Bookkeeping survives; re-reading imports nothing new and reports the collapsed state.
    assert any(ln.obj.get("type") == "file-history-snapshot" for ln in out)
    transcript.write(path, out)
    again = transcript.read(path)
    assert transcript.import_events(session, again) == 0
    assert transcript.collapsed(again) == {"g1u0": "trace", "g1u1.0": "trace"}
    # A later compaction that restores everything brings the text back and drops the header.
    back = transcript.rewrite(again, session, session.units(), plan_for(session))
    kinds = [ln.bonsai["kind"] for ln in back if ln.bonsai]
    assert kinds == ["restored", "restored", "restored"]  # one per thread that was collapsed
    restored = next(ln for ln in back if ln.bonsai)
    assert "user: q0" in restored.obj["message"]["content"]


def test_lines_arriving_during_a_rewrite_are_rechained():
    """Compacting a live session takes minutes, and Claude Code appends the whole time. An
    arrived line still pointing at a replaced uuid makes every earlier message unreachable."""
    msg = {"message": {"role": "user", "content": "x"}}
    mk = lambda raws: [transcript.Line(r, json.loads(r)) for r in raws]
    kept = mk([line("user", "a", None, **msg), line("assistant", "b", "a", **msg)])
    arrived = mk([line("user", "x", "gone", **msg), line("assistant", "y", "x", **msg)])
    out = transcript.rechain(arrived, after=kept)
    assert [o.obj["parentUuid"] for o in out] == ["b", "x"]  # first repointed, second untouched
    ids = {o.uuid for o in kept + out}
    assert all(o.obj["parentUuid"] in ids for o in kept + out if o.obj["parentUuid"])
    assert transcript.rechain([], after=kept) == []


def test_the_last_line_survives_even_when_its_thread_is_pruned(tmp_path):
    """Claude Code appends after a compaction and records whatever uuid was last in the file. If
    the rewrite collapsed that line, the next message points at nothing and orphans everything."""
    path = tmp_path / "s.jsonl"
    path.write_text(jsonl(), encoding="utf-8")
    lines = transcript.read(path)
    s = make_session(tmp_path, size=2, per_unit=2)
    transcript.import_events(s, lines)
    last_event = s.events()[-1]
    s.prune(last_event.id)  # prune the thread holding the final message
    plan = s.plan(budget=200)
    leaf = next(u for u in compact.leaves(s.units()) if last_event.id in u.event_ids)
    assert plan.detail[leaf.id] == "trace"

    out = transcript.rewrite(lines, s, s.units(), plan)
    assert out[-1].uuid == last_event.id  # the anchor Claude Code will point at
    ids = {ln.uuid for ln in out if ln.uuid}
    assert all(
        ln.obj.get("parentUuid") in ids for ln in out if ln.uuid and ln.obj.get("parentUuid")
    )


def test_one_failed_token_count_does_not_kill_the_import(tmp_path, monkeypatch):
    """Counting is thousands of parallel calls on a long transcript, so it is the likeliest
    place to meet a rate limit. One failure estimates that single event instead of losing all
    of them; the budget is then slightly approximate rather than unavailable."""
    session = make_session(tmp_path)
    calls = []

    def flaky(text):
        calls.append(text)
        if len(calls) == 2:
            raise RuntimeError("rate limited")
        return len(text)

    session.count_tokens = flaky
    jsonl = tmp_path / "s.jsonl"
    jsonl.write_text(
        "\n".join(
            json.dumps(
                {
                    **COMMON,
                    "type": "user",
                    "uuid": f"u{i}",
                    "parentUuid": None if i == 0 else f"u{i - 1}",
                    "message": {"role": "user", "content": f"message {i}"},
                }
            )
            for i in range(4)
        ),
        encoding="utf-8",
    )
    added = transcript.import_events(session, transcript.read(jsonl))
    assert added == 4, "every event is imported even though one count failed"
    assert all(e.tokens > 0 for e in session.events())
