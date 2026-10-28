"""Turn lists: a thread names runs of turns, not every index. Parsing them is what these cover.

The real model call is never made here — `_text` is monkeypatched to a canned reply, which is
also how a malformed reply is tested without paying for one.
"""

import json
import re

import pytest
from conftest import make_event

from bonsai import llm

# --- _expand -----------------------------------------------------------------


def test_expand_runs():
    assert llm._expand(["0-2", "5-6"], 10) == [0, 1, 2, 5, 6]


def test_expand_bare_integers_still_work():
    """The old all-indices form must keep parsing: a cached reply may still use it."""
    assert llm._expand([0, 1, 2], 10) == [0, 1, 2]


def test_expand_mixed():
    """A thread that picks a subject up again later: one run, a stray turn, another run."""
    assert llm._expand(["0-2", 7, "9-10"], 20) == [0, 1, 2, 7, 9, 10]


def test_expand_single_turn_run():
    assert llm._expand(["7-7"], 10) == [7]


def test_expand_integer_as_string():
    assert llm._expand(["4"], 10) == [4]


def test_expand_skips_a_reversed_run():
    """ "5-2" says nothing usable, so it is dropped like any other junk entry."""
    assert llm._expand(["5-2", "7"], 10) == [7]


def test_expand_drops_indices_past_the_end():
    """Out of range would reach events[i] and raise a bare IndexError, so it is dropped here."""
    assert llm._expand(["8-12"], 10) == [8, 9]


def test_expand_skips_junk_entries():
    """A stray token is dropped, not fatal.

    Real case: the model returned ["." , "5"] for one thread. Everything else was perfect. With
    unplaced turns now swept into their own thread, losing one entry is visible and recoverable,
    whereas raising threw away a whole paid segmentation over one character.
    """
    assert llm._expand([".", "5"], 10) == [5]
    assert llm._expand(["three", "0-2"], 10) == [0, 1, 2]


def test_expand_keeps_the_in_range_part_of_a_run():
    """ "8-12" on a 10-turn transcript still names turns 8 and 9; only the overshoot goes."""
    assert llm._expand(["8-12", "3"], 10) == [8, 9, 3]


# --- a whole reply -----------------------------------------------------------


def reply(subunits: list[dict]) -> str:
    return json.dumps([{"label": "U", "synopsis": "u", "subunits": subunits}])


def canned(monkeypatch, text: str) -> None:
    """Answer the one model call with `text`, and don't touch ~/.bonsai while testing."""
    monkeypatch.setattr(llm, "_text", lambda *a, **k: text)
    monkeypatch.setattr(llm, "_keep", lambda *a, **k: None)


def test_segment_expands_runs_into_event_ids(monkeypatch):
    events = [make_event(i) for i in range(6)]
    canned(
        monkeypatch,
        reply(
            [
                {"label": "A", "synopsis": "a", "turns": ["0-2"]},
                {"label": "B", "synopsis": "b", "turns": [3, "4-5"]},
            ]
        ),
    )
    units = llm.LLMSegmenter().segment(events)
    by_id = {u.id: u for u in units}
    assert by_id["u0.0"].event_ids == ["e0", "e1", "e2"]
    assert by_id["u0.1"].event_ids == ["e3", "e4", "e5"]


def test_segment_sweeps_a_gap_into_an_unsorted_thread(monkeypatch):
    """A turn no thread claimed is kept, not dropped. Losing it silently is the one unacceptable
    outcome -- it is the exact failure bonsai exists to prevent."""
    events = [make_event(i) for i in range(6)]
    canned(monkeypatch, reply([{"label": "A", "synopsis": "a", "turns": ["0-1", "3-5"]}]))
    units = llm.LLMSegmenter().segment(events)
    covered = {eid for u in units for eid in u.event_ids}
    assert covered == {e.id for e in events}, "every event must land somewhere"
    swept = [u for u in units if u.event_ids == ["e2"]]
    assert len(swept) == 1 and "unsorted" in swept[0].label


def test_segment_dedupes_an_overlap_keeping_the_first_claim(monkeypatch):
    """A turn claimed twice goes to the first thread that asked; the second just loses it."""
    events = [make_event(i) for i in range(6)]
    canned(
        monkeypatch,
        reply(
            [
                {"label": "A", "synopsis": "a", "turns": ["0-3"]},
                {"label": "B", "synopsis": "b", "turns": ["3-5"]},
            ]
        ),
    )
    units = llm.LLMSegmenter().segment(events)
    by_label = {u.label: u for u in units if u.parent_id}
    assert by_label["A"].event_ids == ["e0", "e1", "e2", "e3"]
    assert by_label["B"].event_ids == ["e4", "e5"]
    assert sum(len(u.event_ids) for u in units if u.parent_id) == 6


def test_segment_drops_a_thread_left_empty_by_deduping(monkeypatch):
    events = [make_event(i) for i in range(4)]
    canned(
        monkeypatch,
        reply(
            [
                {"label": "A", "synopsis": "a", "turns": ["0-3"]},
                {"label": "B", "synopsis": "b", "turns": ["1-2"]},
            ]
        ),
    )
    units = llm.LLMSegmenter().segment(events)
    assert "B" not in {u.label for u in units}


def test_json_ignores_trailing_text():
    """The model sometimes emits the array and then carries on writing the conversation.

    Seen on a real 2,753-turn segmentation: valid JSON, then a hallucinated "### turn 2753".
    The array is complete and paid for, so it is parsed and the rest dropped.
    """
    reply = '[{"label": "A", "synopsis": "s", "subunits": []}]\nHere is the plan...\n### turn 99'
    assert llm._json(reply) == [{"label": "A", "synopsis": "s", "subunits": []}]


def test_json_still_rejects_a_truncated_array():
    """Trailing junk is tolerated; an array that never closes is not."""
    with pytest.raises(ValueError):
        llm._json('[{"label": "A", "subunits": [')


def test_expand_accepts_a_bare_run_not_in_a_list():
    """A thread that is one unbroken run: the model writes "turns": "2719-2752", not a list.

    Seen on a real segmentation. Iterating the string would hand _expand single characters, so
    a bare run or number is wrapped rather than rejected.
    """
    assert llm._expand("2719-2722", 3000) == [2719, 2720, 2721, 2722]


def test_expand_accepts_a_bare_integer():
    assert llm._expand(7, 10) == [7]


class _FakeStream:
    """Minimal stand-in for the streaming context manager, with a controllable final message."""

    def __init__(self, message):
        self.message = message

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __iter__(self):
        return iter(())

    def get_final_message(self):
        return self.message


class _Msg:
    def __init__(self, stop_reason, content):
        self.stop_reason, self.content = stop_reason, content
        self.stop_details = None


class _Block:
    def __init__(self, text):
        self.type, self.text = "text", text


def _client_returning(message, monkeypatch):
    class C:
        class messages:
            @staticmethod
            def stream(**kw):
                return _FakeStream(message)

    monkeypatch.setattr(llm, "client", lambda: C)


def test_text_raises_on_max_tokens(monkeypatch):
    """A cut-off reply is never valid JSON; say so instead of handing on a fragment."""
    _client_returning(_Msg("max_tokens", [_Block("[{...")]), monkeypatch)
    with pytest.raises(RuntimeError, match="max_tokens"):
        llm._text("m", None, "u", 100)


def test_text_raises_on_refusal(monkeypatch):
    _client_returning(_Msg("refusal", []), monkeypatch)
    with pytest.raises(RuntimeError, match="refus"):
        llm._text("m", None, "u", 100)


def test_text_raises_when_no_text_block_comes_back(monkeypatch):
    _client_returning(_Msg("end_turn", []), monkeypatch)
    with pytest.raises(RuntimeError, match="no text"):
        llm._text("m", None, "u", 100)


def test_text_returns_the_text_normally(monkeypatch):
    _client_returning(_Msg("end_turn", [_Block("hello")]), monkeypatch)
    assert llm._text("m", None, "u", 100) == "hello"


# --- structured output and inert framing --------------------------------------


def test_json_accepts_the_schema_wrapped_object():
    """Structured output returns {"units": [...]} because a schema's root must be an object."""
    assert llm._json('{"units": [{"label": "A", "subunits": []}]}') == [
        {"label": "A", "subunits": []}
    ]


def test_json_still_accepts_a_bare_array():
    """Replies saved before structured output was switched on must still parse."""
    assert llm._json('[{"label": "A", "subunits": []}]') == [{"label": "A", "subunits": []}]


def test_numbered_does_not_look_like_a_conversation(monkeypatch):
    """The transcript goes in as inert records, not as markdown dialogue.

    Fed 475k tokens formatted as '### turn 0 (user)', the model stopped segmenting and started
    continuing the conversation instead -- three times, returning as little as 24 characters.
    Tagged records read as data to process rather than a script to carry on.
    """
    text = llm._numbered([make_event(0), make_event(1)])
    assert "### turn" not in text
    assert '<turn index="0" role="user">' in text
    assert "</turn>" in text


def test_text_sends_the_schema_when_given_one(monkeypatch):
    seen = {}

    class C:
        class messages:
            @staticmethod
            def stream(**kw):
                seen.update(kw)
                return _FakeStream(_Msg("end_turn", [_Block("{}")]))

    monkeypatch.setattr(llm, "client", lambda: C)
    llm._text("m", None, "u", 100, schema={"type": "object"})
    assert seen["output_config"] == {
        "format": {"type": "json_schema", "schema": {"type": "object"}}
    }


def test_text_omits_output_config_without_a_schema(monkeypatch):
    seen = {}

    class C:
        class messages:
            @staticmethod
            def stream(**kw):
                seen.update(kw)
                return _FakeStream(_Msg("end_turn", [_Block("hi")]))

    monkeypatch.setattr(llm, "client", lambda: C)
    llm._text("m", None, "u", 100)
    assert "output_config" not in seen


# --- scaling the ask to the transcript ----------------------------------------


def test_prompt_uses_the_defaults_on_a_short_transcript():
    """Short transcripts keep the hand-tuned defaults, which is what they were tuned on."""
    p = llm.LLMSegmenter()._prompt(200)
    assert "about 6 UNITS" in p


def test_prompt_asks_for_more_units_on_a_long_transcript():
    """A 2,927-turn transcript got 28 threads at the default ask -- about 105 turns each -- and
    the model simply stopped accounting for the rest. The ask has to grow with the input."""
    p = llm.LLMSegmenter()._prompt(3000)
    units = int(re.search(r"about (\d+) UNITS", p).group(1))
    assert units >= 12, f"3000 turns should ask for many more than 6 units, got {units}"


def test_prompt_never_asks_for_fewer_than_configured():
    """An explicit --units is a floor, never silently lowered."""
    assert "about 30 UNITS" in llm.LLMSegmenter(n_units=30)._prompt(200)


# --- not dying on partly-malformed replies ------------------------------------


def test_subunits_skips_a_malformed_unit_instead_of_raising(capsys):
    """One bad unit should cost that unit, not the whole segmentation.

    The other units are fine and already paid for, and any turns the bad one would have held get
    swept into an unsorted thread, so nothing is lost by carrying on.
    """
    assert llm._subunits({"label": "a"}, 3) == []
    assert "unit 3" in capsys.readouterr().err


def test_segment_survives_one_malformed_unit(monkeypatch):
    events = [make_event(i) for i in range(6)]
    canned(
        monkeypatch,
        json.dumps(
            [
                {
                    "label": "good",
                    "synopsis": "s",
                    "subunits": [{"label": "A", "synopsis": "a", "turns": ["0-2"]}],
                },
                {"label": "broken", "synopsis": "s"},
            ]
        ),
    )
    units = llm.LLMSegmenter().segment(events)
    covered = {eid for u in units for eid in u.event_ids}
    assert covered == {e.id for e in events}
    assert "A" in {u.label for u in units}


def test_expand_treats_a_missing_turns_field_as_empty():
    assert llm._expand(None, 10) == []


def test_segment_rejects_an_empty_transcript():
    """Fail before the API call, with a message that says what is wrong."""
    with pytest.raises(ValueError, match="no events"):
        llm.LLMSegmenter().segment([])


# --- chunking long transcripts ------------------------------------------------


def covering_reply(prompt_user: str) -> str:
    """A valid reply that covers whatever turn count the prompt says, in two threads."""
    n = int(re.search(r"<transcript turns=(\d+)>", prompt_user).group(1))
    half = max(1, n // 2)
    return json.dumps(
        {
            "units": [
                {
                    "label": "U",
                    "synopsis": "s",
                    "subunits": [
                        {"label": "A", "synopsis": "a", "turns": [f"0-{half - 1}"]},
                        {"label": "B", "synopsis": "b", "turns": [f"{half}-{n - 1}"]},
                    ],
                }
            ]
        }
    )


def count_calls(monkeypatch) -> list:
    calls = []

    def fake_text(model, system, user, max_tokens, label="", schema=None):
        calls.append(user)
        return covering_reply(user)

    monkeypatch.setattr(llm, "_text", fake_text)
    monkeypatch.setattr(llm, "_keep", lambda *a, **k: None)
    return calls


def test_short_transcript_is_one_call(monkeypatch):
    """Below the threshold nothing changes -- same single call as before."""
    calls = count_calls(monkeypatch)
    llm.LLMSegmenter().segment([make_event(i) for i in range(100)])
    assert len(calls) == 1


def test_long_transcript_is_split_into_chunks(monkeypatch):
    """The model stopped placing turns at ~1,300 on a 2,943-turn transcript, with 40k of output
    budget left. It ends its turn early rather than being cut off, so the input has to be made
    smaller instead of the instruction louder."""
    calls = count_calls(monkeypatch)
    llm.LLMSegmenter().segment([make_event(i) for i in range(2000)])
    assert len(calls) == 3, f"2000 turns at 800 per chunk should be 3 calls, got {len(calls)}"


def test_chunks_cover_every_event_exactly_once(monkeypatch):
    count_calls(monkeypatch)
    events = [make_event(i) for i in range(2000)]
    units = llm.LLMSegmenter().segment(events)
    seen = [eid for u in units if u.parent_id for eid in u.event_ids]
    assert sorted(seen) == sorted(e.id for e in events)
    assert len(seen) == len(set(seen)), "no event may appear in two threads"


def test_chunked_unit_ids_do_not_collide(monkeypatch):
    """Each chunk numbers its units from zero, so they have to be renumbered on merge."""
    count_calls(monkeypatch)
    units = llm.LLMSegmenter().segment([make_event(i) for i in range(2000)])
    ids = [u.id for u in units]
    assert len(ids) == len(set(ids)), f"duplicate unit ids: {ids}"
    for thread in [u for u in units if u.parent_id]:
        assert thread.id.startswith(thread.parent_id + ".")


# --- an empty reply is retryable, a refusal is not ----------------------------


def test_empty_reply_has_its_own_exception_type():
    assert issubclass(llm.EmptyReply, RuntimeError)


# --- a failed chunk costs that chunk, not the run -----------------------------


def test_a_failing_chunk_becomes_unsorted_turns(monkeypatch):
    """One chunk failing must not discard the chunks that worked.

    Each chunk is a separate paid call. Losing all of them because the third one came back
    malformed is the expensive version of the mistake this whole design is meant to avoid.
    """
    seen = []

    def sometimes_fails(model, system, user, max_tokens, label="", schema=None):
        seen.append(user)
        if len(seen) == 2:
            raise RuntimeError("the model refused this request")
        return covering_reply(user)

    monkeypatch.setattr(llm, "_text", sometimes_fails)
    monkeypatch.setattr(llm, "_keep", lambda *a, **k: None)
    events = [make_event(i) for i in range(2000)]
    units = llm.LLMSegmenter().segment(events)

    covered = [eid for u in units if u.parent_id for eid in u.event_ids]
    assert sorted(covered) == sorted(e.id for e in events), "nothing may be lost"
    unsorted = [u for u in units if u.parent_id and u.label.startswith("unsorted")]
    assert unsorted, "the failed chunk's turns should be kept as unsorted threads"
