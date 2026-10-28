"""The one command that writes outside bonsai's own directory, and how to stop it.

bonsai is meant to sit beside other context-management tools without fighting them. Everything
it does lives in ~/.bonsai/ except compaction, which rewrites the editor's transcript. That is
the only place a conflict with another tool is possible, so it has to be switchable off and
provable from a test rather than from a promise in a README.
"""

import json

import pytest

from bonsai import transcript


def a_transcript(tmp_path):
    common = {"isSidechain": False, "sessionId": "s", "cwd": "C:\\x", "version": "2.1.240"}
    path = tmp_path / "s.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    **common,
                    "type": "user" if i % 2 == 0 else "assistant",
                    "uuid": f"u{i}",
                    "parentUuid": None if i == 0 else f"u{i - 1}",
                    "message": {"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"},
                }
            )
            for i in range(6)
        ),
        encoding="utf-8",
    )
    return path


def test_readonly_blocks_writing_the_transcript(tmp_path, monkeypatch):
    """With BONSAI_READONLY set, the rewrite refuses instead of touching the file."""
    monkeypatch.setenv("BONSAI_READONLY", "1")
    path = a_transcript(tmp_path)
    before = path.read_text(encoding="utf-8")
    with pytest.raises(PermissionError, match="BONSAI_READONLY"):
        transcript.write(path, transcript.read(path))
    assert path.read_text(encoding="utf-8") == before, "the file must be untouched"


def test_writing_works_normally_without_the_switch(tmp_path, monkeypatch):
    monkeypatch.delenv("BONSAI_READONLY", raising=False)
    path = a_transcript(tmp_path)
    lines = transcript.read(path)
    transcript.write(path, lines[:3])
    assert len(transcript.read(path)) == 3


def test_readonly_is_off_by_default(monkeypatch):
    monkeypatch.delenv("BONSAI_READONLY", raising=False)
    assert transcript.read_only() is False


def test_any_non_empty_value_turns_it_on(monkeypatch):
    for value in ("1", "true", "yes", "anything"):
        monkeypatch.setenv("BONSAI_READONLY", value)
        assert transcript.read_only() is True
    monkeypatch.setenv("BONSAI_READONLY", "")
    assert transcript.read_only() is False, "an empty value should not count as set"
