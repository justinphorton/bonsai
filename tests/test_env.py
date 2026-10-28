"""Picking up a local .env, so a fresh clone needs one file and no shell ceremony.

`.gitignore` has always listed `.env`, which implies bonsai reads one. It did not. Every command
needed the key exported by hand first, which is easy to forget and awkward to explain.
"""

import os

import pytest

from bonsai.cli import load_env


@pytest.fixture(autouse=True)
def restore_environment():
    """Put the environment back afterwards.

    load_env writes straight to os.environ, which monkeypatch cannot undo for variables it did
    not set itself -- so without this, a BONSAI_QUIET set here leaked into every later test and
    silenced output they were asserting on.
    """
    before = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(before)


def test_reads_keys_from_a_local_env_file(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=sk-ant-test\n", encoding="utf-8")
    load_env()
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-test"


def test_an_already_set_variable_wins(tmp_path, monkeypatch):
    """A key exported in the shell is the more deliberate choice, so a file never overrides it."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-the-shell")
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=from-the-file\n", encoding="utf-8")
    load_env()
    assert os.environ["ANTHROPIC_API_KEY"] == "from-the-shell"


def test_comments_blanks_and_quotes_are_handled(tmp_path, monkeypatch):
    monkeypatch.delenv("BONSAI_MODEL", raising=False)
    monkeypatch.delenv("BONSAI_QUIET", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        '# a comment\n\nBONSAI_MODEL="claude-opus-5"\n  BONSAI_QUIET = 1 \n', encoding="utf-8"
    )
    load_env()
    assert os.environ["BONSAI_MODEL"] == "claude-opus-5"
    assert os.environ["BONSAI_QUIET"] == "1"


def test_no_env_file_is_fine(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    load_env()  # must not raise


def test_a_malformed_line_is_skipped_not_fatal(tmp_path, monkeypatch):
    monkeypatch.delenv("GOOD", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("this line has no equals sign\nGOOD=yes\n", encoding="utf-8")
    load_env()
    assert os.environ["GOOD"] == "yes"
