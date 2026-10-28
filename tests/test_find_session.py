"""Turning a working directory into Claude Code's project-folder name.

Claude Code names each project folder after the path with every character that is not a letter
or a digit replaced by a dash. Getting this subtly wrong means bonsai simply cannot find the
conversation, with an error that points at the folder it failed to match -- which is what
happened on a vault path containing an apostrophe, spaces and brackets.
"""

import pytest

from bonsai.cli import project_folder

# Left: a real working directory. Right: the folder Claude Code actually created for it.
REAL_PATHS = [
    ("C:\\CompSci\\bonsai", "c--CompSci-bonsai"),
    ("C:\\CompSci\\CS440", "c--CompSci-CS440"),
    ("C:\\CompSci\\CS440-MP3", "c--CompSci-CS440-MP3"),
    ("C:\\WINDOWS\\System32", "C--WINDOWS-System32"),
    ("C:\\Vault\\Justin's Vault (Sync)", "C--Vault-Justin-s-Vault--Sync-"),
]


@pytest.mark.parametrize(("path", "folder"), REAL_PATHS)
def test_matches_the_folder_claude_code_really_made(path, folder):
    assert project_folder(path) == folder.lower()


def test_apostrophes_spaces_and_brackets_all_become_dashes():
    """The vault case: three different characters, each previously left alone."""
    assert project_folder("C:\\a\\Justin's Vault (Sync)") == "c--a-justin-s-vault--sync-"


def test_dots_and_other_punctuation_too():
    assert project_folder("C:\\my.proj\\sub_dir") == "c--my-proj-sub-dir"


def test_case_is_folded_so_matching_is_not_drive_letter_sensitive():
    assert project_folder("c:\\CompSci\\X") == project_folder("C:\\CompSci\\X")
