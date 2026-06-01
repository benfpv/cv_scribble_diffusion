"""Unit tests for the standalone PromptEditor component."""

import pytest

from cv_scribble_diffusion.app.prompt_editor import PromptEditor


def _identity_cursor_index(text, x, cursor, max_chars):
    """Test cursor mapper: clamp the raw x as a character index."""
    return max(0, min(x, len(text)))


def make_editor(initial="hello", max_chars=160, on_commit=None):
    notices = []
    editor = PromptEditor(
        initial_prompt=initial,
        max_chars=max_chars,
        cursor_index_fn=_identity_cursor_index,
        notify=lambda *a, **k: notices.append((a, k)),
        on_commit=on_commit,
    )
    return editor, notices


def test_initial_prompt_is_cleaned_and_committed():
    editor, _ = make_editor("line\none caf\u00e9")
    assert editor.committed() == "line one caf"
    assert editor.editing is False


def test_begin_edit_seeds_draft_from_committed():
    editor, _ = make_editor("sky")
    editor.begin_edit()
    assert editor.editing is True
    assert editor.draft == "sky"
    assert editor.cursor == len("sky")


def test_insert_char_respects_max_chars():
    editor, notices = make_editor("", max_chars=3)
    editor.begin_edit(0)
    for ch in "abcdef":
        editor.insert_char(ch)
    assert editor.draft == "abc"
    assert any("limit reached" in a[0] for a, _ in notices)


def test_commit_returns_changed_and_invokes_callback():
    committed = []
    editor, _ = make_editor("old", on_commit=lambda changed, text: committed.append((changed, text)))
    editor.begin_edit(0)
    editor.draft = "new"
    editor.cursor = 3
    changed = editor.commit()
    assert changed is True
    assert editor.committed() == "new"
    assert committed == [(True, "new")]


def test_commit_unchanged_reports_false():
    editor, _ = make_editor("same")
    editor.begin_edit()
    changed = editor.commit()
    assert changed is False


def test_cancel_restores_committed_prompt():
    editor, _ = make_editor("keep")
    editor.begin_edit(0)
    editor.draft = "discard"
    editor.cancel()
    assert editor.committed() == "keep"
    assert editor.editing is False


def test_select_all_and_delete_selection():
    editor, _ = make_editor("abcdef")
    editor.begin_edit()
    editor.select_all()
    assert editor.selection_bounds() == (0, 6)
    editor.handle_keypress(8, 8)  # Backspace deletes selection
    assert editor.draft == ""


def test_handle_keypress_enter_commits():
    editor, _ = make_editor("a")
    editor.begin_edit(1)
    editor.insert_char("b")
    editor.handle_keypress(13, 13)
    assert editor.committed() == "ab"
    assert editor.editing is False


def test_info_returns_none_when_box_hidden():
    editor, _ = make_editor("x")
    assert editor.info(show_box=False) is None


def test_info_reports_editing_state():
    editor, _ = make_editor("x")
    editor.begin_edit(0)
    editor.select_all()
    info = editor.info(show_box=True)
    assert info is not None
    assert info.editing is True
    assert info.selection_start == 0
    assert info.selection_end == 1


def test_committed_is_thread_safe_accessor():
    editor, _ = make_editor("seed")
    # Editing the draft must not change the committed value until commit.
    editor.begin_edit(0)
    editor.draft = "draft only"
    assert editor.committed() == "seed"
