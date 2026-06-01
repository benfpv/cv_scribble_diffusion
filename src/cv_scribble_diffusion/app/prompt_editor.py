"""Prompt text-editing subsystem for the OpenCV scribble app.

Extracted from ``App`` to isolate cursor/selection/draft/commit mechanics and
the thread-safe committed-prompt accessor used by the diffusion thread. The
editor performs only text mechanics; the owner wires generation side-effects
through the ``on_commit`` callback.
"""

import threading
import time
from typing import Callable, Optional, Tuple

from cv_scribble_diffusion.ui.overlay import PromptInfo


# Type aliases for the injected collaborators.
CursorIndexFn = Callable[[str, int, int, int], int]
NotifyFn = Callable[..., None]
CommitFn = Callable[[bool, str], None]


class PromptEditor:
    """Single-line prompt editor with selection, cursor, and draft/commit.

    The committed prompt is guarded by a lock so the diffusion thread can read
    it via :meth:`committed` while the display thread edits the draft.
    """

    def __init__(
        self,
        initial_prompt: str,
        max_chars: int,
        cursor_index_fn: CursorIndexFn,
        notify: NotifyFn,
        on_commit: Optional[CommitFn] = None,
    ):
        self._max_chars = max_chars
        self._cursor_index_fn = cursor_index_fn
        self._notify = notify
        self._on_commit = on_commit

        self._lock = threading.Lock()
        self._text = self.clean_text(initial_prompt)
        self._draft = self._text
        self._cursor = len(self._draft)
        self._editing = False
        self._selection_anchor: Optional[int] = None
        self._dragging = False

    # -- public state ---------------------------------------------------------

    @property
    def editing(self) -> bool:
        return self._editing

    @property
    def dragging(self) -> bool:
        return self._dragging

    @property
    def draft(self) -> str:
        return self._draft

    @draft.setter
    def draft(self, value: str):
        self._draft = value

    @property
    def cursor(self) -> int:
        return self._cursor

    @cursor.setter
    def cursor(self, value: int):
        self._cursor = value

    @property
    def selection_anchor(self) -> Optional[int]:
        return self._selection_anchor

    @selection_anchor.setter
    def selection_anchor(self, value: Optional[int]):
        self._selection_anchor = value

    def committed(self) -> str:
        """Return the committed prompt for a generation cycle (thread-safe)."""
        with self._lock:
            return self._text

    # -- text helpers ---------------------------------------------------------

    def clean_text(self, text: str) -> str:
        """Keep prompt text single-line, printable ASCII, within the limit."""
        text = text.replace("\r", " ").replace("\n", " ")
        cleaned = "".join(ch for ch in text if 32 <= ord(ch) <= 126)
        return cleaned[:self._max_chars]

    def _clamp_cursor(self, cursor: int) -> int:
        return max(0, min(cursor, len(self._draft)))

    def _set_cursor(self, cursor: int, selecting: bool = False):
        cursor = self._clamp_cursor(cursor)
        if selecting:
            if self._selection_anchor is None:
                self._selection_anchor = self._cursor
        else:
            self._clear_selection()
        self._cursor = cursor

    def selection_bounds(self) -> Optional[Tuple[int, int]]:
        """Return active selection as (start, end), or None."""
        if self._selection_anchor is None:
            return None
        anchor = self._clamp_cursor(self._selection_anchor)
        cursor = self._clamp_cursor(self._cursor)
        if anchor == cursor:
            return None
        return (min(anchor, cursor), max(anchor, cursor))

    def _clear_selection(self):
        self._selection_anchor = None

    def _delete_selection(self) -> bool:
        selection = self.selection_bounds()
        if selection is None:
            return False
        start, end = selection
        self._draft = self._draft[:start] + self._draft[end:]
        self._cursor = start
        self._clear_selection()
        return True

    def select_all(self):
        """Select the complete draft for replacement or deletion."""
        self._cursor = len(self._draft)
        self._selection_anchor = 0 if self._draft else None

    def insert_char(self, ch: str):
        """Insert a printable character at the current cursor."""
        self.insert_text(ch)

    def insert_text(self, text: str):
        """Insert printable text, replacing the active selection if any."""
        text = "".join(ch for ch in text if 32 <= ord(ch) <= 126)
        if not text:
            return
        max_chars = self._max_chars
        selection = self.selection_bounds()
        start, end = selection if selection is not None else (self._cursor, self._cursor)
        available = max_chars - (len(self._draft) - (end - start))
        if available <= 0:
            self._notify(f"Prompt limit reached ({max_chars} chars)")
            return
        insert_text = text[:available]
        self._draft = self._draft[:start] + insert_text + self._draft[end:]
        self._cursor = start + len(insert_text)
        self._clear_selection()
        if len(insert_text) < len(text):
            self._notify(f"Prompt limit reached ({max_chars} chars)")

    # -- focus / commit lifecycle --------------------------------------------

    def begin_edit(self, cursor: Optional[int] = None):
        """Focus the editor and place the cursor."""
        if not self._editing:
            with self._lock:
                self._draft = self._text
        self._set_cursor(len(self._draft) if cursor is None else cursor)
        self._dragging = False
        self._editing = True
        self._notify("Editing prompt: Enter applies, Esc cancels", duration_s=4.0)

    def commit(self) -> bool:
        """Commit the draft. Returns whether the committed prompt changed."""
        new_prompt = self.clean_text(self._draft)
        with self._lock:
            old_prompt = self._text
            self._text = new_prompt
        self._draft = new_prompt
        self._cursor = len(new_prompt)
        self._editing = False
        self._clear_selection()
        self._dragging = False
        changed = new_prompt != old_prompt
        if self._on_commit is not None:
            self._on_commit(changed, new_prompt)
        return changed

    def cancel(self):
        """Discard edits and return to the committed prompt."""
        with self._lock:
            self._draft = self._text
        self._cursor = len(self._draft)
        self._editing = False
        self._clear_selection()
        self._dragging = False
        self._notify("Prompt edit cancelled")

    # -- pointer interaction --------------------------------------------------

    def cursor_from_x(self, x: int) -> int:
        """Translate a window x coordinate to a draft cursor index."""
        return self._cursor_index_fn(self._draft, x, self._cursor, self._max_chars)

    def update_selection_from_x(self, x: int):
        """Extend the selection to the cursor location under *x*."""
        self._set_cursor(self.cursor_from_x(x), selecting=True)

    def finish_selection_drag(self, x: int):
        """End mouse-based selection, clearing empty selections."""
        self.update_selection_from_x(x)
        self._dragging = False
        if self.selection_bounds() is None:
            self._clear_selection()

    def begin_pointer_edit(self, x: int):
        """Focus the editor from a click and start a selection drag."""
        if self._editing:
            text = self._draft
            cursor = self._cursor
        else:
            text = self.committed()
            cursor = len(text)
        cursor = self._cursor_index_fn(text, x, cursor, self._max_chars)
        self.begin_edit(cursor)
        self._selection_anchor = self._cursor
        self._dragging = True

    def handle_double_click(self, x: int):
        """Focus the editor and select all on double-click."""
        self.begin_edit(self.cursor_from_x(x) if self._editing else None)
        self.select_all()
        self._dragging = False

    # -- keyboard -------------------------------------------------------------

    def handle_keypress(self, key_code: int, low_byte: int) -> bool:
        """Handle text editing keys while the field is focused."""
        if low_byte == 1:  # Ctrl+A
            self.select_all()
            return True
        if low_byte == 27:  # Esc
            self.cancel()
            return True
        if low_byte in (13, 10):  # Enter
            self.commit()
            return True
        if low_byte == 8:  # Backspace
            if self._delete_selection():
                return True
            if self._cursor > 0:
                self._draft = (
                    self._draft[:self._cursor - 1] + self._draft[self._cursor:]
                )
                self._cursor -= 1
            return True
        if low_byte == 127 or key_code == 3014656:  # Delete
            if self._delete_selection():
                return True
            if self._cursor < len(self._draft):
                self._draft = (
                    self._draft[:self._cursor] + self._draft[self._cursor + 1:]
                )
            return True
        if key_code == 2424832:  # Left arrow
            selection = self.selection_bounds()
            if selection is not None:
                self._set_cursor(selection[0])
            else:
                self._set_cursor(self._cursor - 1)
            return True
        if key_code == 2555904:  # Right arrow
            selection = self.selection_bounds()
            if selection is not None:
                self._set_cursor(selection[1])
            else:
                self._set_cursor(self._cursor + 1)
            return True
        if key_code == 2359296:  # Home
            self._set_cursor(0)
            return True
        if key_code == 2293760:  # End
            self._set_cursor(len(self._draft))
            return True
        if 32 <= low_byte <= 126:
            self.insert_char(chr(low_byte))
            return True
        return True

    # -- rendering ------------------------------------------------------------

    def info(self, show_box: bool) -> Optional[PromptInfo]:
        """Build prompt editor state for UI rendering."""
        if not show_box:
            return None
        if self._editing:
            text = self._draft
            cursor = self._cursor
        else:
            text = self.committed()
            cursor = len(text)
        cursor_visible = self._editing and int(time.time() * 2) % 2 == 0
        selection = self.selection_bounds() if self._editing else None
        return PromptInfo(
            text=text,
            editing=self._editing,
            cursor=cursor,
            cursor_visible=cursor_visible,
            max_chars=self._max_chars,
            selection_start=selection[0] if selection is not None else 0,
            selection_end=selection[1] if selection is not None else 0,
        )
