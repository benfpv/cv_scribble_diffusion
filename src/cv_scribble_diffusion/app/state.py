"""Thread-safe container for the generation lifecycle state.

The display loop (``App.run``) and the diffusion daemon thread
(``App.async_diffusion``) both read and mutate the generation state. This
module centralizes that shared state behind a single lock so every field
access is atomic and the synchronization policy lives in one place.
"""
import enum
import threading
from typing import Optional


class GenState(enum.Enum):
    """Explicit generation lifecycle states."""
    IDLE = "idle"              # No generation requested
    READY = "ready"            # Strokes present, should generate
    GENERATING = "generating"  # Pipeline currently running
    RESETTING = "resetting"    # Reset requested, awaiting thread drain


class GenerationState:
    """Lock-guarded view of state shared between the display and diffusion threads.

    Each attribute is exposed as a property whose get/set is serialized by a
    single lock. Use :meth:`transition` for atomic conditional state changes
    (compare-and-set) where a plain setter would race.
    """

    def __init__(self, inference_steps: int):
        self._lock = threading.Lock()
        self._gen_state = GenState.IDLE
        self._inference_steps = inference_steps
        self._current_inference_steps = 1
        self._image_size_index = 0
        self._thread_error: Optional[str] = None
        self._gen_count = 0

    @property
    def gen_state(self) -> GenState:
        with self._lock:
            return self._gen_state

    @gen_state.setter
    def gen_state(self, value: GenState):
        with self._lock:
            self._gen_state = value

    @property
    def inference_steps(self) -> int:
        with self._lock:
            return self._inference_steps

    @inference_steps.setter
    def inference_steps(self, value: int):
        with self._lock:
            self._inference_steps = value

    @property
    def current_inference_steps(self) -> int:
        with self._lock:
            return self._current_inference_steps

    @current_inference_steps.setter
    def current_inference_steps(self, value: int):
        with self._lock:
            self._current_inference_steps = value

    @property
    def image_size_index(self) -> int:
        with self._lock:
            return self._image_size_index

    @image_size_index.setter
    def image_size_index(self, value: int):
        with self._lock:
            self._image_size_index = value

    @property
    def thread_error(self) -> Optional[str]:
        with self._lock:
            return self._thread_error

    @thread_error.setter
    def thread_error(self, value: Optional[str]):
        with self._lock:
            self._thread_error = value

    @property
    def gen_count(self) -> int:
        with self._lock:
            return self._gen_count

    @gen_count.setter
    def gen_count(self, value: int):
        with self._lock:
            self._gen_count = value

    def transition(self, expected: GenState, new: GenState) -> bool:
        """Atomically set the state to ``new`` only if it currently equals
        ``expected``. Returns ``True`` when the transition was applied."""
        with self._lock:
            if self._gen_state == expected:
                self._gen_state = new
                return True
            return False
