"""Unit tests for the lock-guarded GenerationState container."""

import threading

from cv_scribble_diffusion.app.state import GenState, GenerationState


def test_defaults_are_initialized():
    state = GenerationState(inference_steps=5)
    assert state.gen_state is GenState.IDLE
    assert state.inference_steps == 5
    assert state.current_inference_steps == 1
    assert state.image_size_index == 0
    assert state.thread_error is None
    assert state.gen_count == 0


def test_properties_round_trip():
    state = GenerationState(inference_steps=1)
    state.gen_state = GenState.GENERATING
    state.inference_steps = 12
    state.current_inference_steps = 8
    state.image_size_index = 3
    state.thread_error = "boom"
    state.gen_count = 7

    assert state.gen_state is GenState.GENERATING
    assert state.inference_steps == 12
    assert state.current_inference_steps == 8
    assert state.image_size_index == 3
    assert state.thread_error == "boom"
    assert state.gen_count == 7


def test_transition_applies_only_on_match():
    state = GenerationState(inference_steps=1)
    assert state.transition(GenState.IDLE, GenState.READY) is True
    assert state.gen_state is GenState.READY
    # No-op when current state does not match expected.
    assert state.transition(GenState.IDLE, GenState.GENERATING) is False
    assert state.gen_state is GenState.READY


def test_concurrent_increments_are_atomic():
    state = GenerationState(inference_steps=1)
    lock = state._lock  # type: ignore[attr-defined]
    iterations = 2000

    def bump():
        for _ in range(iterations):
            with lock:
                state._gen_count += 1  # type: ignore[attr-defined]

    threads = [threading.Thread(target=bump) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert state.gen_count == iterations * 4
