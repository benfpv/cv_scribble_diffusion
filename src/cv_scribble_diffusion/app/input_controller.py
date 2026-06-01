"""Input translation for the application.

Maps raw OpenCV mouse and keyboard events into prompt-editor interactions,
toolbar actions, and canvas strokes. The controller operates against an
``app`` action sink so the translation lives in one place and the display loop
stays thin.
"""
import cv2

from cv_scribble_diffusion.app.state import GenState


class InputController:
    """Translates raw mouse/keyboard events into application actions."""

    def __init__(self, app):
        self.app = app

    # -- keyboard -------------------------------------------------------------

    def handle_keypress(self, key_code: int):
        """Map keyboard input to toolbar-equivalent actions."""
        app = self.app
        if key_code < 0:
            return

        # cv2.waitKeyEx returns a platform-specific code. The low byte covers
        # ASCII/control keys across backends.
        low_byte = key_code & 0xFF

        if app.prompt.editing and app.prompt.handle_keypress(key_code, low_byte):
            return

        if low_byte == 27:  # Esc
            app._request_exit(source="Keyboard")
            return

        if low_byte == 32:  # Space
            app._clear_exit_confirmation()
            if app._gen_state != GenState.RESETTING:
                app._announce("Reset Canvas", source="Keyboard")
                app.reset_canvas()
            return
        if low_byte in (13, 10):  # Enter
            app._clear_exit_confirmation()
            app._save_image()
            return
        if low_byte == 9:  # Tab
            app._clear_exit_confirmation()
            app.mask_visibility_toggle = not app.mask_visibility_toggle
            state = "On" if app.mask_visibility_toggle else "Off"
            app._announce(f"Toggle Mask Visibility [{state}]", source="Keyboard")
            return

        # Undo: support Ctrl+Z (26), plain z/Z, and u/U as fallback.
        if low_byte in (26, ord("z"), ord("Z"), ord("u"), ord("U")):
            app._clear_exit_confirmation()
            app._undo_last_stroke()
            return

        # Arrow keys can vary by backend; support common forms.
        if key_code in (2424832, 81):  # Left
            app._clear_exit_confirmation()
            app._adjust_brush_thickness(-1)
            return
        if key_code in (2555904, 83):  # Right
            app._clear_exit_confirmation()
            app._adjust_brush_thickness(1)
            return

    # -- mouse ----------------------------------------------------------------

    def handle_mouse(self, event, x, y, flags, param):
        """OpenCV mouse callback: toolbar hits and canvas strokes."""
        app = self.app
        cfg = app.cfg
        canvas = app.canvas
        if event == cv2.EVENT_LBUTTONDBLCLK:
            if app.ui.prompt_hit_test(x, y):
                app._clear_exit_confirmation()
                app.prompt.handle_double_click(x)
                return

        if event == cv2.EVENT_LBUTTONDOWN:
            if app.ui.prompt_hit_test(x, y):
                app._clear_exit_confirmation()
                app.prompt.begin_pointer_edit(x)
                return

            if app.prompt.editing:
                app.prompt.commit()

            action = app.ui.hit_test(x, y)
            if action == "exit":
                app._request_exit(source="Toolbar")
                return

            app._clear_exit_confirmation()
            if action == "reset":
                if app._gen_state != GenState.RESETTING:
                    app._announce("Reset Canvas", source="Toolbar")
                    app.reset_canvas()
                return
            elif action == "mask":
                app.mask_visibility_toggle = not app.mask_visibility_toggle
                state = "On" if app.mask_visibility_toggle else "Off"
                app._announce(f"Toggle Mask Visibility [{state}]", source="Toolbar")
                return
            elif action == "save":
                app._save_image()
                return
            elif action == "undo":
                app._undo_last_stroke()
                return
            elif action == "brush_dec":
                app._adjust_brush_thickness(-1)
                return
            elif action == "brush_inc":
                app._adjust_brush_thickness(1)
                return
            elif action == "steps_dec":
                app._adjust_max_inference_steps(-2)
                return
            elif action == "steps_inc":
                app._adjust_max_inference_steps(2)
                return
            elif action == "fps":
                app._cycle_fps()
                return

            coords = app.ui.canvas_coords(x, y)
            if coords is not None:
                if not canvas.drawing:
                    app._undo_stack.append(app._snapshot_state())
                canvas.begin_stroke(*coords)
                app._inference_steps = cfg.inference.min_inference_steps
                app._state.transition(GenState.IDLE, GenState.READY)
        elif event == cv2.EVENT_MOUSEMOVE:
            if app.prompt.dragging:
                app.prompt.update_selection_from_x(x)
                return
            if canvas.drawing:
                coords = app.ui.canvas_coords(x, y)
                if coords is not None:
                    canvas.continue_stroke(*coords)
                    app._inference_steps = cfg.inference.min_inference_steps
                    app._state.transition(GenState.IDLE, GenState.READY)
                else:
                    # If the drag leaves the canvas, preserve the stroke and terminate it.
                    canvas.end_stroke(canvas.prev_x, canvas.prev_y)
        elif event == cv2.EVENT_LBUTTONUP:
            if app.prompt.dragging:
                app.prompt.finish_selection_drag(x)
                return
            coords = app.ui.canvas_coords(x, y)
            if coords is not None:
                canvas.end_stroke(*coords)
            elif canvas.drawing:
                # Mouse-up outside canvas should still finalize drawing state.
                canvas.end_stroke(canvas.prev_x, canvas.prev_y)
