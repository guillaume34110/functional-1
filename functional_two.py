"""Native display and learned menu entry for the Functional 2 candidate."""
import base64
from pathlib import Path

import numpy as np

from model import ACTIONS
from native_render import NativeRenderer
from play import NeuralSession


class FunctionalTwoSession(NeuralSession):
    def __init__(self, release: Path):
        self.renderer = NativeRenderer(release / "native-render.npz")
        with np.load(release / "menus.npz", allow_pickle=False) as bundle:
            self.title = bundle["title"]
            transitions = bundle["transitions"]
        if (self.title.shape != (144, 160) or self.title.dtype != np.uint8 or (self.title > 3).any()
                or transitions.shape != (2, len(ACTIONS)) or transitions.dtype != np.uint8
                or (transitions[0] > 1).any() or not (transitions[0] == 1).any()):
            raise ValueError("invalid Functional 2 menu weights")
        self.title_frame = base64.b64encode(self.title.tobytes()).decode()
        self.starts = transitions[0] == 1
        self.title_logo = self.title[1:118:2, 1::2]
        super().__init__(release / "model.pt", release / "first-piece.npz")

    def reset(self):
        self.previous_low = None
        self.menu_screen = self.title
        return super().reset()

    def result(self, elapsed):
        if self.menu_screen is None:
            low = self.frame[0, 0].numpy()
            if self.previous_low is None or not np.array_equal(low, self.previous_low):
                frame = self.renderer(low)
                self.previous_low = low.copy()
                self.previous_frame = base64.b64encode(frame.tobytes()).decode()
                # The reference returns to the title after the visible top-out animation.
                if np.array_equal(low[:len(self.title_logo)], self.title_logo):
                    self.menu_screen = frame
            encoded = self.previous_frame
        else:
            encoded = self.title_frame if self.menu_screen is self.title else self.previous_frame
        return {"frame": encoded,
                "width": 160, "height": 144, "release": "Functional 2",
                "menu": self.menu_screen is not None,
                "steps": self.steps, "inference_ms": elapsed}

    def step(self, action):
        if self.menu_screen is None:
            return super().step(action)
        if action not in ACTIONS:
            raise ValueError("unknown action")
        if self.starts[ACTIONS.index(action)]:
            self.menu_screen = None
            return super().reset()
        return self.result(0.0)
