"""Export the frozen initial screens and presentation weights for browser inference."""
import argparse
import base64
import json
from pathlib import Path

import torch

from functional_two import FunctionalTwoSession
from model import ACTIONS
from native_render import tile_keys
from play import RELEASE, NeuralSession
from token_model import decode_tokens


def encode(pixels):
    return base64.b64encode(pixels.tobytes()).decode("ascii")


def release_metadata(session):
    model = session.model
    data = {
        "actions": ACTIONS,
        "initial_frame": encode(session.frame.numpy()),
    }
    if session.state is not None:
        data["initial_state"] = session.state.tolist()
    if model.pause_transition is not None:
        with torch.inference_mode():
            panel = decode_tokens(model.pause_panel[None].long(), model.codebook)
            phase = model.pause_phase(session.frame, session.state, torch.tensor([False]))
        data["pause"] = {
            "frame": encode(panel.numpy()),
            "transitions": model.pause_transition.tolist(),
            "resume_action": int(model.pause_resume_action),
            "initial_phase": int(phase[0]),
        }
    if isinstance(session, FunctionalTwoSession):
        data["title"] = encode(session.title)
        data["title_logo"] = encode(session.title_logo)
        data["starts"] = session.starts.tolist()
        data["level_frames"] = [encode(session.frame_for_level(level).numpy()) for level in session.levels]
        data["color_level_10"] = encode(session.frame_for_level(10).numpy())
        data["counter_digits"] = tile_keys(model.counter.glyphs.numpy()).tolist()
        data["native"] = {
            "keys": session.renderer.keys.tolist(),
            "tiles": session.renderer.tiles.reshape(-1, 64).tolist(),
        }
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    sessions = {
        "v1": NeuralSession(RELEASE / "model.pt", RELEASE / "first-piece.npz"),
        "v2": FunctionalTwoSession(RELEASE.with_name("functional-2")),
    }
    args.destination.mkdir(parents=True, exist_ok=True)
    for version, session in sessions.items():
        (args.destination / f"{version}.json").write_text(
            json.dumps(release_metadata(session), separators=(",", ":")) + "\n", encoding="utf-8",
        )


if __name__ == "__main__":
    main()
