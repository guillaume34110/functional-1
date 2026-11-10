"""Fit the V3 title model on transitions from the custom menu specification."""
import argparse
import base64
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn
from torch.nn import functional as F

from browser_assets import encode
from color_art import lettering
from model import ACTIONS


LEVELS = 10
TITLE_ROW = 96
RELEASE = Path(__file__).parent / "releases" / "functional-3"


def target(level, button):
    change = 1 if button in ("right", "up") else -1 if button in ("left", "down") else 0
    return min(LEVELS - 1, max(0, level + change)), int(button in ("start", "a"))


def title_strips():
    color = json.loads((RELEASE / "color.json").read_text())
    raw = base64.b64decode(color["screens"]["title"])
    title = Image.frombytes("P", (160, 144), raw)
    strips = []
    for level in range(1, LEVELS + 1):
        art = title.copy()
        draw = ImageDraw.Draw(art)
        draw.rectangle((17, TITLE_ROW, 142, 130), fill=6)
        draw.line((33, 105, 38, 100), fill=15, width=2)
        draw.line((33, 105, 38, 110), fill=15, width=2)
        draw.line((126, 105, 121, 100), fill=15, width=2)
        draw.line((126, 105, 121, 110), fill=15, width=2)
        lettering(art, 80, 99, f"LEVEL {level}", 5, centered=True)
        lettering(art, 80, 117, "Z / ENTER TO PLAY", 11, centered=True)
        strips.append(encode(np.asarray(art)[TITLE_ROW:]))
    return {"title_row": TITLE_ROW, "strips": strips}


def replay(model, seed):
    rng = random.Random(seed)
    level_basis = torch.eye(LEVELS)
    action_basis = torch.eye(len(ACTIONS))
    navigation = [button for button in ACTIONS if button not in ("a", "start")]
    errors = 0
    for initial in range(LEVELS):
        for _ in range(6):
            level = predicted = initial
            for button in rng.choices(navigation, k=24) + [rng.choice(("a", "start"))]:
                level, start = target(level, button)
                logits = model(torch.cat((level_basis[predicted], action_basis[ACTIONS.index(button)])))
                predicted = int(logits[:LEVELS].argmax())
                errors += (predicted, int(logits[LEVELS:].argmax())) != (level, start)
    return {"episodes": 60, "transitions": 1500, "errors": errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir()
    torch.set_num_threads(1)
    torch.manual_seed(193)
    model = nn.Sequential(nn.Linear(LEVELS + len(ACTIONS), 16), nn.ReLU(), nn.Linear(16, LEVELS + 2))
    inputs = torch.cat((torch.eye(LEVELS).repeat_interleave(len(ACTIONS), dim=0),
                        torch.eye(len(ACTIONS)).repeat(LEVELS, 1)), dim=1)
    expected = torch.tensor([target(level, button) for level in range(LEVELS) for button in ACTIONS])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    for _ in range(500):
        logits = model(inputs)
        loss = F.cross_entropy(logits[:, :LEVELS], expected[:, 0])
        loss = loss + F.cross_entropy(logits[:, LEVELS:], expected[:, 1])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.inference_mode():
        logits = model(inputs)
        predictions = torch.stack((logits[:, :LEVELS].argmax(1), logits[:, LEVELS:].argmax(1)), dim=1)
        if not torch.equal(predictions, expected):
            raise ValueError("Menu model failed to fit the complete transition set")
        metrics = {
            "parameters": sum(p.numel() for p in model.parameters()),
            "train_transitions": len(inputs),
            "validation": replay(model, 2711),
            "test": replay(model, 4813),
        }
        if metrics["validation"]["errors"] or metrics["test"]["errors"]:
            raise ValueError(f"Menu transition replay failed: {metrics}")
        torch.onnx.export(model, inputs[:1], args.output / "menu.onnx", opset_version=18,
                          input_names=["menu_input"], output_names=["logits"])
    (args.output / "menu-title.json").write_text(json.dumps(title_strips(), separators=(",", ":")) + "\n")
    (args.output / "menu-fit.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
