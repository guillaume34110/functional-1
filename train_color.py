"""Reformat the original captures and fit the V3 indexed-color output layer."""
import argparse
import base64
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from collect import COUNTER_FIELDS, DIGITS
from color_art import PALETTE, screens, tile
from native_render import NativeRenderer, patches, tile_keys
from object_annotations import PATTERNS
from token_model import decode_tokens


BOARD = (slice(None), slice(2, 12))
HUD = tuple((y // 8, x // 8 + digit) for _, y, x, length in COUNTER_FIELDS for digit in range(length))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encode(array):
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode("ascii")


class CapturePalette:
    def __init__(self, release):
        self.art = screens()
        self.native = NativeRenderer(release / "native-render.npz")
        self.block_keys = tile_keys(PATTERNS)
        self.sprites = np.stack([tile(kind) for kind in range(7)])
        self.blank_key = tile_keys(np.full((4, 4), 3, dtype=np.uint8)).item()
        self.blank = np.full((8, 8), 6, dtype=np.uint8)
        self.blank[0, 0] = 7
        self.board_shades = np.array([8, 36, 38, 6], dtype=np.uint8)
        self.hud_shades = np.array([5, 4, 3, 1], dtype=np.uint8)
        self.digit_keys = tile_keys(DIGITS[:, 1::2, 1::2])
        with np.load(release / "menus.npz", allow_pickle=False) as bundle:
            self.logo = bundle["title"][1:118:2, 1::2]
        weights = torch.load(release / "model.pt", map_location="cpu", weights_only=True)
        weights = weights.get("model", weights)
        self.pause = decode_tokens(weights["pause_panel"][None].long(), weights["codebook"])[0, 0].numpy()

    def convert(self, source):
        low = patches(source, 4)
        keys = tile_keys(low)
        values = np.repeat(np.repeat(low, 2, axis=-2), 2, axis=-1)
        target = np.broadcast_to(self.art["game"], (len(source), 144, 160)).copy()
        canvas = patches(target, 8)
        board_keys = keys[:, :, 2:12]
        board = self.board_shades[values[:, :, 2:12]]
        board[board_keys == self.blank_key] = self.blank
        preview_keys = keys[:, :5, 13:19]
        preview = canvas[:, 1:6, 13:19]
        for key, sprite in zip(self.block_keys, self.sprites, strict=True):
            board[board_keys == key] = sprite
            preview[preview_keys == key] = sprite
        canvas[:, :, 2:12] = board
        for row, column in HUD:
            key = keys[:, row, column]
            index = self.native.keys.searchsorted(key).clip(max=len(self.native.keys) - 1)
            known = self.native.keys[index] == key
            glyph = np.where(known[:, None, None], self.native.tiles[index], values[:, row, column])
            canvas[:, row, column] = self.hud_shades[glyph]
        playing = np.isin(keys[:, 15, 18], self.digit_keys)
        title = (source[:, :len(self.logo)] == self.logo).all((1, 2))
        pause = (source == self.pause).all((1, 2))
        target[~playing] = self.art["over"]
        target[title] = self.art["title"]
        target[pause] = self.art["pause"]
        return target, playing & ~title & ~pause


def tile_pairs(source, color, playing):
    gray, target = patches(source[playing], 4), patches(color[playing], 8)
    rows, cols = zip(*HUD, strict=True)
    for a, b in ((gray[:, :, 2:12], target[:, :, 2:12]), (gray[:, rows, cols], target[:, rows, cols])):
        keys = tile_keys(a).reshape(-1)
        _, first, counts = np.unique(keys, return_index=True, return_counts=True)
        yield zip(keys[first], b.reshape(-1, 64)[first], counts, strict=True)


def convert_corpus(corpora, destination, palette):
    converted, seen = [], set()
    rom_hash = None
    for corpus in corpora:
        manifest = json.loads((corpus / "manifest.json").read_text())
        if manifest["status"] != "complete":
            raise ValueError(f"incomplete dataset: {corpus}")
        context = manifest["context"]
        rom = context["rom"]["sha256"]
        rom_hash = rom_hash or rom
        if rom_hash != rom or context["frames_per_action"] != 4:
            raise ValueError(f"incompatible ROM or timing: {corpus}")
        for entry in manifest["episodes"]:
            split, prefix = entry["split"], entry["prefix_sha256"]
            if prefix in seen or split not in ("train", "validation", "test"):
                raise ValueError(f"duplicate episode prefix or invalid split: {entry['file']}")
            seen.add(prefix)
            path = (corpus / entry["file"]).resolve()
            if not path.is_relative_to(corpus.resolve()) or digest(path) != entry["sha256"]:
                raise ValueError(f"invalid source episode: {path}")
            with np.load(path, allow_pickle=False) as data:
                source, actions, metadata = data["frames"], data["actions"], data["metadata"]
            if (json.loads(str(metadata))["source"] != "capture" or source.dtype != np.uint8
                    or source.shape != (len(actions) + 1, 72, 80) or source.max() > 3
                    or actions.dtype != np.uint8 or actions.ndim != 1 or actions.max() >= len(context["actions"])):
                raise ValueError(f"invalid capture arrays: {path}")
            color = np.empty((len(source), 144, 160), dtype=np.uint8)
            playing = np.empty(len(source), dtype=bool)
            for start in range(0, len(source), 64):
                chunk = slice(start, start + 64)
                color[chunk], playing[chunk] = palette.convert(source[chunk])
            filename = f"{split}-{len(converted):03}.npz"
            output = destination / filename
            np.savez_compressed(output, source=source, frames=color, actions=actions, playing=playing)
            converted.append({**entry, "file": filename, "source_sha256": entry["sha256"],
                              "sha256": digest(output), "frames": len(source)})
    (destination / "manifest.json").write_text(json.dumps({
        "rom_sha256": rom_hash, "palette": PALETTE, "episodes": converted,
        "shape": [144, 160], "dtype": "uint8", "frames_per_action": 4,
    }, indent=2) + "\n")
    return converted


def fit(destination, episodes):
    votes = (defaultdict(Counter), defaultdict(Counter))
    metrics = defaultdict(Counter)
    for split in ("train", "validation", "test"):
        for entry in episodes:
            if entry["split"] != split:
                continue
            with np.load(destination / entry["file"], allow_pickle=False) as data:
                source, color, playing = data["source"], data["frames"], data["playing"]
            metrics[split].update(episodes=1, frames=len(source), gameplay_frames=int(playing.sum()))
            for bank, pairs in enumerate(tile_pairs(source, color, playing)):
                for key, target, count in pairs:
                    raw = target.tobytes()
                    if split == "train":
                        votes[bank][int(key)][raw] += int(count)
                    else:
                        chosen = tables[bank].get(int(key))
                        metrics[split]["tiles"] += int(count)
                        metrics[split]["covered"] += int(count) * (chosen is not None)
                        metrics[split]["exact"] += int(count) * (chosen == raw)
        if split == "train":
            if any(not bank for bank in votes):
                raise ValueError("no observed training tiles")
            # Maximum-likelihood categorical output for each observed tile in each display role.
            tables = [{key: outcomes.most_common(1)[0][0] for key, outcomes in bank.items()} for bank in votes]
    if not all(metrics[split]["frames"] for split in ("train", "validation", "test")):
        raise ValueError("missing independent dataset split")
    return tables, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, action="append", required=True)
    parser.add_argument("--base", type=Path, default=Path("releases/functional-2"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True)
    dataset = args.output / "dataset"
    dataset.mkdir()
    palette = CapturePalette(args.base)
    records = convert_corpus(args.corpus, dataset, palette)
    tables, metrics = fit(dataset, records)
    if not set(map(int, palette.block_keys)).issubset(tables[0]):
        raise ValueError("training corpus does not cover all seven block textures")
    result = {
        "palette": PALETTE, "screens": {name: encode(image) for name, image in palette.art.items()},
        "banks": [{key: base64.b64encode(value).decode("ascii") for key, value in bank.items()} for bank in tables],
        "blocks": palette.block_keys.tolist(), "hud": HUD, "board_shades": palette.board_shades.tolist(),
        "hud_shades": palette.hud_shades.tolist(),
        "training": {"base_sha256": digest(args.base / "model.pt"),
                     "dataset_sha256": digest(dataset / "manifest.json"), "splits": metrics},
    }
    (args.output / "color.json").write_text(json.dumps(result, separators=(",", ":")) + "\n")
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
