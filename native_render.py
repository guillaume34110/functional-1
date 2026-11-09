"""Restore native-size tile patterns from a paired reference corpus."""
from pathlib import Path

import numpy as np


def patches(frames, size):
    rows, columns = 144 // 8, 160 // 8
    return frames.reshape(-1, rows, size, columns, size).transpose(0, 1, 3, 2, 4)


def tile_keys(tiles):
    shifts = np.arange(0, 32, 2, dtype=np.uint32)
    return (tiles.reshape(*tiles.shape[:-2], 16).astype(np.uint32) << shifts).sum(-1, dtype=np.uint32)


class NativeRenderer:
    def __init__(self, weights: Path):
        with np.load(weights, allow_pickle=False) as data:
            self.keys = data["keys"]
            self.tiles = data["tiles"]
        if (self.keys.ndim != 1 or not len(self.keys) or self.keys.dtype != np.uint32
                or (self.keys[1:] <= self.keys[:-1]).any()
                or self.tiles.shape != (len(self.keys), 8, 8)
                or self.tiles.dtype != np.uint8 or (self.tiles > 3).any()):
            raise ValueError("invalid native renderer vocabulary")

    def __call__(self, frame):
        if frame.shape != (72, 80) or frame.dtype != np.uint8 or (frame > 3).any():
            raise ValueError("expected indexed 80×72 frame")
        low = patches(frame, 4)[0]
        keys = tile_keys(low)
        indices = self.keys.searchsorted(keys).clip(max=len(self.keys) - 1)
        known = self.keys[indices] == keys
        tiles = np.repeat(np.repeat(low, 2, axis=-2), 2, axis=-1)
        tiles[known] = self.tiles[indices[known]]
        return tiles.transpose(0, 2, 1, 3).reshape(144, 160)
