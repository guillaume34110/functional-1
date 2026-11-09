"""Digit readout and shared learned carry for the three visible counters."""
import torch
from torch import nn
from torch.nn import functional as F

from collect import COUNTER_FIELDS


SLOTS = tuple((y // 8, x // 8 + i) for _, y, x, width in COUNTER_FIELDS for i in range(width))
FEATURES = 21


def row_strength(board, object_tiles):
    density = torch.isin(board, object_tiles).float().mean(2)
    return density.pow(32).sum(1, keepdim=True)


class CounterReadout(nn.Module):
    def __init__(self, glyphs, legacy_ids, addends, pulsed=False):
        super().__init__()
        if (glyphs.dtype != torch.uint8 or glyphs.shape != (11, 4, 4) or torch.any(glyphs > 3)
                or len(glyphs.flatten(1).unique(dim=0)) != 11
                or legacy_ids.dtype != torch.long or legacy_ids.shape != (11,)
                or addends.dtype != torch.long or addends.ndim != 3 or addends.shape[1:] != (2, 5)
                or len(addends) < 2 or addends[0].any() or torch.any((addends < 0) | (addends > 9))):
            raise ValueError("invalid counter glyphs or event vocabulary")
        self.register_buffer("glyphs", glyphs)
        self.register_buffer("legacy_ids", legacy_ids)
        self.register_buffer("addends", addends)
        y, x = torch.tensor(SLOTS).T
        self.register_buffer("rows", y, persistent=False)
        self.register_buffer("columns", x, persistent=False)
        self.pulsed = pulsed
        if pulsed:
            if len(addends) < 3 or addends[1, 1].any() or not addends[1, 0].any() or not addends[2:, 1].any(1).all():
                raise ValueError("pulse vocabulary must contain a deposit and clear events")
            self.events = nn.Linear(1, len(addends) - 2)
            self.delay = nn.Embedding(len(addends), 1)
        else:
            self.events = nn.Sequential(nn.Linear(2 * FEATURES, 16), nn.SiLU(), nn.Linear(16, len(addends)))
        self.digit = nn.Sequential(nn.Linear(22, 32), nn.SiLU(), nn.Linear(32, 22))
        self.level = nn.Linear(1, 1)

    def patches(self, frames):
        return frames[:, 0].reshape(len(frames), 18, 4, 20, 4).permute(0, 1, 3, 2, 4)

    def read(self, frames):
        tiles = self.patches(frames)[:, self.rows, self.columns]
        match = (tiles[:, :, None] == self.glyphs).flatten(3).all(3)
        return match.long().argmax(2).masked_fill(~match.any(2), -1)

    def paint(self, frames, digits, use):
        grid = self.patches(frames)
        existing = grid[:, self.rows, self.columns]
        glyphs = torch.where(use[:, None, None, None], self.glyphs[digits.clamp_min(0)], existing)
        positions = (self.rows * grid.size(2) + self.columns)[None, :, None, None].expand_as(glyphs)
        painted = grid.flatten(1, 2).scatter(1, positions, glyphs).reshape_as(grid)
        return painted.permute(0, 1, 3, 2, 4).reshape_as(frames)

    def digit_logits(self, digits, addends, carry):
        features = torch.cat((F.one_hot(digits, 11), F.one_hot(addends, 10), carry[..., None]), dim=-1)
        return self.digit(features.to(self.digit[0].weight.dtype))

    def add(self, digits, addends):
        carry = torch.zeros_like(digits[..., 0])
        result = []
        for column in range(digits.size(-1) - 1, -1, -1):
            decoded = self.digit_logits(digits[..., column], addends[..., column], carry).argmax(-1)
            result.append(decoded % 11)
            carry = decoded // 11
            if column == digits.size(-1) - 1:
                units_carry = carry
        return torch.stack(result[::-1], dim=-1), units_carry

    def forward(self, digits, features):
        return self.apply_code(digits, self.events(features).argmax(1))

    def apply_code(self, digits, code):
        points_lines, carry = self.add(digits[:, :10].reshape(-1, 2, 5), self.addends[code])
        up = (self.level(carry[:, 1, None].to(self.level.weight)) > 0).long()
        level, _ = self.add(digits[:, 10:], F.pad(up, (1, 0)))
        return torch.cat((points_lines.flatten(1), level), dim=1)

    def consume(self, digits, pending, code):
        delayed = self.delay(code)[:, 0] > 0
        for event in (pending, code.masked_fill(delayed, 0)):
            active = event[:, None] != 0
            updated = self.apply_code(digits.masked_fill(~active, 0), event)
            digits = torch.where(active, updated, digits)
        return digits, code.masked_fill(~delayed, 0)
