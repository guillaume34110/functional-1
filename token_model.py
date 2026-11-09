"""Learned transitions of exact 4x4 screen patterns; no game rules or piece logic."""
from pathlib import Path
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from model import ACTIONS


def _validate_codebook(codebook):
    if (not isinstance(codebook, np.ndarray) or codebook.dtype != np.uint8 or
            codebook.ndim != 3 or codebook.shape[1:] != (4, 4) or len(codebook) == 0):
        raise ValueError("codebook must be nonempty uint8 [N,4,4]")
    if np.any(codebook > 3) or len(np.unique(codebook.reshape(len(codebook), 16), axis=0)) != len(codebook):
        raise ValueError("codebook must contain unique palette patterns in 0..3")


def _tiles(frames):
    if (not isinstance(frames, np.ndarray) or frames.dtype != np.uint8 or
            frames.ndim != 3 or frames.shape[1:] != (72, 80) or np.any(frames > 3)):
        raise ValueError("frames must be uint8 palette indices [N,72,80]")
    return frames.reshape(len(frames), 18, 4, 20, 4).transpose(0, 1, 3, 2, 4).reshape(-1, 16)


def build_codebook(frames_iterator):
    """Caller supplies training frames only; validation/test never grow the vocabulary."""
    patterns = set()
    for frames in frames_iterator:
        # Bound temporary arrays even when an entire trajectory is supplied.
        if not isinstance(frames, np.ndarray) or frames.ndim != 3:
            raise ValueError("iterator must yield frame arrays [N,72,80]")
        for start in range(0, len(frames), 256):
            patterns.update(row.tobytes() for row in np.unique(_tiles(frames[start:start + 256]), axis=0))
    if not patterns:
        raise ValueError("cannot build a codebook without training frames")
    return torch.from_numpy(np.frombuffer(b"".join(sorted(patterns)), dtype=np.uint8).copy().reshape(-1, 4, 4))


def encode_frames(frames, codebook_np):
    """Exact pattern lookup; unknown patterns use nearest Hamming match and are counted."""
    _validate_codebook(codebook_np)
    tiles = np.ascontiguousarray(_tiles(frames))
    flat_book = np.ascontiguousarray(codebook_np.reshape(len(codebook_np), 16))
    key_dtype = np.dtype((np.void, 16))
    keys = flat_book.view(key_dtype).ravel()
    query = tiles.view(key_dtype).ravel()
    order = np.argsort(keys)
    positions = np.searchsorted(keys[order], query)
    indices = order[np.minimum(positions, len(keys) - 1)]
    exact = (positions < len(keys)) & (keys[indices] == query)
    unknown = int((~exact).sum())
    if unknown:
        unique, inverse = np.unique(tiles[~exact], axis=0, return_inverse=True)
        nearest = np.empty(len(unique), dtype=np.int64)
        for start in range(0, len(unique), 128):
            distances = (unique[start:start + 128, None, :] != flat_book[None, :, :]).sum(axis=2)
            nearest[start:start + 128] = distances.argmin(axis=1)
        indices[~exact] = nearest[inverse]
    return indices.astype(np.int64, copy=False).reshape(len(frames), 18, 20), unknown


def decode_tokens(tokens, codebook):
    if tokens.dtype != torch.int64 or tokens.ndim != 3 or tuple(tokens.shape[1:]) != (18, 20):
        raise ValueError("tokens must be int64 [B,18,20]")
    if codebook.dtype != torch.uint8 or codebook.ndim != 3 or tuple(codebook.shape[1:]) != (4, 4):
        raise ValueError("codebook must be uint8 [N,4,4]")
    if torch.any(tokens < 0) or torch.any(tokens >= len(codebook)):
        raise ValueError("pattern index outside codebook")
    patches = codebook[tokens]
    return patches.permute(0, 1, 3, 2, 4).reshape(len(tokens), 1, 72, 80)


class TileDynamics(nn.Module):
    state_channels = 16

    def __init__(self, codebook):
        super().__init__()
        if not isinstance(codebook, torch.Tensor):
            raise ValueError("codebook must be a torch uint8 tensor")
        _validate_codebook(codebook.detach().cpu().numpy())
        self.register_buffer("codebook", codebook.detach().clone())
        self.embedding = nn.Embedding(len(codebook), 16)
        self.row_positions = nn.Parameter(torch.randn(18, 16) * 0.02)
        self.column_positions = nn.Parameter(torch.randn(20, 16) * 0.02)
        self.action_embedding = nn.Embedding(len(ACTIONS), 8)
        self.global_attention = nn.MultiheadAttention(16, 2, batch_first=True)
        self.gates = nn.Conv2d(56, 32, 3, padding=1)
        self.candidate = nn.Conv2d(56, 16, 3, padding=1)
        self.readout = nn.Conv2d(16, len(codebook), 1)
        self.copy_strength = nn.Parameter(torch.tensor(2.0))

    def _inputs(self, tokens, actions):
        if tokens.dtype != torch.int64 or tokens.ndim != 3 or tuple(tokens.shape[1:]) != (18, 20):
            raise ValueError("tokens must be int64 [B,18,20]")
        if actions.dtype != torch.int64 or actions.shape != (len(tokens),):
            raise ValueError("actions must be int64 [B]")
        if torch.any(tokens < 0) or torch.any(tokens >= len(self.codebook)):
            raise ValueError("pattern index outside codebook")
        positions = self.row_positions[:, None, :] + self.column_positions[None, :, :]
        features = self.embedding(tokens) + positions
        sequence = features.reshape(len(tokens), 360, 16)
        context, _ = self.global_attention(sequence, sequence, sequence, need_weights=False)
        features = features.permute(0, 3, 1, 2)
        context = context.transpose(1, 2).reshape(len(tokens), 16, 18, 20)
        buttons = self.action_embedding(actions)[:, :, None, None].expand(-1, -1, 18, 20)
        return torch.cat((features, context, buttons), dim=1), features

    def _advance_hidden(self, inputs, state):
        reset, update = self.gates(torch.cat((inputs, state), dim=1)).sigmoid().chunk(2, dim=1)
        candidate = self.candidate(torch.cat((inputs, reset * state), dim=1)).tanh()
        return (1 - update) * state + update * candidate

    def advance(self, tokens, actions, state=None):
        inputs, features = self._inputs(tokens, actions)
        if state is None:
            state = torch.zeros_like(features)
        return self._advance_hidden(inputs, state)

    def forward(self, tokens, actions, state=None):
        state = self.advance(tokens, actions, state)
        copy = F.one_hot(tokens, len(self.codebook)).permute(0, 3, 1, 2).to(state.dtype)
        return self.readout(state) + self.copy_strength * copy, state

    def predict_frame(self, frame, action, state=None):
        """Autoregressive pixel bridge; input patterns outside vocabulary use nearest match."""
        if (frame.dtype != torch.uint8 or frame.ndim != 4 or
                tuple(frame.shape[1:]) != (1, 72, 80) or torch.any(frame > 3)):
            raise ValueError("frame must be palette uint8 [B,1,72,80]")
        patches = frame[:, 0].reshape(len(frame), 18, 4, 20, 4).permute(0, 1, 3, 2, 4).reshape(len(frame), 18, 20, 16)
        distances = (patches[:, :, :, None, :] != self.codebook.reshape(1, 1, 1, -1, 16)).sum(dim=-1)
        tokens = distances.argmin(dim=-1)
        logits, state = self(tokens, action, state)
        return decode_tokens(logits.argmax(dim=1), self.codebook), state


class EventTileDynamics(TileDynamics):
    """Separate whether a location changes from which pattern should be written."""
    def __init__(self, codebook):
        super().__init__(codebook)
        del self.copy_strength
        self.change_head = nn.Conv2d(16, 1, 1)
        nn.init.constant_(self.change_head.bias, -2.0)

    def forward_details(self, tokens, actions, state=None):
        state = self.advance(tokens, actions, state)
        return self.readout(state), self.change_head(state), state

    @staticmethod
    def combine_logits(tokens, write_logits, change_logits):
        """Normalized log probabilities of a stable write/copy mixture."""
        write = F.logsigmoid(change_logits) + F.log_softmax(write_logits, dim=1)
        locations = F.one_hot(tokens, write_logits.shape[1]).permute(0, 3, 1, 2).bool()
        copy = F.logsigmoid(-change_logits).expand_as(write).masked_fill(~locations, -torch.inf)
        return torch.logaddexp(write, copy)

    def forward(self, tokens, actions, state=None):
        write_logits, change_logits, state = self.forward_details(tokens, actions, state)
        return self.combine_logits(tokens, write_logits, change_logits), state


class CalibratedEventDynamics(EventTileDynamics):
    """Keep raw gate logits for weighted BCE, correcting its prior at inference."""
    def __init__(self, codebook):
        super().__init__(codebook)
        self.register_buffer("gate_logit_offset", torch.tensor(math.log(16.0)))

    def forward(self, tokens, actions, state=None):
        write_logits, change_logits, state = self.forward_details(tokens, actions, state)
        return self.combine_logits(tokens, write_logits, change_logits - self.gate_logit_offset), state


class ObservedHistoryDynamics(CalibratedEventDynamics):
    """Expose only past visible patterns, unchanged ages, and held-button durations."""
    state_channels = 20

    def __init__(self, codebook):
        super().__init__(codebook)
        self.gates = nn.Conv2d(80, 32, 3, padding=1)
        self.candidate = nn.Conv2d(80, 16, 3, padding=1)
        # Append new modules after all legacy parameters for optimizer-state migration.
        self.tile_age_embedding = nn.Embedding(256, 4)
        self.held_duration_embedding = nn.Embedding(256, 4)
        self.register_buffer("_observed_history", torch.tensor(True))

    def advance(self, tokens, actions, state=None):
        inputs, features = self._inputs(tokens, actions)
        if state is None:
            hidden = torch.zeros_like(features)
            previous = tokens
            age = torch.zeros_like(tokens)
            held = torch.ones_like(actions)
        else:
            if state.shape != (len(tokens), ObservedHistoryDynamics.state_channels, 18, 20) or not state.is_floating_point():
                raise ValueError("history state must be floating [B,20,18,20]")
            hidden = state[:, :16]
            previous = state[:, 16].detach().long()
            previous_age = state[:, 17].detach().long()
            age = torch.where(tokens == previous, (previous_age + 1).clamp(max=255), 0)
            previous_action = state[:, 19, 0, 0].detach().long()
            previous_held = state[:, 18, 0, 0].detach().long()
            held = torch.where(actions == previous_action, (previous_held + 1).clamp(max=255), 1)
        prior_features = self.embedding(previous).permute(0, 3, 1, 2)
        ages = self.tile_age_embedding(age).permute(0, 3, 1, 2)
        durations = self.held_duration_embedding(held)[:, :, None, None].expand(-1, -1, 18, 20)
        hidden = self._advance_hidden(torch.cat((inputs, prior_features, ages, durations), dim=1), hidden)
        metadata = torch.stack((tokens.to(hidden.dtype), age.to(hidden.dtype),
                                held[:, None, None].expand_as(tokens).to(hidden.dtype),
                                actions[:, None, None].expand_as(tokens).to(hidden.dtype)), dim=1).detach()
        return torch.cat((hidden, metadata), dim=1)

    def forward_details(self, tokens, actions, state=None):
        state = self.advance(tokens, actions, state)
        return self.readout(state[:, :16]), self.change_head(state[:, :16]), state


def expand_history_kernel(weight):
    """Preserve legacy inputs and hidden columns, leaving added feature columns zero."""
    if weight.ndim != 4 or weight.shape[1] not in (56, 80):
        raise ValueError("expected convolution weights with 56 or 80 input channels")
    if weight.shape[1] == 80:
        return weight.clone()
    expanded = weight.new_zeros((weight.shape[0], 80, *weight.shape[2:]))
    expanded[:, :40] = weight[:, :40]
    expanded[:, 64:80] = weight[:, 40:56]
    return expanded


def upgrade_history_weights(old_state_dict):
    """Upgrade a flat calibrated checkpoint without modifying its original tensors."""
    if "gate_logit_offset" not in old_state_dict or "copy_strength" in old_state_dict:
        raise ValueError("upgrade requires calibrated event-model weights")
    reference = old_state_dict["embedding.weight"]
    model = ObservedHistoryDynamics(old_state_dict["codebook"]).to(device=reference.device, dtype=reference.dtype)
    upgraded = model.state_dict()
    extra = {"tile_age_embedding.weight", "held_duration_embedding.weight", "_observed_history"}
    if set(old_state_dict) != set(upgraded) and set(old_state_dict) != set(upgraded) - extra:
        raise ValueError("legacy architecture keys do not match the calibrated model")
    for name, value in old_state_dict.items():
        updated = expand_history_kernel(value) if name in ("gates.weight", "candidate.weight") else value.clone()
        if updated.shape != upgraded[name].shape:
            raise ValueError(f"incompatible legacy tensor: {name}")
        upgraded[name] = updated
    return upgraded


def load_tile_model(weights_path):
    state_dict = torch.load(Path(weights_path), map_location="cpu", weights_only=True)
    if "model" in state_dict:
        state_dict = state_dict["model"]
    if "_object_scene" in state_dict:
        from scene_model import scene_from_weights
        model = scene_from_weights(state_dict)
        return model.eval()
    if "_observed_history" in state_dict:
        model_class = ObservedHistoryDynamics
    elif "gate_logit_offset" in state_dict:
        model_class = CalibratedEventDynamics
    else:
        model_class = EventTileDynamics if "change_head.weight" in state_dict else TileDynamics
    model = model_class(state_dict["codebook"])
    model.load_state_dict(state_dict)
    return model.eval()
