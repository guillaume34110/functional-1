"""Neural object/canvas compositing; geometry only, with no game transition rules."""
import torch
from torch import nn
from torch.nn import functional as F

from model import ACTIONS
from token_model import ObservedHistoryDynamics, decode_tokens


class _ObjectHead(nn.Module):
    def __init__(self, shapes):
        super().__init__()
        self.pose = nn.Conv2d(16, 1, 1)
        self.shape = nn.Linear(16, shapes)
        self.visible = nn.Linear(32, 1)
        nn.init.zeros_(self.visible.weight)
        nn.init.constant_(self.visible.bias, -1)

    def components(self, hidden):
        flattened = hidden.flatten(2)
        pose = self.pose(hidden).flatten(1)
        pooled = torch.einsum("bcp,bp->bc", flattened, pose.softmax(1))
        return self.shape(pooled), pose, pooled

    def forward(self, hidden, valid_poses, joint_offset=None, joint_transform=None):
        shapes, pose, pooled = self.components(hidden)
        joint = shapes[:, :, None] + pose[:, None, :]
        if joint_offset is not None:
            joint = joint + joint_offset
        if joint_transform is not None:
            joint = joint_transform(joint)
        joint = joint.masked_fill(~valid_poses[None], -torch.inf)
        joint = joint.flatten(1).log_softmax(1).reshape_as(joint)
        visible = self.visible(torch.cat((pooled, hidden.flatten(2).mean(2)), dim=1))
        return {"joint": joint, "visible": visible}, pooled


class ObjectSceneDynamics(ObservedHistoryDynamics):
    def __init__(self, codebook, object_shapes, object_tiles):
        super().__init__(codebook)
        self.base_parameter_count = len(list(self.parameters()))
        if (object_shapes.dtype != torch.int64 or object_shapes.ndim != 3 or
                tuple(object_shapes.shape[1:]) != (4, 2) or len(object_shapes) < 1):
            raise ValueError("object_shapes must be int64 [K,4,2]")
        if object_tiles.dtype != torch.int64 or object_tiles.shape != (len(object_shapes),):
            raise ValueError("object_tiles must be int64 [K]")
        if torch.any(object_tiles < 0) or torch.any(object_tiles >= len(codebook)):
            raise ValueError("object texture outside codebook")
        if torch.any(object_shapes < 0) or torch.any(object_shapes.amin(1) != 0):
            raise ValueError("object shapes must have normalized nonnegative offsets")
        if any(len(torch.unique(shape, dim=0)) != 4 for shape in object_shapes):
            raise ValueError("object cells must be distinct")
        blanks = torch.nonzero((codebook.flatten(1) == 3).all(1)).flatten()
        if len(blanks) != 1:
            raise ValueError("scene compositing requires one exact white pattern")
        self.blank_id = int(blanks[0])
        self.register_buffer("object_shapes", object_shapes.detach().clone())
        self.register_buffer("object_tiles", object_tiles.detach().clone())
        self.register_buffer("_object_scene", torch.tensor(True))
        origins = torch.stack(torch.meshgrid(torch.arange(18, device=object_shapes.device),
                                             torch.arange(10, device=object_shapes.device), indexing="ij"), dim=-1).reshape(180, 2)
        coordinates = object_shapes[:, None] + origins[None, :, None]
        valid = ((coordinates[..., 0] < 18) & (coordinates[..., 1] < 10)).all(2)
        if not torch.all(valid.any(1)):
            raise ValueError("every shape must fit the visible canvas")
        masks = torch.zeros(len(object_shapes), 180, 180, device=object_shapes.device)
        indices = (coordinates[..., 0] * 10 + coordinates[..., 1]).clamp(0, 179)
        for cell in range(4):
            masks.scatter_(2, indices[:, :, cell, None], valid[:, :, None].float())
        self.register_buffer("shape_masks", masks.reshape(-1, 180, 18, 10), persistent=False)
        self.register_buffer("valid_poses", valid, persistent=False)
        board = torch.zeros(18, 20, dtype=torch.bool, device=object_shapes.device)
        board[:, 2:12] = True
        self.register_buffer("board_mask", board, persistent=False)
        self.current_head = _ObjectHead(len(object_shapes))
        self.next_head = _ObjectHead(len(object_shapes))
        self.mode_head = nn.Linear(48, 3)
        nn.init.zeros_(self.mode_head.weight)
        with torch.no_grad():
            self.mode_head.bias.copy_(torch.tensor([0.0, 0.0, 6.0]))

    def _next_joint_offset(self, hidden):
        return None

    def _next_details(self, hidden, current):
        return self.next_head(hidden, self.valid_poses, self._next_joint_offset(hidden))

    def _current_hidden(self, tokens, actions, prior_state, transition_hidden):
        return transition_hidden

    def _current_details(self, tokens, hidden):
        return self.current_head(hidden, self.valid_poses)

    def _read_current(self, tokens, hidden, prior_state):
        return self._current_details(tokens, hidden)

    def advance_observed(self, tokens, actions, state=None):
        return self.advance(tokens, actions, state)

    def _scene_mode(self, tokens, actions, pooled, scene):
        return {"mode": self.mode_head(pooled)}

    def scene_details(self, tokens, actions, state=None):
        prior_state = state
        write, gate, state = super().forward_details(tokens, actions, state)
        hidden = state[:, :16, :, 2:12]
        current_hidden = self._current_hidden(tokens, actions, prior_state, hidden)
        current, current_pool = self._read_current(tokens, current_hidden, prior_state)
        future, future_pool = self._next_details(hidden, current)
        pooled = torch.cat((hidden.mean((2, 3)), current_pool, future_pool), dim=1)
        scene = {"write": write, "gate": gate, "state": state, "current": current, "next": future}
        scene.update(self._scene_mode(tokens, actions, pooled, scene))
        return scene

    def _soft_object(self, details):
        probabilities = details["joint"].exp()
        mass = torch.einsum("bkp,kpxy->bkxy", probabilities, self.shape_masks.to(probabilities.dtype))
        mass = mass * details["visible"].sigmoid()[:, :, None, None]
        alpha = F.pad(mass.sum(1, keepdim=True), (2, 8))
        colors = torch.einsum("bkxy,kv->bvxy", mass, F.one_hot(self.object_tiles, len(self.codebook)).to(mass.dtype))
        return alpha, F.pad(colors, (2, 8))

    def _hard_object(self, details):
        joint = details["joint"].masked_fill(~self.valid_poses[None], -torch.inf)
        selected = joint.flatten(1).argmax(1)
        kind, pose = selected // 180, selected % 180
        mask = self.shape_masks[kind, pose].bool() & (details["visible"] > 0)[:, :, None]
        return F.pad(mask, (2, 8)), self.object_tiles[kind]

    def _rewrite_board(self, tokens, details, rewritten, hard):
        return rewritten

    def _select_rewritten_rows(self, details, background, rewritten, hard):
        return rewritten

    def _hard_rewrite_candidates(self, tokens, details, gate):
        current_mask, _ = self._hard_object(details["current"])
        background = torch.where(current_mask, self.blank_id, tokens)
        rewritten = self.combine_logits(background, details["write"], gate).argmax(1)
        return background, self._rewrite_board(tokens, details, rewritten, hard=True)

    def _hard_background(self, tokens, details, gate):
        background, rewritten = self._hard_rewrite_candidates(tokens, details, gate)
        rewritten = self._select_rewritten_rows(details, background, rewritten, hard=True)
        mode = details["mode"].argmax(1)[:, None, None]
        return torch.where(mode == 0, background, torch.where(mode == 1, tokens, rewritten))

    def decode_scene(self, tokens, details, hard=False):
        gate = details["gate"] - self.gate_logit_offset
        if hard:
            board = self._hard_background(tokens, details, gate)
            raw_ui = self.combine_logits(tokens, details["write"], gate).argmax(1)
            background = torch.where(self.board_mask[None], board, raw_ui)
            future_mask, color = self._hard_object(details["next"])
            return torch.where(future_mask, color[:, None, None], background)
        original = F.one_hot(tokens, len(self.codebook)).permute(0, 3, 1, 2).to(details["write"].dtype)
        current_alpha, _ = self._soft_object(details["current"])
        blank = original.new_zeros(1, len(self.codebook), 1, 1)
        blank[:, self.blank_id] = 1
        background = original * (1 - current_alpha) + blank * current_alpha
        probability = gate.sigmoid()
        writes = details["write"].softmax(1)
        rewritten = (1 - probability) * background + probability * writes
        rewritten = self._rewrite_board(tokens, details, rewritten, hard=False)
        rewritten = self._select_rewritten_rows(details, background, rewritten, hard=False)
        mode = details["mode"].softmax(1)[:, :, None, None, None]
        board = mode[:, 0] * background + mode[:, 1] * original + mode[:, 2] * rewritten
        raw_ui = (1 - probability) * original + probability * writes
        background = torch.where(self.board_mask[None, None], board, raw_ui)
        future_alpha, colors = self._soft_object(details["next"])
        final = ((1 - future_alpha) * background + colors).clamp_min(1e-9)
        return (final / final.sum(1, keepdim=True)).log().contiguous()

    def forward(self, tokens, actions, state=None):
        details = self.scene_details(tokens, actions, state)
        return self.decode_scene(tokens, details, hard=False), details["state"]

    def rendered_state(self, details, prediction):
        return details["state"]

    def predict_tokens(self, tokens, actions, state=None):
        details = self.scene_details(tokens, actions, state)
        prediction = self.decode_scene(tokens, details, hard=True)
        return prediction, self.rendered_state(details, prediction)

    def predict_frame(self, frame, action, state=None):
        predicted, state = self.predict_tokens(self.frame_tokens(frame), action, state)
        return decode_tokens(predicted, self.codebook), state

    def frame_tokens(self, frame):
        if (frame.dtype != torch.uint8 or frame.ndim != 4 or
                tuple(frame.shape[1:]) != (1, 72, 80) or torch.any(frame > 3)):
            raise ValueError("frame must be palette uint8 [B,1,72,80]")
        patches = frame[:, 0].reshape(len(frame), 18, 4, 20, 4).permute(0, 1, 3, 2, 4).reshape(len(frame), 18, 20, 16)
        distances = (patches[:, :, :, None] != self.codebook.reshape(1, 1, 1, -1, 16)).sum(-1)
        return distances.argmin(-1)


class ExtendedSceneDynamics(ObjectSceneDynamics):
    """Additional learned recurrent capacity using only displayed patterns/buttons."""
    state_channels = 36

    def __init__(self, codebook, object_shapes, object_tiles):
        super().__init__(codebook, object_shapes, object_tiles)
        self.legacy_parameter_count = len(list(self.parameters()))
        self.extra_gates = nn.Conv2d(56, 32, 3, padding=1)
        self.extra_candidate = nn.Conv2d(56, 16, 3, padding=1)
        self.extra_projection = nn.Conv2d(16, 16, 1)
        nn.init.zeros_(self.extra_projection.weight); nn.init.zeros_(self.extra_projection.bias)
        self.register_buffer("_extended_scene", torch.tensor(True))

    def advance(self, tokens, actions, state=None):
        if state is not None and (state.shape != (len(tokens), 36, 18, 20) or not state.is_floating_point()):
            raise ValueError("extended scene state must be floating[B,36,18,20]")
        legacy = super().advance(tokens, actions, None if state is None else state[:, :20])
        old = legacy[:, :16]
        extra = torch.zeros_like(old) if state is None else state[:, 20:36]
        patterns = self.embedding(tokens).permute(0, 3, 1, 2)
        buttons = self.action_embedding(actions)[:, :, None, None].expand(-1, -1, 18, 20)
        inputs = torch.cat((old, patterns, buttons), 1)
        reset, update = self.extra_gates(torch.cat((inputs, extra), 1)).sigmoid().chunk(2, 1)
        candidate = self.extra_candidate(torch.cat((inputs, reset * extra), 1)).tanh()
        extra = (1 - update) * extra + update * candidate
        fused = old + self.extra_projection(extra)
        return torch.cat((fused, legacy[:, 16:20], extra), 1)


class RelativeSceneDynamics(ObjectSceneDynamics):
    """Learned generic rigid raster transport mixed with the absolute future joint."""
    def __init__(self, codebook, object_shapes, object_tiles):
        super().__init__(codebook, object_shapes, object_tiles)
        self.legacy_parameter_count = len(list(self.parameters()))
        self.motion_delta = nn.Linear(40, 49)
        nn.init.normal_(self.motion_delta.weight, std=.01)
        nn.init.zeros_(self.motion_delta.bias)
        with torch.no_grad(): self.motion_delta.bias[24] = 2
        self.motion_gate = nn.Linear(40, 1)
        nn.init.zeros_(self.motion_gate.weight); nn.init.constant_(self.motion_gate.bias, -4)
        self.register_buffer("_relative_scene", torch.tensor(True))

    def transport_joint(self, joint, delta):
        probabilities = joint.exp().reshape(len(joint), len(self.object_shapes), 18, 10)
        mixture = torch.zeros_like(probabilities)
        for index, (dy, dx) in enumerate(( (dy, dx) for dy in range(-3, 4) for dx in range(-3, 4) )):
            y0, y1 = max(0, -dy), min(18, 18 - dy)
            x0, x1 = max(0, -dx), min(10, 10 - dx)
            shifted = F.pad(probabilities[:, :, y0:y1, x0:x1],
                            (x0 + dx, 10 - x1 - dx, y0 + dy, 18 - y1 - dy))
            mixture = mixture + shifted * delta[:, index, None, None, None]
        mixture = mixture.flatten(2) * self.valid_poses[None].to(mixture.dtype)
        mass = mixture.sum((1, 2), keepdim=True)
        fallback = mass[:, 0, 0] == 0
        safe = mixture / mass.clamp_min(torch.finfo(mixture.dtype).tiny)
        # Numerically empty transport reuses current probability only to keep logs finite;
        # scene_details selects the absolute future joint for these samples.
        safe = torch.where(fallback[:, None, None], joint.exp(), safe)
        log_joint = safe.clamp_min(torch.finfo(safe.dtype).tiny).log().masked_fill(~self.valid_poses[None], -torch.inf)
        log_joint = log_joint - log_joint.flatten(1).logsumexp(1)[:, None, None]
        return log_joint, mass[:, 0, 0], fallback

    def scene_details(self, tokens, actions, state=None):
        details = super().scene_details(tokens, actions, state)
        hidden = details["state"][:, :16, :, 2:12]
        pose = self.current_head.pose(hidden).flatten(1)
        pool = torch.einsum("bcp,bp->bc", hidden.flatten(2), pose.softmax(1))
        features = torch.cat((pool, hidden.mean((2, 3)), self.action_embedding(actions)), 1)
        delta_logits = self.motion_delta(features)
        delta = delta_logits.softmax(1)
        gate = self.motion_gate(features)
        transported, mass, fallback = self.transport_joint(details["current"]["joint"], delta)
        absolute = details["next"]["joint"]
        valid = self.valid_poses[None]
        # Avoid logaddexp(-inf,-inf): its derivative is undefined at invalid geometry.
        a = absolute.masked_fill(~valid, 0) + F.logsigmoid(-gate)[:, :, None]
        b = transported.masked_fill(~valid, 0) + F.logsigmoid(gate)[:, :, None]
        joint = torch.logaddexp(a, b).masked_fill(~valid, -torch.inf)
        joint = torch.where(fallback[:, None, None], absolute, joint)
        details["next"] = {**details["next"], "joint": joint}
        details["transport"] = {"alpha": gate.sigmoid(), "retained_mass": mass[:, None],
                                "identity_delta": delta[:, 24:25], "numerical_fallback": fallback[:, None],
                                "delta_logits": delta_logits, "gate_logits": gate}
        return details


class FootprintSceneDynamics(ObjectSceneDynamics):
    """Score a candidate using learned features on its four rendered cells."""

    def __init__(self, codebook, object_shapes, object_tiles):
        super().__init__(codebook, object_shapes, object_tiles)
        self.legacy_parameter_count = len(list(self.parameters()))
        self.footprint_score = nn.Conv2d(16, len(object_shapes), 1)
        nn.init.zeros_(self.footprint_score.weight)
        nn.init.zeros_(self.footprint_score.bias)
        cells = self.shape_masks.flatten(2).topk(4, dim=-1).indices
        self.register_buffer("footprint_cells", cells, persistent=False)
        self.register_buffer("_footprint_scene", torch.tensor(True))

    def _next_joint_offset(self, hidden):
        scores = self.footprint_score(hidden).flatten(2)
        shapes = torch.arange(scores.shape[1], device=scores.device)[:, None, None]
        return scores[:, shapes, self.footprint_cells].mean(3)


class PoseRerankSceneDynamics(ObjectSceneDynamics):
    """Refine the three most likely future joints using observed features."""

    def __init__(self, codebook, object_shapes, object_tiles):
        super().__init__(codebook, object_shapes, object_tiles)
        self.legacy_parameter_count = len(list(self.parameters()))
        self.pose_reranker = nn.Sequential(nn.Linear(20, 16), nn.Tanh(), nn.Linear(16, 1, bias=False))
        nn.init.zeros_(self.pose_reranker[-1].weight)
        self.register_buffer("rerank_cells", self.shape_masks.flatten(2).topk(4, dim=-1).indices, persistent=False)
        self.register_buffer("_pose_rerank_scene", torch.tensor(True))

    def _rerank_joint(self, joint, hidden, current_joint):
        valid = self.valid_poses.flatten()
        scores = joint.flatten(1)
        _, candidates = scores.masked_fill(~valid, -torch.inf).topk(3, dim=1)
        candidate_valid = valid[candidates]
        shape, pose = candidates // 180, candidates % 180
        current = current_joint.flatten(1).argmax(1)
        current_pose = current[:, None] % 180
        row_delta = (pose // 10 - current_pose // 10).to(hidden.dtype) / 17
        col_delta = (pose % 10 - current_pose % 10).to(hidden.dtype) / 9
        same_shape = (shape == current[:, None] // 180).to(hidden.dtype)
        candidate_score = scores.gather(1, candidates).masked_fill(~candidate_valid, 0)
        cells = self.rerank_cells[shape, pose]
        board = hidden.flatten(2).transpose(1, 2)
        batch = torch.arange(len(hidden), device=hidden.device)[:, None, None]
        local = board[batch, cells].mean(2)
        context = torch.stack((row_delta, col_delta, same_shape, candidate_score), dim=-1)
        correction = self.pose_reranker(torch.cat((local, context), dim=-1)).squeeze(-1)
        # Preserve the shortlist's total mass so other joints keep their probabilities.
        selected_logits = candidate_score.masked_fill(~candidate_valid, -torch.inf)
        mass_change = (selected_logits + correction).logsumexp(1, keepdim=True) - selected_logits.logsumexp(1, keepdim=True)
        correction = correction - mass_change
        correction = correction.masked_fill(~candidate_valid, 0)
        return scores.scatter_add(1, candidates, correction).reshape_as(joint)

    def _next_details(self, hidden, current):
        def rerank(joint):
            return self._rerank_joint(joint, hidden, current["joint"])
        return self.next_head(hidden, self.valid_poses, joint_transform=rerank)


class ActionInvariantCurrentSceneDynamics(ObjectSceneDynamics):
    """Read the current object using a fixed action condition."""

    def __init__(self, codebook, object_shapes, object_tiles):
        super().__init__(codebook, object_shapes, object_tiles)
        self.register_buffer("_action_invariant_current_scene", torch.tensor(True))

    def _current_hidden(self, tokens, actions, prior_state, transition_hidden):
        observation = self.advance(tokens, torch.zeros_like(actions), prior_state)
        return observation[:, :16, :, 2:12]


class LearnedOverlapSceneDynamics(ObjectSceneDynamics):
    """Learn a soft pose cost from observed tiles and the predicted current object."""

    def __init__(self, codebook, object_shapes, object_tiles):
        super().__init__(codebook, object_shapes, object_tiles)
        self.legacy_parameter_count = len(list(self.parameters()))
        self.overlap_head = nn.Sequential(nn.Conv2d(18, 8, 3, padding=1), nn.SiLU(), nn.Conv2d(8, 1, 1))
        nn.init.zeros_(self.overlap_head[-1].weight)
        nn.init.zeros_(self.overlap_head[-1].bias)
        self.register_buffer("overlap_cells", self.shape_masks.flatten(2).topk(4, dim=-1).indices, persistent=False)
        self.register_buffer("_learned_overlap_scene", torch.tensor(True))
        self.register_buffer("overlap_scale", torch.tensor(1.0))

    def scene_details(self, tokens, actions, state=None):
        details = super().scene_details(tokens, actions, state)
        hidden = details["state"][:, :16, :, 2:12]
        current = details["current"]
        current_mass = current["joint"].exp() * current["visible"].sigmoid()[:, :, None]
        soft_object = torch.einsum("bkp,kpxy->bxy", current_mass, self.shape_masks.to(hidden.dtype))
        occupied = (tokens[:, :, 2:12] != self.blank_id).to(hidden.dtype)
        features = torch.cat((hidden, occupied[:, None], soft_object[:, None]), dim=1)
        cell_logits = self.overlap_head(features).flatten(1)
        logits = cell_logits[:, self.overlap_cells].amax(3)
        details["overlap_logits"] = logits

        joint = details["next"]["joint"]
        copy_probability = details["mode"].softmax(1)[:, 0, None, None]
        offset = F.softplus(logits.new_zeros(())) - F.softplus(logits)
        adjusted = joint + copy_probability * offset * self.overlap_scale
        normalizer = adjusted.flatten(1).logsumexp(1) - joint.flatten(1).logsumexp(1)
        details["next"]["joint"] = adjusted - normalizer[:, None, None]
        return details


class SpawnPreviewSceneDynamics(LearnedOverlapSceneDynamics):
    """Read the visible preview panel to refine the next shape during composition."""

    def __init__(self, codebook, object_shapes, object_tiles):
        super().__init__(codebook, object_shapes, object_tiles)
        self.preview_base_parameter_count = len(list(self.parameters()))
        self.spawn_preview_head = nn.Sequential(nn.Linear(64, 16), nn.SiLU(), nn.Linear(16, len(object_shapes)))
        nn.init.zeros_(self.spawn_preview_head[-1].weight)
        nn.init.zeros_(self.spawn_preview_head[-1].bias)
        self.register_buffer("_spawn_preview_scene", torch.tensor(True))

    def _preview_features(self, tokens):
        preview = self.embedding(tokens[:, :6, 13:19]).permute(0, 3, 1, 2)
        return F.adaptive_avg_pool2d(preview, (2, 2)).flatten(1)

    def _preview_shape_scores(self, tokens):
        return self.spawn_preview_head(self._preview_features(tokens))

    def scene_details(self, tokens, actions, state=None):
        details = super().scene_details(tokens, actions, state)
        compose_probability = details["mode"].softmax(1)[:, 1, None, None]
        offset = self._preview_shape_scores(tokens)[:, :, None] * compose_probability
        joint = details["next"]["joint"]
        adjusted = joint + offset
        normalizer = adjusted.flatten(1).logsumexp(1) - joint.flatten(1).logsumexp(1)
        details["next"]["joint"] = adjusted - normalizer[:, None, None]
        details["preview_joint_offset"] = offset
        return details


def _clock_features(state, button_embedding):
    board = state[:, :, :, 2:12]
    pooled = board[:, :16].mean((2, 3))
    age = board[:, 17].amin((1, 2))[:, None] / 255
    held = state[:, 18, 0, 0, None] / 255
    return torch.cat((pooled, age, held, button_embedding), dim=1)


class MotionClockSceneDynamics(SpawnPreviewSceneDynamics):
    """Use observed screen age and recurrent features to score a pose change."""

    def __init__(self, codebook, object_shapes, object_tiles):
        super().__init__(codebook, object_shapes, object_tiles)
        self.motion_clock_head = nn.Linear(26, 1)
        nn.init.zeros_(self.motion_clock_head.weight)
        nn.init.zeros_(self.motion_clock_head.bias)
        self.register_buffer("_motion_clock_scene", torch.tensor(True))

    def motion_clock_features(self, state, actions):
        return _clock_features(state, self.action_embedding(actions))

    def motion_candidate_weights(self, details):
        joint = details["current"]["joint"]
        current = joint.flatten(1).argmax(1, keepdim=True)
        moving = torch.ones_like(joint.flatten(1), dtype=torch.bool).scatter_(1, current, False).reshape_as(joint)
        return (moving & self.valid_poses) * details["mode"].softmax(1)[:, 0, None, None]

    def scene_details(self, tokens, actions, state=None):
        details = super().scene_details(tokens, actions, state)
        features = self.motion_clock_features(details["state"], actions)
        offset = self.motion_clock_head(features)[:, :, None] * self.motion_candidate_weights(details)
        joint = details["next"]["joint"]
        adjusted = joint + offset
        normalizer = adjusted.flatten(1).logsumexp(1) - joint.flatten(1).logsumexp(1)
        details["next"]["joint"] = adjusted - normalizer[:, None, None]
        details["motion_clock_features"] = features
        details["motion_joint_offset"] = offset
        return details


class PreviewRenderSceneDynamics(SpawnPreviewSceneDynamics):
    """Render only the visible preview crop as a learned categorical motif."""

    def __init__(self, codebook, object_shapes, object_tiles, preview_patterns):
        super().__init__(codebook, object_shapes, object_tiles)
        if (preview_patterns.dtype != torch.long or preview_patterns.ndim != 3
                or preview_patterns.shape[1:] != (6, 6) or not 1 <= len(preview_patterns) <= 64
                or torch.any((preview_patterns < 0) | (preview_patterns >= len(codebook)))
                or len(torch.unique(preview_patterns.flatten(1), dim=0)) != len(preview_patterns)):
            raise ValueError("expected 1..64 unique preview token patterns [P,6,6]")
        tiles = torch.unique(object_tiles)
        texture_counts = (preview_patterns[..., None] == tiles).sum((1, 2))
        if torch.any((texture_counts.sum(1) != 4) | (texture_counts.amax(1) != 4)):
            raise ValueError("each preview pattern must show four tiles of a single texture")
        self.register_buffer("preview_patterns", preview_patterns)
        self.register_buffer("_preview_render_scene", torch.tensor(True))
        self.register_buffer("preview_pattern_probs",
                             F.one_hot(preview_patterns, len(codebook)).permute(0, 3, 1, 2).float(),
                             persistent=False)
        self.preview_render_head = nn.Sequential(nn.Linear(80, 16), nn.SiLU(), nn.Linear(16, len(preview_patterns) + 1))
        nn.init.zeros_(self.preview_render_head[-1].weight)
        nn.init.zeros_(self.preview_render_head[-1].bias)
        with torch.no_grad():
            self.preview_render_head[-1].bias[0] = 2

    def preview_render_features(self, tokens, state, mode=None):
        hidden = state[:, :16, :, 2:12].mean((2, 3))
        return torch.cat((hidden, self._preview_features(tokens)), dim=1)

    def scene_details(self, tokens, actions, state=None):
        details = super().scene_details(tokens, actions, state)
        features = self.preview_render_features(tokens, details["state"], details["mode"])
        details["preview_render_logits"] = self.preview_render_head(features)
        return details

    def _preview_choice(self, crop, logits):
        return logits.argmax(1)

    def _preview_distribution(self, crop, logits):
        return logits.softmax(1)

    def decode_scene(self, tokens, details, hard=False):
        rendered = super().decode_scene(tokens, details, hard)
        crop = tokens[:, :6, 13:19]
        logits = details["preview_render_logits"]
        if hard:
            choice = self._preview_choice(crop, logits)
            pattern = self.preview_patterns[(choice - 1).clamp_min(0)]
            rendered[:, :6, 13:19] = torch.where(choice[:, None, None] == 0, crop, pattern)
        else:
            probability = self._preview_distribution(crop, logits)
            copy = F.one_hot(crop, len(self.codebook)).permute(0, 3, 1, 2).to(probability.dtype)
            pattern = torch.einsum("bp,pvyx->bvyx", probability[:, 1:], self.preview_pattern_probs.to(probability.dtype))
            crop_probs = (pattern + probability[:, :1, None, None] * copy).clamp_min(1e-9)
            rendered = rendered.clone()
            rendered[:, :, :6, 13:19] = (crop_probs / crop_probs.sum(1, keepdim=True)).log()
        return rendered


class FactorizedPreviewRenderSceneDynamics(PreviewRenderSceneDynamics):
    """Predict whether to change the preview, then choose its replacement."""

    def __init__(self, codebook, object_shapes, object_tiles, preview_patterns):
        super().__init__(codebook, object_shapes, object_tiles, preview_patterns)
        if len(preview_patterns) < 2:
            raise ValueError("factorized preview requires at least two distinct motifs")
        self.preview_render_head[0] = nn.Linear(83, 16)
        with torch.no_grad():
            self.preview_render_head[-1].bias[0] = -2
        self.register_buffer("_factorized_preview_render_scene", torch.tensor(True))

    def preview_render_features(self, tokens, state, mode=None):
        features = super().preview_render_features(tokens, state)
        return torch.cat((features, mode.softmax(1)), dim=1)

    def _replacement_logits(self, crop, logits):
        already_visible = (crop[:, None] == self.preview_patterns).flatten(2).all(2)
        return logits[:, 1:].masked_fill(already_visible, -torch.inf)

    def _preview_choice(self, crop, logits):
        replacement = self._replacement_logits(crop, logits).argmax(1) + 1
        return torch.where(logits[:, 0] > 0, replacement, 0)

    def _preview_distribution(self, crop, logits):
        change = logits[:, :1].sigmoid()
        replacement = self._replacement_logits(crop, logits).softmax(1)
        return torch.cat((1 - change, change * replacement), dim=1)


class ConsistencySceneDynamics(FactorizedPreviewRenderSceneDynamics):
    """Learn when a predicted object changes shape or horizontal origin."""

    def __init__(self, codebook, object_shapes, object_tiles, preview_patterns):
        super().__init__(codebook, object_shapes, object_tiles, preview_patterns)
        self.consistency_head = nn.Linear(26, 2)
        nn.init.zeros_(self.consistency_head.weight)
        nn.init.zeros_(self.consistency_head.bias)
        self.register_buffer("_consistency_scene", torch.tensor(True))

    def consistency_features(self, state, actions):
        return _clock_features(state, self.action_embedding(actions))

    def consistency_group(self, current_joint):
        current = current_joint.flatten(1).argmax(1)
        shape = torch.arange(current_joint.size(1), device=current_joint.device)
        column = torch.arange(180, device=current_joint.device) % 10
        shape_change = shape[None, :, None] != (current // 180)[:, None, None]
        column_change = column[None, None, :] != (current % 10)[:, None, None]
        return shape_change.long() + 2 * column_change.long()

    def scene_details(self, tokens, actions, state=None):
        details = super().scene_details(tokens, actions, state)
        group = self.consistency_group(details["current"]["joint"])
        shape, column = self.consistency_head(self.consistency_features(details["state"], actions)).unbind(1)
        offset = shape[:, None, None] * ((group & 1) != 0) + column[:, None, None] * ((group & 2) != 0)
        offset = offset * self.valid_poses * details["mode"].softmax(1)[:, 0, None, None]
        joint = details["next"]["joint"]
        adjusted = joint + offset
        normalizer = adjusted.flatten(1).logsumexp(1) - joint.flatten(1).logsumexp(1)
        details["next"]["joint"] = adjusted - normalizer[:, None, None]
        details["consistency_joint_offset"] = offset
        return details


class ModeCorrectionSceneDynamics(ConsistencySceneDynamics):
    """Add learned residual mode scores before mode-dependent composition."""

    def __init__(self, codebook, object_shapes, object_tiles, preview_patterns):
        super().__init__(codebook, object_shapes, object_tiles, preview_patterns)
        self.mode_correction_head = nn.Linear(58, 3)
        nn.init.zeros_(self.mode_correction_head.weight)
        nn.init.zeros_(self.mode_correction_head.bias)
        self.register_buffer("_mode_correction_scene", torch.tensor(True))

    def _scene_mode(self, tokens, actions, pooled, scene):
        base = self.mode_head(pooled)
        clock = _clock_features(scene["state"], self.action_embedding(actions))
        features = torch.cat((clock, pooled[:, 16:]), dim=1)
        return {"mode": base + self.mode_correction_head(features), "mode_base_logits": base,
                "mode_correction_features": features}


class _SpatialModeHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.rows = nn.Conv1d(17, 8, 3, padding=1)
        self.readout = nn.Linear(16, 3)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(self, features):
        rows = F.silu(self.rows(features))
        return self.readout(torch.cat((rows.mean(2), rows.amax(2)), dim=1))


class SpatialModeSceneDynamics(ModeCorrectionSceneDynamics):
    """Refine mode probabilities from observed and recurrent row features."""

    def __init__(self, codebook, object_shapes, object_tiles, preview_patterns):
        super().__init__(codebook, object_shapes, object_tiles, preview_patterns)
        self.spatial_mode_head = _SpatialModeHead()
        self.register_buffer("_spatial_mode_scene", torch.tensor(True))

    def _scene_mode(self, tokens, actions, pooled, scene):
        details = super()._scene_mode(tokens, actions, pooled, scene)
        hidden = scene["state"][:, :16, :, 2:12].mean(3)
        occupancy = (tokens[:, :, 2:12] != self.blank_id).to(hidden.dtype).mean(2)
        features = torch.cat((hidden, occupancy[:, None]), dim=1)
        base = details["mode"]
        details.update(mode=base + self.spatial_mode_head(features),
                       spatial_mode_base_logits=base, spatial_mode_features=features)
        return details


class _RowTransportHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = nn.Linear(17, 8)
        self.key = nn.Linear(17, 8)
        self.blank = nn.Linear(17, 1)
        self.gate = nn.Linear(34, 1)
        self.relative_bias = nn.Parameter(torch.zeros(35))
        rows = torch.arange(18)
        self.register_buffer("relative_index", rows[None] - rows[:, None] + 17, persistent=False)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -4.)

    def forward(self, features):
        scores = self.query(features) @ self.key(features).transpose(1, 2)
        scores = scores * (8 ** -0.5) + self.relative_bias[self.relative_index]
        rows = torch.cat((scores, self.blank(features)), dim=2)
        pooled = torch.cat((features.mean(1), features.amax(1)), dim=1)
        return rows, self.gate(pooled)[:, 0]


class RowTransportSceneDynamics(ModeCorrectionSceneDynamics):
    """Learn to transport visible source rows within the rewrite layer."""

    def __init__(self, codebook, object_shapes, object_tiles, preview_patterns):
        super().__init__(codebook, object_shapes, object_tiles, preview_patterns)
        self.row_transport_head = _RowTransportHead()
        self.register_buffer("_row_transport_scene", torch.tensor(True))

    def row_transport_features(self, tokens, details):
        hidden = details["state"][:, :16, :, 2:12].mean(3).transpose(1, 2)
        occupancy = (tokens[:, :, 2:12] != self.blank_id).to(hidden.dtype).mean(2, keepdim=True)
        return torch.cat((hidden, occupancy), dim=2)

    def scene_details(self, tokens, actions, state=None):
        details = super().scene_details(tokens, actions, state)
        rows, gate = self.row_transport_head(self.row_transport_features(tokens, details))
        details.update(row_transport_logits=rows, row_transport_gate=gate)
        return details

    def row_sources(self, tokens):
        board = tokens[:, :, 2:12]
        return torch.cat((board, torch.full_like(board[:, :1], self.blank_id)), dim=1)

    def transported_rows(self, tokens, details):
        selected = details["row_transport_logits"].argmax(2)
        return self.row_sources(tokens).gather(1, selected[:, :, None].expand(-1, -1, 10))

    def _rewrite_board(self, tokens, details, rewritten, hard):
        gate = details.get("render_transport_gate", details["row_transport_gate"])
        result = rewritten.clone()
        if hard:
            transported = self.transported_rows(tokens, details)
            result[:, :, 2:12] = torch.where(gate[:, None, None] > 0, transported, rewritten[:, :, 2:12])
        else:
            source_probs = F.one_hot(self.row_sources(tokens), len(self.codebook)).to(rewritten.dtype)
            transported = torch.einsum("btr,brxv->bvtx", details["row_transport_logits"].softmax(2), source_probs)
            alpha = gate.sigmoid()[:, None, None, None]
            result[:, :, :, 2:12] = rewritten[:, :, :, 2:12] * (1 - alpha) + transported * alpha
        return result


class ContactModeSceneDynamics(RowTransportSceneDynamics):
    """Estimate visible contact to condition the learned composition scores."""

    def __init__(self, codebook, object_shapes, object_tiles, preview_patterns):
        super().__init__(codebook, object_shapes, object_tiles, preview_patterns)
        self.contact_head = nn.Linear(9, 1)
        self.register_buffer("_contact_mode_scene", torch.tensor(True))

    def contact_features(self, tokens, mask):
        foreground = mask[:, :, 2:12]
        occupied = (tokens[:, :, 2:12] != self.blank_id) & ~foreground
        neighborhood = F.pad(occupied[:, None].float(), (1, 1, 1, 0))
        neighborhood = F.unfold(F.pad(neighborhood, (0, 0, 0, 1), value=1), kernel_size=3)
        return (neighborhood * foreground.flatten(1)[:, None]).amax(2)

    def _scene_mode(self, tokens, actions, pooled, scene):
        result = super()._scene_mode(tokens, actions, pooled, scene)
        mask, color = self._hard_object(scene["current"])
        visible = mask.flatten(1).any(1) & ((tokens == color[:, None, None]) | ~mask).flatten(1).all(1)
        neighbors = self.contact_features(tokens, mask)
        logit = self.contact_head(neighbors)[:, 0]
        offset = F.logsigmoid(logit) * visible
        result["mode"] = result["mode"] + F.pad(offset[:, None], (1, 1))
        result["contact_logit"] = logit
        result["contact_offset"] = offset
        result["contact_visible"] = visible
        result["contact_neighbors"] = neighbors
        return result


class JointEventSceneDynamics(ContactModeSceneDynamics):
    """Score composition and object pose together using learned relative biases."""

    def __init__(self, codebook, object_shapes, object_tiles, preview_patterns):
        super().__init__(codebook, object_shapes, object_tiles, preview_patterns)
        self.event_row = nn.Embedding(35, 3)
        self.event_column = nn.Embedding(19, 3)
        self.event_shape = nn.Embedding(2, 3)
        self.event_contact = nn.Linear(1, 3)
        for parameter in (*self.event_row.parameters(), *self.event_column.parameters(),
                          *self.event_shape.parameters(), *self.event_contact.parameters()):
            nn.init.zeros_(parameter)
        self.register_buffer("event_poses", torch.arange(len(object_shapes) * 180), persistent=False)
        self.register_buffer("_joint_event_scene", torch.tensor(True))

    def event_scores(self, mode, future, current, contact, visible):
        relative_row = self.event_poses % 180 // 10 - current[:, None] % 180 // 10
        relative_column = self.event_poses % 10 - current[:, None] % 10
        changed_shape = (self.event_poses // 180 != current[:, None] // 180).long()
        offset = (self.event_row(relative_row + 17) + self.event_column(relative_column + 9)
                  + self.event_shape(changed_shape) + self.event_contact(contact[:, None])[:, None])
        return mode[:, :, None] + future[:, None, :] + offset.transpose(1, 2) * visible[:, None, None]

    def _refine_events(self, scores, details, actions, current):
        return scores

    def scene_details(self, tokens, actions, state=None):
        details = super().scene_details(tokens, actions, state)
        mode, joint = details["mode"], details["next"]["joint"]
        current = details["current"]["joint"].flatten(1).argmax(1)
        details["previous_action"] = (torch.full_like(actions, len(ACTIONS)) if state is None
                                      else state[:, 19, 0, 0].long())
        details["previous_board"] = tokens[:, :, 2:12] if state is None else state[:, 16, :, 2:12].long()
        scores = self.event_scores(mode, joint.flatten(1), current,
                                   details["contact_logit"].sigmoid(), details["contact_visible"])
        scores = self._refine_events(scores, details, actions, current)
        best_mode = scores.amax(2)
        selected = scores[torch.arange(len(scores), device=scores.device), best_mode.argmax(1)]
        details.update(event_scores=scores, event_base_mode=mode, event_base_joint=joint, mode=best_mode)
        details["next"]["joint"] = selected.log_softmax(1).reshape_as(joint)
        return details

    def decode_scene(self, tokens, details, hard=False):
        if hard:
            return super().decode_scene(tokens, details, hard=True)
        scores = details["event_scores"]
        log_weights = scores.logsumexp(2).log_softmax(1)
        mixtures = []
        for mode in range(3):
            mode_logits = torch.full_like(details["mode"], -torch.inf)
            mode_logits[:, mode] = 0
            future = {**details["next"], "joint": scores[:, mode].log_softmax(1).reshape_as(details["next"]["joint"])}
            rendered = super().decode_scene(tokens, {**details, "mode": mode_logits, "next": future})
            mixtures.append(rendered + log_weights[:, mode, None, None, None])
        return torch.stack(mixtures).logsumexp(0).contiguous()


class TimedEventSceneDynamics(JointEventSceneDynamics):
    """Learn the waiting time of the local downward move within the copy event."""

    def __init__(self, codebook, object_shapes, object_tiles, preview_patterns):
        super().__init__(codebook, object_shapes, object_tiles, preview_patterns)
        self.drop_clock = nn.Linear(4, 3)
        self.register_buffer("_timed_event_scene", torch.tensor(True))

    def drop_log_odds(self, row, age, held):
        features = torch.stack((row / 17, row == 0, row == 1, held.log1p() / 5.5), dim=1)
        intercept, slope, immediate = self.drop_clock(features).unbind(1)
        return torch.where(age == 0, immediate, intercept + F.softplus(slope) * age)

    def _refine_events(self, scores, details, actions, current):
        down = (current + 10).clamp_max(scores.size(2) - 1)
        state = details["state"]
        eligible = (actions == ACTIONS.index("down")) & (state[:, 18, 0, 0] > 1) & details["contact_visible"]
        eligible &= (current // 180 == down // 180) & (current != down) & self.valid_poses.flatten()[down]
        rows = eligible.nonzero().flatten()
        candidates = torch.stack((current[rows], down[rows]), dim=1)
        pair = scores[rows[:, None], 0, candidates]
        mask = self.shape_masks.flatten(0, 1)[current[rows]].bool()
        age = state[rows, 17, :, 2:12].masked_fill(~mask, 255).amin((1, 2))
        logits = self.drop_log_odds(current[rows] % 180 // 10, age, state[rows, 18, 0, 0])
        float_probability = (-details["contact_logit"][rows]).sigmoid()
        odds = torch.lerp(pair[:, 1] - pair[:, 0], logits, float_probability)
        adjusted = F.pad(odds[:, None], (1, 0)).log_softmax(1) + pair.logsumexp(1, keepdim=True)
        result = scores.clone()
        result[rows[:, None], 0, candidates] = adjusted
        return result


class CanvasOverlapSceneDynamics(TimedEventSceneDynamics):
    """Score how much a candidate would overlap its mode's visible canvas."""

    def __init__(self, codebook, object_shapes, object_tiles, preview_patterns):
        super().__init__(codebook, object_shapes, object_tiles, preview_patterns)
        self.canvas_overlap_log_scale = nn.Parameter(torch.zeros(2))
        self.register_buffer("_canvas_overlap_scene", torch.tensor(True))

    def canvas_overlap(self, details):
        occupied = details["state"][:, 16, :, 2:12] != self.blank_id
        foreground, _ = self._hard_object(details["current"])
        copy = occupied & ~foreground[:, :, 2:12]
        canvases = torch.stack((copy, occupied), dim=1).flatten(2)
        return canvases[:, :, self.overlap_cells.flatten(0, 1)].sum(3)

    def _refine_events(self, scores, details, actions, current):
        scores = super()._refine_events(scores, details, actions, current)
        overlap = self.canvas_overlap(details)
        strength = F.softplus(self.canvas_overlap_log_scale)[None, :, None]
        visible = details["next"]["visible"][:, :, None] > 0
        return torch.cat((scores[:, :2] - strength * overlap * visible, scores[:, 2:]), dim=1)


class TrackedCanvasSceneDynamics(CanvasOverlapSceneDynamics):
    """Read the current piece with evidence from the previous rendered pose."""

    state_channels = ObservedHistoryDynamics.state_channels + 1

    def __init__(self, codebook, object_shapes, object_tiles, preview_patterns):
        super().__init__(codebook, object_shapes, object_tiles, preview_patterns)
        self.pose_carry = nn.Linear(3, 1)
        self.register_buffer("_tracked_canvas_scene", torch.tensor(True))

    def advance(self, tokens, actions, state=None):
        channels = ObservedHistoryDynamics.state_channels
        if state is not None and (state.ndim != 4 or state.size(1) not in (channels, self.state_channels)):
            raise ValueError("tracking state must preserve the observed history channels")
        base = super().advance(tokens, actions, None if state is None else state[:, :channels])
        return torch.cat((base, torch.full_like(base[:, :1], -1)), dim=1)

    def advance_observed(self, tokens, actions, state=None):
        return self.scene_details(tokens, actions, state)["state"]

    def carry_features(self, tokens, current, previous):
        joint = current["joint"].flatten(1)
        best = joint.argmax(1)
        poses = torch.stack((previous.clamp_min(0), best), dim=1)
        cells = self.overlap_cells.flatten(0, 1)[poses]
        board = tokens[:, :, 2:12].flatten(1)
        colors = self.object_tiles[poses // 180]
        painted = (board.gather(1, cells.flatten(1)).reshape_as(cells) == colors[:, :, None]).float().mean(2)
        gap = joint.gather(1, poses[:, :1])[:, 0] - joint.amax(1)
        features = torch.stack((gap.clamp(-20, 0) / 20, current["visible"][:, 0].sigmoid(), painted[:, 1]), dim=1)
        enabled = (previous >= 0) & (painted[:, 0] == 1) & self.valid_poses.flatten()[poses[:, 0]]
        return features, enabled

    def carry_logit(self, tokens, prior_state, features):
        return self.pose_carry(features)

    def _read_current(self, tokens, hidden, prior_state):
        current, pool = super()._read_current(tokens, hidden, prior_state)
        if prior_state is None or prior_state.size(1) != self.state_channels:
            return current, pool
        previous = prior_state[:, -1, 0, 0].long()
        features, enabled = self.carry_features(tokens, current, previous)
        gate = self.carry_logit(tokens, prior_state, features)
        joint = current["joint"].flatten(1)
        carry = previous.clamp_min(0)[:, None]
        adjusted = joint + F.logsigmoid(-gate)
        selected = torch.logaddexp(adjusted.gather(1, carry), F.logsigmoid(gate))
        adjusted = adjusted.scatter(1, carry, selected)
        no_piece = F.logsigmoid(-current["visible"]) + F.logsigmoid(-gate)
        visible = (-no_piece.expm1()).log() - no_piece
        current["joint"] = torch.where(enabled[:, None], adjusted, joint).reshape_as(current["joint"])
        current["visible"] = torch.where(enabled[:, None], visible, current["visible"])
        return current, pool

    def scene_details(self, tokens, actions, state=None):
        details = super().scene_details(tokens, actions, state)
        future = details["next"]
        pose = future["joint"].flatten(1).argmax(1)
        carry = torch.where(future["visible"][:, 0] > 0, pose, -1)
        history = details["state"][:, :-1]
        plane = carry[:, None, None, None].to(history).expand_as(history[:, :1])
        details["state"] = torch.cat((history, plane), dim=1)
        return details


class ContinuationSceneDynamics(TrackedCanvasSceneDynamics):
    """Condition the next event on button history and visible contact."""

    def __init__(self, codebook, object_shapes, object_tiles, preview_patterns):
        super().__init__(codebook, object_shapes, object_tiles, preview_patterns)
        self.orientation_transition = nn.Embedding((len(ACTIONS) + 1) * len(ACTIONS), 1)
        nn.init.zeros_(self.orientation_transition.weight)
        self.compose_contact_scale = nn.Parameter(torch.zeros(()))
        self.register_buffer("_continuation_scene", torch.tensor(True))

    def continuation_offsets(self, actions, previous, contact):
        orientation = self.orientation_transition(previous * len(ACTIONS) + actions)[:, 0]
        compose = F.softplus(self.compose_contact_scale) * F.logsigmoid(contact)
        return torch.stack((torch.zeros_like(compose), orientation, compose, torch.zeros_like(compose)), dim=1)

    def _refine_events(self, scores, details, actions, current):
        scores = super()._refine_events(scores, details, actions, current)
        offsets = self.continuation_offsets(actions, details["previous_action"], details["contact_logit"])
        active = details["contact_visible"][:, None] & (details["next"]["visible"] > 0)
        same = self.event_poses // 180 == current[:, None] // 180
        copy = scores[:, 0] + offsets[:, 1:2] * active * ~same
        compose = scores[:, 1] + offsets[:, 2:3] * active
        return torch.stack((copy, compose, scores[:, 2]), dim=1)


class RelativeProposalSceneDynamics(ContinuationSceneDynamics):
    """Propose a local translation as an alternative to the absolute copy pose."""

    def __init__(self, codebook, object_shapes, object_tiles, preview_patterns):
        super().__init__(codebook, object_shapes, object_tiles, preview_patterns)
        self.relative_proposal = nn.Sequential(nn.Linear(37, 16), nn.SiLU(), nn.Linear(16, 50))
        self.proposal_gate_calibration = nn.Parameter(torch.zeros(2))
        with torch.no_grad():
            self.relative_proposal[-1].bias[-1] = -3
        offsets = torch.arange(-3, 4)
        self.register_buffer("proposal_offsets", torch.cartesian_prod(offsets, offsets), persistent=False)
        self.register_buffer("_relative_proposal_scene", torch.tensor(True))

    def proposal_features(self, details, actions, current):
        state = details["state"]
        mask = self.shape_masks.flatten(0, 1)[current].bool()
        local = (state[:, :16, :, 2:12] * mask[:, None]).sum((2, 3)) / 4
        age = state[:, 17, :, 2:12].masked_fill(~mask, 255).amin((1, 2))
        previous = details["previous_action"]
        previous_embedding = self.action_embedding(previous.clamp_max(len(ACTIONS) - 1))
        previous_embedding = previous_embedding * (previous < len(ACTIONS))[:, None]
        coordinates = torch.stack((current % 180 // 10 / 17, current % 10 / 9, age.log1p() / 5.5,
                                   state[:, 18, 0, 0].log1p() / 5.5, details["contact_logit"].sigmoid()), dim=1)
        return torch.cat((local, self.action_embedding(actions), previous_embedding, coordinates), dim=1)

    def proposal_gate_delta(self, scores, current, actions, previous):
        changed = scores[:, 0].argmax(1) // 180 != current // 180
        orientation = self.orientation_transition(previous * len(ACTIONS) + actions)[:, 0]
        bias, weight = self.proposal_gate_calibration.unbind()
        return (bias - weight * orientation * changed)[:, None]

    def local_positions(self, current):
        row = current[:, None] % 180 // 10 + self.proposal_offsets[:, 0]
        col = current[:, None] % 10 + self.proposal_offsets[:, 1]
        return row.clamp(0, 17) * 10 + col.clamp(0, 9), (row >= 0) & (row < 18) & (col >= 0) & (col < 10)

    def local_proposal(self, copy, translation, features, current):
        positions, in_view = self.local_positions(current)
        indices = current[:, None] // 180 * 180 + positions
        valid = self.valid_poses.flatten()
        local = translation.masked_fill(~(in_view & valid[indices]), -torch.inf).log_softmax(1)
        return torch.full_like(copy, -torch.inf).scatter_reduce(1, indices, local, reduce="amax")

    def absolute_proposal(self, copy, current):
        return copy - copy.logsumexp(1, keepdim=True)

    def adjust_proposal(self, gate, absolute, relative, current, actions, neighbors):
        return gate, relative

    def blend_copy(self, scores, features, current, active, gate_delta=0, *, actions=None, neighbors=None):
        copy = scores[:, 0]
        output = self.relative_proposal(features)
        gate = output[:, -1:] + gate_delta
        relative = self.local_proposal(copy, output[:, :-1], features, current)
        absolute = self.absolute_proposal(copy, current)
        gate, relative = self.adjust_proposal(gate, absolute, relative, current, actions, neighbors)
        valid = self.valid_poses.flatten()
        mixture = torch.logaddexp(absolute.masked_fill(~valid, 0) + F.logsigmoid(-gate),
                                  relative.masked_fill(~valid, 0) + F.logsigmoid(gate))
        mixture = mixture.masked_fill(~valid, -torch.inf) + copy.logsumexp(1, keepdim=True)
        chosen = torch.where(active[:, None], mixture, copy)
        return torch.cat((chosen[:, None], scores[:, 1:]), dim=1)

    def _refine_events(self, scores, details, actions, current):
        features = self.proposal_features(details, actions, current)
        active = details["contact_visible"] & (details["next"]["visible"][:, 0] > 0)
        gate = self.proposal_gate_delta(scores, current, actions, details["previous_action"])
        scores = self.blend_copy(scores, features, current, active, gate, actions=actions,
                                 neighbors=details["contact_neighbors"])
        return super()._refine_events(scores, details, actions, current)


class OrientedProposalSceneDynamics(RelativeProposalSceneDynamics):
    """Propose local orientations that retain the tracked piece's visible style."""

    def __init__(self, codebook, object_shapes, object_tiles, preview_patterns):
        super().__init__(codebook, object_shapes, object_tiles, preview_patterns)
        self.proposal_shape_embedding = nn.Embedding(len(object_shapes), 4)
        self.oriented_proposal = nn.Sequential(nn.Linear(41, 16), nn.SiLU(), nn.Linear(16, len(object_shapes) + 14))
        self.proposal_style_scale = nn.Parameter(torch.zeros(()))
        self.oriented_command_gate = nn.Embedding((len(ACTIONS) + 1) * len(ACTIONS), 1)
        nn.init.zeros_(self.oriented_command_gate.weight)
        self.motion_agreement_gate = nn.Embedding((len(ACTIONS) + 1) * len(ACTIONS), 3)
        nn.init.zeros_(self.motion_agreement_gate.weight)
        self.register_buffer("_oriented_proposal_scene", torch.tensor(True))

    def local_proposal(self, copy, translation, features, current):
        kind = current // 180
        output = self.oriented_proposal(torch.cat((features, self.proposal_shape_embedding(kind)), dim=1))
        shape, row, col = output.split((len(self.object_shapes), 7, 7), dim=1)
        rotation_shift = row[:, self.proposal_offsets[:, 0] + 3] + col[:, self.proposal_offsets[:, 1] + 3]
        kinds = torch.arange(len(self.object_shapes), device=current.device)
        rotating = kinds[None] != kind[:, None]
        relative = translation[:, None] + shape[:, :, None] + rotation_shift[:, None] * rotating[:, :, None]
        positions, in_view = self.local_positions(current)
        indices = kinds[None, :, None] * 180 + positions[:, None]
        style = self.object_tiles[None] == self.object_tiles[kind, None]
        available = in_view[:, None] & style[:, :, None] & self.valid_poses.flatten()[indices]
        relative = relative.masked_fill(~available, -torch.inf).flatten(1).log_softmax(1)
        return torch.full_like(copy, -torch.inf).scatter_reduce(1, indices.flatten(1), relative, reduce="amax")

    def absolute_proposal(self, copy, current):
        style = self.object_tiles[self.event_poses // 180] == self.object_tiles[current // 180, None]
        adjusted = copy - F.softplus(self.proposal_style_scale) * ~style
        return adjusted.log_softmax(1)

    def proposal_gate_delta(self, scores, current, actions, previous):
        gate = super().proposal_gate_delta(scores, current, actions, previous)
        buttons = previous * len(ACTIONS) + actions
        agreement = self.motion_agreement(scores, current)
        return gate + self.oriented_command_gate(buttons) + self.motion_agreement_gate(buttons).gather(1, agreement[:, None])

    def motion_agreement(self, scores, current):
        absolute = scores[:, 0].argmax(1)
        span = self.proposal_offsets.abs().amax()
        row = (absolute % 180 // 10 - current % 180 // 10).abs()
        column = (absolute % 10 - current % 10).abs()
        changed = (absolute // 180 != current // 180).long()
        return torch.where((row > span) | (column > span), 2, changed)


class RewriteEvidenceSceneDynamics(OrientedProposalSceneDynamics):
    """Calibrate the rewrite score from visible board and transport evidence."""

    def __init__(self, codebook, object_shapes, object_tiles, preview_patterns):
        super().__init__(codebook, object_shapes, object_tiles, preview_patterns)
        self.rewrite_evidence = nn.Sequential(nn.Linear(10, 8), nn.SiLU(), nn.Linear(8, 1))
        nn.init.constant_(self.rewrite_evidence[-1].bias, 4)
        self.register_buffer("_rewrite_evidence_scene", torch.tensor(True))

    def rewrite_features(self, details):
        current = details["state"][:, 16, :, 2:12].long()
        previous = details["previous_board"]
        occupied = current != self.blank_id
        fullest = occupied.float().mean(2).topk(2, dim=1).values
        older = (previous != self.blank_id).float().mean(2).amax(1)
        unusual = []
        for board in (current, previous):
            special = (board != self.blank_id) & ~torch.isin(board, self.object_tiles)
            unusual.append(special.float().mean(2).amax(1))
        return torch.stack((fullest[:, 0], fullest[:, 1], older, *unusual,
                            occupied[:, :4].float().mean((1, 2)),
                            details["row_transport_gate"].clamp(-6, 6) / 6,
                            details["contact_logit"].sigmoid(), details["contact_visible"].float(),
                            details["next"]["visible"][:, 0].sigmoid()), dim=1)

    def _refine_events(self, scores, details, actions, current):
        scores = super()._refine_events(scores, details, actions, current)
        adjustment = F.logsigmoid(self.rewrite_evidence(self.rewrite_features(details)))
        return torch.cat((scores[:, :2], scores[:, 2:] + adjustment[:, :, None]), dim=1)


class FeedbackCarrySceneDynamics(RewriteEvidenceSceneDynamics):
    """Condition carry confidence on the board actually emitted at the previous step."""

    def __init__(self, codebook, object_shapes, object_tiles, preview_patterns):
        super().__init__(codebook, object_shapes, object_tiles, preview_patterns)
        self.carry_agreement = nn.Linear(3, 1, bias=False)
        nn.init.zeros_(self.carry_agreement.weight)
        self.register_buffer("_feedback_carry_scene", torch.tensor(True))

    def agreement_features(self, tokens, prior_state, carry_features):
        feedback = prior_state[:, -1]
        known = feedback[:, 0, 1] == -2
        different = (feedback[:, :, 2:12] != tokens[:, :, 2:12]).float().mean((1, 2))
        different = torch.where(known, different, torch.ones_like(different))
        changed = (different > 0).float()
        return torch.stack((changed, different, changed * carry_features[:, 2]), dim=1)

    def carry_logit(self, tokens, prior_state, features):
        return super().carry_logit(tokens, prior_state, features) + self.carry_agreement(
            self.agreement_features(tokens, prior_state, features))

    def rendered_state(self, details, prediction):
        state = details["state"]
        feedback = torch.full_like(state[:, -1:], -1)
        feedback[:, 0, 0, 0] = state[:, -1, 0, 0]
        feedback[:, 0, 0, 1] = -2
        feedback[:, 0, :, 2:12] = prediction[:, :, 2:12]
        return torch.cat((state[:, :-1], feedback), dim=1)

    def advance_observed(self, tokens, actions, state=None):
        return self.predict_tokens(tokens, actions, state)[1]


class ReplayEventSceneDynamics(FeedbackCarrySceneDynamics):
    """Calibrate joint event scores from proved continuations of short rollouts."""

    def __init__(self, codebook, object_shapes, object_tiles, preview_patterns):
        super().__init__(codebook, object_shapes, object_tiles, preview_patterns)
        self.replay_calibration = nn.Sequential(nn.Linear(47, 12), nn.SiLU(), nn.Linear(12, 19))
        nn.init.zeros_(self.replay_calibration[-1].weight)
        nn.init.zeros_(self.replay_calibration[-1].bias)
        self.register_buffer("_replay_event_scene", torch.tensor(True))

    def replay_features(self, details, actions, current):
        tracked = details["contact_visible"] & (details["next"]["visible"][:, 0] > 0)
        features = torch.cat((self.proposal_features(details, actions, current), self.rewrite_features(details)), dim=1)
        return features, tracked

    def replay_pose_bins(self, current):
        pose = self.event_poses[None]
        source = current[:, None]
        row = (pose % 180 // 10 - source % 180 // 10).clamp(-3, 3) + 3
        col = (pose % 10 - source % 10).clamp(-3, 3) + 3
        changed = pose // 180 != source // 180
        return (row * 7 + col) * 2 + changed

    def replay_offsets(self, features, tracked):
        mode, row, col, orientation = self.replay_calibration(features).split((3, 7, 7, 2), dim=1)
        motion = row[:, :, None, None] + col[:, None, :, None] + orientation[:, None, None, :]
        return torch.cat((mode[:, :1] + motion.flatten(1) * tracked[:, None], mode[:, 1:]), dim=1)

    def _refine_events(self, scores, details, actions, current):
        scores = super()._refine_events(scores, details, actions, current)
        features, tracked = self.replay_features(details, actions, current)
        details["replay_features"] = features
        offsets = self.replay_offsets(features, tracked)
        copy = scores[:, 0] + offsets.gather(1, self.replay_pose_bins(current))
        return torch.cat((copy[:, None], scores[:, 1:] + offsets[:, -2:, None]), dim=1)


class ClearRoutingSceneDynamics(ReplayEventSceneDynamics):
    """Learn which rendered background to use while retaining per-mode pose scores."""

    def __init__(self, codebook, object_shapes, object_tiles, preview_patterns):
        super().__init__(codebook, object_shapes, object_tiles, preview_patterns)
        self.clear_routing = nn.Linear(51, 4)
        nn.init.zeros_(self.clear_routing.weight)
        nn.init.zeros_(self.clear_routing.bias)
        self.register_buffer("_clear_routing_scene", torch.tensor(True))

    def clear_routing_features(self, details):
        tokens = details["state"][:, 16].long()
        board = tokens[:, :, 2:12]
        moved = (self.transported_rows(tokens, details) != board).any(2)
        confidence = details["row_transport_logits"].softmax(2).amax(2)
        change = (board != details["previous_board"]).float().mean((1, 2))
        row_evidence = torch.stack((moved.float().mean(1), confidence.amin(1),
                                    (confidence * moved).sum(1) / moved.sum(1).clamp_min(1), change), dim=1)
        return torch.cat((details["replay_features"], row_evidence), dim=1)

    def _refine_events(self, scores, details, actions, current):
        scores = super()._refine_events(scores, details, actions, current)
        correction = self.clear_routing(self.clear_routing_features(details))
        details["render_transport_gate"] = details["row_transport_gate"] + correction[:, 3]
        return scores + correction[:, :3, None]


class ClearAnimationSceneDynamics(ClearRoutingSceneDynamics):
    """Learn a visible row sequence with a latched source and an elapsed phase."""

    phase_limit = 52

    def __init__(self, codebook, object_shapes, object_tiles, preview_patterns, flash_id):
        super().__init__(codebook, object_shapes, object_tiles, preview_patterns)
        if not 0 <= flash_id < len(codebook) or flash_id == self.blank_id:
            raise ValueError("invalid observed animation tile")
        self.register_buffer("animation_flash_id", torch.tensor(flash_id))
        self.animation_onset = nn.Sequential(nn.Linear(11, 16), nn.SiLU(), nn.Linear(16, 1))
        nn.init.zeros_(self.animation_onset[-1].weight)
        nn.init.constant_(self.animation_onset[-1].bias, -8.)
        self.animation_time = nn.Embedding(self.phase_limit, 8)
        self.animation_rows = nn.Sequential(nn.Linear(33, 24), nn.SiLU(), nn.Linear(24, 11))
        self.animation_end = nn.Sequential(nn.Linear(21, 16), nn.SiLU(), nn.Linear(16, 2))
        self.animation_origin = nn.Linear(29, 28)
        self.animation_shape = nn.Linear(64, len(object_shapes))
        nn.init.zeros_(self.animation_shape.weight)
        nn.init.zeros_(self.animation_shape.bias)
        rows = torch.arange(18)[:, None] + torch.arange(-4, 5)
        self.register_buffer("animation_row_indices", rows, persistent=False)
        self.register_buffer("_clear_animation_scene", torch.tensor(True))

    def onset_features(self, board, previous):
        ordinary = torch.isin(board, self.object_tiles)
        density = ordinary.float().mean(2)
        fullest, rows = density.topk(4, dim=1)
        unchanged = (board == previous).float().mean(2)
        unusual = (~ordinary & (board != self.blank_id)).float().mean((1, 2))
        summary = torch.stack((density.mean(1), unusual, 1 - unchanged.mean(1)), dim=1)
        return torch.cat((fullest.pow(16), unchanged.gather(1, rows), summary), dim=1)

    def animation_start_logit(self, tokens, details):
        features = self.onset_features(tokens[:, :, 2:12], details["previous_board"])
        return self.animation_onset(features)[:, 0]

    def animation_features(self, source, age, actions):
        board = source[:, :, 2:12]
        density = torch.isin(board, self.object_tiles).float().mean(2)
        time = self.animation_time(age.clamp_max(self.phase_limit - 1))
        phase = torch.cat((time, density.topk(4, dim=1).values.pow(16), age[:, None].float() / self.phase_limit,
                           self.action_embedding(actions)), dim=1)
        return phase, density

    def animation_row_choices(self, source):
        board = source[:, :, 2:12]
        nearby = board[:, self.animation_row_indices.clamp(0, 17)]
        blank = torch.full_like(nearby[:, :, :1], self.blank_id)
        flash = torch.ones_like(blank) * self.animation_flash_id
        return torch.cat((nearby, blank, flash), dim=2)

    def animation_row_scores(self, phase, density):
        nearby = F.pad(density, (4, 4)).unfold(1, 9, 1)
        peaks = density.pow(16)
        prefix = peaks.cumsum(1)
        suffix = peaks.sum(1, keepdim=True) - prefix + peaks
        row = torch.arange(18, device=density.device, dtype=density.dtype)[None].expand_as(density) / 17
        local = torch.stack((prefix / 4, suffix / 4, row), dim=2)
        logits = self.animation_rows(torch.cat((phase[:, None].expand(-1, 18, -1), nearby, local), dim=2))
        invalid = (self.animation_row_indices < 0) | (self.animation_row_indices >= 18)
        return logits.masked_fill(F.pad(invalid, (0, 2)), -torch.inf)

    def animation_spawn(self, source, phase, previous):
        preview = self._preview_features(source)
        shape = self.spawn_preview_head(preview) + self.animation_shape(preview)
        embedding = self.action_embedding(previous.clamp_max(len(ACTIONS) - 1))
        embedding = embedding * (previous < len(ACTIONS))[:, None]
        row, col = self.animation_origin(torch.cat((phase, embedding), dim=1)).split((18, 10), dim=1)
        pose = (row[:, :, None] + col[:, None, :]).flatten(1)
        joint = (shape[:, :, None] + pose[:, None]).masked_fill(~self.valid_poses, -torch.inf)
        return joint.flatten(1).log_softmax(1).reshape_as(joint)

    def animation_end_logits(self, phase, age):
        return self.animation_end(phase).unbind(1)

    def animation_output_phase(self, phase, ending):
        return phase

    def advance(self, tokens, actions, state=None):
        if state is not None and state.size(1) == self.state_channels + 1:
            state = state[:, :self.state_channels]
        return super().advance(tokens, actions, state)

    def scene_details(self, tokens, actions, state=None):
        memory = None
        if state is not None and state.size(1) == self.state_channels + 1:
            state, memory = state[:, :self.state_channels], state[:, -1]
        details = super().scene_details(tokens, actions, state)
        age = torch.zeros_like(actions) if memory is None else memory[:, 0, 0].long()
        onset = self.animation_start_logit(tokens, details)
        active = ((age > 0) | (onset > 0)).nonzero().flatten()
        next_memory = torch.zeros_like(details["state"][:, -1:])
        details["animation_memory"] = next_memory
        if not len(active):
            return details
        source = tokens[active] if memory is None else torch.where(
            age[active, None, None] > 0, memory[active].long(), tokens[active])
        phase, density = self.animation_features(source, age[active], actions[active])
        end, show = self.animation_end_logits(phase, age[active])
        ending = end > 0
        visible = ending & (show > 0)
        phase = self.animation_output_phase(phase, ending)
        joint = self.animation_spawn(source, phase, details["previous_action"][active])
        rows = self.animation_row_choices(source)
        scores = self.animation_row_scores(phase, density)
        details["animation"] = {"active": active, "ending": ending, "rows": rows, "scores": scores,
                                "pose": torch.where(visible, joint.flatten(1).argmax(1), -1)}
        details["mode"] = details["mode"].clone()
        details["mode"][active] = details["mode"].new_tensor([-1., -1., 1.])
        details["next"]["joint"] = details["next"]["joint"].index_copy(0, active, joint)
        visibility = (visible[:, None].to(joint) * 2 - 1) * 20
        details["next"]["visible"] = details["next"]["visible"].index_copy(0, active, visibility)
        waiting = active[~ending]
        next_memory[waiting, 0] = source[~ending].to(next_memory)
        next_memory[waiting, 0, 0, 0] = (age[waiting] + 1).to(next_memory)
        return details

    def _hard_background(self, tokens, details, gate):
        background = super()._hard_background(tokens, details, gate)
        animation = details.get("animation")
        if animation is not None:
            chosen = animation["scores"].argmax(2)
            rows = animation["rows"].gather(2, chosen[:, :, None, None].expand(-1, -1, 1, 10))
            background[animation["active"], :, 2:12] = rows[:, :, 0]
        return background

    def decode_scene(self, tokens, details, hard=False):
        rendered = super().decode_scene(tokens, details, hard)
        animation = details.get("animation")
        if animation is None:
            return rendered
        active = animation["active"]
        waiting = active[~animation["ending"]]
        if hard:
            rendered[waiting, :, :2] = tokens[waiting, :, :2]
            rendered[waiting, :, 12:] = tokens[waiting, :, 12:]
        else:
            rendered = rendered.clone()
            palette = F.one_hot(animation["rows"], len(self.codebook)).to(rendered.dtype)
            background = torch.einsum("brs,brsxv->bvrx", animation["scores"].softmax(2), palette)
            foreground = {key: value[active] for key, value in details["next"].items() if key in ("joint", "visible")}
            mask, color = self._soft_object(foreground)
            board = background * (1 - mask[:, :, :, 2:12]) + color[:, :, :, 2:12]
            held = F.one_hot(tokens[waiting], len(self.codebook)).permute(0, 3, 1, 2).to(rendered)
            rendered[waiting] = held.clamp_min(1e-9).log()
            rendered[active, :, :, 2:12] = board.clamp_min(1e-9).log()
        return rendered

    def rendered_state(self, details, prediction):
        state = super().rendered_state(details, prediction)
        animation = details.get("animation")
        if animation is not None:
            state[animation["active"], -1, 0, 0] = animation["pose"].to(state)
        return torch.cat((state, details["animation_memory"]), dim=1)

    def forward(self, tokens, actions, state=None):
        details = self.scene_details(tokens, actions, state)
        rendered = self.decode_scene(tokens, details)
        return rendered, self.rendered_state(details, rendered.argmax(1))


class ClearDurationSceneDynamics(ClearAnimationSceneDynamics):
    """Predict a duration from the latched source to keep phase completion monotone."""

    def __init__(self, codebook, object_shapes, object_tiles, preview_patterns, flash_id):
        super().__init__(codebook, object_shapes, object_tiles, preview_patterns, flash_id)
        self.register_buffer("_clear_duration_scene", torch.tensor(True))

    def clear_duration(self, phase):
        neutral = torch.zeros(len(phase), dtype=torch.long, device=phase.device)
        # The timing head sees the locked board with a fixed age and button.
        source = torch.cat((self.animation_time(neutral), phase[:, 8:12], phase.new_zeros(len(phase), 1),
                            self.action_embedding(neutral)), dim=1)
        return self.animation_end(source)[:, 0].sigmoid() * (self.phase_limit - 1)

    def animation_output_phase(self, phase, ending):
        terminal = torch.full((len(phase),), self.phase_limit - 1, dtype=torch.long, device=phase.device)
        features = torch.cat((self.animation_time(terminal), phase[:, 8:12], phase.new_ones(len(phase), 1),
                              phase[:, 13:]), dim=1)
        return torch.where(ending[:, None], features, phase)

    def animation_end_logits(self, phase, age):
        terminal = self.animation_output_phase(phase, torch.ones_like(age, dtype=torch.bool))
        show = self.animation_end(terminal)[:, 1]
        return 2 * (age.float() + .5 - self.clear_duration(phase)), show


class TerminalSceneDynamics(ClearDurationSceneDynamics):
    """Recognize a terminal phase and render its observed wipe and panel."""

    def __init__(self, codebook, object_shapes, object_tiles, preview_patterns, flash_id, fill_id, panels):
        super().__init__(codebook, object_shapes, object_tiles, preview_patterns, flash_id)
        if not 0 <= fill_id < len(codebook) or fill_id == self.blank_id:
            raise ValueError("invalid observed terminal wipe tile")
        if (panels.ndim != 3 or panels.shape[1:] != (18, 20) or not 1 <= len(panels) <= 16
                or torch.any((panels < 0) | (panels >= len(codebook)))):
            raise ValueError("invalid observed terminal panels")
        self.register_buffer("terminal_fill_id", torch.tensor(fill_id))
        self.register_buffer("terminal_panels", panels.to(torch.uint8 if len(codebook) <= 256 else torch.long))
        self.terminal_onset = nn.Sequential(nn.Linear(19, 12), nn.SiLU(), nn.Linear(12, 1))
        self.terminal_duration = nn.Linear(12, 1)
        self.terminal_wipe = nn.Sequential(nn.Linear(15, 12), nn.SiLU(), nn.Linear(12, 1))
        self.terminal_menu = nn.Sequential(nn.Linear(len(panels) + 17, 12), nn.SiLU(), nn.Linear(12, len(panels)))
        nn.init.zeros_(self.terminal_onset[-1].weight)
        nn.init.constant_(self.terminal_onset[-1].bias, -8.)
        self.register_buffer("_terminal_scene", torch.tensor(True))

    def terminal_features(self, board, age):
        occupied = torch.isin(board, self.object_tiles)
        upper = occupied[:, :4]
        density = upper.float().mean(2)
        central = upper[:, :, 2:8].float().mean(2)
        support = (upper & occupied[:, 1:5]).float().mean(2)
        elapsed = age[:, :4].float().log1p() / 5.5
        population = upper.sum((1, 2)).clamp_min(1)
        mean_age = (elapsed * upper).sum((1, 2)) / population
        youngest = elapsed.masked_fill(~upper, 2).amin((1, 2)).clamp_max(1)
        changing = ((age[:, :4] == 0) & upper).sum((1, 2)) / population
        unusual = (~occupied & (board != self.blank_id)).float().mean((1, 2))
        summary = torch.stack((occupied[:, 4:].float().mean((1, 2)),
                               occupied[:, 4:].any(2).float().mean(1), occupied.float().mean((1, 2)),
                               mean_age, youngest, changing, unusual), dim=1)
        return torch.cat((density, central, support, summary), dim=1)

    def terminal_phase(self, source, age):
        board = source[:, :, 2:12]
        profile = self.terminal_features(board, torch.zeros_like(board))[:, :12]
        duration = self.terminal_duration(profile)[:, 0].sigmoid() * 40
        time = torch.stack((age.float() / 40, (age - duration) / 40), dim=1)
        features = torch.cat((profile, time), dim=1)
        row = torch.arange(18, device=board.device, dtype=features.dtype) / 17
        row = row[None, :, None].expand(len(board), -1, -1)
        wipe = self.terminal_wipe(torch.cat((features[:, None].expand(-1, 18, -1), row), dim=2))[:, :, 0]
        return wipe, duration

    def terminal_menu_scores(self, tokens, actions, elapsed):
        matches = (tokens[:, None] == self.terminal_panels[None]).flatten(2).all(2)
        known = matches.long().argmax(1)
        index = torch.where(matches.any(1), known, len(self.terminal_panels))
        panel = F.one_hot(index, len(self.terminal_panels) + 1).float()
        time = F.one_hot(elapsed.clamp(0, 7), 8).float()
        return self.terminal_menu(torch.cat((panel, time, self.action_embedding(actions)), dim=1))

    def terminal_start_logit(self, tokens, details):
        features = self.terminal_features(tokens[:, :, 2:12], details["state"][:, 17, :, 2:12])
        return self.terminal_onset(features)[:, 0]

    def scene_details(self, tokens, actions, state=None):
        details = super().scene_details(tokens, actions, state)
        memory = state[:, -1] if state is not None and state.size(1) == self.state_channels + 1 else None
        age = torch.zeros_like(actions) if memory is None else (-memory[:, 0, 0].long()).clamp_min(0)
        onset = self.terminal_start_logit(tokens, details) > 0
        animation = details.get("animation")
        if animation is not None:
            onset[animation["active"]] = False
        active = ((age > 0) | onset).nonzero().flatten()
        if not len(active):
            return details
        source = tokens[active] if memory is None else torch.where(
            age[active, None, None] > 0, memory[active].long(), tokens[active])
        wipe, duration = self.terminal_phase(source, age[active])
        panel = 2 * (age[active].float() + .5 - duration)
        menu = self.terminal_menu_scores(tokens[active], actions[active], (panel / 2).floor().long())
        details["terminal"] = {"active": active, "source": source[:, :, 2:12], "wipe": wipe, "panel": panel,
                               "menu": menu, "chosen": self.terminal_panels[menu.argmax(1)].long()}
        details["next"]["visible"] = details["next"]["visible"].clone()
        details["next"]["visible"][active] = -20
        details["mode"] = details["mode"].clone()
        details["mode"][active] = details["mode"].new_tensor([-1., -1., 1.])
        details["animation_memory"][active, 0] = source.to(details["animation_memory"])
        details["animation_memory"][active, 0, 0, 0] = -(age[active] + 1).clamp_max(self.phase_limit).float()
        return details

    def _hard_background(self, tokens, details, gate):
        background = super()._hard_background(tokens, details, gate)
        terminal = details.get("terminal")
        if terminal is not None:
            wipe = torch.where(terminal["wipe"][:, :, None] > 0, self.terminal_fill_id, terminal["source"])
            background[terminal["active"], :, 2:12] = torch.where(
                terminal["panel"][:, None, None] > 0, terminal["chosen"][:, :, 2:12], wipe)
        return background

    def decode_scene(self, tokens, details, hard=False):
        rendered = super().decode_scene(tokens, details, hard)
        terminal = details.get("terminal")
        if terminal is None:
            return rendered
        active = terminal["active"]
        if hard:
            rendered[active, :, :2] = tokens[active, :, :2]
            rendered[active, :, 12:] = tokens[active, :, 12:]
            showing = terminal["panel"] > 0
            rendered[active[showing]] = terminal["chosen"][showing]
        else:
            palette = len(self.codebook)
            display = F.one_hot(tokens[active], palette).permute(0, 3, 1, 2).to(rendered)
            source = F.one_hot(terminal["source"], palette).permute(0, 3, 1, 2).to(rendered)
            fill = F.one_hot(self.terminal_fill_id, palette).to(rendered)[None, :, None, None]
            display[:, :, :, 2:12] = torch.lerp(source, fill, terminal["wipe"][:, None, :, None].sigmoid())
            patterns = F.one_hot(self.terminal_panels.long(), palette).permute(0, 3, 1, 2).to(rendered)
            panel = torch.einsum("bp,pvrc->bvrc", terminal["menu"].softmax(1), patterns)
            display = torch.lerp(display, panel, terminal["panel"][:, None, None, None].sigmoid())
            rendered = rendered.clone()
            rendered[active] = display.clamp_min(1e-9).log()
        return rendered

    def rendered_state(self, details, prediction):
        state = super().rendered_state(details, prediction)
        terminal = details.get("terminal")
        if terminal is not None:
            state[terminal["active"], -2, 0, 0] = -1
        return state


class TerminalContactSceneDynamics(TerminalSceneDynamics):
    """Calibrate a terminal proposal using the currently tracked foreground."""

    def __init__(self, codebook, object_shapes, object_tiles, preview_patterns, flash_id, fill_id, panels):
        super().__init__(codebook, object_shapes, object_tiles, preview_patterns, flash_id, fill_id, panels)
        self.terminal_contact_gate = nn.Linear(5, 1, bias=False)
        nn.init.zeros_(self.terminal_contact_gate.weight)
        self.register_buffer("_terminal_contact_scene", torch.tensor(True))

    def terminal_contact_features(self, tokens, details):
        known = details["contact_visible"]
        supported = details["contact_neighbors"][:, 7] > 0
        mask, _ = self._hard_object(details["current"])
        occupied = torch.isin(tokens[:, :, 2:12], self.object_tiles)
        pile = occupied & ~(mask[:, :, 2:12] & known[:, None, None])
        proposal, _ = self._hard_object(details["next"])
        overlap = (pile & proposal[:, :, 2:12]).float().sum((1, 2)) / 4
        copy = details["mode"].argmax(1) == 0
        return torch.stack((known & ~supported, known & supported, ~known, pile[:, :4].float().mean((1, 2)),
                            overlap * copy), dim=1).float()

    def terminal_start_logit(self, tokens, details):
        base = super().terminal_start_logit(tokens, details)
        return base + self.terminal_contact_gate(self.terminal_contact_features(tokens, details))[:, 0]


class ClearOnsetSceneDynamics(TerminalContactSceneDynamics):
    """Learn whether visible row fullness depends on a moving foreground."""

    def __init__(self, codebook, object_shapes, object_tiles, preview_patterns, flash_id, fill_id, panels):
        super().__init__(codebook, object_shapes, object_tiles, preview_patterns, flash_id, fill_id, panels)
        self.animation_contact_gate = nn.Linear(5, 1, bias=False)
        nn.init.zeros_(self.animation_contact_gate.weight)
        self.register_buffer("_clear_onset_scene", torch.tensor(True))

    def animation_contact_features(self, tokens, details):
        board = tokens[:, :, 2:12]
        occupied = torch.isin(board, self.object_tiles)
        known = details["contact_visible"]
        support = details["contact_neighbors"][:, 7] > 0
        mask, _ = self._hard_object(details["current"])
        foreground = mask[:, :, 2:12] & known[:, None, None]
        strength = occupied.float().mean(2).pow(16)
        pile = (occupied & ~foreground).float().mean(2).pow(16)
        contribution = (strength - pile).sum(1)
        changed = ((board != details["previous_board"]) & foreground).float().sum((1, 2)) / 4
        return torch.stack((contribution * ~support, contribution * support, strength.sum(1) * ~known,
                            pile.sum(1), contribution * changed), dim=1)

    def animation_start_logit(self, tokens, details):
        base = super().animation_start_logit(tokens, details)
        return base + self.animation_contact_gate(self.animation_contact_features(tokens, details))[:, 0]


class ComposeContactSceneDynamics(ClearOnsetSceneDynamics):
    """Calibrate retained-foreground proposals using visible support."""

    def __init__(self, codebook, object_shapes, object_tiles, preview_patterns, flash_id, fill_id, panels):
        super().__init__(codebook, object_shapes, object_tiles, preview_patterns, flash_id, fill_id, panels)
        self.compose_contact_gate = nn.Linear(1, 1, bias=False)
        nn.init.zeros_(self.compose_contact_gate.weight)
        self.register_buffer("_compose_contact_scene", torch.tensor(True))

    def compose_contact_signal(self, details):
        return details["contact_visible"][:, None] * (1 - details["contact_neighbors"][:, 7:8])

    def _refine_events(self, scores, details, actions, current):
        scores = super()._refine_events(scores, details, actions, current)
        offset = self.compose_contact_gate(self.compose_contact_signal(details))
        return scores + F.pad(offset, (1, 1))[:, :, None]


class CounterSceneDynamics(ComposeContactSceneDynamics):
    """Add a legible, learned HUD at the pixel bridge of the frozen scene model."""

    def __init__(self, codebook, object_shapes, object_tiles, preview_patterns, flash_id, fill_id, panels,
                 glyphs, legacy_ids, addends, pulsed=False, calibrated=0, a_rotation=None, a_proposal_gate=False,
                 a_placement=None, pause=None):
        from counter_model import CounterReadout

        super().__init__(codebook, object_shapes, object_tiles, preview_patterns, flash_id, fill_id, panels)
        self.counter = CounterReadout(glyphs, legacy_ids, addends, pulsed)
        if calibrated not in (0, 2, 3):
            raise ValueError("invalid scene continuity coefficient count")
        self.scene_continuity = nn.Parameter(torch.zeros(calibrated)) if calibrated else None
        target, odds = (None, None) if a_rotation is None else a_rotation
        if target is not None:
            size = (len(object_shapes),)
            if (target.dtype != torch.long or target.shape != size or odds.shape != size
                    or torch.any((target < -1) | (target >= size[0])) or not torch.isfinite(odds).all()
                    or torch.any(odds < 0)):
                raise ValueError("invalid learned A orientation vocabulary")
        self.register_buffer("a_rotation_target", target)
        self.register_buffer("a_rotation_odds", odds)
        if a_proposal_gate and target is None:
            raise ValueError("the A proposal gate needs learned orientations")
        self.register_buffer("_a_proposal_gate", torch.tensor(True) if a_proposal_gate else None)
        column, column_odds = (None, None) if a_placement is None else a_placement
        if column is not None:
            size = (len(object_shapes),)
            if (not a_proposal_gate or column.dtype != torch.long or column.shape != size
                    or column_odds.shape != size or not torch.isfinite(column_odds).all()
                    or torch.any(column_odds < 0) or torch.any(column.abs() > 3)):
                raise ValueError("invalid learned A placement vocabulary")
        self.register_buffer("a_column_offset", column)
        self.register_buffer("a_column_odds", column_odds)
        if torch.any((legacy_ids < 0) | (legacy_ids >= len(codebook))):
            raise ValueError("legacy counter glyph is outside the scene codec")
        panel, transition, resume = (None, None, None) if pause is None else pause
        if pause is not None and (
                panel.dtype != torch.uint8 or panel.shape != (18, 20) or torch.any(panel.long() >= len(codebook))
                or transition.shape != (3, len(ACTIONS)) or not transition.is_floating_point()
                or not torch.isfinite(transition).all()
                or resume.dtype != torch.long or resume.ndim != 0 or not 0 <= int(resume) < len(ACTIONS)):
            raise ValueError("invalid learned pause readout")
        self.register_buffer("pause_panel", panel)
        self.register_buffer("pause_resume_action", resume)
        self.pause_transition = nn.Parameter(torch.zeros_like(transition)) if pause is not None else None
        self.register_buffer("_counter_scene", torch.tensor(True))

    def rotation_placement_visible(self, current, neighbors):
        col = current % 10
        right = col + self.object_shapes[current // 180, :, 1].amax(1)
        return ~neighbors.bool().any(1) & (col > 0) & (right < 9)

    def adjust_proposal(self, gate, absolute, relative, current, actions, neighbors):
        if self._a_proposal_gate is None or actions is None or not (actions == ACTIONS.index("a")).any():
            return gate, relative
        kind = current // 180
        target = self.a_rotation_target[kind]
        style = self.object_tiles[self.event_poses // 180] == self.object_tiles[kind, None]
        absolute_kind = absolute.masked_fill(~style, -torch.inf).argmax(1) // 180
        local_kind = relative.argmax(1) // 180
        odds = self.a_rotation_odds[kind]
        use = ((actions == ACTIONS.index("a")) & (target >= 0) & (odds > 0)
               & (absolute_kind != kind) & (absolute_kind != target) & (local_kind == target))
        gate = torch.where(use[:, None], odds[:, None], gate)
        if self.a_column_offset is None or neighbors is None:
            return gate, relative
        place = use & self.rotation_placement_visible(current, neighbors)
        group = (self.event_poses // 180 == target[:, None]) & place[:, None]
        column = current % 10 + self.a_column_offset[kind]
        preferred = group & (self.event_poses % 10 == column[:, None])
        shifted = relative + self.a_column_odds[kind, None] * preferred
        before = relative.masked_fill(~group, -torch.inf).logsumexp(1)
        after = shifted.masked_fill(~group, -torch.inf).logsumexp(1)
        scale = torch.where(place, before - after, 0)
        return gate, torch.where(group, shifted + scale[:, None], relative)

    def continuity_signals(self, tokens, details):
        known = details["contact_visible"].clone()
        for phase in ("animation", "terminal"):
            if phase in details:
                known[details[phase]["active"]] = False
        boards = torch.stack((tokens[:, :, 2:12], details["previous_board"]), dim=1)
        occupied = torch.isin(boards, self.object_tiles)
        ordinary = (occupied | (boards == self.blank_id)).flatten(1).all(1)
        full_row = occupied.all(3).flatten(1).any(1)
        rewrite_context = known & ordinary & ~full_row & (details["contact_neighbors"][:, 7] > 0)
        current = details["current"]["joint"].flatten(1).argmax(1)
        changed_style = self.object_tiles[self.event_poses // 180] != self.object_tiles[current // 180, None]
        return known, changed_style, rewrite_context

    def lateral_switch(self, tokens, actions, details, known):
        previous = details["previous_action"]
        left, right, down = (ACTIONS.index(name) for name in ("left", "right", "down"))
        active = (known & (actions == down) & ((previous == left) | (previous == right))).nonzero().flatten()
        proved = torch.zeros_like(known)
        if not len(active):
            return proved
        pose = details["current"]["joint"][active].flatten(1).argmax(1)
        kind, position = pose // 180, pose % 180
        row, col = position // 10, position % 10
        prior_col = col + torch.where(previous[active] == left, 1, -1)
        prior_row = torch.stack((row, row - 1), dim=1)
        prior_pos = prior_row.clamp_min(0) * 10 + prior_col.clamp(0, 9)[:, None]
        mask = self.shape_masks[kind, position].bool()
        earlier_mask = self.shape_masks[kind[:, None], prior_pos].bool()
        earlier = details["previous_board"][active, None]
        color = self.object_tiles[kind, None, None, None]
        painted = ((earlier == color) | ~earlier_mask).flatten(2).all(2)
        background = tokens[active, :, 2:12].masked_fill(mask, self.blank_id)
        unchanged = (earlier.masked_fill(earlier_mask, self.blank_id) == background[:, None]).flatten(2).all(2)
        valid = self.valid_poses[kind[:, None], prior_pos] & (prior_row >= 0)
        proved[active] = ((painted & unchanged & valid).sum(1) == 1) & (prior_col >= 0) & (prior_col < 10)
        return proved

    def scene_details(self, tokens, actions, state=None):
        details = super().scene_details(tokens, actions, state)
        if self.scene_continuity is None:
            return details
        known, changed_style, rewrite = self.continuity_signals(tokens, details)
        style_scale, rewrite_scale = self.scene_continuity[:2].unbind()
        scores = details["event_scores"]
        old = scores[:, 0]
        copy = old + style_scale * (changed_style & known[:, None])
        if len(self.scene_continuity) == 3:
            switched = self.lateral_switch(tokens, actions, details, known)
            current_col = details["current"]["joint"].flatten(1).argmax(1) % 10
            lateral = (self.event_poses % 10 != current_col[:, None]) & switched[:, None]
            copy = copy + self.scene_continuity[2] * lateral
        if self.a_rotation_target is not None and self._a_proposal_gate is None:
            kind = details["current"]["joint"].flatten(1).argmax(1) // 180
            a, noop = ACTIONS.index("a"), ACTIONS.index("noop")
            command = (actions == a) | ((actions == noop) & (details["previous_action"] == a))
            target = self.a_rotation_target[kind]
            changing = ~changed_style & (self.event_poses // 180 != kind[:, None])
            use = known & command & (target >= 0)
            eligible = changing & use[:, None]
            preferred = (self.event_poses // 180 == target[:, None]) & eligible
            shifted = copy + self.a_rotation_odds[kind, None] * preferred
            base_peak = copy.masked_fill(~eligible, -torch.inf).amax(1)
            adjusted_peak = shifted.masked_fill(~eligible, -torch.inf).amax(1)
            offset = torch.where(eligible.any(1), base_peak - adjusted_peak, 0)
            copy = torch.where(eligible, shifted + offset[:, None], copy)
        # The conditional identity correction leaves the composition score of copy intact.
        copy = copy + (old.amax(1) - copy.amax(1))[:, None]
        scores = torch.stack((copy, scores[:, 1], scores[:, 2] + (rewrite_scale * rewrite)[:, None]), dim=1)
        mode = scores.amax(2)
        selected = scores[torch.arange(len(scores), device=scores.device), mode.argmax(1)]
        joint = selected.log_softmax(1).reshape_as(details["next"]["joint"])
        details["next"]["joint"] = torch.where(known[:, None, None], joint, details["next"]["joint"])
        details["mode"] = torch.where(known[:, None], mode, details["mode"])
        details["event_scores"] = scores
        return details

    def counter_features(self, tokens, actions, details, prediction):
        old = details["current"]["joint"].flatten(1).argmax(1)
        new = details["next"]["joint"].flatten(1).argmax(1)
        change = (torch.isin(prediction[:, :, 2:12], self.object_tiles).sum((1, 2))
                  - torch.isin(tokens[:, :, 2:12], self.object_tiles).sum((1, 2)))
        phase = details["mode"].new_zeros((len(tokens), 3))
        if (animation := details.get("animation")) is not None:
            active = animation["active"]
            phase[active, 0] = 1
            phase[active, 1] = animation["ending"].to(phase)
            density = torch.isin(animation["rows"][:, :, 4], self.object_tiles).float().mean(2)
            phase[active, 2] = density.pow(16).sum(1) / 4
        style = self.object_tiles[old // 180] != self.object_tiles[new // 180]
        movement = torch.stack((details["contact_visible"], details["contact_neighbors"][:, 7],
                                style, (old % 180 // 10 - new % 180 // 10) / 17, change / 10), dim=1)
        return torch.cat((details["mode"].softmax(1), details["current"]["visible"].sigmoid(),
                          details["next"]["visible"].sigmoid(), movement, phase,
                          F.one_hot(actions, len(ACTIONS))), dim=1)

    def predict_frame(self, frame, action, state=None):
        return self._pixel_step(frame, action, state, observed=False)

    def advance_observed_frame(self, frame, action, state=None):
        return self._pixel_step(frame, action, state, observed=True)[1]

    def pause_phase(self, frame, state, paused):
        digits = self.counter.read(frame)
        unavailable = ~(digits >= 0).all(1) | (digits[:, [4, 9, 11]] == 10).any(1)
        if state is not None and state.size(1) > self.state_channels:
            unavailable |= state[:, self.state_channels, 0, 0] != 0
        return torch.where(paused, 1, 2 * unavailable.long())

    def _pixel_step(self, frame, action, state, observed):
        if self.pause_transition is None:
            return self._counter_step(frame, action, state, observed)
        channels = self.state_channels + 2
        carried = state is not None and state.size(1) == channels + 3
        if not carried and not (self.pause_transition[0, action] > 0).any():
            return self._counter_step(frame, action, state, observed)
        core = state[:, :channels] if carried else state
        width = torch.full_like(action, 0 if core is None else core.size(1))
        paused = torch.zeros_like(action, dtype=torch.bool)
        if carried:
            saved_width = state[:, channels, 0, 0].long()
            paused = saved_width >= 0
            width = torch.where(paused, saved_width, width)
        hold = self.pause_transition[self.pause_phase(frame, core, paused), action] > 0
        if not (hold | paused).any():
            return self._counter_step(frame, action, core, observed)

        # Pack eight 2-bit pixels per float32 cell so the suspended screen is lossless.
        shifts = torch.arange(0, 16, 2, device=frame.device)
        resume = paused & ~hold
        if resume.any():
            frame = frame.clone()
            packed = state[resume, -2:].long()
            frame[resume] = ((packed[..., None] >> shifts) & 3).reshape(-1, 1, 72, 80).to(frame)
            action = torch.where(resume, self.pause_resume_action, action)

        output = torch.empty_like(frame)
        next_core = frame.new_zeros((len(frame), channels, 18, 20), dtype=torch.float32)
        for size in width[~hold].unique().tolist():
            live = ~hold & (width == size)
            memory = core[live, :size] if size else None
            output[live], next_core[live] = self._counter_step(frame[live], action[live], memory, observed)
        if not hold.any():
            return output, next_core

        output[hold] = decode_tokens(self.pause_panel[None].long(), self.codebook)
        if core is not None:
            next_core[hold, :core.size(1)] = core[hold]
        memory = next_core.new_zeros((len(frame), 3, 18, 20))
        memory[:, 0, 0, 0] = width.masked_fill(~hold, -1)
        if carried:
            held = hold & paused
            memory[held, 1:] = state[held, -2:]
        entering = hold & ~paused
        pixels = frame[entering].reshape(-1, 2, 18, 20, 8).long()
        memory[entering, 1:] = (pixels << shifts).sum(-1).to(memory)
        return output, torch.cat((next_core, memory), dim=1)

    def rendered_counter_event(self, tokens, details, prediction):
        from counter_model import row_strength

        board, result = tokens[:, :, 2:12], prediction[:, :, 2:12]
        overlay, color = self._hard_object(details["next"])
        overlay = overlay[:, :, 2:12]
        composition = torch.where(overlay, color[:, None, None], board)
        painted = (overlay.sum((1, 2)) == 4) & (~overlay | (board == self.blank_id)).flatten(1).all(1)
        deposit = ((details["mode"].argmax(1) == 1) & details["contact_visible"]
                   & (details["contact_neighbors"][:, 7] > 0) & painted
                   & (result == composition).flatten(1).all(1))
        code = deposit.long()
        if (animation := details.get("animation")) is not None:
            ending = animation["ending"]
            active = animation["active"][ending]
            source = animation["rows"][ending, :, 4]
            occupied = torch.isin(result[active], self.object_tiles)
            compacted = (occupied | (result[active] == self.blank_id)).flatten(1).all(1)
            compacted &= occupied.sum((1, 2)) < torch.isin(source, self.object_tiles).sum((1, 2))
            learned = self.counter.events(row_strength(source, self.object_tiles)).argmax(1) + 2
            code[active] = learned.masked_fill(~compacted, 0)
        if (terminal := details.get("terminal")) is not None:
            code[terminal["active"]] = 0
        return code

    def _counter_step(self, frame, action, state, observed):
        from counter_model import FEATURES

        tokens = self.frame_tokens(frame)
        digits = self.counter.read(frame)
        rows, cols = self.counter.rows, self.counter.columns
        legacy = torch.where(digits >= 0, self.counter.legacy_ids[digits.clamp_min(0)], tokens[:, rows, cols])
        width = 1 if self.counter.pulsed else FEATURES
        previous = frame.new_zeros((len(frame), width), dtype=torch.float32)
        replay = torch.zeros_like(action, dtype=torch.bool)
        if state is not None and state.size(1) == self.state_channels + 2:
            memory = state[:, -1].flatten(1)
            previous = memory[:, :width]
            if not observed:
                cached, displayed = memory[:, width:width + 2 * len(rows)].long().chunk(2, dim=1)
                replay = (displayed >= 0).all(1) & (digits == displayed).all(1)
                legacy = torch.where(replay[:, None], cached, legacy)
            state = state[:, :-1]
        tokens[:, rows, cols] = legacy
        details = self.scene_details(tokens, action, state)
        prediction = self.decode_scene(tokens, details, hard=True)
        core = self.rendered_state(details, prediction)
        use = (digits >= 0).all(1) & (digits[:, [4, 9, 11]] != 10).all(1)
        if (terminal := details.get("terminal")) is not None:
            use[terminal["active"][terminal["panel"] > 0]] = False
        updated = digits.clamp_min(0)
        if self.counter.pulsed:
            pending = torch.zeros_like(action)
            if not observed:
                code = self.rendered_counter_event(tokens, details, prediction).masked_fill(~use, 0)
                prior = previous[:, 0].long().masked_fill(~(use & replay), 0)
                updated, pending = self.counter.consume(updated, prior, code)
            features = pending[:, None]
        else:
            features = self.counter_features(tokens, action, details, prediction)
            updated = self.counter(updated, torch.cat((previous, features), dim=1))
        rendered = self.counter.paint(decode_tokens(prediction, self.codebook), updated, use)
        # Retain the frozen renderer's inputs when the readout repaints its own feedback.
        displayed = updated.masked_fill(~(use & (not observed))[:, None], -1)
        memory = torch.cat((features, prediction[:, rows, cols], displayed), dim=1)
        memory = F.pad(memory, (0, 18 * 20 - memory.size(1))).reshape(len(frame), 1, 18, 20)
        return rendered, torch.cat((core, memory), dim=1)


class RowMotionClockSceneDynamics(RowTransportSceneDynamics):
    def __init__(self, codebook, object_shapes, object_tiles, preview_patterns):
        super().__init__(codebook, object_shapes, object_tiles, preview_patterns)
        self.motion_clock_head = nn.Linear(26, 1)
        nn.init.zeros_(self.motion_clock_head.weight)
        nn.init.zeros_(self.motion_clock_head.bias)
        self.register_buffer("_row_motion_clock_scene", torch.tensor(True))

    def motion_clock_features(self, state, actions):
        return _clock_features(state, self.action_embedding(actions))

    def motion_candidates(self, current_joint, actions):
        current = current_joint.flatten(1).argmax(1, keepdim=True)
        poses = torch.arange(current_joint[0].numel(), device=current_joint.device)[None, :]
        down = (poses // 180 == current // 180) & (poses % 10 == current % 10) & (poses > current)
        return (down & (actions[:, None] == 3)).reshape_as(current_joint) & self.valid_poses

    def scene_details(self, tokens, actions, state=None):
        details = super().scene_details(tokens, actions, state)
        features = self.motion_clock_features(details["state"], actions)
        candidates = self.motion_candidates(details["current"]["joint"], actions)
        offset = self.motion_clock_head(features)[:, :, None] * candidates * details["mode"].softmax(1)[:, 0, None, None]
        joint = details["next"]["joint"]
        adjusted = joint + offset
        normalizer = adjusted.flatten(1).logsumexp(1) - joint.flatten(1).logsumexp(1)
        details["next"]["joint"] = adjusted - normalizer[:, None, None]
        details["motion_joint_offset"] = offset
        return details


class RewriteModeSceneDynamics(RowTransportSceneDynamics):
    """Refine modes using the network's proposed pixel rewrite."""

    def __init__(self, codebook, object_shapes, object_tiles, preview_patterns):
        super().__init__(codebook, object_shapes, object_tiles, preview_patterns)
        self.rewrite_mode_head = nn.Linear(10, 3)
        nn.init.zeros_(self.rewrite_mode_head.weight)
        nn.init.zeros_(self.rewrite_mode_head.bias)
        self.register_buffer("_rewrite_mode_scene", torch.tensor(True))

    def rewrite_mode_features(self, tokens, scene):
        current_mask, _ = self._hard_object(scene["current"])
        screen = tokens[:, :, 2:12]
        background = screen.masked_fill(current_mask[:, :, 2:12], self.blank_id)
        gate = scene["gate"][:, :, :, 2:12] - self.gate_logit_offset
        log_probs = self.combine_logits(background, scene["write"][:, :, :, 2:12], gate)
        probability = log_probs.exp()
        proposal = log_probs.argmax(1)
        hard_screen = (proposal != screen).to(probability.dtype)
        hard_background = (proposal != background).to(probability.dtype)
        soft_screen = 1 - probability.gather(1, screen[:, None])[:, 0]
        soft_background = 1 - probability.gather(1, background[:, None])[:, 0]
        confidence = probability.amax(1)
        entropy = -(probability * log_probs.clamp_min(-100)).sum(1)
        per_row = soft_background.mean(2)
        change = gate.sigmoid()
        return torch.stack((hard_screen.mean((1, 2)), hard_background.mean((1, 2)),
                            soft_screen.mean((1, 2)), soft_background.mean((1, 2)),
                            confidence.mean((1, 2)), entropy.mean((1, 2)), per_row.amax(1),
                            per_row.std(1, correction=0), change.mean((1, 2, 3)),
                            change.amax((1, 2, 3))), dim=1)

    def _scene_mode(self, tokens, actions, pooled, scene):
        details = super()._scene_mode(tokens, actions, pooled, scene)
        features = self.rewrite_mode_features(tokens, scene)
        base = details["mode"]
        details.update(mode=base + self.rewrite_mode_head(features),
                       rewrite_mode_base_logits=base, rewrite_mode_features=features)
        return details


class CurrentReadoutSceneDynamics(RowTransportSceneDynamics):
    """Score a current-object candidate using its observed footprint appearance."""

    def __init__(self, codebook, object_shapes, object_tiles, preview_patterns):
        super().__init__(codebook, object_shapes, object_tiles, preview_patterns)
        self.current_readout_head = nn.Sequential(nn.Linear(3, 8), nn.SiLU(), nn.Linear(8, 1, bias=False))
        nn.init.zeros_(self.current_readout_head[-1].weight)
        self.register_buffer("_current_readout_scene", torch.tensor(True))
        self.register_buffer("current_readout_cells", self.shape_masks.flatten(2).topk(4, dim=-1).indices,
                             persistent=False)

    def current_readout_lookup(self):
        embedding = F.normalize(self.embedding.weight, dim=1)
        return embedding[self.object_tiles] @ embedding.T

    def current_readout_features(self, board, lookup):
        cells = board.flatten(1)[:, self.current_readout_cells]
        kinds = torch.arange(len(self.object_tiles), device=board.device)[None, :, None, None]
        texture = (cells == self.object_tiles[None, :, None, None]).to(lookup.dtype).mean(3)
        similarity = lookup[kinds, cells]
        return torch.stack((texture, similarity.mean(3), similarity.amin(3)), dim=3)

    def correct_current_joint(self, joint, board, lookup):
        features = self.current_readout_features(board, lookup)
        adjusted = joint + self.current_readout_head(features)[:, :, :, 0]
        normalizer = adjusted.flatten(1).logsumexp(1) - joint.flatten(1).logsumexp(1)
        return adjusted - normalizer[:, None, None]

    def _current_details(self, tokens, hidden):
        current, pool = super()._current_details(tokens, hidden)
        current["joint"] = self.correct_current_joint(current["joint"], tokens[:, :, 2:12],
                                                      self.current_readout_lookup())
        return current, pool


class _RowPreserveHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.rows = nn.Conv1d(18, 8, 3, padding=1)
        self.readout = nn.Conv1d(8, 1, 1)
        nn.init.zeros_(self.readout.weight)
        nn.init.constant_(self.readout.bias, -4.)

    def forward(self, features):
        return self.readout(F.silu(self.rows(features.transpose(1, 2))))[:, 0]


class RowPreserveSceneDynamics(RowTransportSceneDynamics):
    """Choose per row whether to retain the visible background or rewrite it."""

    def __init__(self, codebook, object_shapes, object_tiles, preview_patterns):
        super().__init__(codebook, object_shapes, object_tiles, preview_patterns)
        self.row_preserve_head = _RowPreserveHead()
        self.register_buffer("_row_preserve_scene", torch.tensor(True))

    def row_preserve_features(self, tokens, details):
        rows = self.row_transport_features(tokens, details)
        change = (details["gate"][:, 0, :, 2:12] - self.gate_logit_offset).sigmoid().mean(2, keepdim=True)
        return torch.cat((rows, change), dim=2)

    def row_preserve_candidates(self, tokens, details):
        return self._hard_rewrite_candidates(tokens, details, details["gate"] - self.gate_logit_offset)

    def scene_details(self, tokens, actions, state=None):
        details = super().scene_details(tokens, actions, state)
        features = self.row_preserve_features(tokens, details)
        details.update(row_preserve_features=features, row_preserve_logits=self.row_preserve_head(features))
        return details

    def _select_rewritten_rows(self, details, background, rewritten, hard):
        logits = details["row_preserve_logits"]
        if hard:
            return torch.where(logits[:, :, None] > 0, background, rewritten)
        preserve = logits.sigmoid()[:, None, :, None]
        return preserve * background + (1 - preserve) * rewritten


SCENE_FAMILIES = {
    "_extended_scene": ExtendedSceneDynamics,
    "_relative_scene": RelativeSceneDynamics,
    "_footprint_scene": FootprintSceneDynamics,
    "_pose_rerank_scene": PoseRerankSceneDynamics,
    "_action_invariant_current_scene": ActionInvariantCurrentSceneDynamics,
    "_learned_overlap_scene": LearnedOverlapSceneDynamics,
    "_spawn_preview_scene": SpawnPreviewSceneDynamics,
    "_motion_clock_scene": MotionClockSceneDynamics,
    "_preview_render_scene": PreviewRenderSceneDynamics,
    "_factorized_preview_render_scene": FactorizedPreviewRenderSceneDynamics,
    "_consistency_scene": ConsistencySceneDynamics,
    "_mode_correction_scene": ModeCorrectionSceneDynamics,
    "_spatial_mode_scene": SpatialModeSceneDynamics,
    "_row_transport_scene": RowTransportSceneDynamics,
    "_contact_mode_scene": ContactModeSceneDynamics,
    "_joint_event_scene": JointEventSceneDynamics,
    "_timed_event_scene": TimedEventSceneDynamics,
    "_canvas_overlap_scene": CanvasOverlapSceneDynamics,
    "_tracked_canvas_scene": TrackedCanvasSceneDynamics,
    "_continuation_scene": ContinuationSceneDynamics,
    "_relative_proposal_scene": RelativeProposalSceneDynamics,
    "_oriented_proposal_scene": OrientedProposalSceneDynamics,
    "_rewrite_evidence_scene": RewriteEvidenceSceneDynamics,
    "_feedback_carry_scene": FeedbackCarrySceneDynamics,
    "_replay_event_scene": ReplayEventSceneDynamics,
    "_clear_routing_scene": ClearRoutingSceneDynamics,
    "_clear_animation_scene": ClearAnimationSceneDynamics,
    "_clear_duration_scene": ClearDurationSceneDynamics,
    "_terminal_scene": TerminalSceneDynamics,
    "_terminal_contact_scene": TerminalContactSceneDynamics,
    "_clear_onset_scene": ClearOnsetSceneDynamics,
    "_compose_contact_scene": ComposeContactSceneDynamics,
    "_counter_scene": CounterSceneDynamics,
    "_row_motion_clock_scene": RowMotionClockSceneDynamics,
    "_rewrite_mode_scene": RewriteModeSceneDynamics,
    "_current_readout_scene": CurrentReadoutSceneDynamics,
    "_row_preserve_scene": RowPreserveSceneDynamics,
}


def scene_from_weights(weights):
    if "model" in weights: weights = weights["model"]
    selected = [family for marker, family in SCENE_FAMILIES.items() if marker in weights]
    if CounterSceneDynamics in selected:
        if ComposeContactSceneDynamics not in selected:
            raise ValueError("counter checkpoint must preserve the compose contact marker")
        selected.remove(ComposeContactSceneDynamics)
    if any(issubclass(family, ComposeContactSceneDynamics) for family in selected):
        if ClearOnsetSceneDynamics not in selected:
            raise ValueError("compose contact checkpoint must preserve the clear onset marker")
        selected.remove(ClearOnsetSceneDynamics)
    if any(issubclass(family, ClearOnsetSceneDynamics) for family in selected):
        if TerminalContactSceneDynamics not in selected:
            raise ValueError("clear onset checkpoint must preserve the terminal contact marker")
        selected.remove(TerminalContactSceneDynamics)
    if any(issubclass(family, TerminalContactSceneDynamics) for family in selected):
        if TerminalSceneDynamics not in selected:
            raise ValueError("terminal contact checkpoint must preserve the terminal marker")
        selected.remove(TerminalSceneDynamics)
    if any(issubclass(family, TerminalSceneDynamics) for family in selected):
        if ClearDurationSceneDynamics not in selected:
            raise ValueError("terminal checkpoint must preserve the clear duration marker")
        selected.remove(ClearDurationSceneDynamics)
    if any(issubclass(family, ClearDurationSceneDynamics) for family in selected):
        if ClearAnimationSceneDynamics not in selected:
            raise ValueError("clear duration checkpoint must preserve the clear animation marker")
        selected.remove(ClearAnimationSceneDynamics)
    if any(issubclass(family, ClearAnimationSceneDynamics) for family in selected):
        if ClearRoutingSceneDynamics not in selected:
            raise ValueError("clear animation checkpoint must preserve the clear routing marker")
        selected.remove(ClearRoutingSceneDynamics)
    if any(issubclass(family, ClearRoutingSceneDynamics) for family in selected):
        if ReplayEventSceneDynamics not in selected:
            raise ValueError("clear routing checkpoint must preserve the replay event marker")
        selected.remove(ReplayEventSceneDynamics)
    if any(issubclass(family, ReplayEventSceneDynamics) for family in selected):
        if FeedbackCarrySceneDynamics not in selected:
            raise ValueError("replay event checkpoint must preserve the feedback carry marker")
        selected.remove(FeedbackCarrySceneDynamics)
    if any(issubclass(family, FeedbackCarrySceneDynamics) for family in selected):
        if RewriteEvidenceSceneDynamics not in selected:
            raise ValueError("feedback carry checkpoint must preserve the rewrite evidence marker")
        selected.remove(RewriteEvidenceSceneDynamics)
    if any(issubclass(family, RewriteEvidenceSceneDynamics) for family in selected):
        if OrientedProposalSceneDynamics not in selected:
            raise ValueError("rewrite evidence checkpoint must preserve the oriented proposal marker")
        selected.remove(OrientedProposalSceneDynamics)
    if any(issubclass(family, OrientedProposalSceneDynamics) for family in selected):
        if RelativeProposalSceneDynamics not in selected:
            raise ValueError("oriented proposal checkpoint must preserve the relative proposal marker")
        selected.remove(RelativeProposalSceneDynamics)
    if any(issubclass(family, RelativeProposalSceneDynamics) for family in selected):
        if ContinuationSceneDynamics not in selected:
            raise ValueError("relative proposal checkpoint must preserve the continuation marker")
        selected.remove(ContinuationSceneDynamics)
    if any(issubclass(family, ContinuationSceneDynamics) for family in selected):
        if TrackedCanvasSceneDynamics not in selected:
            raise ValueError("continuation checkpoint must preserve the tracked canvas marker")
        selected.remove(TrackedCanvasSceneDynamics)
    if any(issubclass(family, TrackedCanvasSceneDynamics) for family in selected):
        if CanvasOverlapSceneDynamics not in selected:
            raise ValueError("tracked canvas checkpoint must preserve the canvas overlap marker")
        selected.remove(CanvasOverlapSceneDynamics)
    if any(issubclass(family, CanvasOverlapSceneDynamics) for family in selected):
        if TimedEventSceneDynamics not in selected:
            raise ValueError("canvas overlap checkpoint must preserve the timed event marker")
        selected.remove(TimedEventSceneDynamics)
    if any(issubclass(family, TimedEventSceneDynamics) for family in selected):
        if JointEventSceneDynamics not in selected:
            raise ValueError("timed event checkpoint must preserve the joint event marker")
        selected.remove(JointEventSceneDynamics)
    if any(issubclass(family, JointEventSceneDynamics) for family in selected):
        if ContactModeSceneDynamics not in selected:
            raise ValueError("joint event checkpoint must preserve the contact mode marker")
        selected.remove(ContactModeSceneDynamics)
    if any(issubclass(family, (ContactModeSceneDynamics, RowMotionClockSceneDynamics, RewriteModeSceneDynamics,
                               CurrentReadoutSceneDynamics, RowPreserveSceneDynamics)) for family in selected):
        if RowTransportSceneDynamics not in selected:
            raise ValueError("derived scene checkpoint must preserve the row transport marker")
        selected.remove(RowTransportSceneDynamics)
    if any(issubclass(family, (SpatialModeSceneDynamics, RowTransportSceneDynamics)) for family in selected):
        if ModeCorrectionSceneDynamics not in selected:
            raise ValueError("spatial mode checkpoint must preserve the mode correction marker")
        selected.remove(ModeCorrectionSceneDynamics)
    if any(issubclass(family, ModeCorrectionSceneDynamics) for family in selected):
        if ConsistencySceneDynamics not in selected:
            raise ValueError("mode correction checkpoint must preserve the consistency marker")
        selected.remove(ConsistencySceneDynamics)
    if any(issubclass(family, ConsistencySceneDynamics) for family in selected):
        if FactorizedPreviewRenderSceneDynamics not in selected:
            raise ValueError("consistency checkpoint must preserve the factorized preview marker")
        selected.remove(FactorizedPreviewRenderSceneDynamics)
    if any(issubclass(family, FactorizedPreviewRenderSceneDynamics) for family in selected):
        if PreviewRenderSceneDynamics not in selected:
            raise ValueError("factorized checkpoint must preserve the preview renderer marker")
        selected.remove(PreviewRenderSceneDynamics)
    if any(issubclass(family, (MotionClockSceneDynamics, PreviewRenderSceneDynamics)) for family in selected):
        if SpawnPreviewSceneDynamics not in selected or LearnedOverlapSceneDynamics not in selected:
            raise ValueError("derived scene checkpoint must preserve the preview and overlap markers")
        selected.remove(SpawnPreviewSceneDynamics)
        selected.remove(LearnedOverlapSceneDynamics)
    if SpawnPreviewSceneDynamics in selected:
        if LearnedOverlapSceneDynamics not in selected:
            raise ValueError("spawn preview checkpoint must preserve the learned overlap marker")
        selected.remove(LearnedOverlapSceneDynamics)
    if len(selected) > 1:
        raise ValueError("incompatible scene architecture markers")
    family = selected[0] if selected else ObjectSceneDynamics
    if issubclass(family, LearnedOverlapSceneDynamics):
        scale = weights.get("overlap_scale", torch.tensor(1.0))
        if (not isinstance(scale, torch.Tensor) or scale.ndim != 0 or scale.is_complex()
                or scale.dtype == torch.bool or not torch.isfinite(scale.to(dtype=torch.float32)) or scale < 0):
            raise ValueError("overlap_scale must be a finite nonnegative scalar tensor")
        if "overlap_scale" not in weights:
            weights = {**weights, "overlap_scale": scale}
    with torch.random.fork_rng(devices=[]):
        arguments = (weights["codebook"], weights["object_shapes"], weights["object_tiles"])
        if issubclass(family, PreviewRenderSceneDynamics):
            if "preview_patterns" not in weights:
                raise ValueError("preview renderer checkpoint is missing its pattern dictionary")
            arguments += (weights["preview_patterns"],)
        if issubclass(family, ClearAnimationSceneDynamics):
            arguments += (int(weights["animation_flash_id"]),)
        if issubclass(family, TerminalSceneDynamics):
            arguments += (int(weights["terminal_fill_id"]), weights["terminal_panels"].long())
        if family is CounterSceneDynamics:
            coefficients = weights.get("scene_continuity")
            if coefficients is not None and (coefficients.ndim != 1 or coefficients.numel() not in (2, 3)
                                             or not torch.isfinite(coefficients).all()):
                raise ValueError("invalid fitted scene continuity coefficients")
            arguments += (weights["counter.glyphs"], weights["counter.legacy_ids"], weights["counter.addends"],
                          "counter.delay.weight" in weights, 0 if coefficients is None else len(coefficients),
                          (weights["a_rotation_target"], weights["a_rotation_odds"])
                          if "a_rotation_target" in weights else None, "_a_proposal_gate" in weights,
                          (weights["a_column_offset"], weights["a_column_odds"])
                          if "a_column_offset" in weights else None,
                          (weights["pause_panel"], weights["pause_transition"], weights["pause_resume_action"])
                          if "pause_transition" in weights else None)
        model = family(*arguments)
    if family is RelativeProposalSceneDynamics and "proposal_gate_calibration" not in weights:
        weights = {**weights, "proposal_gate_calibration": model.proposal_gate_calibration.detach()}
    if family is OrientedProposalSceneDynamics and "oriented_command_gate.weight" not in weights:
        weights = {**weights, "oriented_command_gate.weight": model.oriented_command_gate.weight.detach()}
    if family is OrientedProposalSceneDynamics and "motion_agreement_gate.weight" not in weights:
        weights = {**weights, "motion_agreement_gate.weight": model.motion_agreement_gate.weight.detach()}
    model.load_state_dict(weights, strict=True)
    return model


def expand_scene_weights(weights):
    if "_object_scene" not in weights or any(marker in weights for marker in SCENE_FAMILIES):
        raise ValueError("memory expansion requires a legacy scene checkpoint")
    model = ExtendedSceneDynamics(weights["codebook"], weights["object_shapes"], weights["object_tiles"])
    expanded = model.state_dict()
    for name, value in weights.items():
        if name not in expanded or value.shape != expanded[name].shape:
            raise ValueError(f"invalid legacy scene tensor: {name}")
        expanded[name] = value.clone()
    model.load_state_dict(expanded, strict=True)
    return model


def relative_scene_weights(weights):
    if "_object_scene" not in weights or any(marker in weights for marker in SCENE_FAMILIES):
        raise ValueError("relative motion requires a legacy scene checkpoint")
    model = RelativeSceneDynamics(weights["codebook"], weights["object_shapes"], weights["object_tiles"])
    upgraded = model.state_dict()
    for name, value in weights.items():
        if name not in upgraded or value.shape != upgraded[name].shape: raise ValueError(f"invalid legacy tensor: {name}")
        upgraded[name] = value.clone()
    model.load_state_dict(upgraded, strict=True)
    return model


def footprint_scene_weights(weights):
    if "_object_scene" not in weights or any(marker in weights for marker in SCENE_FAMILIES):
        raise ValueError("footprint migration requires a legacy scene checkpoint")
    model = FootprintSceneDynamics(weights["codebook"], weights["object_shapes"], weights["object_tiles"])
    migrated = model.state_dict()
    additions = {"footprint_score.weight", "footprint_score.bias", "_footprint_scene"}
    if set(weights) != set(migrated) - additions:
        raise ValueError("legacy scene checkpoint has unexpected or missing tensors")
    migrated.update(weights)
    model.load_state_dict(migrated, strict=True)
    return model


def pose_rerank_scene_weights(weights):
    if "_object_scene" not in weights or any(marker in weights for marker in SCENE_FAMILIES):
        raise ValueError("pose rerank migration requires a legacy scene checkpoint")
    model = PoseRerankSceneDynamics(weights["codebook"], weights["object_shapes"], weights["object_tiles"])
    migrated = model.state_dict()
    additions = {key for key in migrated if key.startswith("pose_reranker.") or key == "_pose_rerank_scene"}
    if set(weights) != set(migrated) - additions:
        raise ValueError("legacy scene checkpoint has unexpected or missing tensors")
    migrated.update(weights)
    model.load_state_dict(migrated, strict=True)
    return model


def action_invariant_current_scene_weights(weights):
    if "_object_scene" not in weights or any(marker in weights for marker in SCENE_FAMILIES):
        raise ValueError("action-invariant current migration requires a legacy scene checkpoint")
    marker = torch.tensor(True, device=weights["_object_scene"].device)
    return scene_from_weights({**weights, "_action_invariant_current_scene": marker})


def learned_overlap_scene_weights(weights):
    if "_object_scene" not in weights or any(marker in weights for marker in SCENE_FAMILIES):
        raise ValueError("learned overlap migration requires a legacy scene checkpoint")
    model = LearnedOverlapSceneDynamics(weights["codebook"], weights["object_shapes"], weights["object_tiles"])
    migrated = model.state_dict()
    additions = {key for key in migrated if key.startswith("overlap_head.") or key in ("_learned_overlap_scene", "overlap_scale")}
    if set(weights) != set(migrated) - additions:
        raise ValueError("legacy scene checkpoint has unexpected or missing tensors")
    migrated.update(weights)
    model.load_state_dict(migrated, strict=True)
    return model


def spawn_preview_scene_weights(weights):
    markers = {marker for marker in SCENE_FAMILIES if marker in weights}
    if "_object_scene" not in weights or markers != {"_learned_overlap_scene"}:
        raise ValueError("spawn preview migration requires a learned overlap checkpoint")
    base = scene_from_weights(weights)
    model = SpawnPreviewSceneDynamics(base.codebook, base.object_shapes, base.object_tiles)
    migrated = model.state_dict()
    migrated.update(base.state_dict())
    model.load_state_dict(migrated, strict=True)
    return model


def motion_clock_scene_weights(weights):
    markers = {marker for marker in SCENE_FAMILIES if marker in weights}
    if "_object_scene" not in weights or markers != {"_learned_overlap_scene", "_spawn_preview_scene"}:
        raise ValueError("motion clock migration requires a spawn preview checkpoint")
    base = scene_from_weights(weights)
    model = MotionClockSceneDynamics(base.codebook, base.object_shapes, base.object_tiles)
    migrated = model.state_dict()
    migrated.update(base.state_dict())
    model.load_state_dict(migrated, strict=True)
    return model


def preview_render_scene_weights(weights, patterns, factorized=False):
    markers = {marker for marker in SCENE_FAMILIES if marker in weights}
    if "_object_scene" not in weights or markers != {"_learned_overlap_scene", "_spawn_preview_scene"}:
        raise ValueError("preview renderer migration requires a spawn preview checkpoint")
    base = scene_from_weights(weights)
    family = FactorizedPreviewRenderSceneDynamics if factorized else PreviewRenderSceneDynamics
    model = family(base.codebook, base.object_shapes, base.object_tiles, patterns)
    migrated = model.state_dict()
    migrated.update(base.state_dict())
    model.load_state_dict(migrated, strict=True)
    return model


def consistency_scene_weights(weights):
    markers = {marker for marker in SCENE_FAMILIES if marker in weights}
    expected = {"_learned_overlap_scene", "_spawn_preview_scene", "_preview_render_scene",
                "_factorized_preview_render_scene"}
    if "_object_scene" not in weights or markers != expected:
        raise ValueError("consistency migration requires a factorized preview checkpoint")
    base = scene_from_weights(weights)
    model = ConsistencySceneDynamics(base.codebook, base.object_shapes, base.object_tiles, base.preview_patterns)
    migrated = model.state_dict()
    migrated.update(base.state_dict())
    model.load_state_dict(migrated, strict=True)
    return model


def mode_correction_scene_weights(weights):
    markers = {marker for marker in SCENE_FAMILIES if marker in weights}
    expected = {"_learned_overlap_scene", "_spawn_preview_scene", "_preview_render_scene",
                "_factorized_preview_render_scene", "_consistency_scene"}
    if "_object_scene" not in weights or markers != expected:
        raise ValueError("mode correction migration requires a consistency checkpoint")
    base = scene_from_weights(weights)
    model = ModeCorrectionSceneDynamics(base.codebook, base.object_shapes, base.object_tiles, base.preview_patterns)
    migrated = model.state_dict()
    migrated.update(base.state_dict())
    model.load_state_dict(migrated, strict=True)
    return model


def spatial_mode_scene_weights(weights):
    markers = {marker for marker in SCENE_FAMILIES if marker in weights}
    expected = {"_learned_overlap_scene", "_spawn_preview_scene", "_preview_render_scene",
                "_factorized_preview_render_scene", "_consistency_scene", "_mode_correction_scene"}
    if "_object_scene" not in weights or markers != expected:
        raise ValueError("spatial mode migration requires a mode correction checkpoint")
    base = scene_from_weights(weights)
    model = SpatialModeSceneDynamics(base.codebook, base.object_shapes, base.object_tiles, base.preview_patterns)
    migrated = model.state_dict()
    migrated.update(base.state_dict())
    model.load_state_dict(migrated, strict=True)
    return model


def row_transport_scene_weights(weights):
    markers = {marker for marker in SCENE_FAMILIES if marker in weights}
    expected = {"_learned_overlap_scene", "_spawn_preview_scene", "_preview_render_scene",
                "_factorized_preview_render_scene", "_consistency_scene", "_mode_correction_scene"}
    if "_object_scene" not in weights or markers != expected:
        raise ValueError("row transport migration requires a mode correction checkpoint")
    base = scene_from_weights(weights)
    model = RowTransportSceneDynamics(base.codebook, base.object_shapes, base.object_tiles, base.preview_patterns)
    migrated = model.state_dict()
    migrated.update(base.state_dict())
    model.load_state_dict(migrated, strict=True)
    return model


def contact_mode_scene_weights(weights):
    base = scene_from_weights(weights)
    if type(base) is not RowTransportSceneDynamics:
        raise ValueError("contact mode requires a row transport checkpoint")
    model = ContactModeSceneDynamics(base.codebook, base.object_shapes, base.object_tiles, base.preview_patterns)
    model.load_state_dict({**model.state_dict(), **base.state_dict()})
    return model


def joint_event_scene_weights(weights):
    base = scene_from_weights(weights)
    if type(base) is not ContactModeSceneDynamics:
        raise ValueError("joint event training requires a contact mode checkpoint")
    model = JointEventSceneDynamics(base.codebook, base.object_shapes, base.object_tiles, base.preview_patterns)
    model.load_state_dict({**model.state_dict(), **base.state_dict()})
    return model


def timed_event_scene_weights(weights):
    base = scene_from_weights(weights)
    if type(base) is not JointEventSceneDynamics:
        raise ValueError("timed event training requires a joint event checkpoint")
    model = TimedEventSceneDynamics(base.codebook, base.object_shapes, base.object_tiles, base.preview_patterns)
    model.load_state_dict({**model.state_dict(), **base.state_dict()})
    return model


def canvas_overlap_scene_weights(weights):
    base = scene_from_weights(weights)
    if type(base) is not TimedEventSceneDynamics:
        raise ValueError("canvas overlap training requires a timed event checkpoint")
    model = CanvasOverlapSceneDynamics(base.codebook, base.object_shapes, base.object_tiles, base.preview_patterns)
    model.load_state_dict({**model.state_dict(), **base.state_dict()})
    return model


def tracked_canvas_scene_weights(weights):
    base = scene_from_weights(weights)
    if type(base) is not CanvasOverlapSceneDynamics:
        raise ValueError("tracked canvas training requires a canvas overlap checkpoint")
    model = TrackedCanvasSceneDynamics(base.codebook, base.object_shapes, base.object_tiles, base.preview_patterns)
    model.load_state_dict({**model.state_dict(), **base.state_dict()})
    return model


def continuation_scene_weights(weights):
    base = scene_from_weights(weights)
    if type(base) is not TrackedCanvasSceneDynamics:
        raise ValueError("continuation training requires a tracked canvas checkpoint")
    model = ContinuationSceneDynamics(base.codebook, base.object_shapes, base.object_tiles, base.preview_patterns)
    model.load_state_dict({**model.state_dict(), **base.state_dict()})
    return model


def relative_proposal_scene_weights(weights):
    base = scene_from_weights(weights)
    if type(base) is not ContinuationSceneDynamics:
        raise ValueError("relative proposal training requires a continuation checkpoint")
    model = RelativeProposalSceneDynamics(base.codebook, base.object_shapes, base.object_tiles, base.preview_patterns)
    model.load_state_dict({**model.state_dict(), **base.state_dict()})
    return model


def oriented_proposal_scene_weights(weights):
    base = scene_from_weights(weights)
    if type(base) is not RelativeProposalSceneDynamics:
        raise ValueError("oriented proposal training requires a relative proposal checkpoint")
    model = OrientedProposalSceneDynamics(base.codebook, base.object_shapes, base.object_tiles, base.preview_patterns)
    model.load_state_dict({**model.state_dict(), **base.state_dict()})
    return model


def rewrite_evidence_scene_weights(weights):
    base = scene_from_weights(weights)
    if type(base) is not OrientedProposalSceneDynamics:
        raise ValueError("rewrite evidence training requires an oriented proposal checkpoint")
    model = RewriteEvidenceSceneDynamics(base.codebook, base.object_shapes, base.object_tiles, base.preview_patterns)
    model.load_state_dict({**model.state_dict(), **base.state_dict()})
    return model


def feedback_carry_scene_weights(weights):
    base = scene_from_weights(weights)
    if type(base) is not RewriteEvidenceSceneDynamics:
        raise ValueError("feedback carry training requires a rewrite evidence checkpoint")
    model = FeedbackCarrySceneDynamics(base.codebook, base.object_shapes, base.object_tiles, base.preview_patterns)
    model.load_state_dict({**model.state_dict(), **base.state_dict()})
    return model


def replay_event_scene_weights(weights):
    base = scene_from_weights(weights)
    if type(base) is not FeedbackCarrySceneDynamics:
        raise ValueError("replay event training requires a feedback carry checkpoint")
    model = ReplayEventSceneDynamics(base.codebook, base.object_shapes, base.object_tiles, base.preview_patterns)
    model.load_state_dict({**model.state_dict(), **base.state_dict()})
    return model


def clear_routing_scene_weights(weights):
    base = scene_from_weights(weights)
    if type(base) is not ReplayEventSceneDynamics:
        raise ValueError("clear routing training requires a replay event checkpoint")
    model = ClearRoutingSceneDynamics(base.codebook, base.object_shapes, base.object_tiles, base.preview_patterns)
    model.load_state_dict({**model.state_dict(), **base.state_dict()})
    return model


def clear_animation_scene_weights(weights, flash_id):
    base = scene_from_weights(weights)
    if type(base) is not ClearRoutingSceneDynamics:
        raise ValueError("clear animation training requires a clear routing checkpoint")
    model = ClearAnimationSceneDynamics(base.codebook, base.object_shapes, base.object_tiles, base.preview_patterns, flash_id)
    model.load_state_dict({**model.state_dict(), **base.state_dict()})
    return model


def clear_duration_scene_weights(weights, flash_id):
    base = scene_from_weights(weights)
    if type(base) is not ClearAnimationSceneDynamics or int(base.animation_flash_id) != flash_id:
        raise ValueError("clear duration training requires the same observed animation tile")
    model = ClearDurationSceneDynamics(base.codebook, base.object_shapes, base.object_tiles, base.preview_patterns, flash_id)
    model.load_state_dict({**model.state_dict(), **base.state_dict()})
    return model


def terminal_scene_weights(weights, fill_id, panels):
    base = scene_from_weights(weights)
    if type(base) is not ClearDurationSceneDynamics:
        raise ValueError("terminal training requires a learned clear duration checkpoint")
    model = TerminalSceneDynamics(base.codebook, base.object_shapes, base.object_tiles, base.preview_patterns,
                                 int(base.animation_flash_id), fill_id, panels)
    model.load_state_dict({**model.state_dict(), **base.state_dict()})
    return model


def terminal_contact_scene_weights(weights):
    base = scene_from_weights(weights)
    if type(base) is not TerminalSceneDynamics:
        raise ValueError("terminal contact training requires a fitted terminal checkpoint")
    model = TerminalContactSceneDynamics(base.codebook, base.object_shapes, base.object_tiles, base.preview_patterns,
                                        int(base.animation_flash_id), int(base.terminal_fill_id), base.terminal_panels.long())
    model.load_state_dict({**model.state_dict(), **base.state_dict()})
    return model


def clear_onset_scene_weights(weights):
    base = scene_from_weights(weights)
    if type(base) is not TerminalContactSceneDynamics:
        raise ValueError("clear onset training requires a terminal contact checkpoint")
    model = ClearOnsetSceneDynamics(base.codebook, base.object_shapes, base.object_tiles, base.preview_patterns,
                                    int(base.animation_flash_id), int(base.terminal_fill_id), base.terminal_panels.long())
    model.load_state_dict({**model.state_dict(), **base.state_dict()})
    return model


def compose_contact_scene_weights(weights):
    base = scene_from_weights(weights)
    if type(base) is not ClearOnsetSceneDynamics:
        raise ValueError("compose contact training requires a fitted clear onset checkpoint")
    model = ComposeContactSceneDynamics(base.codebook, base.object_shapes, base.object_tiles, base.preview_patterns,
                                       int(base.animation_flash_id), int(base.terminal_fill_id), base.terminal_panels.long())
    model.load_state_dict({**model.state_dict(), **base.state_dict()})
    return model


def row_motion_clock_scene_weights(weights):
    base = scene_from_weights(weights)
    if type(base) is not RowTransportSceneDynamics:
        raise ValueError("row motion clock requires a row transport checkpoint")
    model = RowMotionClockSceneDynamics(base.codebook, base.object_shapes, base.object_tiles, base.preview_patterns)
    model.load_state_dict({**model.state_dict(), **base.state_dict()})
    return model


def rewrite_mode_scene_weights(weights):
    markers = {marker for marker in SCENE_FAMILIES if marker in weights}
    expected = {"_learned_overlap_scene", "_spawn_preview_scene", "_preview_render_scene",
                "_factorized_preview_render_scene", "_consistency_scene", "_mode_correction_scene",
                "_row_transport_scene"}
    if "_object_scene" not in weights or markers != expected:
        raise ValueError("rewrite mode migration requires a row transport checkpoint")
    base = scene_from_weights(weights)
    model = RewriteModeSceneDynamics(base.codebook, base.object_shapes, base.object_tiles, base.preview_patterns)
    migrated = model.state_dict()
    migrated.update(base.state_dict())
    model.load_state_dict(migrated, strict=True)
    return model


def current_readout_scene_weights(weights):
    markers = {marker for marker in SCENE_FAMILIES if marker in weights}
    expected = {"_learned_overlap_scene", "_spawn_preview_scene", "_preview_render_scene",
                "_factorized_preview_render_scene", "_consistency_scene", "_mode_correction_scene",
                "_row_transport_scene"}
    if "_object_scene" not in weights or markers != expected:
        raise ValueError("current readout migration requires a row transport checkpoint")
    base = scene_from_weights(weights)
    model = CurrentReadoutSceneDynamics(base.codebook, base.object_shapes, base.object_tiles, base.preview_patterns)
    migrated = model.state_dict()
    migrated.update(base.state_dict())
    model.load_state_dict(migrated, strict=True)
    return model


def row_preserve_scene_weights(weights):
    markers = {marker for marker in SCENE_FAMILIES if marker in weights}
    expected = {"_learned_overlap_scene", "_spawn_preview_scene", "_preview_render_scene",
                "_factorized_preview_render_scene", "_consistency_scene", "_mode_correction_scene",
                "_row_transport_scene"}
    if "_object_scene" not in weights or markers != expected:
        raise ValueError("row preserve migration requires a row transport checkpoint")
    base = scene_from_weights(weights)
    model = RowPreserveSceneDynamics(base.codebook, base.object_shapes, base.object_tiles, base.preview_patterns)
    migrated = model.state_dict()
    migrated.update(base.state_dict())
    model.load_state_dict(migrated, strict=True)
    return model


def expand_scene_state(state):
    if not state.is_floating_point() or state.ndim != 4 or tuple(state.shape[1:]) != (20, 18, 20):
        raise ValueError("memory migration requires floating[B,20,18,20]")
    return torch.cat((state.clone(), state.new_zeros(len(state), 16, 18, 20)), 1)


def upgrade_scene_weights(old_observed_flat, shapes, tiles):
    """Keep every learned legacy tensor, initializing only the graphic object heads."""
    if "_observed_history" not in old_observed_flat or "_object_scene" in old_observed_flat:
        raise ValueError("upgrade requires flat observed-history model weights")
    reference = old_observed_flat["embedding.weight"]
    model = ObjectSceneDynamics(old_observed_flat["codebook"], shapes, tiles).to(device=reference.device, dtype=reference.dtype)
    upgraded = model.state_dict()
    for name, value in old_observed_flat.items():
        if name not in upgraded or value.shape != upgraded[name].shape:
            raise ValueError(f"legacy architecture tensor mismatch: {name}")
        upgraded[name] = value.clone()
    return upgraded
