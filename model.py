"""Action-conditioned pixel dynamics; buttons are supplied by the player."""
import torch
from torch import nn
from torch.nn import functional as F

ACTIONS = ("noop", "left", "right", "down", "a", "b", "start", "up")


class NeuralTetris(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 4, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(4, 8, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.gates = nn.Conv2d(40, 16, 3, padding=1)
        self.candidate = nn.Conv2d(40, 8, 3, padding=1)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 4, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(4, 4, 4, stride=2, padding=1),
        )

    def forward(self, frame, action, state=None):
        """Palette indices uint8 [B,1,72,80], int64 [B], state [B,8,18,20].

        Return next-screen logits [B,4,72,80] and new recurrent state.
        """
        if frame.dtype != torch.uint8 or frame.ndim != 4 or tuple(frame.shape[1:]) != (1, 72, 80):
            raise ValueError("frame must be palette indices uint8 [B,1,72,80]")
        if action.dtype != torch.int64 or action.shape != (frame.shape[0],):
            raise ValueError("action must be int64 [B]")
        if torch.any(frame > 3):
            raise ValueError("palette indices must be in 0..3")
        features = self.encoder(frame.float() / 3.0)
        if state is None:
            state = torch.zeros_like(features)
        buttons = F.one_hot(action, len(ACTIONS)).to(features.dtype)[:, :, None, None].expand(-1, -1, 18, 20)
        # Rows/columns communicate immediately for line clears and distant screen changes.
        rows = features.mean(dim=3, keepdim=True).expand_as(features)
        columns = features.mean(dim=2, keepdim=True).expand_as(features)
        inputs = torch.cat((features, rows, columns, buttons), dim=1)
        reset, update = self.gates(torch.cat((inputs, state), dim=1)).sigmoid().chunk(2, dim=1)
        candidate = self.candidate(torch.cat((inputs, reset * state), dim=1)).tanh()
        state = (1 - update) * state + update * candidate
        # Identity skip preserves static pixels while the recurrent decoder learns changes.
        copy_logits = F.one_hot(frame[:, 0].long(), 4).permute(0, 3, 1, 2).float() * 2.0
        return self.decoder(state) + copy_logits, state
