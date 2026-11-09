"""Verify a real Metal checkpoint can continue through one optimizer step on CPU."""
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np
import torch
from token_model import CalibratedEventDynamics, encode_frames
from train_stateful import calibrated_loss, checkpoint_baseline


def main():
    started = time.monotonic()
    torch.set_num_threads(1)
    path = Path("weights/stateful-last-checkpoint.pt")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    assert checkpoint["config"]["device"] == "mps" and checkpoint["step"] == 8501
    manifest = json.loads(Path("data/manifest.json").read_text())
    entries = [e for e in manifest["episodes"] if e["split"] == "train"]
    model = CalibratedEventDynamics(checkpoint["model"]["codebook"])
    model.load_state_dict(checkpoint["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    optimizer.load_state_dict(checkpoint["optimizer"])
    torch.set_rng_state(checkpoint["torch_rng"])
    for slot, (episode_index, cursor) in enumerate(checkpoint["stream_cursors"]):
        with np.load(Path("data") / entries[episode_index]["file"], allow_pickle=False) as data:
            if len(data["actions"]) - cursor < 32:
                continue
            frames, unknown = encode_frames(data["frames"][cursor:cursor + 33], model.codebook.numpy())
            buttons = data["actions"][cursor:cursor + 32].copy()
        assert unknown == 0
        break
    else:
        raise RuntimeError("no full current window available for cross-backend check")
    frames = torch.from_numpy(frames)[None]
    buttons = torch.from_numpy(buttons).long()[None]
    state = checkpoint["stream_state"][slot:slot + 1].clone()
    optimizer.zero_grad(set_to_none=True)
    total = torch.zeros(())
    for t in range(32):
        write, gate, state = model.forward_details(frames[:, t], buttons[:, t], state)
        total = total + calibrated_loss(write, gate, frames[:, t + 1], frames[:, t], model.codebook, torch.tensor([True]))
    loss = total / 32
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
    optimizer.step()
    assert torch.isfinite(loss) and torch.isfinite(state).all()
    assert all(p.device.type == "cpu" and p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    result = {"status": "compatible", "source_checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "source_step": checkpoint["step"], "backend": "cpu", "torch": str(torch.__version__),
              "loss": float(loss.detach()), "elapsed_seconds": time.monotonic() - started,
              "selection_baseline": checkpoint_baseline(checkpoint), "state_shape": list(state.shape),
              "peak_rss_kib_linux": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    Path("cross-backend.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
