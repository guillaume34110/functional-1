"""Bounded compatibility/timing check for the existing local Metal installation."""
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from token_model import CalibratedEventDynamics, encode_frames
from train_stateful import calibrated_loss


def main():
    output = Path(os.environ["TASK_WORKSPACE"])
    if not torch.backends.mps.is_available():
        raise RuntimeError("Metal unavailable; no implicit CPU fallback")
    torch.set_num_threads(2)
    checkpoint = torch.load("weights/events-last-checkpoint.pt", map_location="cpu", weights_only=True)
    model = CalibratedEventDynamics(checkpoint["model"]["codebook"])
    missing = model.load_state_dict(checkpoint["model"], strict=False)
    assert set(missing.missing_keys) == {"gate_logit_offset"} and not missing.unexpected_keys
    manifest = json.loads(Path("data/manifest.json").read_text())
    source = next(e for e in manifest["episodes"] if e["split"] == "train")
    with np.load(Path("data") / source["file"], allow_pickle=False) as data:
        encoded, missing = encode_frames(data["frames"][:65], model.codebook.numpy())
        assert not missing
        actions = torch.from_numpy(data["actions"][:64].copy()).long()[None].expand(4, -1).to("mps")
    frames = torch.from_numpy(encoded)[None].expand(4, -1, -1, -1).to("mps")
    model.to("mps").train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    optimizer.load_state_dict(checkpoint["optimizer"])
    times = []
    for iteration in range(3):
        torch.mps.synchronize()
        started = time.perf_counter()
        state = None
        with torch.no_grad():
            for t in range(32):
                _, state = model(frames[:, t], actions[:, t], state)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.zeros((), device="mps")
        for t in range(32, 64):
            write, gate, state = model.forward_details(frames[:, t], actions[:, t], state)
            loss = loss + calibrated_loss(write, gate, frames[:, t + 1], frames[:, t], model.codebook,
                                          torch.ones(4, dtype=torch.bool, device="mps"))
        loss = loss / 32
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
        optimizer.step()
        torch.mps.synchronize()
        assert torch.isfinite(loss)
        times.append(time.perf_counter() - started)
    result = {"device": "mps", "torch": str(torch.__version__), "batch": 4, "unroll": 32,
              "warmup": 32, "update_seconds": times, "allocated_bytes": torch.mps.current_allocated_memory(),
              "driver_bytes": torch.mps.driver_allocated_memory(), "last_loss": float(loss.detach().cpu())}
    (output / "mps-benchmark.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
