# Functional 1

The [Functional 2 release](docs/functional-2-baseline.md#functional-2-release) adds a native 160×144 display, the title and pause menus, and automatic gravity. The hosted application opens [V2](https://tetris.progsoft.eu/) by default. Select [V1](https://tetris.progsoft.eu/v1) to compare with the original release. Each version keeps its own game state.

Functional 1 is a small action-conditioned game model. You choose the buttons and the model produces the next screen from the current image and recurrent memory. The released checkpoint is the 329,982-byte V85 model from the verified human playthrough.

## Play locally

```sh
python -m pip install -r requirements.txt
python play.py
```

Open [the local game](http://127.0.0.1:8765/). Use the arrow keys to move and soft drop, Z and X to rotate, and Enter for the in-game pause. The application also has a separate pause button that stops requests. The release directory contains the verified checkpoint, the first playable reference screen, and their SHA-256 digests.

![Functional 1 desktop design concept](docs/design/functional-1-desktop.png)

[Mobile design concept](docs/design/functional-1-mobile.png)

## Container deployment

Build the root Dockerfile and route traffic to port 8765. The health endpoint is `/health`. The container verifies both release manifests and serves V2 at `/` and V1 at `/v1` with bounded, in-memory visitor sessions. A restart clears active games.

```sh
docker build -t functional-1 .
docker run --rm -p 8765:8765 functional-1
```

## Training

1. Capture Adjustris v1.1 with PyBoy. Record four Game Boy frames per command and reduce the 160×144 screen to 80×72 with four shades. Each trajectory stores screens and the corresponding button sequence. Divide the recordings by episode before extracting windows.
2. Train the image dynamics from the observed screen, current button and causal history. The early baseline used a convolutional GRU with categorical pixel loss. The released scene model uses a learned tile representation, recurrent visual features and separate predictions for the moving piece, stack, preview and screen transitions.
3. Supplement recorded play with controlled branches from verified screen prefixes. Use these examples to study ambiguous movement, rotation, contact and line clearing. Fit corrections on training examples and check them on separate validation branches.
4. Replay native screenshots to recover small HUD digits that the reduced raster aliases. Learn when a rendered deposit or completed line clear updates the counter, then carry the digits through a learned decimal readout.
5. Preserve the V84 gameplay tensors and fit 24 pause transition coefficients on nine training sequences for V85. Keep the previous screen and recurrent state while paused. Compare the pause panel and the first resumed frame to the reference on five separate validation games.

The ROM supplies training observations. The interactive server loads the release and feeds its own generated screens back into the model. Each browser gets its own recurrent state.

## What worked

- Predicting changes in a compact tile space helped retain the stack over long rollouts. A local proposal learned from actual button interventions corrected rotations and horizontal placement that the whole-board predictor missed.
- Checking animation and counters separately made it possible to distinguish a late visible lock from a wrong score increment. The final comparison found agreement on 51 counter sequences by one step after confirmation.
- The strongest end-to-end result was a human-controlled game from the first piece to the terminal menu. It ran for 2,534 model steps, cleared five lines and reached 540 points. The replay verified the compacted board on all five clears.

Browser validation of the model measured 27.5 ms p95 inference over 256 steps on the development Mac. The full game also exposed one premature lock, and fixed ROM input replays can diverge early because the learned timing differs.

## Reference

Adjustris v1.1 is the CC0 reference game used for captures. [Upstream project and license](https://github.com/tbsp/Adjustris)
