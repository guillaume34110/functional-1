# Build a playable screen model for another game

The player supplies the controls. The model receives the current screen, one button command, and recurrent memory, then predicts the next screen and memory. The emulator records reference data during development. The shipped browser runs the model locally.

## 1. Define the capture contract

Choose a reference build you can record and redistribute under its license. Pin its checksum, the emulator version, the ordered button list, the number of emulator ticks per command, and the frame format. Keep this contract next to the captures, as in [context.json](../context.json). Pick a resolution that preserves the smallest cues that affect play. If tiny HUD digits alias at that resolution, save native frames for a separate display readout.

Adapt the capture loop in [run.py](../run.py) and the play profiles in [collect.py](../collect.py). The current reader expects an NPZ containing `frames`, `actions`, and JSON `metadata` with the matching context. For `T` actions, save `T + 1` frames. The action at index `t` must take frame `t` to frame `t + 1`. The example uses unsigned 8-bit 80×72 frames with four palette indices. A different raster, action set, or cadence also changes the model, input validation, and browser tensor shapes.

## 2. Record and split complete episodes

Start independent runs with different seeds or boot states. Capture normal play and the cases a player is likely to expose, including long holds, quick reversals, contact with boundaries, idle timing, pause, restart, and the end screen. Save the opening frame, the input log, the seed, and a file digest. Split by independent run before making windows. Keep branches from the same source run in the same split, and reserve a test split for the final assessment.

These commands show the capture and baseline training steps for the bundled reference game:

```sh
python -m pip install -r requirements.txt
python collect.py --output output/reference-corpus --episodes 64 --steps 1400 \
  --profile heuristic --profile exploration --profile topout --profile pause --profile sustained
python train.py --data output/reference-corpus --output output/pixel-baseline
```

Use fresh output paths. Before a full training run, check that the reader can replay one episode, that the labels align with visible actions, and that every split contains the events and sequence lengths used in evaluation.

## 3. Train and fix failures with paired examples

Start with the compact recurrent baseline in [model.py](../model.py). Warm its state on a real prefix, then feed its own predictions back during evaluation. Track both changed pixels and meaningful game events. A model can score well on pixels while missing a button press because most of the screen stayed still.

For an ambiguous failure, replay to the same verified screen and fork a short continuation with each relevant button and with no input. Compare when the branches first diverge, which parts of the screen change, and what should remain stable. Use snapshots to reproduce the branches during capture. Keep emulator state out of model inputs. The [rotation collector](../collect_v2_rotations.py) and [wall rotation fitter](../train_wall_rotation.py) show how to collect and check these examples for Adjustris. Reuse that method for the new game's own interactions.

Change the smallest part that explains the failure. In Adjustris, the learned tile representation helped preserve the stack, and separate readouts improved the piece preview and HUD. Fit those corrections only on training cases, then compare them against the baseline on held-out inputs. For a menu option that changes difficulty, verify the first playable screen and the actual timing it selects.

For a color edition like V3, keep the recurrent model's original screen representation as its input and fit a display readout on recolored training frames. Carry over the episode splits and compare the readout to recolored validation and test frames. The browser can share the exported dynamics and load the color readout as a separate asset. [V3 training and results](../README.md#v3-color-training)

## 4. Verify the browser build and release it

Freeze the chosen checkpoint, a verified opening frame or short observed prefix, the action order, any display readouts, and their checksums. Adapt [browser_export.py](../browser_export.py) and [the browser worker](../web/local-game.mjs) to export and run the new model's initial and recurrent steps. Check that the ONNX outputs agree with the Python model on representative inputs, including the recurrent state and the screens around a menu or pause transition.

Replay held-out runs from the opening to the end screen and play a full game manually. Verify that a fresh game resets state, that the controls work at the chosen cadence, and that the browser continues playing after disconnecting the network once its assets load. The [complete game replay](../checks/v2_games.py) and [browser check](../checks/browser.py) are examples to adapt. Measure throughput on the target device.

Test quick taps that begin and end between model steps as well as held inputs. In V3 the worker queues those taps, and a quick Down tap supplies the two consecutive input samples needed to start a soft drop. Check that the title screen still treats its arrows as single presses.

The [Dockerfile](../Dockerfile) verifies the release checksums, exports the models, and serves static assets. Run the container on port 8765 and check `/health` before opening the game. For releases, use the same tested artifact and promote only the exact revision whose CI passed. The [deployment workflow](../.github/workflows/deploy.yml) advances the branch watched by Coolify after a successful main-branch build.
