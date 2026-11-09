# Functional 2 reference baseline

Run the offline benchmark from the repository root.

```sh
PYTHONPATH=. python checks/reference_game.py output/reference-bench
```

The benchmark verifies the Adjustris ROM hash and matches the first screen of the Functional 1 release. It captures the native 160×144 title screen and gameplay, replays the same buttons through the checkpoint, and saves aligned screenshots, arrays and metrics. The pixel-driven test player also plays an adaptive game against the model. This code is a test driver. It is outside the application runtime.

| Scenario | Reference | Functional 1 |
| --- | --- | --- |
| No input, 256 steps | First descent at step 16, second at step 32 | Board frozen for all 256 steps |
| Held left and right with both rotation buttons | Native reference captures include wall-adjacent and partially visible spawn poses | First board divergence at step 5 |
| Mixed controls, 1,024 steps | Eight lines cleared with the adaptive reference driver | Zero lines with those same buttons, one line with the driver reacting to the generated screen |

Nearest-neighbor round-tripping the reference through 80×72 changes about 22% of the native HUD pixels in all three cases. The initial frame already loses small text detail. Later generated frames also change panel borders and preview details. The native output target can address sampling loss. Timing, pose continuity and screen transitions need their own targets.

The next qualification should include the title-to-play transition, settings navigation, idle descent through a supported lock, both rotations against both walls, line clearing, terminal display and restart. Keep these benchmark replays separate from training. Report adaptive play separately from exact command replays, since control timing affects the resulting stack.

## Idle-gravity candidate

`train_idle_gravity.py` fits a transition threshold from the observed first-piece poses in the gravity captures. The selected age is 15, which moves the piece on the 16th step. It classified 4,896 training transitions and 1,664 validation transitions without error. The released Functional 1 tensors are the starting checkpoint.

In the independent 256-step idle benchmark, the candidate reproduces every board, including the starting screen. It reaches its first lock on step 272. All 610 frames of the held-Down replay match Functional 1. The candidate still needs wall-rotation and mixed-input qualification before release.

## Menu and native display candidate

```sh
python play.py --functional-2 output/v2-menu-candidate --port 18787
```

The candidate starts at the captured 160×144 title. The title readout learns the A and START responses from ROM interventions. The existing neural pause handles in-game START, and the model's terminal sequence returns to the title once the learned display shows the title logo. Chrome exercised the title, keyboard entry, native canvas, restart and a held-Down game through top-out and back to play. This candidate uses the first piece set. The ROM's other sets and piece editor are outside the current candidate.

The wall captures provide 864 fully visible training rotations near the sides. A key formed from texture and orientation, action, column and a coarse height bucket has an unambiguous outcome for 66 of 74 observed contexts. Those keys cover 170 of 252 held-out rotations and match all 170 observed outcomes. The ambiguous and unseen keys need additional evidence before a general wall correction can be qualified.

## Wall and opening candidates

With an observed prefix, the fitted wall readout increased exact rotation frames from 115 to 193 out of 252. The wall-only rollout drifted earlier than the gravity candidate on the fixed sequence. Adding the learned lateral repeat cadence and retaining the gravity clock across horizontal input reproduced the first 24 steps of that sequence exactly. The idle comparison still matched all 257 board images, including the seed.

The first remaining divergence is A at the left wall on step 25. The same visible pose admits two pivot offsets in the training data. All three first-rotation examples for that pose agree with the reference outcome, while later rotations include a second offset. The table currently drops that ambiguous key. A causal distinction between first and later turns is needed to qualify this case.

The fixed open-loop wall replay has 34 exact board images with the opening readout, up from 27 in the gravity candidate. The adaptive player cleared zero lines with this opening candidate, versus one with the gravity candidate. The tested menu bundle remains the launch candidate while the experimental wall and opening checkpoints are evaluated.

## Playable stack-preserving candidate

```sh
python play.py --functional-2 output/v2-stack-candidate --port 18787
```

The first-turn readout preserves a causal bit for the first attempted rotation of the opening piece. It corrects the left-wall rotation at step 25. The fixed wall replay matches the reference through step 32 after accounting for the gravity step immediately following a rotation.

The stack guard rejects ordinary-board rewrites without a full row and deposits without visible support. It selects a legal proposal from the model scores. This prevents the reproducible corruption at step 96. The adaptive player now completes a game in 693 steps with eight clear events confirmed by visible row compaction. The longest static board before the title is 18 frames. One of two separate reference clear windows was reproduced. The other remains a timing or rendering discrepancy.

The 186-test Linux suite passes. Chrome exercises the candidate at 160×144, including the title, keyboard start, stable pause, reset, top-out and restart. Open-loop wall timing still diverges from the reference, and gravity under a continuously held direction beyond the opening piece needs independent qualification.

The sustained-input benchmark reproduces the freeze in the stack candidate. A later experimental checkpoint matches the reference under both held directions through step 271 and reaches a supported lock at step 272. Its right-input replay continues across three deposits, and the adaptive game still clears eight lines. Held left exposes a separate spawn fault after the first lock. The rendered piece stack gains four cells repeatedly and returns to the title at step 316, while the reference remains in play. This checkpoint remains experimental.

## Functional 2 release

From the repository root, run the local release and open [the game](http://127.0.0.1:18787/).

```sh
python play.py --functional-2 releases/functional-2 --port 18787
```

Press Enter or Z from the title. The release keeps the title, pause and return-to-title sequence and displays the generated game at 160×144. The native readout reduced the early HUD reconstruction error from 22.1% to 0.85% on generated screens. The package contains its model, menu readout, native readout, seed and SHA-256 manifest.

- Each 800-step held-direction replay reaches three visible deposits. The first supported lock matches the ROM at step 272. The longest static board is 16 frames in both directions. The learned spawn readout preserves the positions observed in six held-out first deposits, and the repeat readout matches all 48 held-out repeat examples. The model carries the pose it actually drew into the next step.
- All 252 held-out complete-pose wall rotations retain a valid four-cell piece. The exact board matches are 42/66 for left A, 48/68 for left B, 48/48 for right A and 55/70 for right B. Some pivot and timing differences remain.
- The adaptive visual player completes a 638-step game and clears four lines, verified from the compacted board. The fixed no-input replay reproduces all 257 board images through step 256. The full 186-test suite and the Chrome title, pause, top-out and restart test pass on the release code.

For a fresh ROM comparison or long held-input test, use the release checkpoint with the offline drivers.

```sh
PYTHONPATH=. python checks/reference_game.py output/reference-v2 --checkpoint releases/functional-2/model.pt
PYTHONPATH=. python checks/held_wall.py releases/functional-2/model.pt output/held-wall-v2
```

The menu launches the first piece set. Other sets and the Adjustris piece editor are outside this release.
