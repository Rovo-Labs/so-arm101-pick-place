# gpudad_pickcube_world_reconstructed -- static world

This repository is our clone of the environment behind
[gpudad/so101_pick_cube](https://huggingface.co/datasets/gpudad/so101_pick_cube),
GPU-DAD's SO-101 pick-and-place data set. We rebuilt that scene in MuJoCo, and
used the replica to generate a 1,000-episode pick-and-place dataset of our own:
[DATASET NAME](INSERT LINK WHEN READY). Details below.

## The original

This is a MuJoCo replica of the scene in
[gpudad/so101_pick_cube](https://huggingface.co/datasets/gpudad/so101_pick_cube)
by [gpudad](https://huggingface.co/gpudad). All fidelity numbers below are
measured against frames from that dataset.

![The original scene from all three cameras](figures/original_3views.png)

*Episode 44, frame 16 — from the original
[gpudad/so101_pick_cube](https://huggingface.co/datasets/gpudad/so101_pick_cube) data set*

## The Replicated Scene

![The reconstruction from all three cameras](figures/reconstruction_3views.png)

*The same episode and frame, rendered here, in our replicated world*

### Look at it

One command, no setup:

    uv run run.py

That writes `world.png` -- the three calibrated camera views, side by side --
and opens it. [uv](https://docs.astral.sh/uv/) fetches Python and MuJoCo for
you on the first run, so there is nothing to install first. It takes a few
seconds the first time and is instant after that. It needs no display, so it
works the same over SSH, in a container, or in CI.

`world.png` is the picture above. Not "similar to" -- the same pixels:

    uv run run.py --check

    checking against figures/reconstruction_3views.png
      FRONT     identical
      OVERHEAD  identical
      WRIST     identical
    run.py reproduces the published figure exactly (0 of 299568 pixels differ).

`run.py` puts the world in the state recorded at episode 44, frame 16 -- the
arm's recorded joint angles, with the cube and bin at the positions fitted for
this world -- and renders each camera from the scene it was colour-matched to.
There is no keyframe in the XML, so those numbers live at the top of `run.py`
if you want to look at a different state.

| command | what you get |
| --- | --- |
| `uv run run.py` | all three calibrated views -> `world.png` |
| `uv run run.py --check` | proof the render matches the figure above |
| `uv run run.py --camera front` | one view (`front`, `overhead`, `wrist`) |
| `uv run run.py --camera free` | an orbit view of the whole table |
| `uv run run.py --window` | an interactive window -- drag to orbit, scroll to zoom, Tab to cycle cameras |
| `uv run run.py --size 2048` | bigger, for slides |
| `uv run run.py --out shot.png` | write somewhere else |

Run `uv run run.py --help` for the full list. `--window` needs a display and a
graphics driver; every other command runs anywhere.

Prefer to drive it yourself? The scenes are plain MJCF and open in any MuJoCo
viewer:

    python -m mujoco.viewer --mjcf=assets/scene_front.xml

## The dataset

A companion dataset of **1,000 episodes** of the pick-and-place cube task,
recorded entirely in this world. Same task, same scene, same renders in every
episode -- the only thing that varies is what drives the arm.

| episodes | controller |
| --- | --- |
| 100 | GPU-DAD's recorded action data, replayed in our replicated world |
| 400 | a SmolVLA policy trained on GPU-DAD's own data, acting in our replicated world |
| 500 | our inverse-kinematics expert |

[DATASET NAME](INSERT LINK WHEN READY)

## Files

| file | what it is | MAE vs real |
| --- | --- | --- |
| `assets/scene_front.xml` | Colour-matched to GPU-DAD's FRONT camera. | 10.58 |
| `assets/scene_overhead.xml` | Colour-matched to GPU-DAD's OVERHEAD camera. | 4.60 |
| `assets/scene_wrist.xml` | Colour-matched to GPU-DAD's WRIST camera. | 12.87 |

MAE is mean absolute pixel error, 0-255 per channel, against real GPU-DAD frames. Lower is closer.

## How close is our Replicated world to the original?

A policy trained on GPU-DAD's own images scores **25/50** in this world. A policy
trained on this reconstruction scores **25/50**. See [RESULTS.md](RESULTS.md).

## Why there is more than one scene

The three scenes differ in paint and lamps only -- geometry and physics are
identical across all of them. To run a policy, drive ONE scene as the physics truth and mirror its state into the other two, rendering each camera view from the scene that was colour-matched to it.

## How to cite

If you use this world, please cite it:

    Rasheed, A., Gallimore, K., Subbiah, V. and Johnson, E. (2026).
    gpudad_pickcube_world_reconstructed: a MuJoCo reconstruction of the
    GPU-DAD SO-101 pick-cube scene (v1.0.0). Rovo Labs.
    https://github.com/Rovo-Labs/gpu-dad-clone

Machine-readable metadata lives in [CITATION.cff](CITATION.cff) -- GitHub's
"Cite this repository" button reads it directly. Released under Apache-2.0;
see `LICENSE` and `NOTICE`.
