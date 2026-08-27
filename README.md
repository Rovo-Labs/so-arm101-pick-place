# gpudad_pickcube_world_reconstructed -- static world

This repository is our clone of the environment behind
[gpudad/so101_pick_cube](https://huggingface.co/datasets/gpudad/so101_pick_cube),
GPU-DAD's SO-101 pick-and-place data set. We rebuilt that scene in MuJoCo, and
used the replica to generate a 1,000-episode pick-and-place dataset of our own:
[DATASET NAME](INSERT LINK WHEN READY). Details below.

## The original Environment

This is a MuJoCo replica of the scene in
[gpudad/so101_pick_cube](https://huggingface.co/datasets/gpudad/so101_pick_cube)
by [gpudad](https://huggingface.co/gpudad). All fidelity numbers below are
measured against frames from that dataset.

![The original scene from all three cameras](figures/original_3views.png)

*Episode 44, frame 16 — from the original
[gpudad/so101_pick_cube](https://huggingface.co/datasets/gpudad/so101_pick_cube) data set*

## The Replicated Environment

![The reconstruction from all three cameras](figures/reconstruction_3views.png)

*The same episode and frame, rendered here, in our replicated world*

### How to View the Environment

    uv run run.py

Writes `world.png`: renders the scene seen from all three cameras --
front, overhead and wrist -- side by side.

| command | what you get |
| --- | --- |
| `uv run run.py` | all three views -> `world.png` |
| `uv run run.py --camera front` | camera view (`front`, `overhead`, `wrist`) |
| `uv run run.py --window` | an interactive window |
| `uv run run.py --help` | lists all options |

Already have MuJoCo? Drop the `uv run` and use Python directly:

    python run.py

Needs Python 3.10+ with `mujoco>=3.10,<3.12`, `pillow` and `glfw`.

The scenes are plain MJCF, so you can skip `run.py` altogether and open one in
MuJoCo's own viewer:

    python -m mujoco.viewer --mjcf=assets/scene_front.xml

That gives you the scene as-loaded, without the recorded arm and cube pose that
`run.py` applies.

## The Dataset

A companion dataset of **1,000 episodes** of the pick-and-place cube task,
recorded entirely in this replicated environment. 

| episodes | controller |
| --- | --- |
| 100 | GPU-DAD's recorded action data, replayed in our replicated world |
| 400 | a SmolVLA policy trained on GPU-DAD's own data, acting in our replicated world |
| 500 | our inverse-kinematics expert, acting in our replicated world |

[DATASET NAME](INSERT LINK WHEN READY)

## Environment Files

| file | what it is | MAE vs real |
| --- | --- | --- |
| `assets/scene_front.xml` | Color-matched to GPU-DAD's FRONT camera. | 10.58 |
| `assets/scene_overhead.xml` | Color-matched to GPU-DAD's OVERHEAD camera. | 4.60 |
| `assets/scene_wrist.xml` | Color-matched to GPU-DAD's WRIST camera. | 12.87 |

MAE is mean absolute pixel error, 0-255 per channel, against real GPU-DAD frames. Lower is closer.

## How close is our Replicated Environment to the original?

A policy trained on GPU-DAD's own images scores **25/50** in this world. A policy
trained on this reconstruction scores **25/50**. See [RESULTS.md](RESULTS.md).

## Why there is More than One Environment File

The three scenes differ in paint and lamps only -- geometry and physics are
identical across all of them. To run a policy, drive ONE scene as the physics truth and mirror its state into the other two, rendering each camera view from the scene that was color-matched to it.

## How to Cite

If you use this environment, please cite it:

    Rasheed A., Gallimore K., Subbiah V., Johnson E. (2026).
    gpudad_pickcube_world_reconstructed: a MuJoCo reconstruction of the
    GPU-DAD SO-101 pick-cube scene (version 1.0.0).
    URL: https://github.com/Rovo-Labs/gpu-dad-clone

Machine-readable metadata lives in [CITATION.cff](CITATION.cff) -- GitHub's
"Cite this repository" button reads it directly. Released under Apache-2.0;
see `LICENSE` and `NOTICE`.
