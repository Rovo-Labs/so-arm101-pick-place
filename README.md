# gpudad_pickcube_world_reconstructed -- static world

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

A frozen MuJoCo scene. Open a scene file in any MuJoCo viewer.

    python -m mujoco.viewer --mjcf=scene_front.xml

## Files

| file | what it is | MAE vs real |
| --- | --- | --- |
| `scene_front.xml` | Colour-matched to GPU-DAD's FRONT camera. | 10.58 |
| `scene_overhead.xml` | Colour-matched to GPU-DAD's OVERHEAD camera. | 4.60 |
| `scene_wrist.xml` | Colour-matched to GPU-DAD's WRIST camera. | 12.87 |

MAE is mean absolute pixel error, 0-255 per channel, against real GPU-DAD frames. Lower is closer.

## How close is our Replicated world to the original?

A policy trained on GPU-DAD's own images scores **25/50** in this world. A policy
trained on this reconstruction scores **25/50**. See [RESULTS.md](RESULTS.md).

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

## Why there is more than one scene

The three scenes differ in paint and lamps only -- geometry and physics are
identical across all of them. To run a policy, drive ONE scene as the physics truth and mirror its state into the other two, rendering each camera view from the scene that was colour-matched to it.

## How to cite

If you use this world, please cite it:

    Gallimore, K. and Rasheed, A. (2026). gpudad_pickcube_world_reconstructed:
    a MuJoCo reconstruction of the GPU-DAD SO-101 pick-cube scene (v1.0.0).
    Rovo Labs. https://github.com/Rovo-Labs/gpu-dad-clone

Machine-readable metadata lives in [CITATION.cff](CITATION.cff) -- GitHub's
"Cite this repository" button reads it directly. Released under Apache-2.0;
see `LICENSE` and `NOTICE`.
