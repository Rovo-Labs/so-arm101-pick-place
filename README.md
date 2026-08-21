# gpudad_pickcube_world_reconstructed -- static world

## The original

This is a MuJoCo replica of the scene in
[gpudad/so101_pick_cube](https://huggingface.co/datasets/gpudad/so101_pick_cube)
by [gpudad](https://huggingface.co/gpudad). All fidelity numbers below are
measured against frames from that dataset.

## The Replicated Scene

A frozen MuJoCo scene. Open a scene file in any MuJoCo viewer.

    python -m mujoco.viewer --mjcf=scene_freeroam.xml

## Files

| file | what it is | MAE vs real |
| --- | --- | --- |
| `scene_freeroam.xml` | START HERE. Fly the camera anywhere; geometry is the scene's real geometry. | -- |
| `scene_front.xml` | Colour-matched to GPU-DAD's FRONT camera. | 10.58 |
| `scene_overhead.xml` | Colour-matched to GPU-DAD's OVERHEAD camera. | 4.60 |
| `scene_wrist.xml` | Colour-matched to GPU-DAD's WRIST camera. | 12.87 |
| `so101_new_calib.xml` | the SO-101 arm, with the wrist camera and base shift baked in | -- |
| `assets/` | robot part meshes (STL), unmodified from so101-nexus-core | -- |
| `floor_checker.png` | floor texture for scene_freeroam.xml, scene_overhead.xml, scene_wrist.xml | -- |
| `floor_checker_front.png` | floor texture for scene_front.xml | -- |
| `LICENSE` | Apache-2.0 | -- |
| `CITATION.cff` | how to cite this work | -- |
| `NOTICE` | attribution and our modifications | -- |

## Why there is more than one scene

A MuJoCo camera is only a position and a lens angle -- it carries no colour.
Lights and materials belong to the model, so one file can hold many cameras
but only ONE palette. Matching three real cameras that each auto-white-balance
differently therefore takes three files. They differ in paint and lamps only;
geometry and physics are identical across all of them.

You are never locked to a named camera. Every viewer lets you fly freely in
any of these files; the named cameras are bookmarks.

## Notes

Every scene here shares one robot (so101_new_calib.xml) and one assets/ folder.

**The near clipping plane.** Every scene cuts away anything closer than 19.5mm to whichever camera you are looking through. That is why the wrist camera sees the gripper sliced open -- it sits inside the gripper -- and why a camera further away sees a normal solid robot. It is not a wrist-only setting; it applies to all of them.

**Where the cube and bin are.** In the dataset, the cube and bin move to a different spot in every episode. Those positions are NOT stored in these files -- the simulation sets them when an episode starts. So if you open a scene on its own, the cube and bin sit at one default spot rather than where they were in any particular episode. That is expected, not a fault.

For the exact per-episode cube and bin positions, see the published dataset:
[DATASET NAME](INSERT LINK WHEN READY).


## Provenance and licence

The SO-101 robot description and all 13 STL meshes come from
**so101-nexus-core 0.3.12** (Apache-2.0). The
meshes are byte-identical to upstream. `so101_new_calib.xml` is modified in two
places -- the robot base position and the wrist camera pose/fovy -- both listed
explicitly in `NOTICE`.

Everything else here -- the scene files, the camera and lighting fits, the
floor textures -- is original work, released under Apache-2.0 to
match. See `LICENSE` and `NOTICE`.

MAE is mean absolute pixel error, 0-255 per channel, against real GPU-DAD frames. Lower is closer.
