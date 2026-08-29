"""Watch the IK expert pick the cube up and drop it in the bin.

    uv run example.py                 a new random placement, filmed
    uv run example.py --seed 7        that exact placement again
    uv run example.py --camera wrist  film one camera instead of all three

Writes example.gif: the whole attempt, every control step, from all three
cameras side by side. The expert itself lives in ik_expert.py; this file only
decides where the cube and the bin go, and holds the cameras.

uv fetches Python and MuJoCo on the first run, so there is nothing to install.
Get uv from https://docs.astral.sh/uv/  --  or use any Python 3.10+:

    pip install "mujoco>=3.10,<3.12" "pillow>=10.1" numpy && python example.py
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ik_expert import ASSETS, FPS, Expert

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger("example")

OUTPUT = "example.gif"

# Each camera was colour-matched to exactly one scene, so each view is rendered
# from the scene it was fitted to and the pairing cannot be got wrong. The
# three scenes carry identical geometry and physics, so ONE of them is driven
# as the truth and its state is mirrored into the other two purely to be
# photographed -- which is the recipe the README prescribes for running a
# policy in this world.
SCENES = {
    "front": ("scene_front.xml", "front"),
    "overhead": ("scene_overhead.xml", "overhead"),
    "wrist": ("scene_wrist.xml", "wrist_cam"),
}
DRIVER = "front"  # the scene the physics actually runs in

# Where the cube and the bin may land, in metres, in the world frame. These are
# the boxes the data set was collected over. The robot base is at x = -0.1024,
# so +x is away from the robot; the cube lands on the +y side, the bin on -y.
CUBE_X = (0.15, 0.19)
CUBE_Y = (0.03, 0.07)
BIN_X = (0.15, 0.19)
BIN_Y = (-0.07, -0.03)

SEED_RANGE = 1_000_000  # a drawn placement seed is in [0, SEED_RANGE)

SIZE = 320  # pixels per camera
STRIDE = 2  # keep every other control step, so the file stays small
GUTTER = 12
BAND = 30  # caption strip above the views, as in run.py's world.png


class Film:
    """Renders one or more cameras of the driven scene, step by step."""

    def __init__(self, expert: Expert, views: list[str]) -> None:
        self.expert = expert
        self.views = views
        self.frames: list[Image.Image] = []
        self.steps = 0
        self.renderers: dict[str, tuple[mujoco.Renderer, mujoco.MjData, str]] = {}
        for view in views:
            filename, camera = SCENES[view]
            if view == DRIVER:
                model, data = expert.model, expert.data
            else:
                # A render-only copy of the same world in that view's paint.
                model = mujoco.MjModel.from_xml_path(str(ASSETS / filename))
                same = (
                    model.nq == expert.model.nq
                    and model.nmocap == expert.model.nmocap
                )
                if not same:
                    raise RuntimeError(f"{filename} is not the driver's world")
                data = mujoco.MjData(model)
            visual = model.vis.global_
            visual.offwidth = max(visual.offwidth, SIZE)
            visual.offheight = max(visual.offheight, SIZE)
            self.renderers[view] = (
                mujoco.Renderer(model, height=SIZE, width=SIZE),
                data,
                camera,
            )

    def close(self) -> None:
        for renderer, _, _ in self.renderers.values():
            renderer.close()

    def shoot(self) -> None:
        """Photograph the current state from every view, and lay them out."""
        tiles = []
        for view in self.views:
            renderer, data, camera = self.renderers[view]
            if view != DRIVER:
                # Mirror the driver's state across. Only the pose is copied --
                # the driver is never read back from, so nothing here can
                # perturb the physics.
                data.qpos[:] = self.expert.data.qpos
                data.qvel[:] = self.expert.data.qvel
                data.mocap_pos[:] = self.expert.data.mocap_pos
                data.mocap_quat[:] = self.expert.data.mocap_quat
                mujoco.mj_forward(renderer.model, data)
            renderer.update_scene(data, camera=camera)
            tiles.append(Image.fromarray(renderer.render()))
        self.frames.append(self._compose(tiles))

    def _compose(self, tiles: list[Image.Image]) -> Image.Image:
        if len(tiles) == 1:
            return tiles[0]
        font_px = max(9, round(BAND * 0.5))
        try:
            font = ImageFont.load_default(size=font_px)
        except TypeError:  # pragma: no cover -- Pillow < 10.1
            font = ImageFont.load_default()
        width = SIZE * len(tiles) + GUTTER * (len(tiles) - 1)
        sheet = Image.new("RGB", (width, BAND + SIZE), (255, 255, 255))
        draw = ImageDraw.Draw(sheet)
        for index, (view, tile) in enumerate(zip(self.views, tiles, strict=True)):
            left = index * (SIZE + GUTTER)
            sheet.paste(tile, (left, BAND))
            draw.text(
                (left, round((BAND - font_px) / 2)),
                view.upper(),
                fill=(51, 51, 51),
                font=font,
            )
        return sheet

    def on_step(self) -> None:
        self.steps += 1
        if self.steps % STRIDE == 0:
            self.shoot()


def seed_value(raw: str) -> int:
    """Accept any whole number from zero up, and say so if it is not one."""
    try:
        value = int(raw)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{raw!r} is not a whole number") from None
    if value < 0:
        raise argparse.ArgumentTypeError(f"a seed is zero or greater, not {value}")
    return value


def reveal(path: Path) -> None:
    """Open the finished film in the platform viewer. Never fails the run.

    Skipped when stdout is not a terminal, so piping or scripting this never
    tries to open a window.
    """
    if not sys.stdout.isatty():
        return
    if sys.platform == "win32":
        with contextlib.suppress(OSError):
            os.startfile(path)
        return
    opener = "open" if sys.platform == "darwin" else "xdg-open"
    if shutil.which(opener):
        with contextlib.suppress(OSError, subprocess.SubprocessError):
            subprocess.run([opener, str(path)], timeout=5, check=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="example.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--seed",
        type=seed_value,
        default=None,
        metavar="N",
        help="play this specific placement. N is any whole number from 0 up. "
        f"Without it, a seed is drawn at random from the first {SEED_RANGE:,}.",
    )
    parser.add_argument(
        "--camera",
        choices=["all", *sorted(SCENES)],
        default="all",
        help="film all three cameras side by side, or just one (default: all)",
    )
    args = parser.parse_args()

    # Without --seed the placement is new every run. The seed is drawn and
    # printed either way, so a run worth showing someone else can always be
    # replayed exactly.
    seed = args.seed
    if seed is None:
        seed = int(np.random.default_rng().integers(0, SEED_RANGE))
    rng = np.random.default_rng(seed)
    cube_xy = (float(rng.uniform(*CUBE_X)), float(rng.uniform(*CUBE_Y)))
    bin_xy = (float(rng.uniform(*BIN_X)), float(rng.uniform(*BIN_Y)))

    views = list(SCENES) if args.camera == "all" else [args.camera]
    expert = Expert(SCENES[DRIVER][0])
    film = Film(expert, views)
    try:
        expert.on_step = film.on_step
        result = expert.run(cube_xy, bin_xy)
    finally:
        expert.on_step = None
        film.close()

    # The verdict is the point of the run, so it leads. A failure is a real
    # outcome here, not an error -- the expert is dynamics-aware and can drop
    # the cube -- so it is reported plainly and the exit status stays 0.
    if result.success:
        logger.info("SUCCESS  the cube ended up in the bin")
    else:
        logger.info(f"FAILED   {result.verdict}")
    logger.info(
        f"  placement seed {seed}   "
        f"cube ({cube_xy[0]:.3f}, {cube_xy[1]:+.3f})   "
        f"bin ({bin_xy[0]:.3f}, {bin_xy[1]:+.3f})   "
        f"{result.steps} steps"
    )
    if args.seed is None:
        logger.info(f"  replay it with:  uv run example.py --seed {seed}")

    target = Path.cwd() / OUTPUT
    if not os.access(target.parent, os.W_OK):
        logger.error(f"example.py: cannot write to {target.parent}")
        raise SystemExit(1)
    # One palette for the whole film, and no dithering. These renders are
    # flat-shaded, so dithering buys nothing visible and costs about two thirds
    # of the file: 5.3 MB becomes 1.9 MB. 256 is the GIF maximum, so no colour
    # is lost.
    palette = film.frames[0].quantize(colors=256, method=Image.MEDIANCUT)
    reduced = [f.quantize(palette=palette, dither=Image.NONE) for f in film.frames]
    try:
        reduced[0].save(
            target,
            save_all=True,
            append_images=reduced[1:],
            duration=round(1000 * STRIDE / FPS),
            loop=0,
            optimize=True,
        )
    except OSError as error:
        logger.error(f"example.py: cannot write {target}: {error}")
        raise SystemExit(1) from error
    first = film.frames[0]
    logger.info(
        f"  wrote {target.name}  ({len(film.frames)} frames, "
        f"{first.width}x{first.height}, {target.stat().st_size / 1e6:.1f} MB)"
    )
    reveal(target)


if __name__ == "__main__":
    main()
