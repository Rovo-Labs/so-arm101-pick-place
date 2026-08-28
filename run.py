"""Look at the reconstructed GPU-DAD pick-cube world.

    uv run run.py            writes world.png -- the three camera views
    uv run run.py --window   opens an interactive window instead

uv fetches Python and MuJoCo on the first run, so there is nothing to install.
Get uv from https://docs.astral.sh/uv/  --  or use any Python 3.10+:

    pip install "mujoco>=3.10,<3.12" "pillow>=10.1" glfw && python run.py
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import NoReturn

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger("run")

MISSING_DEPENDENCIES = """\
run.py needs MuJoCo and Pillow, and they are not installed here.
  ({error})

The easy fix -- uv fetches Python and everything else for you:

    uv run run.py

Get uv from https://docs.astral.sh/uv/

Without uv, use any Python 3.10 or newer:

    python3.12 -m venv .venv
    .venv/bin/pip install "mujoco>=3.10,<3.12" "pillow>=10.1" glfw
    .venv/bin/python run.py

macOS ships Python 3.9, which cannot install MuJoCo 3.10 at all."""

NO_DISPLAY = """\
run.py: could not open a window -- {problem}.
{said}
  This usually means the machine has no display: a server, a container,
  or an SSH session without X forwarding.

  Render an image instead. It works anywhere:
      uv run run.py"""

try:
    import mujoco
    from PIL import Image, ImageDraw, ImageFont
except ImportError as error:  # pragma: no cover -- the no-dependencies path
    logger.error(MISSING_DEPENDENCIES.format(error=error))
    raise SystemExit(1) from error


ASSETS = Path(__file__).resolve().parent / "assets"
OUTPUT = "world.png"

# Each camera was colour-matched to exactly one scene; the scenes differ in
# paint and lighting only. run.py renders every view from the scene it was
# fitted to, so the pairing cannot be got wrong -- hence no --scene flag.
SCENES = {
    "front": ("scene_front.xml", "front"),
    "overhead": ("scene_overhead.xml", "overhead"),
    "wrist": ("scene_wrist.xml", "wrist_cam"),
}
WINDOW_SCENE = "scene_front.xml"
CAMERA_ALIASES = {"wrist_cam": "wrist"}
# The scenes also declare an "overhead_dummy" camera (pos "0 0 1", no
# orientation, no fovy). It is an uncalibrated placeholder, not the fitted
# overhead camera, so run.py never offers it. Do not "fix" its absence.
PLACEHOLDER_CAMERAS = frozenset({"overhead_dummy"})

# The scenes declare offwidth/offheight 512, and mujoco.Renderer raises above
# that rather than clamping, so this is the size everything renders at.
SIZE = 512
# Sheet proportions, taken from figures/reconstruction_3views.png so the two
# can be held side by side.
GUTTER = 16
BAND = 36

# --------------------------------------------------------------------------
# The state below is GPU-DAD episode 44, frame 16 -- the frame shown in
# figures/reconstruction_3views.png. It is recorded data, not a pose invented
# for this script:
#   POSE_ARM  = the dataset's recorded observation.state for that frame,
#               converted to MuJoCo joint angles (radians).
#   POSE_CUBE = the cube placement fitted for THIS world by the v5 visual
#               placement refinement (worst mean centroid error 0.66 px).
#   POSE_BIN  = the container placement from the same refinement. The bin is a
#               mocap body, so it moves via data.mocap_pos, never qpos.
# Together these reproduce the published figure exactly. Edit them to look at
# a different state -- there is no keyframe in the XML to fall back on.
# See NOTICE for the provenance of POSE_ARM.
# --------------------------------------------------------------------------
POSE_ARM = (
    -0.3590147473493621,
    -0.12949600332158023,
    0.2737756618497921,
    1.427106105633954,
    -0.002630974254689465,
    0.5999985948846367,
)
POSE_CUBE = (0.1330528530145393, 0.0597912330345744, 0.02, 1.0, 0.0, 0.0, 0.0)
POSE_BIN = (0.09120804242790284, -0.01489297435760058, 0.0)


def die(problem: str, remedy: str = "") -> NoReturn:
    """Report a user-facing failure and exit 1 without a traceback."""
    logger.error(f"run.py: {problem}")
    if remedy:
        logger.error(f"  {remedy}")
    raise SystemExit(1)


def apply_pose(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Put the world into the recorded episode-44 frame-16 state."""
    data.qpos[0:6] = POSE_ARM
    data.ctrl[:] = POSE_ARM  # the sts3215 servos are kp=998 position
    data.qpos[6:13] = POSE_CUBE  # actuators: without this mirror they yank
    data.qvel[:] = 0.0  # the arm back upright on the first step
    data.mocap_pos[0] = POSE_BIN
    mujoco.mj_forward(model, data)
    worst = min((data.contact[i].dist for i in range(data.ncon)), default=0.0)
    if worst < -0.001:
        logger.warning(
            f"warning: the posed state overlaps the scene by {-worst * 1000:.1f} mm. "
            "The world XML may have changed since POSE_* were fitted."
        )


def load(filename: str) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Compile a scene XML and pose it at the recorded episode-44 frame."""
    path = ASSETS / filename
    if not path.is_file():
        die(
            f"cannot find {path}.",
            "Run run.py from inside the repository, or keep it next to the "
            "assets folder.",
        )
    try:
        model = mujoco.MjModel.from_xml_path(str(path))
    except (ValueError, mujoco.FatalError) as error:
        die(f"MuJoCo could not load {path}: {error}")
    data = mujoco.MjData(model)
    apply_pose(model=model, data=data)
    return model, data


def render(filename: str, camera: str) -> Image.Image:
    """Render one square view from the scene it was colour-matched to."""
    model, data = load(filename)
    # Widen the compiled model's offscreen buffer if a scene ever declares a
    # smaller one -- never the XML, which is read-only here.
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, SIZE)
    model.vis.global_.offheight = max(model.vis.global_.offheight, SIZE)
    try:
        with mujoco.Renderer(model, height=SIZE, width=SIZE) as renderer:
            renderer.update_scene(data, camera=camera)
            return Image.fromarray(renderer.render())
    except (ValueError, mujoco.FatalError) as error:
        die(f"the renderer refused a {SIZE}x{SIZE} image: {error}")


def compose_sheet() -> Image.Image:
    """Render all three views and lay them out like the published figure."""
    font_px = max(10, round(BAND * 0.5))
    try:
        font = ImageFont.load_default(size=font_px)
    except TypeError:  # pragma: no cover -- Pillow < 10.1
        font = ImageFont.load_default()
    sheet = Image.new("RGB", (SIZE * 3 + GUTTER * 2, BAND + SIZE), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for index, (name, (filename, camera)) in enumerate(SCENES.items()):
        left = index * (SIZE + GUTTER)
        sheet.paste(render(filename=filename, camera=camera), (left, BAND))
        draw.text(
            (left, round((BAND - font_px) / 2)),
            name.upper(),
            fill=(51, 51, 51),
            font=font,
        )
    return sheet


def reveal(path: Path, allowed: bool) -> None:
    """Open the written image in the platform viewer. Never fails the run."""
    if not allowed or not sys.stdout.isatty():
        return
    if sys.platform == "win32":
        with contextlib.suppress(OSError):
            os.startfile(path)
        return
    opener = "open" if sys.platform == "darwin" else "xdg-open"
    if shutil.which(opener):
        with contextlib.suppress(OSError, subprocess.SubprocessError):
            subprocess.run([opener, str(path)], timeout=5, check=False)


def run_window() -> None:
    """Open an interactive GLFW window on the scene."""
    try:
        import glfw  # noqa: PLC0415 -- GUI-only; the headless path must never need it
    except (ImportError, OSError) as error:
        die(
            f"glfw is missing or could not load ({error}).",
            "Render an image instead -- it needs no display and works anywhere:  "
            "uv run run.py",
        )

    errors: list[str] = []

    def no_display(problem: str) -> NoReturn:
        said = f"  GLFW said: {errors[-1]}\n" if errors else ""
        logger.error(NO_DISPLAY.format(problem=problem, said=said))
        raise SystemExit(1)

    model, data = load(WINDOW_SCENE)
    glfw.set_error_callback(lambda code, text: errors.append(f"{code}: {text}"))
    if not glfw.init():
        no_display("GLFW would not start")
    window = glfw.create_window(1200, 900, "so-arm101-pick-place -- run.py", None, None)
    if not window:
        glfw.terminate()
        no_display("GLFW would not open a window")
    glfw.make_context_current(window)
    glfw.swap_interval(1)

    # MjrContext MUST be built after make_context_current -- this order is
    # load-bearing; reversing it segfaults or renders nothing.
    cam, opt = mujoco.MjvCamera(), mujoco.MjvOption()
    mujoco.mjv_defaultCamera(cam)
    mujoco.mjv_defaultOption(opt)
    scene = mujoco.MjvScene(model, maxgeom=10000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
    mujoco.mjv_defaultFreeCamera(model, cam)

    cameras: list[str | None] = [None]
    for index in range(model.ncam):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, index)
        if name and name not in PLACEHOLDER_CAMERAS:
            cameras.append(name)
    state = {"camera": 0, "paused": False}
    mouse = {"x": 0.0, "y": 0.0, "left": False, "right": False, "middle": False}

    def on_button(win: object, button: int, act: int, mods: int) -> None:
        for name, code in (
            ("left", glfw.MOUSE_BUTTON_LEFT),
            ("right", glfw.MOUSE_BUTTON_RIGHT),
            ("middle", glfw.MOUSE_BUTTON_MIDDLE),
        ):
            mouse[name] = glfw.get_mouse_button(win, code) == glfw.PRESS
        mouse["x"], mouse["y"] = glfw.get_cursor_pos(win)

    def on_move(win: object, xpos: float, ypos: float) -> None:
        dx, dy = xpos - mouse["x"], ypos - mouse["y"]
        mouse["x"], mouse["y"] = xpos, ypos
        if not (mouse["left"] or mouse["right"] or mouse["middle"]):
            return
        _, height = glfw.get_window_size(win)
        shift = (
            glfw.get_key(win, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
            or glfw.get_key(win, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
        )
        mjmouse = mujoco.mjtMouse
        if mouse["right"]:
            action = mjmouse.mjMOUSE_MOVE_H if shift else mjmouse.mjMOUSE_MOVE_V
        elif mouse["left"]:
            action = mjmouse.mjMOUSE_ROTATE_H if shift else mjmouse.mjMOUSE_ROTATE_V
        else:
            action = mjmouse.mjMOUSE_ZOOM
        move_camera(action, dx / height, dy / height)

    def on_scroll(win: object, xoff: float, yoff: float) -> None:
        move_camera(mujoco.mjtMouse.mjMOUSE_ZOOM, 0.0, -0.05 * yoff)

    def on_key(win: object, key: int, scancode: int, act: int, mods: int) -> None:
        if act != glfw.PRESS:
            return
        if key in (glfw.KEY_ESCAPE, glfw.KEY_Q):
            glfw.set_window_should_close(win, True)
        elif key == glfw.KEY_SPACE:
            state["paused"] = not state["paused"]
        elif key == glfw.KEY_R:
            apply_pose(model=model, data=data)
        elif key == glfw.KEY_TAB:
            state["camera"] = (state["camera"] + 1) % len(cameras)
            name = cameras[state["camera"]]
            if name is None:
                mujoco.mjv_defaultFreeCamera(model, cam)
                cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            else:
                cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                cam.fixedcamid = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_CAMERA, name
                )

    # Prime the scene BEFORE any input callback can reach it. mjv_moveCamera
    # reads the scene's frustum, and on a freshly constructed MjvScene that is
    # still zero -- dragging the mouse in the first frame otherwise dies with
    # "mjv_cameraInModel: mjvScene frustum_near too small".
    mujoco.mjv_updateScene(
        model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene
    )

    # mujoco 3.11 removed the scene argument from mjv_moveCamera, and the
    # dependency range allows both 3.10 and 3.11. Resolve which one this build
    # wants ONCE, using a zero-delta call that cannot move anything, rather
    # than guessing or paying a try/except on every mouse event.
    try:
        mujoco.mjv_moveCamera(model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0.0, 0.0, cam)

        def move_camera(action: int, dx: float, dy: float) -> None:
            mujoco.mjv_moveCamera(model, action, dx, dy, cam)

    except TypeError:  # mujoco 3.10 still wants the scene

        def move_camera(action: int, dx: float, dy: float) -> None:
            mujoco.mjv_moveCamera(model, action, dx, dy, scene, cam)

    glfw.set_mouse_button_callback(window, on_button)
    glfw.set_cursor_pos_callback(window, on_move)
    glfw.set_scroll_callback(window, on_scroll)
    glfw.set_key_callback(window, on_key)
    logger.info("drag to orbit, shift+drag to pan, scroll to zoom")
    logger.info("Tab next camera | Space pause | R reset | Esc quit")

    started = time.perf_counter() - data.time
    while not glfw.window_should_close(window):
        if state["paused"]:
            started = time.perf_counter() - data.time
        else:
            # Catch physics up to wall-clock; one step per frame would run at
            # about an eighth of real speed. The cap stops a stall from
            # accruing debt it then tries to simulate all at once.
            steps = 0
            while data.time < time.perf_counter() - started and steps < 200:
                mujoco.mj_step(model, data)
                steps += 1
            if steps >= 200:
                started = time.perf_counter() - data.time
        width, height = glfw.get_framebuffer_size(window)  # framebuffer -> Retina
        viewport = mujoco.MjrRect(0, 0, width, height)
        mujoco.mjv_updateScene(
            model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene
        )
        mujoco.mjr_render(viewport, scene, context)
        mujoco.mjr_overlay(
            mujoco.mjtFont.mjFONT_NORMAL,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            viewport,
            "drag\nshift+drag\nscroll\nTab\nSpace\nR\nEsc",
            "orbit\npan\nzoom\nnext camera\npause\nreset\nquit",
            context,
        )
        mujoco.mjr_overlay(
            mujoco.mjtFont.mjFONT_NORMAL,
            mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
            viewport,
            "camera\nscene\nsim time",
            f"{cameras[state['camera']] or 'free'}\n{WINDOW_SCENE}\n{data.time:6.2f} s",
            context,
        )
        glfw.swap_buffers(window)
        glfw.poll_events()
    glfw.terminate()


def parse_camera(raw: str) -> str:
    """Normalise --camera and reject the uncalibrated placeholder."""
    text = raw.strip().lower()
    text = CAMERA_ALIASES.get(text, text)
    if text in PLACEHOLDER_CAMERAS:
        die(
            f"'{raw}' is an uncalibrated placeholder camera in the XML, not the "
            "fitted overhead view.",
            "Use --camera overhead.",
        )
    if text not in {"all", *SCENES}:
        die(
            f"there is no camera called '{raw}'.",
            f"Choose one of: all, {', '.join(sorted(SCENES))}.",
        )
    return text


def main() -> None:
    """Render the world, or open a window on it."""
    parser = argparse.ArgumentParser(
        prog="run.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  uv run run.py                  all three views -> world.png\n"
            "  uv run run.py --camera front   just the front view\n"
            "  uv run run.py --window         fly around it\n"
        ),
    )
    parser.add_argument(
        "--camera",
        type=parse_camera,
        default="all",
        help="which view to save: all, front, overhead or wrist (default: all)",
    )
    parser.add_argument(
        "--window",
        action="store_true",
        help="open a live window to move around in, instead of saving a picture",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="do not launch the platform image viewer",
    )
    args = parser.parse_args()

    if args.window:
        run_window()
        return

    target = Path.cwd() / OUTPUT
    if not os.access(target.parent, os.W_OK):
        die(
            f"cannot write to {target.parent}: Permission denied",
            "Run run.py from a folder you can write to.",
        )
    if args.camera == "all":
        image = compose_sheet()
    else:
        filename, camera = SCENES[args.camera]
        image = render(filename=filename, camera=camera)

    try:
        image.save(target)
    except OSError as error:
        die(f"cannot write {target}: {error}")
    logger.info(f"wrote {target}  ({image.width}x{image.height})")
    reveal(path=target, allowed=not args.no_open)


if __name__ == "__main__":
    main()
