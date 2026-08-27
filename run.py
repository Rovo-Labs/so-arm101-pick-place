#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "mujoco>=3.10,<3.12",
#     "pillow>=10.1",
#     "glfw>=2.6",
# ]
# ///
"""Look at the reconstructed GPU-DAD pick-cube world.

    uv run run.py            writes world.png -- the three calibrated views
    uv run run.py --window   opens an interactive window instead
    uv run run.py --check    proves the render matches the published figure

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


def _below_warning(record: logging.LogRecord) -> bool:
    """Route INFO to stdout and WARNING/ERROR to stderr."""
    return record.levelno < logging.WARNING


_stdout = logging.StreamHandler(stream=sys.stdout)
_stdout.addFilter(filter=_below_warning)
_stderr = logging.StreamHandler(stream=sys.stderr)
_stderr.setLevel(level=logging.WARNING)
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[_stdout, _stderr],
)
logger = logging.getLogger("run")


try:
    import mujoco
    from PIL import Image, ImageChops, ImageDraw, ImageFont
except ImportError as error:  # pragma: no cover -- the no-dependencies path
    logger.error("run.py needs MuJoCo and Pillow, and they are not installed here.")
    logger.error(f"  ({error})")
    logger.error("")
    logger.error("The easy fix -- uv fetches Python and everything else for you:")
    logger.error("")
    logger.error("    uv run run.py")
    logger.error("")
    logger.error("Get uv from https://docs.astral.sh/uv/")
    logger.error("")
    logger.error("Without uv, use any Python 3.10 or newer:")
    logger.error("")
    logger.error("    python3.12 -m venv .venv")
    logger.error('    .venv/bin/pip install "mujoco>=3.10,<3.12" "pillow>=10.1" glfw')
    logger.error("    .venv/bin/python run.py")
    logger.error("")
    logger.error("macOS ships Python 3.9, which cannot install MuJoCo 3.10 at all.")
    raise SystemExit(1) from error


HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
FIGURE = HERE / "figures" / "reconstruction_3views.png"

# Each camera was colour-matched to exactly one scene; the scenes differ in
# paint and lamps only (README, "Why there is more than one scene"). run.py
# renders every view from the scene it was fitted to, so the pairing cannot be
# got wrong. That is why there is no --scene flag.
SCENES = {
    "front": ("scene_front.xml", "front"),
    "overhead": ("scene_overhead.xml", "overhead"),
    "wrist": ("scene_wrist.xml", "wrist_cam"),
}
SHEET_ORDER = ("front", "overhead", "wrist")
TRUTH_SCENE = "scene_front.xml"
CAMERA_ALIASES = {"wrist_cam": "wrist"}
# assets/*.xml also declare a camera named "overhead_dummy" (pos "0 0 1", no
# orientation, no fovy). It is an uncalibrated placeholder, not the fitted
# overhead camera, so run.py never offers it. Do not "fix" its absence.
PLACEHOLDER_CAMERAS = frozenset({"overhead_dummy"})

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
# Rendered at 512x512 and downscaled to 316 with LANCZOS, these reproduce the
# published figure bit for bit -- `uv run run.py --check` proves it. If you
# edit the scene XMLs, --check is your regression test.
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

DEFAULT_SIZE = 512
MIN_SIZE = 64
MAX_SIZE = 4096
FIGURE_PANEL = 316
FIGURE_GUTTER = 10
FIGURE_BAND = 22


def die(problem: str, remedy: str = "") -> NoReturn:
    """Report a user-facing failure and exit 1 without a traceback."""
    logger.error(f"run.py: {problem}")
    if remedy:
        logger.error(f"  {remedy}")
    raise SystemExit(1)


def scene_path(filename: str) -> Path:
    """Return the path to a scene XML, failing clearly if assets/ is absent."""
    path = ASSETS / filename
    if not path.is_file():
        die(
            f"cannot find {path}.",
            "Run run.py from inside the repository, or keep it next to the "
            "assets folder.",
        )
    return path


def load(filename: str) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Compile a scene XML and return a posed (model, data) pair."""
    path = scene_path(filename)
    try:
        model = mujoco.MjModel.from_xml_path(str(path))
    except (ValueError, mujoco.FatalError) as error:
        die(f"MuJoCo could not load {path}: {error}")
    data = mujoco.MjData(model)
    apply_pose(model=model, data=data)
    return model, data


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
            "The world XML may have changed since POSE_* were fitted; "
            "run `uv run run.py --check`."
        )


def mirror_state(
    source: mujoco.MjData,
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> None:
    """Copy the truth scene's state into another scene's data.

    This is the method the README prescribes under "Why there is more than one
    scene": drive ONE scene as the physics truth and mirror its state into the
    others, rendering each view from the scene it was colour-matched to.
    """
    data.qpos[:] = source.qpos
    data.qvel[:] = source.qvel
    data.ctrl[:] = source.ctrl
    if model.na:
        data.act[:] = source.act
    data.mocap_pos[:] = source.mocap_pos
    data.mocap_quat[:] = source.mocap_quat
    data.time = source.time
    mujoco.mj_forward(model, data)


def render(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    size: int,
    camera: str | None,
) -> Image.Image:
    """Render one square image, raising the offscreen buffer if it is too small."""
    # The scenes declare offwidth/offheight 512. Renderer() RAISES above that
    # rather than clamping, so patch the compiled model -- never the XML. max()
    # so a small --size cannot shrink the buffer.
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, size)
    model.vis.global_.offheight = max(model.vis.global_.offheight, size)
    try:
        with mujoco.Renderer(model, height=size, width=size) as renderer:
            if camera is None:
                renderer.update_scene(data)
            else:
                renderer.update_scene(data, camera=camera)
            return Image.fromarray(renderer.render())
    except (ValueError, mujoco.FatalError) as error:
        die(
            f"the renderer refused a {size}x{size} image: {error}",
            f"Try a smaller --size; {DEFAULT_SIZE}x{DEFAULT_SIZE} works everywhere.",
        )


def render_calibrated(size: int) -> dict[str, Image.Image]:
    """Render each calibrated view from the scene it was colour-matched to."""
    truth_model, truth_data = load(TRUTH_SCENE)
    panels: dict[str, Image.Image] = {}
    for name in SHEET_ORDER:
        filename, camera = SCENES[name]
        model, data = load(filename)
        mirror_state(source=truth_data, model=model, data=data)
        panels[name] = render(model=model, data=data, size=size, camera=camera)
    del truth_model
    return panels


def compose_sheet(panels: dict[str, Image.Image], size: int) -> Image.Image:
    """Lay the three views out side by side, mirroring the published figure."""
    gutter = max(6, round(size * FIGURE_GUTTER / FIGURE_PANEL))
    band = max(14, round(size * FIGURE_BAND / FIGURE_PANEL))
    font_px = max(10, round(band * 0.5))
    try:
        font = ImageFont.load_default(size=font_px)
    except TypeError:  # pragma: no cover -- Pillow < 10.1
        font = ImageFont.load_default()
    sheet = Image.new("RGB", (size * 3 + gutter * 2, band + size), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for index, name in enumerate(SHEET_ORDER):
        left = index * (size + gutter)
        sheet.paste(panels[name], (left, band))
        draw.text(
            (left, round((band - font_px) / 2)),
            name.upper(),
            fill=(51, 51, 51),
            font=font,
        )
    return sheet


def resolve_out(raw: str | None) -> Path:
    """Work out where to write, creating the parent directory if needed."""
    target = Path(raw).expanduser() if raw else Path("world.png")
    if not target.is_absolute():
        target = Path.cwd() / target
    # normpath, NOT resolve() -- do not rewrite the user's path through symlinks
    target = Path(os.path.normpath(target))
    if target.is_dir() or target.suffix == "":
        target = target / "world.png"
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        die(
            f"cannot create {target.parent}: {error}",
            "Pick another folder with --out.",
        )
    if not os.access(target.parent, os.W_OK):
        die(
            f"cannot write to {target.parent}: Permission denied",
            "Pick another folder with --out.",
        )
    return target


def reveal(path: Path, allowed: bool) -> None:
    """Open the written image in the platform viewer. Never fails the run."""
    if not allowed or not sys.stdout.isatty():
        return
    if sys.platform == "win32":
        with contextlib.suppress(OSError):
            os.startfile(path)
        return
    opener = "open" if sys.platform == "darwin" else "xdg-open"
    if not shutil.which(opener):
        return
    with contextlib.suppress(OSError, subprocess.SubprocessError):
        subprocess.run([opener, str(path)], timeout=5, check=False)


def do_check() -> None:
    """Compare a fresh render against the published figure, pixel for pixel."""
    if not FIGURE.is_file():
        die(
            f"cannot find {FIGURE}.",
            "Run run.py from inside the repository.",
        )
    reference = Image.open(FIGURE).convert("RGB")
    expected = (
        FIGURE_PANEL * 3 + FIGURE_GUTTER * 2,
        FIGURE_BAND + FIGURE_PANEL,
    )
    if reference.size != expected:
        die(
            f"{FIGURE.name} is {reference.size[0]}x{reference.size[1]}, "
            f"expected {expected[0]}x{expected[1]}.",
            "The published figure has changed; POSE_* and --check need refitting.",
        )
    logger.info(f"checking against {FIGURE.relative_to(HERE)}")
    panels = render_calibrated(size=DEFAULT_SIZE)
    failures = 0
    for index, name in enumerate(SHEET_ORDER):
        left = index * (FIGURE_PANEL + FIGURE_GUTTER)
        crop = reference.crop(
            (left, FIGURE_BAND, left + FIGURE_PANEL, FIGURE_BAND + FIGURE_PANEL)
        )
        got = panels[name].resize(
            (FIGURE_PANEL, FIGURE_PANEL), Image.Resampling.LANCZOS
        )
        diff = ImageChops.difference(got, crop)
        if diff.getbbox() is None:
            logger.info(f"  {name.upper():<9} identical")
        else:
            worst = max(channel.getextrema()[1] for channel in diff.split())
            logger.info(
                f"  {name.upper():<9} differs -- largest channel difference {worst}"
            )
            failures += 1
    total = FIGURE_PANEL * FIGURE_PANEL * len(SHEET_ORDER)
    if failures:
        logger.error(f"FAILED: the render no longer matches {FIGURE.name}.")
        logger.error(
            "Most likely the MuJoCo version changed: 3.12 moves the floor texture, "
            "which is"
        )
        logger.error(
            "why run.py pins mujoco<3.12. Otherwise assets/ or POSE_* has been edited."
        )
        raise SystemExit(1)
    logger.info(
        f"run.py reproduces the published figure exactly (0 of {total} pixels differ)."
    )


def die_no_display(problem: str, errors: list[str]) -> NoReturn:
    """Explain that a window is impossible here, and offer the path that works."""
    logger.error(f"run.py: could not open a window -- {problem}.")
    if errors:
        logger.error(f"  GLFW said: {errors[-1]}")
    logger.error("")
    logger.error(
        "  This usually means the machine has no display: a server, a container,"
    )
    logger.error("  or an SSH session without X forwarding.")
    logger.error("")
    logger.error("  Render an image instead. It works anywhere:")
    logger.error("      uv run run.py")
    raise SystemExit(1)


def run_window() -> None:
    """Open an interactive GLFW window on the truth scene."""
    try:
        import glfw  # noqa: PLC0415 -- GUI-only; the headless path must never need it
    except (ImportError, OSError) as error:
        die(
            f"glfw is missing or could not load ({error}).",
            "Render an image instead -- it needs no display and works anywhere:  "
            "uv run run.py",
        )

    model, data = load(TRUTH_SCENE)

    errors: list[str] = []
    glfw.set_error_callback(lambda code, text: errors.append(f"{code}: {text}"))
    if not glfw.init():
        die_no_display("GLFW would not start", errors)
    window = glfw.create_window(
        1200, 900, "gpudad_pickcube_world -- run.py", None, None
    )
    if not window:
        glfw.terminate()
        die_no_display("GLFW would not open a window", errors)
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

    def select_camera() -> None:
        name = cameras[state["camera"]]
        if name is None:
            mujoco.mjv_defaultFreeCamera(model, cam)
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        else:
            cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            cam.fixedcamid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)

    mouse = {"x": 0.0, "y": 0.0, "left": False, "right": False, "middle": False}

    def on_button(win: object, button: int, act: int, mods: int) -> None:
        mouse["left"] = glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        mouse["right"] = (
            glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        )
        mouse["middle"] = (
            glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
        )
        mouse["x"], mouse["y"] = glfw.get_cursor_pos(win)

    def on_move(win: object, xpos: float, ypos: float) -> None:
        dx = xpos - mouse["x"]
        dy = ypos - mouse["y"]
        mouse["x"], mouse["y"] = xpos, ypos
        if not (mouse["left"] or mouse["right"] or mouse["middle"]):
            return
        _, height = glfw.get_window_size(win)
        shift = (
            glfw.get_key(win, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
            or glfw.get_key(win, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
        )
        if mouse["right"]:
            action = (
                mujoco.mjtMouse.mjMOUSE_MOVE_H
                if shift
                else mujoco.mjtMouse.mjMOUSE_MOVE_V
            )
        elif mouse["left"]:
            action = (
                mujoco.mjtMouse.mjMOUSE_ROTATE_H
                if shift
                else mujoco.mjtMouse.mjMOUSE_ROTATE_V
            )
        else:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM
        mujoco.mjv_moveCamera(model, action, dx / height, dy / height, scene, cam)

    def on_scroll(win: object, xoff: float, yoff: float) -> None:
        mujoco.mjv_moveCamera(
            model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0.0, -0.05 * yoff, scene, cam
        )

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
            select_camera()

    # Prime the scene BEFORE any input callback can reach it. mjv_moveCamera
    # reads the scene's frustum, and on a freshly constructed MjvScene that is
    # still zero -- dragging the mouse in the first frame otherwise dies with
    # "mjv_cameraInModel: mjvScene frustum_near too small".
    mujoco.mjv_updateScene(
        model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene
    )

    glfw.set_mouse_button_callback(window, on_button)
    glfw.set_cursor_pos_callback(window, on_move)
    glfw.set_scroll_callback(window, on_scroll)
    glfw.set_key_callback(window, on_key)

    logger.info("drag to orbit, shift+drag to pan, scroll to zoom")
    logger.info("Tab next camera | Space pause | R reset | Esc quit")

    keys_text = "drag\nshift+drag\nscroll\nTab\nSpace\nR\nEsc"
    actions_text = "orbit\npan\nzoom\nnext camera\npause\nreset\nquit"
    started = time.perf_counter() - data.time
    while not glfw.window_should_close(window):
        if state["paused"]:
            started = time.perf_counter() - data.time
        else:
            budget = 0
            while data.time < time.perf_counter() - started and budget < 200:
                mujoco.mj_step(model, data)
                budget += 1
            if budget >= 200:  # a stall must not accrue debt
                started = time.perf_counter() - data.time
        width, height = glfw.get_framebuffer_size(window)  # framebuffer, not window
        viewport = mujoco.MjrRect(0, 0, width, height)  # -> correct on Retina
        mujoco.mjv_updateScene(
            model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene
        )
        mujoco.mjr_render(viewport, scene, context)
        mujoco.mjr_overlay(
            mujoco.mjtFont.mjFONT_NORMAL,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            viewport,
            keys_text,
            actions_text,
            context,
        )
        mujoco.mjr_overlay(
            mujoco.mjtFont.mjFONT_NORMAL,
            mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
            viewport,
            "camera\nscene\nsim time",
            f"{cameras[state['camera']] or 'free'}\n{TRUTH_SCENE}\n{data.time:6.2f} s",
            context,
        )
        glfw.swap_buffers(window)
        glfw.poll_events()
    glfw.terminate()


def positive_square(raw: str) -> int:
    """Parse --size, explaining why renders are square rather than WxH."""
    text = raw.strip().lower()
    if "x" in text or "," in text:
        die(
            f"--size takes one number, not '{raw}'.",
            "Renders are square: MuJoCo's field of view is vertical, so the "
            "calibrated cameras are only correct at 1:1. Try --size 1024.",
        )
    try:
        value = int(text)
    except ValueError:
        die(f"--size takes a number, not '{raw}'.")
    if not MIN_SIZE <= value <= MAX_SIZE:
        die(f"--size must be between {MIN_SIZE} and {MAX_SIZE}, got {value}.")
    return value


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
    if text not in {"all", "free", *SCENES}:
        die(
            f"there is no camera called '{raw}'.",
            f"Choose one of: all, free, {', '.join(sorted(SCENES))}.",
        )
    return text


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser, epilog and all."""
    parser = argparse.ArgumentParser(
        prog="run.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  uv run run.py                  all three views -> world.png\n"
            "  uv run run.py --camera free    an orbit view of the whole table\n"
            "  uv run run.py --size 2048      bigger, for slides\n"
            "  uv run run.py --window         fly around it\n"
            "  uv run run.py --check          prove it matches the figure\n"
        ),
    )
    parser.add_argument(
        "--camera",
        type=parse_camera,
        default=None,
        help="all, free, front, overhead or wrist (default: all)",
    )
    parser.add_argument(
        "--size",
        type=positive_square,
        default=None,
        metavar="N",
        help=f"panel size in pixels, square (default: {DEFAULT_SIZE})",
    )
    parser.add_argument(
        "--out",
        default=None,
        metavar="PATH",
        help="where to write the image (default: world.png)",
    )
    parser.add_argument(
        "--window",
        action="store_true",
        help="open an interactive window instead of writing a file",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="prove the render matches figures/reconstruction_3views.png",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="do not launch the platform image viewer",
    )
    return parser


def main() -> None:
    """Render the world, check it, or open a window on it."""
    args = build_parser().parse_args()

    if args.window:
        for flag, value in (("--size", args.size), ("--out", args.out)):
            if value is not None:
                die(
                    f"{flag} only applies to rendered images.",
                    "The window is resizable -- just drag its corner.",
                )
        if args.check:
            die("--window and --check cannot be combined.", "Run them separately.")
        run_window()
        return

    if args.check:
        for flag, value in (
            ("--camera", args.camera),
            ("--size", args.size),
            ("--out", args.out),
        ):
            if value is not None:
                die(
                    f"--check writes nothing, so {flag} has no effect.",
                    f"Drop {flag}, or drop --check.",
                )
        do_check()
        return

    size = args.size if args.size is not None else DEFAULT_SIZE
    camera = args.camera if args.camera is not None else "all"
    target = resolve_out(args.out)

    logger.info(
        "GPU-DAD episode 44, frame 16 -- the frame in figures/reconstruction_3views.png"
    )
    if camera == "all":
        for name in SHEET_ORDER:
            filename, cam_name = SCENES[name]
            logger.info(f"  {cam_name:<10} from assets/{filename}")
        image = compose_sheet(panels=render_calibrated(size=size), size=size)
    elif camera == "free":
        model, data = load(TRUTH_SCENE)
        logger.info(f"  free camera from assets/{TRUTH_SCENE}")
        image = render(model=model, data=data, size=size, camera=None)
    else:
        filename, cam_name = SCENES[camera]
        truth_model, truth_data = load(TRUTH_SCENE)
        model, data = load(filename)
        mirror_state(source=truth_data, model=model, data=data)
        logger.info(f"  {cam_name:<10} from assets/{filename}")
        image = render(model=model, data=data, size=size, camera=cam_name)
        del truth_model

    try:
        image.save(target)
    except OSError as error:
        die(f"cannot write {target}: {error}", "Pick another folder with --out.")
    logger.info(f"wrote {target}  ({image.width}x{image.height})")
    logger.info("")
    logger.info("Fly around it:  uv run run.py --window")
    logger.info("Orbit view:     uv run run.py --camera free")
    logger.info("Prove it matches the published figure:  uv run run.py --check")
    reveal(path=target, allowed=not args.no_open)


if __name__ == "__main__":
    main()
