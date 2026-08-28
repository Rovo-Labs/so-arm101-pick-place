"""The inverse-kinematics expert that drives the SO-101 through a pick and place.

`example.py` runs it. This file is the controller behind the 500
inverse-kinematics episodes in rovolabs/so-arm101-pick-place, lifted from the
collector that recorded them and stripped of the data-collection machinery --
no LeRobot writer, no dataset annotations, no state traces.

The expert is kinematic only at the planning layer: IK proposes a joint vector
per waypoint, and every motion between waypoints goes through MuJoCo's position
actuators and stepped physics. It cannot teleport the gripper, and it cannot
will the cube into the bin -- a slipped grasp is a failed episode.

Three things about this arm defeat a textbook position-only solver, and each has
an answer below:

  * SERVO DROOP. The arm joints are fixed-gain position actuators, so at rest
    they settle wherever kp * (ctrl - qpos) balances the pose's gravity torque
    -- below the command, by up to ~0.04 rad on shoulder_lift at full reach.
    Commanding the IK target verbatim therefore misses far grasps
    deterministically. execute_waypoint() settles, measures the error, folds it
    into a running per-episode offset, and repeats.
  * WRIST FLATTENING. shoulder_lift, elbow_flex and wrist_flex are parallel
    hinges, so their sum IS the gripper's pitch. Left free, the solver spends
    that sum buying reach and arrives with the wrist flat, which stands the
    gripper's housing on top of the cube instead of closing around it. IK
    therefore holds the sum at GRASP_PITCH and solves position in the reduced
    pan/lift/elbow space.
  * GRASP HEADING. wrist_roll does not steer the gripper at all -- it only
    spins the jaws about the approach axis -- so squareness to a cube face has
    to be asked for. A Newton loop on wrist_roll aligns the grasp axis with the
    nearest face.

There is no end-effector site in the model, so IK targets the MIDPOINT OF THE
TWO FINGER PADS.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

ASSETS = Path(__file__).resolve().parent / "assets"

# The arm contributes six hinge joints and the cube a freejoint, so qpos is
# [6 joints, 7 cube] and ctrl is the same six joints in the same order.
JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
ARM = slice(0, 5)  # the five joints IK may move
GRIP = 5  # the gripper is commanded, never solved for

HOME = np.array([0.0, -0.7, 1.2, 0.4, 1.0, 0.6])

# Gripper commands, in radians of the jaw hinge.
APPROACH_GRIP = 0.6  # open, on the way down
CLOSED_GRIP = 0.0  # commanded shut; the cube stops the jaws well short
RELEASE_GRIP = 0.9  # wider than the approach, to let go cleanly

# shoulder_lift + elbow_flex + wrist_flex held constant through the grasp.
# Under free position-only IK that sum correlates -0.986 with cube reach -- the
# solver flattens the wrist to buy reach -- and the grasps that did lift all sat
# near 1.0. Holding it keeps the fingers straddling the cube at every reach.
GRASP_PITCH = 1.0

# Height of the PAD MIDPOINT above the table, in metres. The descent and the
# lift are laddered rather than taken in one step: joint-space interpolation
# between distant waypoints bows sideways by more than the ~10 mm of finger
# clearance and punts the cube (measured: unlifted grasps had it knocked a mean
# of 38 mm away). Short steps keep each segment near-vertical.
PREGRASP_Z = 0.100
DESCEND_LADDER = (0.075, 0.055, 0.042, 0.030)
LIFT_LADDER = (0.045, 0.065, 0.085)
CARRY_Z = 0.085
PLACE_Z = 0.040
CLEAR_Z = 0.115
RETREAT_Z = 0.130

# Controller: a 30 Hz command rate, and a per-step cap that keeps the commanded
# joint vector from moving further than the servos can actually follow.
FPS = 30
ARM_CAP = 0.02  # rad per control step, per arm joint
GRIP_CAP = 0.083  # rad per control step, gripper
MAX_PHASE_STEPS = 120  # give up on a waypoint after this many steps
RELEASE_DWELL_STEPS = 15  # hold still after opening, so the cube settles
# The verdict is about where things COME TO REST, so the episode runs on for a
# beat after the arm retreats and is judged then. Judging the instant the arm
# stops calls a cube that is still rolling into the bin a failure.
ADJUDICATION_STEPS = 60

# Servo droop compensation.
SERVO_TOL = 0.01  # rad, largest arm-joint error accepted at a waypoint
SERVO_ROUNDS = 3  # compensation passes after the first ramp
SERVO_OFFSET_CAP = 0.2  # rad, largest command offset from the IK target
SERVO_SETTLE_TOL = 1e-4  # rad, per-step qpos change that counts as settled
SERVO_SETTLE_STEPS = 30  # control steps allowed for settling
# Strict verification applies only where the arm moves through free space. The
# contact phases inherit the offset learned at the previous free-space waypoint
# as feed-forward, but are not strict-failed: their residual error is contact,
# not droop, and fighting it would crush the grasp.
FREE_SPACE_PHASES = frozenset(
    {"pregrasp", "lift", "move_to_bin", "vertical_clear", "retreat"}
)

# Damped least squares.
IK_ITERATIONS = 150
IK_TOLERANCE = 0.002  # metres
IK_DAMPING = 0.02
IK_MAX_STEP = 0.08  # rad, per joint, per iteration
IK_EPSILON = 1e-4  # rad, central-difference step for the Jacobian
HEADING_ROUNDS = 6
HEADING_TOL = 3.0  # degrees off the nearest cube face

# Letting go means: jaws open at least this far, and clear of the cube and the
# bin rim by at least this much, touching neither.
LIFTED_Z = 0.055  # cube centre height that counts as picked up
MIN_RELEASE_GRIP = 0.55
MIN_RELEASE_CLEARANCE = 0.015


@dataclass
class Result:
    """What one episode did, whether or not it worked."""

    cube_xy: tuple[float, float]
    bin_xy: tuple[float, float]
    steps: int = 0
    failed_phase: str | None = None
    lifted: bool = False
    in_bin: bool = False
    released: bool = False

    @property
    def success(self) -> bool:
        return (
            self.failed_phase is None
            and self.lifted
            and self.in_bin
            and self.released
        )

    @property
    def verdict(self) -> str:
        if self.success:
            return "success"
        if self.failed_phase is not None:
            return f"stalled in {self.failed_phase}"
        if not self.lifted:
            return "never lifted the cube"
        if not self.in_bin:
            return "cube not in bin"
        return "never let go"


class Expert:
    """The IK expert, and the world it acts in."""

    def __init__(self, scene: str = "scene_front.xml") -> None:
        self.model = mujoco.MjModel.from_xml_path(str(ASSETS / scene))
        self.data = mujoco.MjData(self.model)
        # A throwaway MjData for evaluating candidate poses during IK; solving
        # in self.data would trample the live state.
        self.ik = mujoco.MjData(self.model)

        actuators = tuple(
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            for i in range(self.model.nu)
        )
        if actuators != JOINTS:
            raise RuntimeError(f"expected actuators {JOINTS}, found {actuators}")

        # One control step is however many physics steps fit in 1/FPS.
        self.substeps = max(1, round(1.0 / (FPS * self.model.opt.timestep)))
        self.ctrl_low = self.model.actuator_ctrlrange[:, 0].copy()
        self.ctrl_high = self.model.actuator_ctrlrange[:, 1].copy()
        self.offset = np.zeros(5)  # the running servo-droop correction
        # Set this to a zero-argument callable to be notified after every
        # control step. example.py uses it to film the episode; nothing in
        # here knows or cares what a camera is.
        self.on_step: object = None

        self.cube_body = self._id(mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.cube_geom = self._id(mujoco.mjtObj.mjOBJ_GEOM, "cube")
        self.cube_qpos = int(
            self.model.jnt_qposadr[self.model.body_jntadr[self.cube_body]]
        )
        self.cube_half = float(self.model.geom_size[self.cube_geom][2])

        # The bin has no freejoint: it is a mocap body, immovable by design,
        # because the real bin never shifts a pixel in the footage we copied.
        # It is placed through mocap_pos, never qpos.
        self.bin_body = self._id(mujoco.mjtObj.mjOBJ_BODY, "container")
        self.bin_mocap = int(self.model.body_mocapid[self.bin_body])
        self.bin_geoms = {
            self._id(mujoco.mjtObj.mjOBJ_GEOM, f"bin_{i}") for i in range(5)
        }
        # The bin's size is a fitted value that lives in the XML, so its floor,
        # rim and inside width are measured off the compiled model, not assumed.
        floor = self._id(mujoco.mjtObj.mjOBJ_GEOM, "bin_0")
        wall = self._id(mujoco.mjtObj.mjOBJ_GEOM, "bin_1")
        self.bin_floor_z = float(
            self.model.geom_pos[floor][2] + self.model.geom_size[floor][2]
        )
        self.bin_rim_z = float(
            self.model.geom_pos[wall][2] + self.model.geom_size[wall][2]
        )
        self.bin_inner = float(
            abs(self.model.geom_pos[wall][0]) - self.model.geom_size[wall][0]
        )

        self.pads = np.array(
            [
                self._id(mujoco.mjtObj.mjOBJ_GEOM, "static_finger_pad"),
                self._id(mujoco.mjtObj.mjOBJ_GEOM, "moving_finger_pad"),
            ]
        )
        self.gripper_geoms = self._subtree_geoms(
            self._id(mujoco.mjtObj.mjOBJ_BODY, "gripper")
        )

    def _id(self, objtype: mujoco.mjtObj, name: str) -> int:
        index = mujoco.mj_name2id(self.model, objtype, name)
        if index < 0:
            raise RuntimeError(f"the scene has no {name}")
        return index

    def _subtree_geoms(self, body: int) -> set[int]:
        """Every geom on this body or any body descended from it."""
        bodies = {body}
        for candidate in range(self.model.nbody):
            parent = candidate
            while parent > 0:
                if parent in bodies:
                    bodies.add(candidate)
                    break
                parent = int(self.model.body_parentid[parent])
        return {
            geom
            for geom in range(self.model.ngeom)
            if int(self.model.geom_bodyid[geom]) in bodies
        }

    # -- the world ---------------------------------------------------------

    def reset(self, cube_xy: tuple[float, float], bin_xy: tuple[float, float]) -> None:
        """Put the arm at HOME and drop the cube and the bin at their spots."""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:6] = HOME
        # Mirror the pose into ctrl. Without this the kp=998 position servos
        # read a zero command on the first step and yank the arm upright.
        self.data.ctrl[:] = np.clip(HOME, self.ctrl_low, self.ctrl_high)
        # A shade proud of the table, so the cube lands rather than starting
        # the episode interpenetrating it.
        self.data.qpos[self.cube_qpos : self.cube_qpos + 7] = [
            cube_xy[0], cube_xy[1], self.cube_half + 0.007, 1.0, 0.0, 0.0, 0.0,
        ]
        self.data.mocap_pos[self.bin_mocap] = [bin_xy[0], bin_xy[1], 0.0]
        self.data.mocap_quat[self.bin_mocap] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(self.model, self.data)
        self._advance()
        self.offset = np.zeros(5)

    def _advance(self) -> None:
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)
        if self.on_step is not None:
            self.on_step()

    def pad_midpoint(self) -> np.ndarray:
        """Where the two finger pads meet, right now."""
        return 0.5 * self.data.geom_xpos[self.pads].sum(axis=0)

    # -- inverse kinematics ------------------------------------------------

    def _midpoint_at(self, q: np.ndarray) -> np.ndarray:
        """Pad midpoint for a candidate joint vector."""
        self.ik.qpos[:6] = q
        mujoco.mj_kinematics(self.model, self.ik)
        return 0.5 * self.ik.geom_xpos[self.pads].sum(axis=0)

    def _jacobian(self, q: np.ndarray) -> np.ndarray:
        """3x5 pad-midpoint Jacobian, by central differences.

        MuJoCo can hand this over analytically instead, via mj_jacGeom on the
        two pads -- and the two agree to 4e-10, at about a fifth of the cost.
        It is deliberately not used. This world is contact-rich enough that a
        difference that small changes which side of a marginal grasp an episode
        lands on: swapping the two flips whole episodes between "cube in the
        bin" and "cube launched off the table". Central differences are what
        the published episodes were collected with, so they are what runs here.
        """
        jac = np.zeros((3, 5))
        for joint in range(5):
            ahead, behind = q.copy(), q.copy()
            ahead[joint] += IK_EPSILON
            behind[joint] -= IK_EPSILON
            jac[:, joint] = (
                self._midpoint_at(ahead) - self._midpoint_at(behind)
            ) / (2 * IK_EPSILON)
        return jac

    def _heading_error(self, q: np.ndarray) -> float:
        """Degrees between the grasp axis and the nearest cube-face direction.

        The cube is square, so the two face-aligned families are 90 degrees
        apart and either will do -- hence the 90-degree wrap.
        """
        self._midpoint_at(q)
        pads = self.ik.geom_xpos[self.pads]
        span = pads[0][:2] - pads[1][:2]
        heading = np.degrees(np.arctan2(span[1], span[0]))
        return float(((heading + 45.0) % 90.0) - 45.0)

    def solve(
        self,
        target: np.ndarray,
        seed: np.ndarray,
        grip: float,
        wrist_roll: float | None = None,
        pitch: float | None = None,
        align: bool = False,
    ) -> np.ndarray:
        """Joint vector putting the pad midpoint at target, holding grip.

        With `pitch` given, wrist_flex stops being free: it is pinned to
        pitch - shoulder_lift - elbow_flex, and position is solved in the
        reduced pan/lift/elbow space with the wrist_flex column folded into the
        other two. That leaves wrist_roll for `align` to spend on squaring the
        grasp to a cube face.

        Damped least squares: dq = J^T (J J^T + k^2 I)^-1 e. The damping stops
        the arm lunging near a singularity, where an undamped pseudo-inverse
        asks for infinite joint velocity.
        """
        q = np.asarray(seed, dtype=np.float64).copy()
        q[GRIP] = grip
        if wrist_roll is not None:
            q[4] = wrist_roll
        if pitch is not None:
            q[3] = np.clip(pitch - q[1] - q[2], self.ctrl_low[3], self.ctrl_high[3])

        def position(q: np.ndarray) -> np.ndarray:
            for _ in range(IK_ITERATIONS):
                error = target - self._midpoint_at(q)
                if np.linalg.norm(error) < IK_TOLERANCE:
                    break
                jac = self._jacobian(q)
                if pitch is None:
                    step = jac.T @ np.linalg.solve(
                        jac @ jac.T + IK_DAMPING**2 * np.eye(3), error
                    )
                    q[ARM] += np.clip(step, -IK_MAX_STEP, IK_MAX_STEP)
                else:
                    # wrist_flex is dependent, so its column joins the two it
                    # depends on and the system shrinks to pan/lift/elbow.
                    reduced = np.column_stack(
                        [jac[:, 0], jac[:, 1] - jac[:, 3], jac[:, 2] - jac[:, 3]]
                    )
                    step = reduced.T @ np.linalg.solve(
                        reduced @ reduced.T + IK_DAMPING**2 * np.eye(3), error
                    )
                    q[:3] += np.clip(step, -IK_MAX_STEP, IK_MAX_STEP)
                    q[3] = pitch - q[1] - q[2]
                q = np.clip(q, self.ctrl_low, self.ctrl_high)
                q[GRIP] = grip
            return q

        q = position(q)
        if pitch is not None and align:
            # Newton on wrist_roll, re-solving position after each change
            # because rolling the gripper also moves the pad midpoint.
            for _ in range(HEADING_ROUNDS):
                error = self._heading_error(q)
                if abs(error) < HEADING_TOL:
                    break
                probe = q.copy()
                probe[4] = np.clip(probe[4] + 0.05, self.ctrl_low[4], self.ctrl_high[4])
                gain = (self._heading_error(probe) - error) / 0.05
                if abs(gain) < 1.0:
                    gain = np.sign(gain) or 1.0
                q[4] = np.clip(
                    q[4] - error / gain, self.ctrl_low[4], self.ctrl_high[4]
                )
                q = position(q)
        return q

    def plan(
        self, cube_xy: tuple[float, float], bin_xy: tuple[float, float]
    ) -> list[tuple[str, np.ndarray]]:
        """The waypoints of a pick and place, solved back to back.

        Seeding each solve from the previous solution keeps consecutive
        waypoints in the same branch of the arm's null space, so the elbow does
        not flip halfway through a carry.
        """
        cube = np.asarray(cube_xy, dtype=np.float64)
        destination = np.asarray(bin_xy, dtype=np.float64)
        seed = HOME.copy()
        waypoints: list[tuple[str, np.ndarray]] = []

        def reach(
            name: str,
            xy: np.ndarray,
            z: float,
            grip: float,
            roll: float | None = None,
            pitch: float | None = None,
            align: bool = False,
        ) -> None:
            nonlocal seed
            seed = self.solve(
                np.array([xy[0], xy[1], z]), seed, grip, roll, pitch, align
            )
            waypoints.append((name, seed.copy()))

        def squeeze(name: str, grip: float) -> None:
            nonlocal seed
            seed = seed.copy()
            seed[GRIP] = grip
            waypoints.append((name, seed.copy()))

        reach("pregrasp", cube, PREGRASP_Z, APPROACH_GRIP, 1.0, GRASP_PITCH, True)
        for z in DESCEND_LADDER:
            reach("descend", cube, z, APPROACH_GRIP, None, GRASP_PITCH, True)
        squeeze("close", CLOSED_GRIP)
        for z in LIFT_LADDER:
            reach("lift", cube, z, CLOSED_GRIP, None, GRASP_PITCH)
        reach("move_to_bin", destination, CARRY_Z, CLOSED_GRIP, None, GRASP_PITCH)
        reach("place", destination, PLACE_Z, CLOSED_GRIP)
        squeeze("open", RELEASE_GRIP)
        reach("vertical_clear", destination, CLEAR_Z, RELEASE_GRIP)
        reach("retreat", destination, RETREAT_Z, RELEASE_GRIP)
        return waypoints

    # -- execution ---------------------------------------------------------

    def _ramp(self, command: np.ndarray, result: Result) -> bool:
        """Walk ctrl towards command a capped slice at a time."""
        caps = np.array([ARM_CAP] * 5 + [GRIP_CAP])
        for _ in range(MAX_PHASE_STEPS):
            gap = command - self.data.ctrl[:6]
            if np.max(np.abs(gap)) < 1e-3:
                return True
            self.data.ctrl[:6] = np.clip(
                self.data.ctrl[:6] + np.clip(gap, -caps, caps),
                self.ctrl_low,
                self.ctrl_high,
            )
            self._advance()
            result.steps += 1
        return False

    def _settle(self, result: Result) -> None:
        """Step until the arm stops moving, so its droop can be measured."""
        previous = self.data.qpos[:6].copy()
        for _ in range(SERVO_SETTLE_STEPS):
            self._advance()
            result.steps += 1
            current = self.data.qpos[:6].copy()
            if np.max(np.abs(current - previous)[:5]) < SERVO_SETTLE_TOL:
                return
            previous = current

    def execute(self, phase: str, target: np.ndarray, result: Result) -> bool:
        """Drive until the JOINTS reach the target, not merely the command.

        Free-space phases ramp, settle, measure the steady-state error, fold it
        into the running offset and go again. Contact phases apply the offset
        already learned as feed-forward but accept command-arrival, because
        their residual error is the cube, not droop.
        """
        strict = phase in FREE_SPACE_PHASES
        command = np.clip(
            np.concatenate([target[:5] + self.offset, target[5:]]),
            self.ctrl_low,
            self.ctrl_high,
        )
        for _ in range(1 + (SERVO_ROUNDS if strict else 0)):
            if not self._ramp(command, result):
                return False
            self._settle(result)
            if not strict:
                return True
            error = target[:5] - self.data.qpos[:5]
            if np.max(np.abs(error)) < SERVO_TOL:
                self.offset = np.clip(
                    command[:5] - target[:5], -SERVO_OFFSET_CAP, SERVO_OFFSET_CAP
                )
                return True
            self.offset = np.clip(
                self.offset + error, -SERVO_OFFSET_CAP, SERVO_OFFSET_CAP
            )
            command = command.copy()
            command[:5] = np.clip(
                target[:5] + self.offset, self.ctrl_low[:5], self.ctrl_high[:5]
            )
        return False

    def run(
        self,
        cube_xy: tuple[float, float],
        bin_xy: tuple[float, float],
        after_phase: object = None,
    ) -> Result:
        """Fly one episode.

        after_phase, if given, is called with each phase name as it completes
        (and with "spawn" for the reset state) -- that is how example.py takes
        its photographs without this module knowing anything about images.
        """
        result = Result(cube_xy=cube_xy, bin_xy=bin_xy)
        self.reset(cube_xy, bin_xy)
        if after_phase is not None:
            after_phase("spawn")
        for phase, target in self.plan(cube_xy, bin_xy):
            if not self.execute(phase, target, result):
                result.failed_phase = phase
                break
            if phase == "close":
                result.lifted = False
            if phase == "lift":
                result.lifted = self.holding_cube()
            if phase == "open":
                for _ in range(RELEASE_DWELL_STEPS):
                    self._advance()
                    result.steps += 1
            if after_phase is not None:
                after_phase(phase)
        for _ in range(ADJUDICATION_STEPS):
            self._advance()
        result.in_bin = self.cube_in_bin()
        result.released = self.let_go()
        return result

    # -- did it work? ------------------------------------------------------

    def holding_cube(self) -> bool:
        """Cube clear of the table AND still in the jaws."""
        off_table = self.data.xpos[self.cube_body][2] > LIFTED_Z
        return bool(off_table and self._touching({self.cube_geom}, self.gripper_geoms))

    def cube_in_bin(self) -> bool:
        """Inside the bin's footprint, above its floor, below its rim.

        The rim test is the one that matters: a cube perched on the wall tops
        passes the footprint test just as well as one sitting on the floor.
        """
        cube = self.data.xpos[self.cube_body]
        container = self.data.xpos[self.bin_body]
        return bool(
            np.all(np.abs(cube[:2] - container[:2]) < self.bin_inner)
            and cube[2] > container[2] + self.bin_floor_z
            and cube[2] < container[2] + self.bin_rim_z + self.cube_half
        )

    def let_go(self) -> bool:
        """Jaws open, lifted clear, and touching neither cube nor bin."""
        cube_top = float(self.data.xpos[self.cube_body][2] + self.cube_half)
        rim = float(self.data.xpos[self.bin_body][2] + self.bin_rim_z)
        pads_at = float(np.min(self.data.geom_xpos[self.pads, 2]))
        holding = self._touching(self.gripper_geoms, self.bin_geoms | {self.cube_geom})
        return bool(
            self.data.qpos[GRIP] >= MIN_RELEASE_GRIP
            and pads_at > max(cube_top, rim) + MIN_RELEASE_CLEARANCE
            and not holding
        )

    def _touching(self, left: set[int], right: set[int]) -> bool:
        for i in range(self.data.ncon):
            first = int(self.data.contact[i].geom1)
            second = int(self.data.contact[i].geom2)
            if (first in left and second in right) or (
                second in left and first in right
            ):
                return True
        return False
