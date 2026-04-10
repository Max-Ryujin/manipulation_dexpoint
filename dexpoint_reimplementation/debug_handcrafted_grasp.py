"""Run a handcrafted grasp-and-lift policy and log reward-debug signals."""

import argparse
import csv
import imageio
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

if "MUJOCO_GL" not in os.environ and not os.environ.get("DISPLAY"):
    os.environ["MUJOCO_GL"] = "egl"

import mujoco
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from franka_gym_env import FrankaGymEnvironment
from tasks import GraspingTask, create_task_config
from manipulation import ControllerStatus
from ycb_scene import DEFAULT_YCB_OBJECT_ROOT


# For the attachment_site frame, this rotates the approach axis to point down the
# world -Z direction, producing a vertical top grasp rather than a side grasp.
TOP_DOWN_QUAT = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "debug_handcrafted_grasp_runs"
INFO_KEYS = [
    "reward_total",
    "distance_reward",
    "orientation_reward",
    "grasp_reward",
    "lift_reward",
    "goal_reward",
    "goal_height_reward",
    "time_penalty",
    "success_bonus",
    "reach_distance",
    "ee_target_xy_distance",
    "ee_target_z_distance",
    "gripper_opening_width",
    "gripper_open_fraction",
    "gripper_actuator_force",
    "orientation_down_alignment",
    "distance_score",
    "orientation_score",
    "caging_score",
    "force_score",
    "goal_distance",
    "goal_xy_distance",
    "goal_z_distance",
    "goal_height_distance",
    "target_height_above_table",
    "target_lift",
    "lift_progress",
    "between_fingers_score",
    "grasp_resistance_score",
    "target_between_fingers",
    "grasp_detected",
    "is_success",
    "goal_reward_active",
]


@dataclass
class Phase:
    name: str
    kind: str
    timeout_steps: int
    pose_fn: Optional[Callable[[FrankaGymEnvironment], Tuple[np.ndarray, np.ndarray]]] = None
    gripper_command: Optional[str] = None
    dwell_steps: int = 0


def build_environment(
    seed: Optional[int],
    camera_height: int = 240,
    camera_width: int = 320,
    overlay_pointcloud: bool = False,
) -> FrankaGymEnvironment:
    env = FrankaGymEnvironment(
        xml_path=None,
        task_name=GraspingTask.NAME,
        ycb_object_root=DEFAULT_YCB_OBJECT_ROOT.as_posix(),
        num_points=512,
        camera_height=camera_height,
        camera_width=camera_width,
        rate=200.0,
        frame_skip=10,
        visualize_pointclouds=overlay_pointcloud,
    )
    env.configure_task(
        create_task_config(
            GraspingTask.NAME,
            max_episode_steps=800,
            target_body_name=env.target_body_name,
        )
    )
    if seed is not None:
        env.seed(seed)
    return env


def seconds_to_steps(env: FrankaGymEnvironment, seconds: float) -> int:
    dt = env.env.model.opt.timestep * env.env.frame_skip
    return max(1, int(np.ceil(seconds / dt)))


def get_site_pose(env: FrankaGymEnvironment, site_name: str) -> Tuple[np.ndarray, np.ndarray]:
    site_id = env.env.model.site(site_name).id
    position = env.env.data.site_xpos[site_id].copy().astype(np.float32)
    rotation_matrix = env.env.data.site_xmat[site_id].reshape(3, 3).copy()
    quaternion = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quaternion, rotation_matrix.reshape(-1))
    return position, quaternion.astype(np.float32)


def get_body_quaternion(env: FrankaGymEnvironment, body_name: str) -> np.ndarray:
    return env.env.get_object_orientation(body_name).astype(np.float32)


def make_motion_phase(
    name: str,
    timeout_steps: int,
    z_offset: float,
) -> Phase:
    def _pose_fn(env: FrankaGymEnvironment) -> Tuple[np.ndarray, np.ndarray]:
        target_pos = env.get_target_position().astype(np.float64)
        pose = target_pos.copy()
        pose[2] += z_offset
        return pose, TOP_DOWN_QUAT.copy()

    return Phase(name=name, kind="motion", timeout_steps=timeout_steps, pose_fn=_pose_fn)


def build_phase_sequence(env: FrankaGymEnvironment, args: argparse.Namespace) -> List[Phase]:
    return [
        make_motion_phase("move_above_can", seconds_to_steps(env, args.move_timeout), args.approach_height),
        Phase(
            name="open_gripper",
            kind="gripper",
            timeout_steps=seconds_to_steps(env, args.gripper_dwell),
            gripper_command="open",
            dwell_steps=seconds_to_steps(env, args.gripper_dwell),
        ),
        make_motion_phase("descend_to_grasp", seconds_to_steps(env, args.move_timeout), args.grasp_height),
        Phase(
            name="close_gripper",
            kind="gripper",
            timeout_steps=seconds_to_steps(env, args.grasp_dwell),
            gripper_command="close",
            dwell_steps=seconds_to_steps(env, args.grasp_dwell),
        ),
        Phase(
            name="settle_grasp",
            kind="dwell",
            timeout_steps=seconds_to_steps(env, args.settle_dwell),
            dwell_steps=seconds_to_steps(env, args.settle_dwell),
        ),
        make_motion_phase("lift_can", seconds_to_steps(env, args.move_timeout), args.lift_height),
    ]


def flatten_xyz(prefix: str, value: np.ndarray) -> Dict[str, float]:
    return {
        f"{prefix}_x": float(value[0]),
        f"{prefix}_y": float(value[1]),
        f"{prefix}_z": float(value[2]),
    }


def flatten_quat(prefix: str, value: np.ndarray) -> Dict[str, float]:
    return {
        f"{prefix}_qw": float(value[0]),
        f"{prefix}_qx": float(value[1]),
        f"{prefix}_qy": float(value[2]),
        f"{prefix}_qz": float(value[3]),
    }


def build_row(
    env: FrankaGymEnvironment,
    episode_index: int,
    phase_index: int,
    phase: Phase,
    reward: float,
    done: bool,
    info: Dict[str, Any],
) -> Dict[str, Any]:
    attachment_position, attachment_quat = get_site_pose(env, "attachment_site")
    gripper_site_position, gripper_site_quat = get_site_pose(env, "gripper")
    ee_position = env.get_end_effector_position()
    left_finger_pos, right_finger_pos = env.get_finger_positions()
    target_pos, target_quat = env.get_target_pose()
    hand_quat = get_body_quaternion(env, "hand")
    joint_state = env.env.data.qpos[: env.robot_dof].copy()
    ctrl_state = env.env.data.ctrl[: env.robot_dof].copy()

    row: Dict[str, Any] = {
        "episode": episode_index,
        "phase_index": phase_index,
        "phase": phase.name,
        "phase_kind": phase.kind,
        "sim_time": float(env.env.sim_time),
        "step_count": int(env.step_count),
        "controller_status": env.env.controller.get_status().value,
        "reward": float(reward),
        "done": bool(done),
        "pointcloud_empty": bool(info.get("pointcloud_empty", False)),
        "pointcloud_size": int(info.get("pointcloud_size", env._last_pointcloud_size)),
        "contact_count": int(info.get("contact_count", env.env.data.ncon)),
        "max_joint_speed": float(
            info.get("max_joint_speed", np.max(np.abs(env.env.data.qvel[: env.robot_dof])))
        ),
        "gripper_ctrl_command": float(ctrl_state[7]),
        "gripper_joint_position": float(env.get_gripper_joint_position()),
        "gripper_open_fraction": float(env.get_gripper_open_fraction()),
        "gripper_opening_width": float(env.get_gripper_opening_width()),
        "gripper_actuator_force": float(env.get_gripper_actuator_force()),
    }
    row.update(flatten_xyz("ee_position", ee_position))
    row.update(flatten_xyz("attachment_site_position", attachment_position))
    row.update(flatten_quat("attachment_site", attachment_quat))
    row.update(flatten_xyz("gripper_site_position", gripper_site_position))
    row.update(flatten_quat("gripper_site", gripper_site_quat))
    row.update(flatten_xyz("left_finger_position", left_finger_pos))
    row.update(flatten_xyz("right_finger_position", right_finger_pos))
    row.update(flatten_xyz("target_position", target_pos.astype(np.float32)))
    row.update(flatten_quat("target", target_quat.astype(np.float32)))
    row.update(flatten_quat("hand", hand_quat))

    for joint_index in range(env.robot_dof):
        row[f"joint_qpos_{joint_index}"] = float(joint_state[joint_index])
        row[f"ctrl_{joint_index}"] = float(ctrl_state[joint_index])

    for key in INFO_KEYS:
        row[key] = info.get(key)

    row.setdefault("phase_gripper_command", "")
    row.setdefault("phase_target_position_x", None)
    row.setdefault("phase_target_position_y", None)
    row.setdefault("phase_target_position_z", None)
    row.setdefault("phase_target_qw", None)
    row.setdefault("phase_target_qx", None)
    row.setdefault("phase_target_qy", None)
    row.setdefault("phase_target_qz", None)

    return row


def collect_env_step(env: FrankaGymEnvironment) -> Tuple[float, bool, Dict[str, Any]]:
    env.step_count += 1
    reward, done, info = env._compute_reward_and_done()
    info = dict(info)
    info.setdefault("pointcloud_empty", False)
    info.setdefault("pointcloud_size", 0)
    info["contact_count"] = int(env.env.data.ncon)
    info["max_joint_speed"] = float(np.max(np.abs(env.env.data.qvel[: env.robot_dof])))

    if not done:
        failure_info = env._check_failure_termination()
        if failure_info is not None:
            reward = float(env.task_config.get("failure_penalty", env.failure_penalty))
            done = True
            info.update(failure_info)
            info["failure_penalty"] = reward

    if env.step_count >= env.max_episode_steps:
        done = True
        info["step_limit_reached"] = True
    return float(reward), bool(done), info


def render_debug_frame(
    env: FrankaGymEnvironment, overlay_pointcloud: bool = False
) -> Optional[np.ndarray]:
    if overlay_pointcloud:
        return env.render_with_pointcloud(mode="rgb_array")
    return env.render(mode="rgb_array")


def command_motion_phase(
    env: FrankaGymEnvironment,
    phase: Phase,
    step_size: float,
) -> Dict[str, Any]:
    if phase.pose_fn is None:
        raise ValueError(f"Motion phase '{phase.name}' requires pose_fn")

    target_position, target_quat = phase.pose_fn(env)
    # add minimal offset to the target position to avoid hitting the can with one of the fingers when descending
    target_position[1] += 0.01
    ik = env.env.get_ik()
    ik.update_configuration(env.env.data.qpos)
    dt = env.env.model.opt.timestep * env.env.frame_skip
    ik.set_target_position(target_position, target_quat)
    converged = ik.converge_ik(dt)
    if not converged:
        raise RuntimeError(
            f"IK did not converge for phase '{phase.name}' at target {target_position.tolist()}"
        )

    # The controller keeps gripper and arm progress in a single status enum.
    # After a close command it can remain in GRASPING indefinitely while still
    # holding the desired gripper ctrl value, which blocks subsequent arm motion.
    if env.env.controller.get_status() == ControllerStatus.GRASPING:
        env.env.controller.stop()

    env.env.controller.move_to_incremental(ik.configuration.q[:7], step_size=step_size)
    return {
        "target_position": target_position,
        "target_quat": target_quat,
    }


def command_gripper_phase(env: FrankaGymEnvironment, phase: Phase) -> None:
    if phase.gripper_command == "open":
        env.env.controller.open_gripper()
        return
    if phase.gripper_command == "close":
        env.env.controller.close_gripper()
        return
    raise ValueError(f"Unknown gripper command for phase '{phase.name}': {phase.gripper_command}")


def advance_phase_if_ready(
    env: FrankaGymEnvironment,
    phase: Phase,
    phase_started_step: int,
) -> bool:
    elapsed = env.step_count - phase_started_step
    if phase.kind == "motion":
        return elapsed > 0 and env.env.controller.get_status() == ControllerStatus.IDLE
    if phase.kind in {"gripper", "dwell"}:
        return elapsed >= phase.dwell_steps
    raise ValueError(f"Unknown phase kind: {phase.kind}")


def run_episode(
    env: FrankaGymEnvironment,
    episode_index: int,
    csv_writer: Any,
    args: argparse.Namespace,
    video_writer: Optional[Any] = None,
) -> Dict[str, Any]:
    env.reset()
    if video_writer is not None:
        frame = render_debug_frame(env, overlay_pointcloud=args.overlay_pointcloud)
        if frame is not None:
            video_writer.append_data(frame)
    phases = build_phase_sequence(env, args)
    phase_index = 0
    phase = phases[phase_index]
    phase_started = False
    phase_started_step = env.step_count
    phase_command: Dict[str, Any] = {}
    last_reward = 0.0
    last_info: Dict[str, Any] = {}

    while True:
        if not phase_started:
            print(f"[episode {episode_index}] phase {phase_index + 1}/{len(phases)}: {phase.name}")
            phase_started_step = env.step_count
            if phase.kind == "motion":
                phase_command = command_motion_phase(env, phase, args.arm_step_size)
            elif phase.kind == "gripper":
                command_gripper_phase(env, phase)
                phase_command = {"gripper_command": phase.gripper_command}
            elif phase.kind == "dwell":
                phase_command = {}
            else:
                raise ValueError(f"Unknown phase kind: {phase.kind}")
            phase_started = True

        env.env.controller.step()
        env.env.step()
        reward, done, info = collect_env_step(env)
        if video_writer is not None:
            frame = render_debug_frame(env, overlay_pointcloud=args.overlay_pointcloud)
            if frame is not None:
                video_writer.append_data(frame)
        last_reward = reward
        last_info = info

        row = build_row(env, episode_index, phase_index, phase, reward, done, info)
        for key, value in phase_command.items():
            if key == "target_position":
                row.update(flatten_xyz("phase_target_position", value.astype(np.float32)))
            elif key == "target_quat":
                row.update(flatten_quat("phase_target", value.astype(np.float32)))
            else:
                row[f"phase_{key}"] = value
        csv_writer.writerow(row)

        elapsed = env.step_count - phase_started_step
        if phase.kind == "motion" and elapsed > 0 and elapsed % 10 == 0:
            print(
                f"[episode {episode_index}] {phase.name} progress "
                f"step={elapsed}/{phase.timeout_steps} "
                f"remaining_waypoints={len(env.env.controller.trajectory)}"
            )
        if elapsed > phase.timeout_steps:
            info["phase_timeout"] = True
            info["timed_out_phase"] = phase.name
            print(f"[episode {episode_index}] timeout in phase '{phase.name}' after {elapsed} steps")
            return {
                "done": True,
                "reward": reward,
                "info": info,
                "completed_phases": phase_index,
            }

        if done:
            print(
                f"[episode {episode_index}] done reward={reward:.4f} "
                f"lift={info.get('target_lift', 0.0):.4f} success={info.get('is_success', False)}"
            )
            return {
                "done": done,
                "reward": reward,
                "info": info,
                "completed_phases": phase_index + 1,
            }

        if advance_phase_if_ready(env, phase, phase_started_step):
            phase_index += 1
            if phase_index >= len(phases):
                print(
                    f"[episode {episode_index}] scripted sequence complete "
                    f"reward={reward:.4f} lift={info.get('target_lift', 0.0):.4f}"
                )
                return {
                    "done": done,
                    "reward": reward,
                    "info": info,
                    "completed_phases": phase_index,
                }
            phase = phases[phase_index]
            phase_started = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run")
    parser.add_argument("--seed", type=int, default=None, help="Seed for environment randomization")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for CSV logs",
    )
    parser.add_argument(
        "--video-name",
        type=str,
        default="handcrafted_grasp.mp4",
        help="Filename for the saved RGB rollout video",
    )
    parser.add_argument(
        "--video-fps",
        type=float,
        default=20.0,
        help="Frames per second for the saved rollout video",
    )
    parser.add_argument(
        "--camera-width",
        type=int,
        default=320,
        help="Camera render width for saved debug video frames",
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=240,
        help="Camera render height for saved debug video frames",
    )
    parser.add_argument(
        "--overlay-pointcloud",
        action="store_true",
        help="Overlay the sampled point cloud on saved video frames. Disabled by default because it makes scripted debugging much slower.",
    )
    parser.add_argument(
        "--approach-height",
        type=float,
        default=0.12,
        help="Meters above the can center for the pre-grasp approach",
    )
    parser.add_argument(
        "--grasp-height",
        type=float,
        default=0.03,
        help="Meters above the can center for the grasp target",
    )
    parser.add_argument(
        "--lift-height",
        type=float,
        default=0.25,
        help="Meters above the can center for the lift target",
    )
    parser.add_argument(
        "--move-timeout",
        type=float,
        default=30.0,
        help="Seconds allowed for each arm motion phase",
    )
    parser.add_argument(
        "--gripper-dwell",
        type=float,
        default=0.65,
        help="Seconds to dwell after the open-gripper command",
    )
    parser.add_argument(
        "--grasp-dwell",
        type=float,
        default=1.0,
        help="Seconds to dwell after the close-gripper command",
    )
    parser.add_argument(
        "--settle-dwell",
        type=float,
        default=0.5,
        help="Seconds to hold the closed grasp before lifting",
    )
    parser.add_argument(
        "--arm-step-size",
        type=float,
        default=0.1,
        help="Joint interpolation step size passed to the position controller",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "handcrafted_grasp_log.csv"
    video_path = run_dir / args.video_name

    env = build_environment(
        args.seed,
        camera_height=args.camera_height,
        camera_width=args.camera_width,
        overlay_pointcloud=args.overlay_pointcloud,
    )

    print(f"Logging to {csv_path}")
    print(f"Saving video to {video_path}")
    try:
        with csv_path.open("w", newline="") as handle, imageio.get_writer(
            video_path,
            fps=args.video_fps,
        ) as video_writer:
            writer: Optional[csv.DictWriter] = None
            summaries = []

            class StreamingWriter:
                def writerow(self, row: Dict[str, Any]) -> None:
                    nonlocal writer
                    if writer is None:
                        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
                        writer.writeheader()
                    writer.writerow(row)
                    handle.flush()

            for episode_index in range(args.episodes):
                summary = run_episode(
                    env,
                    episode_index,
                    StreamingWriter(),
                    args,
                    video_writer=video_writer,
                )
                summaries.append(summary)

            for index, summary in enumerate(summaries):
                info = summary["info"]
                print(
                    f"episode={index} reward={summary['reward']:.4f} completed_phases={summary['completed_phases']} "
                    f"success={info.get('is_success', False)} lift={info.get('target_lift', 0.0):.4f}"
                )
    finally:
        env.close()


if __name__ == "__main__":
    main()