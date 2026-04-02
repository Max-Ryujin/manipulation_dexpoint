"""Gym Environment wrapper for Franka robot with point cloud observations."""

import math
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import gym
import imageio
import numpy as np
from gym import spaces

from manipulation import FrankaEnvironment
from manipulation.perception import MujocoCamera
from pointcloud_observation import (
    center_and_sample_pointcloud,
    collect_fused_pointcloud,
    get_workspace_configuration,
)
from ycb_scene import (
    DEFAULT_YCB_OBJECT_ROOT,
    ensure_single_object_ycb_scene,
    load_ycb_object_spec,
)


class FrankaGymEnvironment(gym.Env):
    """
    Gym wrapper for FrankaEnvironment with point cloud observations.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        xml_path: Optional[str] = None,
        task_name: str = "grasping",
        num_points: int = 512,
        camera_height: int = 480,
        camera_width: int = 640,
        camera_names: Optional[list] = None,
        rate: float = 200.0,
        frame_skip: int = 10,
        randomize_target_pose: bool = True,
        ycb_object_root: str = DEFAULT_YCB_OBJECT_ROOT.as_posix(),
        target_scale: float = 0.9,
        target_body_name: str = "target_object",
        target_drop_height_range: Tuple[float, float] = (0.0, 0.03),
        goal_height_range: Tuple[float, float] = (0.1, 0.3),
        visualize_pointclouds: bool = False,
        pointcloud_point_size: int = 1,
        pointcloud_alpha: float = 0.2,
        capture_episode_frames: bool = False,
    ):
        """
        Initialize the Franka gym environment.

        Args:
            xml_path: Path to MuJoCo XML scene file
            task_name: Task identifier
            num_points: Target number of points in merged point cloud
            camera_height: Camera render height (pixels)
            camera_width: Camera render width (pixels)
            camera_names: List of camera names to use. If None, uses defaults.
            rate: Simulation frequency (Hz)
            randomize_target_pose: Whether to randomize can placement each episode
            ycb_object_root: Path to the YCB object directory to load into the scene
            target_scale: Uniform scale applied to the target object's visual and collision geometry
            target_body_name: Body name used for the grasp target inside MuJoCo
            target_drop_height_range: Extra randomized drop height range above table
            visualize_pointclouds: Whether to visualize point clouds in render output
            pointcloud_point_size: Size of rendered point cloud dots in pixels (default: 4)
            pointcloud_alpha: Transparency of point cloud overlay (0-1, default: 0.7)
            capture_episode_frames: Whether to capture RGB frames on every step for debug videos
        """
        self.task_name = task_name
        self.num_points = num_points
        self.camera_height = camera_height
        self.camera_width = camera_width
        self.rate = rate
        self.frame_skip = frame_skip
        self.randomize_target_pose = randomize_target_pose
        self.visualize_pointclouds = visualize_pointclouds
        self.pointcloud_point_size = pointcloud_point_size
        self.pointcloud_alpha = pointcloud_alpha
        self.capture_episode_frames = capture_episode_frames
        self.robot_dof = 8
        self.joint_dim = self.robot_dof
        self.target_scale = float(target_scale)
        self.target_body_name = target_body_name
        self.target_drop_height_range = target_drop_height_range
        self.goal_height_range = goal_height_range
        self._rng = np.random.default_rng()
        self.goal_position: np.ndarray = np.zeros(3, dtype=np.float32)

        self.ycb_object_root = Path(ycb_object_root).resolve()
        self.target_spec = load_ycb_object_spec(
            self.ycb_object_root, scale=self.target_scale
        )

        if xml_path is None:
            scene_xml_path, _ = ensure_single_object_ycb_scene(
                self.ycb_object_root,
                scale=self.target_scale,
            )
            self.xml_path = scene_xml_path.as_posix()
        else:
            self.xml_path = xml_path

        # Set default camera names if not provided
        if camera_names is None:
            self.camera_names = ["top_camera", "side_camera", "front_camera"]
        else:
            self.camera_names = camera_names

        # Initialize environment
        self.env = FrankaEnvironment(
            self.xml_path, rate=rate, frame_skip=self.frame_skip
        )
        self.camera = MujocoCamera(self.env, width=camera_width, height=camera_height)
        self.hand_body_id = self.env.model.body("hand").id
        self.attachment_site_id = self.env.model.site("attachment_site").id

        self.ctrl_min = self.env.model.actuator_ctrlrange[: self.robot_dof, 0]
        self.ctrl_max = self.env.model.actuator_ctrlrange[: self.robot_dof, 1]

        self.workspace_bounds, self.workspace_info = get_workspace_configuration(
            self.env.model
        )
        self.table_height = float(self.workspace_info["table_height"])
        self.target_rest_height = self.table_height + float(
            self.target_spec.half_extents[2]
        )
        self.success_lift_height = 0.08

        self.env.add_collision_exception(self.target_body_name)
        self._sample_goal_position()

        # Observation space: Dict with point cloud + joint state + ee_position + goal_position
        self.observation_space = spaces.Dict(
            {
                "pointcloud": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(num_points, 3), dtype=np.float32
                ),
                "joint_state": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.joint_dim,), dtype=np.float32
                ),
                "ee_position": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
                ),
                "goal_position": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
                ),
            }
        )

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.robot_dof,), dtype=np.float32
        )

        # Task-specific attributes
        self.task_config = {}
        self.max_episode_steps = 800
        self.step_count = 0

        # Frame buffering for edge case debugging
        self.frame_buffer = deque(maxlen=self.max_episode_steps)
        self.episode_frames = []
        self._last_valid_observation = self._create_empty_observation()
        self._last_pointcloud_empty = False
        self._last_pointcloud_size = 0

    def reset(self) -> Dict[str, np.ndarray]:
        """Reset environment and return initial observation."""
        self.env.reset()
        self.env.clear_collision_exceptions()
        self.env.add_collision_exception(self.target_body_name)

        if self.randomize_target_pose:
            self._reset_target_pose()

        self._sample_goal_position()

        # Step a few times to settle simulation
        for _ in range(10):
            self.env.step()

        target_pos = self.get_target_position()
        self.target_rest_height = float(target_pos[2])

        self.step_count = 0

        # Clear frame buffer for new episode
        self.frame_buffer.clear()
        self.episode_frames = []
        self._last_valid_observation = self._create_empty_observation()
        self._last_pointcloud_empty = False
        self._last_pointcloud_size = 0

        obs = self._get_observation()
        return obs

    def seed(self, seed: Optional[int] = None) -> List[Optional[int]]:
        """Seed the environment RNG used for object and goal sampling."""
        self._rng = np.random.default_rng(seed)
        return [seed]

    def _create_empty_observation(self) -> Dict[str, np.ndarray]:
        return {
            "pointcloud": np.zeros((self.num_points, 3), dtype=np.float32),
            "joint_state": np.zeros((self.joint_dim,), dtype=np.float32),
            "ee_position": np.zeros((3,), dtype=np.float32),
            "goal_position": np.zeros((3,), dtype=np.float32),
        }

    def _reset_target_pose(self) -> None:
        bounds = self.workspace_bounds
        placement_margin = float(self.target_spec.placement_radius + 0.02)

        min_x = bounds["min_x"] + placement_margin
        max_x = bounds["max_x"] - placement_margin
        min_y = bounds["min_y"] + placement_margin
        max_y = bounds["max_y"] - placement_margin

        if min_x >= max_x:
            x = 0.5 * (bounds["min_x"] + bounds["max_x"])
        else:
            x = float(self._rng.uniform(min_x, max_x))

        if min_y >= max_y:
            y = 0.5 * (bounds["min_y"] + bounds["max_y"])
        else:
            y = float(self._rng.uniform(min_y, max_y))

        drop_height = float(
            self._rng.uniform(
                self.target_drop_height_range[0], self.target_drop_height_range[1]
            )
        )
        z = self.table_height + float(self.target_spec.half_extents[2]) + drop_height
        yaw = float(self._rng.uniform(-math.pi, math.pi))
        quat = np.array(
            [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)], dtype=np.float64
        )

        self.env.set_object_pose(
            self.target_body_name,
            pos=np.array([x, y, z], dtype=np.float64),
            quat=quat,
        )
        self.env.reset_velocities()
        self.env.forward()

    def _sample_goal_position(self) -> None:
        """Randomly sample a goal XYZ the robot should bring the object to."""
        bounds = self.workspace_bounds
        x = float(self._rng.uniform(bounds["min_x"], bounds["max_x"]))
        y = float(self._rng.uniform(bounds["min_y"], bounds["max_y"]))
        z = self.table_height + float(
            self._rng.uniform(self.goal_height_range[0], self.goal_height_range[1])
        )
        self.goal_position = np.array([x, y, z], dtype=np.float32)

    def get_target_position(self) -> np.ndarray:
        return self.env.get_object_position(self.target_body_name)

    def get_target_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.env.get_object_pose(self.target_body_name)

    def get_finger_positions(self) -> Tuple[np.ndarray, np.ndarray]:
        left_finger_pos = self.env.get_object_position("left_finger").astype(np.float32)
        right_finger_pos = self.env.get_object_position("right_finger").astype(
            np.float32
        )
        return left_finger_pos, right_finger_pos

    def get_gripper_opening_width(self) -> float:
        left_finger_pos, right_finger_pos = self.get_finger_positions()
        return float(np.linalg.norm(right_finger_pos - left_finger_pos))

    def get_gripper_actuator_force(self) -> float:
        return float(self.env.data.actuator_force[7])

    def get_end_effector_position(self) -> np.ndarray:
        return self.env.data.site_xpos[self.attachment_site_id].astype(np.float32)

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        """
        Execute one environment step.

        Args:
            action: 8D array of normalized joint position commands

        Returns:
            obs: Current observation (Dict with pointcloud and joint_state)
            reward: Task-specific reward
            done: Whether episode is finished
            info: Info dict with task-specific metrics
        """

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape != (self.robot_dof,):
            raise ValueError(
                f"Expected action shape {(self.robot_dof,)}, got {action.shape}"
            )

        arm_action = action[:7]
        gripper_action = action[7]

        current_arm_qpos = self.env.data.qpos[:7].copy()

        rl_dt = self.env.model.opt.timestep * self.env.frame_skip

        target_arm_qpos = current_arm_qpos + (1.5 * arm_action * rl_dt)

        target_gripper_ctrl = ((gripper_action + 1.0) / 2.0) * (
            self.ctrl_max[7] - self.ctrl_min[7]
        ) + self.ctrl_min[7]
        unclipped_target_ctrl = np.append(target_arm_qpos, target_gripper_ctrl)
        target_ctrl = np.clip(unclipped_target_ctrl, self.ctrl_min, self.ctrl_max)
        action_metrics = {
            "action_l2": float(np.linalg.norm(action)),
            "arm_action_l2": float(np.linalg.norm(arm_action)),
            "gripper_action": float(gripper_action),
            "action_saturation_fraction": float(np.mean(np.abs(action) >= 0.95)),
            "ctrl_clip_fraction": float(
                np.mean(~np.isclose(target_ctrl, unclipped_target_ctrl, atol=1e-6))
            ),
            "gripper_ctrl_command": float(target_ctrl[7]),
        }
        self.env.data.ctrl[:8] = target_ctrl

        # Step simulation
        self.env.step()
        self.step_count += 1

        if self.capture_episode_frames:
            frame = self.render(mode="rgb_array")
            if frame is not None:
                self.episode_frames.append(frame)

        # Get observation
        obs = self._get_observation()

        if self._last_pointcloud_empty:
            info = {
                "pointcloud_empty": True,
                "pointcloud_size": self._last_pointcloud_size,
                "contact_count": int(self.env.data.ncon),
                "max_joint_speed": float(
                    np.max(np.abs(self.env.data.qvel[: self.robot_dof]))
                ),
                "episode_failure_reason": "empty_pointcloud",
            }
            info.update(action_metrics)
            done = True
            reward = 0.0
            self.episode_frames = []
            return obs, reward, done, info

        # Compute task-specific reward and done signal
        reward, done, info = self._compute_reward_and_done()
        info.setdefault("pointcloud_empty", False)
        info["pointcloud_size"] = self._last_pointcloud_size
        info["contact_count"] = int(self.env.data.ncon)
        info["max_joint_speed"] = float(
            np.max(np.abs(self.env.data.qvel[: self.robot_dof]))
        )
        info.update(action_metrics)

        # Episode termination on max steps
        if self.step_count >= self.max_episode_steps:
            done = True
            info["step_limit_reached"] = True

        # Clear frames on successful episode completion
        if done:
            self.episode_frames = []

        return obs, reward, done, info

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Extract point cloud and proprioceptive observations from all cameras."""
        merged_points = collect_fused_pointcloud(
            self.env,
            self.camera,
            camera_names=self.camera_names,
            num_points=self.num_points,
            camera_height=self.camera_height,
            camera_width=self.camera_width,
            workspace_bounds=self.workspace_bounds,
            table_height=self.table_height,
        )
        self._last_pointcloud_size = int(len(merged_points))

        if len(merged_points) == 0:
            hand_pos = self.env.data.xpos[self.hand_body_id].copy()
            max_joint_speed = float(
                np.max(np.abs(self.env.data.qvel[: self.robot_dof]))
            )
            print(
                "[FrankaGymEnvironment] no valid points "
                f"step={self.step_count} hand_pos={np.array2string(hand_pos, precision=3)} "
                f"max_joint_speed={max_joint_speed:.3f} contacts={int(self.env.data.ncon)}"
            )

            # Save debug images and video of current episode
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_dir = Path(f"debug_no_points_{timestamp}")
            debug_dir.mkdir(exist_ok=True)

            # Save individual frames from each camera
            for name in self.camera_names:
                image = self.render(mode=f"camera_{name}")
                cv2.imwrite(str(debug_dir / f"debug_{name}.png"), image)

            # Save episode video if frames were captured
            if len(self.episode_frames) > 0:
                video_path = debug_dir / f"episode_video_{timestamp}.mp4"
                imageio.mimwrite(str(video_path), self.episode_frames, fps=30)

            self._last_pointcloud_empty = True
            joint_state = self.env.data.qpos[: self.robot_dof].astype(np.float32)
            ee_position = self.get_end_effector_position()
            fallback_obs = {
                "pointcloud": self._last_valid_observation["pointcloud"].copy(),
                "joint_state": joint_state,
                "ee_position": ee_position,
                "goal_position": self.goal_position.copy(),
            }
            self._last_valid_observation = fallback_obs
            return fallback_obs

        pointcloud = center_and_sample_pointcloud(merged_points, self.num_points)
        self._last_pointcloud_empty = False

        # Extract joint state (positions only)
        joint_state = self.env.data.qpos[: self.robot_dof].astype(np.float32)

        # Extract end-effector position and stable goal position
        ee_position = self.get_end_effector_position()

        obs = {
            "pointcloud": pointcloud,
            "joint_state": joint_state,
            "ee_position": ee_position,
            "goal_position": self.goal_position.copy(),
        }
        self._last_valid_observation = {
            "pointcloud": pointcloud.copy(),
            "joint_state": joint_state.copy(),
            "ee_position": ee_position.copy(),
            "goal_position": self.goal_position.copy(),
        }
        return obs

    def _compute_reward_and_done(self) -> Tuple[float, bool, Dict[str, Any]]:
        """
        Compute task-specific reward and done signal.

        Returns:
            reward: Scalar reward value
            done: Boolean done flag
            info: Info dictionary with task metrics
        """
        reward = 0.0
        done = False
        info = {}

        if self.task_config.get("reward_fn"):
            reward, done, info = self.task_config["reward_fn"](self)

        return reward, done, info

    def configure_task(self, task_config: Dict[str, Any]):
        """
        Configure task-specific parameters and reward functions.

        Args:
            task_config: Dictionary with task configuration:
                - 'max_episode_steps': int
                - 'reward_fn': callable(env) -> (reward, done, info)
                - Other task-specific parameters
        """
        self.task_config = task_config
        self.max_episode_steps = task_config.get("max_episode_steps", 500)
        self.randomize_target_pose = task_config.get(
            "randomize_target_pose", self.randomize_target_pose
        )

    def set_task_reward(self, reward_fn):
        """Set a custom reward function."""
        self.task_config["reward_fn"] = reward_fn

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """
        Render the environment.

        Args:
            mode: Rendering mode. Options:
                - "human": Launch interactive MuJoCo viewer
                - "rgb_array": Return RGB image from first camera
                - "camera_<name>": Return RGB from specific camera

        Returns:
            RGB image as numpy array (HxWx3) if mode is "rgb_array" or "camera_*",
            None if mode is "human"
        """
        if mode == "human":
            # Launch interactive MuJoCo viewer
            if self.env.viewer is None:
                self.env.launch_viewer()
            return None

        elif mode == "rgb_array":
            rgb = self.camera.render_rgb(
                self.camera_names[0],
                width=self.camera_width,
                height=self.camera_height,
            )
            return rgb

        elif mode.startswith("camera_"):
            camera_name = mode[7:]  # Remove "camera_" prefix
            rgb = self.camera.render_rgb(
                camera_name, width=self.camera_width, height=self.camera_height
            )
            return rgb

        else:
            print(f"Warning: Unknown render mode '{mode}'")
            return None

    def render_with_pointcloud(
        self, camera_name: Optional[str] = None, mode: str = "rgb_array"
    ) -> Optional[np.ndarray]:
        if not self.visualize_pointclouds:
            return self.render(mode=mode)

        if camera_name is None:
            camera_name = self.camera_names[1]

        try:
            # Get RGB image
            rgb = self.camera.render_rgb(
                camera_name,
                width=self.camera_width,
                height=self.camera_height,
            )

            if rgb is None:
                return None

            # Get point cloud for this camera
            # Use only the points from this specific camera to avoid confusion
            samples_per_camera = 50 * self.num_points
            points, colors, pixels, depths = self.camera.get_pointcloud(
                camera_name,
                width=self.camera_width,
                height=self.camera_height,
                num_samples=samples_per_camera,
                min_depth=0.1,
                max_depth=3.0,
            )

            min_x = self.workspace_bounds["min_x"]
            max_x = self.workspace_bounds["max_x"]
            min_y = self.workspace_bounds["min_y"]
            max_y = self.workspace_bounds["max_y"]
            min_z = self.table_height

            hand_id = self.env.model.body("hand").id
            hand_pos = self.env.data.xpos[hand_id]
            sphere_radius = 0.12

            workspace_mask = (
                (points[:, 0] >= min_x)
                & (points[:, 0] <= max_x)
                & (points[:, 1] >= min_y)
                & (points[:, 1] <= max_y)
                & (points[:, 2] > (min_z + 0.01))
                & (points[:, 2] < min_z + 0.2)
            )

            dist_to_hand = np.linalg.norm(points - hand_pos, axis=1)
            hand_mask = dist_to_hand < sphere_radius

            # Combine both regions
            mask = workspace_mask | hand_mask

            points = points[mask]
            colors = colors[mask]
            pixels = pixels[mask]
            depths = depths[mask]

            # make colors darker for better visibility
            colors = (colors * 0.6).astype(np.uint8)

            order = np.argsort(depths)

            # Create overlay image and a standalone point-cloud canvas.
            overlay = rgb.copy().astype(float)
            pointcloud_only = np.full_like(rgb, 255, dtype=np.uint8)

            # Draw points in depth order
            for idx in order:
                u, v = pixels[idx]

                cv2.circle(
                    overlay,
                    (int(u), int(v)),
                    self.pointcloud_point_size,
                    tuple(int(c) for c in colors[idx]),
                    -1,
                )
                cv2.circle(
                    pointcloud_only,
                    (int(u), int(v)),
                    self.pointcloud_point_size,
                    tuple(int(c) for c in colors[idx]),
                    -1,
                )

            # Blend overlay with original RGB using alpha
            rgb_out = (
                self.pointcloud_alpha * overlay
                + (1 - self.pointcloud_alpha) * rgb.astype(float)
            ).astype(np.uint8)

            return np.concatenate([rgb_out, pointcloud_only], axis=1)

        except Exception as e:
            print(f"Warning: Failed to render point cloud overlay: {e}")
            return self.render(mode=mode)

    def close(self):
        """Close environment and viewer."""
        if self.env.viewer is not None:
            self.env.viewer.close()

        if self.camera is not None:
            self.camera.close()
