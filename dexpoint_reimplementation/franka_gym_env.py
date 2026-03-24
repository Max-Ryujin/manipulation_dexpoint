"""Gym Environment wrapper for Franka robot with point cloud observations."""

import numpy as np
import gym
from gym import spaces
from typing import Dict, Tuple, Optional, Any, List
from pathlib import Path
import cv2
from collections import deque
from datetime import datetime
import imageio

from manipulation import FrankaEnvironment
from manipulation.perception import MujocoCamera
from manipulation.symbolic.domains.blocks import BlocksDomain, BlocksStateManager
from pointcloud_observation import (
    collect_fused_pointcloud,
    center_and_sample_pointcloud,
)


class FrankaGymEnvironment(gym.Env):
    """
    Gym wrapper for FrankaEnvironment with point cloud observations.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        xml_path: str,
        task_name: str = "blocks_world",
        num_points: int = 512,
        camera_height: int = 480,
        camera_width: int = 640,
        camera_names: Optional[list] = None,
        rate: float = 200.0,
        frame_skip: int = 10,
        randomize_blocks: bool = True,
        n_blocks_to_place: int = 3,
        visualize_pointclouds: bool = False,
        pointcloud_point_size: int = 1,
        pointcloud_alpha: float = 0.2,
    ):
        """
        Initialize the Franka gym environment.

        Args:
            xml_path: Path to MuJoCo XML scene file
            task_name: Task identifier ('blocks_world' or 'grasping')
            num_points: Target number of points in merged point cloud
            camera_height: Camera render height (pixels)
            camera_width: Camera render width (pixels)
            camera_names: List of camera names to use. If None, uses defaults.
            rate: Simulation frequency (Hz)
            randomize_blocks: Whether to randomize block placement each episode
            n_blocks_to_place: Number of blocks to place on table (default: 2)
            visualize_pointclouds: Whether to visualize point clouds in render output
            pointcloud_point_size: Size of rendered point cloud dots in pixels (default: 4)
            pointcloud_alpha: Transparency of point cloud overlay (0-1, default: 0.7)
        """
        self.xml_path = xml_path
        self.task_name = task_name
        self.num_points = num_points
        self.camera_height = camera_height
        self.camera_width = camera_width
        self.rate = rate
        self.frame_skip = frame_skip
        self.randomize_blocks = randomize_blocks
        self.n_blocks_to_place = n_blocks_to_place
        self.visualize_pointclouds = visualize_pointclouds
        self.pointcloud_point_size = pointcloud_point_size
        self.pointcloud_alpha = pointcloud_alpha
        self.robot_dof = 8
        self.joint_dim = self.robot_dof

        # Set default camera names if not provided
        if camera_names is None:
            self.camera_names = ["top_camera", "side_camera", "front_camera"]
        else:
            self.camera_names = camera_names

        # Initialize environment
        self.env = FrankaEnvironment(xml_path, rate=rate, frame_skip=self.frame_skip)
        self.camera = MujocoCamera(self.env, width=camera_width, height=camera_height)

        self.ctrl_min = self.env.model.actuator_ctrlrange[: self.robot_dof, 0]
        self.ctrl_max = self.env.model.actuator_ctrlrange[: self.robot_dof, 1]

        # Initialize blocks domain and state manager for randomized block placement
        self.domain = BlocksDomain(self.env.model)
        self.state_manager = BlocksStateManager(self.domain, self.env)

        # Observation space: Dict with point cloud + joint state
        self.observation_space = spaces.Dict(
            {
                "pointcloud": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(num_points, 3), dtype=np.float32
                ),
                "joint_state": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.joint_dim,), dtype=np.float32
                ),
            }
        )

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.robot_dof,), dtype=np.float32
        )

        # Task-specific attributes
        self.task_config = {}
        self.max_episode_steps = 1000
        self.step_count = 0

        # Frame buffering for edge case debugging
        self.frame_buffer = deque(maxlen=self.max_episode_steps)
        self.episode_frames = []

    def reset(self) -> Dict[str, np.ndarray]:
        """Reset environment and return initial observation."""
        self.env.reset()

        # Load blocks based on task configuration
        if self.state_manager is not None:

            self.state_manager.sample_random_state(
                n_blocks=self.n_blocks_to_place, include_platforms=True
            )

        # Step a few times to settle simulation
        for _ in range(10):
            self.env.step()

        self.step_count = 0

        # Clear frame buffer for new episode
        self.frame_buffer.clear()
        self.episode_frames = []

        obs = self._get_observation()
        return obs

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

        arm_action = action[:7]
        gripper_action = action[7]

        current_arm_qpos = self.env.data.qpos[:7].copy()

        rl_dt = self.env.model.opt.timestep * self.env.frame_skip

        target_arm_qpos = current_arm_qpos + (1.5 * arm_action * rl_dt)

        target_gripper_ctrl = ((gripper_action + 1.0) / 2.0) * (
            self.ctrl_max[7] - self.ctrl_min[7]
        ) + self.ctrl_min[7]
        target_ctrl = np.append(target_arm_qpos, target_gripper_ctrl)

        target_ctrl = np.clip(target_ctrl, self.ctrl_min, self.ctrl_max)
        self.env.data.ctrl[:8] = target_ctrl

        # Step simulation
        self.env.step()
        self.step_count += 1

        # Capture frame for episode buffer
        frame = self.render(mode="rgb_array")
        if frame is not None:
            self.episode_frames.append(frame)

        # Get observation
        obs = self._get_observation()

        # Compute task-specific reward and done signal
        reward, done, info = self._compute_reward_and_done()

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
        bounds = self.domain.get_working_bounds()
        info = self.domain.get_domain_info()
        merged_points = collect_fused_pointcloud(
            self.env,
            self.camera,
            camera_names=self.camera_names,
            num_points=self.num_points,
            camera_height=self.camera_height,
            camera_width=self.camera_width,
            workspace_bounds=bounds,
            table_height=info["table_height"],
        )

        if len(merged_points) == 0:
            hand_id = self.env.model.body("hand").id
            hand_pos = self.env.data.xpos[hand_id].copy()
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

            return None

        pointcloud = center_and_sample_pointcloud(merged_points, self.num_points)

        # Extract joint state (positions only)
        joint_state = self.env.data.qpos[: self.robot_dof].astype(np.float32)

        return {
            "pointcloud": pointcloud,
            "joint_state": joint_state,
        }

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
        self.n_blocks_to_place = task_config.get(
            "n_blocks_to_place", self.n_blocks_to_place
        )
        self.randomize_blocks = task_config.get(
            "randomize_blocks", self.randomize_blocks
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
            samples_per_camera = 15 * self.num_points
            points, colors, pixels, depths = self.camera.get_pointcloud(
                camera_name,
                width=self.camera_width,
                height=self.camera_height,
                num_samples=samples_per_camera,
                min_depth=0.1,
                max_depth=3.0,
            )

            bounds = self.domain.get_working_bounds()
            info = self.domain.get_domain_info()
            min_x = bounds["min_x"]
            max_x = bounds["max_x"]
            min_y = bounds["min_y"]
            max_y = bounds["max_y"]
            min_z = info["table_height"]

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

            # Create overlay image (copy of RGB)
            overlay = rgb.copy().astype(float)

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

            # Blend overlay with original RGB using alpha
            rgb_out = (
                self.pointcloud_alpha * overlay
                + (1 - self.pointcloud_alpha) * rgb.astype(float)
            ).astype(np.uint8)

            return rgb_out

        except Exception as e:
            print(f"Warning: Failed to render point cloud overlay: {e}")
            return self.render(mode=mode)

    def close(self):
        """Close environment and viewer."""
        if self.env.viewer is not None:
            self.env.viewer.close()

        if self.camera is not None:
            self.camera.close()
