"""Gym Environment wrapper for Franka robot with point cloud observations."""

import math
import os
import time
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
    collect_fused_pointcloud,
    collect_fused_pointcloud_for_training,
    get_default_camera_names,
    get_pointcloud_samples_per_camera,
    get_workspace_configuration,
    POINTCLOUD_OVERSAMPLE_FACTOR,
    sample_pointcloud,
)
from ycb_scene import (
    DEFAULT_YCB_OBJECT_ROOT,
    YCB_ASSET_SOURCE_YCB_SIM,
    ensure_single_object_ycb_scene,
    load_ycb_object_spec,
)


class FrankaGymEnvironment(gym.Env):
    """
    Gym wrapper for FrankaEnvironment with point cloud observations.
    """

    metadata = {"render_modes": []}

    @staticmethod
    def _quat_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
        quat = np.asarray(quat, dtype=np.float64)
        quat_norm = float(np.linalg.norm(quat))
        if quat_norm <= 0.0:
            return np.eye(3, dtype=np.float64)

        w, x, y, z = quat / quat_norm
        return np.array(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )

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
        ycb_asset_source: str = YCB_ASSET_SOURCE_YCB_SIM,
        target_scale: float = 1.0,
        target_body_name: str = "target_object",
        target_drop_height_range: Tuple[float, float] = (0.0, 0.03),
        goal_height_range: Tuple[float, float] = (0.1, 0.3),
        visualize_pointclouds: bool = False,
        pointcloud_point_size: int = 1,
        pointcloud_alpha: float = 0.2,
        capture_episode_frames: bool = False,
        use_depth_only_pointcloud: bool = False,
        ycb_object_names: Optional[List[str]] = None,
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
            ycb_asset_source: Which YCB asset source to use ('ycb_sim' or 'raw')
            target_scale: Uniform scale applied to the target object's visual and collision geometry
            target_body_name: Body name used for the grasp target inside MuJoCo
            target_drop_height_range: Extra randomized drop height range above table
            visualize_pointclouds: Whether to visualize point clouds in render output
            pointcloud_point_size: Size of rendered point cloud dots in pixels (default: 4)
            pointcloud_alpha: Transparency of point cloud overlay (0-1, default: 0.7)
            capture_episode_frames: Whether to capture RGB frames on every step for debug videos
            ycb_object_names: Optional list of YCB object folder names to randomly sample per
                episode (e.g. ["005_tomato_soup_can", "006_mustard_bottle"]).  When provided
                with more than one entry, the environment keeps a cached MuJoCo runtime for each
                object and swaps between them across episodes.  Names are resolved relative to
                the parent directory of *ycb_object_root*.
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
        self.use_depth_only_pointcloud = use_depth_only_pointcloud
        self.robot_dof = 8
        self.joint_dim = self.robot_dof
        self.target_scale = float(target_scale)
        self.ycb_asset_source = ycb_asset_source
        self.target_body_name = target_body_name
        self.target_drop_height_range = target_drop_height_range
        self.goal_height_range = goal_height_range
        self._rng = np.random.default_rng()
        self.goal_position: np.ndarray = np.zeros(3, dtype=np.float32)

        requested_object_names = list(ycb_object_names or [])
        resolved_ycb_object_root = Path(ycb_object_root).resolve()
        if requested_object_names:
            resolved_ycb_object_root = resolved_ycb_object_root.parent / requested_object_names[0]

        self.ycb_object_root = resolved_ycb_object_root
        self.target_spec = load_ycb_object_spec(
            self.ycb_object_root,
            scale=self.target_scale,
            source=self.ycb_asset_source,
        )

        if xml_path is None:
            scene_xml_path, _ = ensure_single_object_ycb_scene(
                self.ycb_object_root,
                scale=self.target_scale,
                source=self.ycb_asset_source,
            )
            self.xml_path = scene_xml_path.as_posix()
        else:
            self.xml_path = xml_path

        # Set default camera names if not provided
        if camera_names is None:
            self.camera_names = get_default_camera_names()
        else:
            self.camera_names = list(camera_names)

        # Initialize environment
        self.env = FrankaEnvironment(
            self.xml_path, rate=rate, frame_skip=self.frame_skip
        )
        self.env.set_viewer_marker_callback(self._get_viewer_debug_markers)
        self.camera = MujocoCamera(self.env, width=camera_width, height=camera_height)
        self.hand_body_id = self.env.model.body("hand").id
        self.attachment_site_id = self.env.model.site("attachment_site").id

        self.ctrl_min = self.env.model.actuator_ctrlrange[: self.robot_dof, 0]
        self.ctrl_max = self.env.model.actuator_ctrlrange[: self.robot_dof, 1]
        self.gripper_ctrl_min = float(self.ctrl_min[7])
        self.gripper_ctrl_max = float(self.ctrl_max[7])

        self.workspace_bounds, self.workspace_info = get_workspace_configuration(
            self.env.model
        )
        self.table_height = float(self.workspace_info["table_height"])
        self.table_body_name = str(self.workspace_info["table_body_name"])
        self.target_id = int(self.env.model.body(self.target_body_name).id)
        self.table_body_id = int(self.env.model.body(self.table_body_name).id)
        self.target_rest_height = self.table_height + float(
            self.target_spec.rest_offset_z
        )
        self.success_lift_height = 0.08
        self.failure_penalty = -1.0
        self.failure_xy_margin = 0.05
        self.failure_z_margin = 0.01

        # Task-specific attributes must exist before any initialization code
        # consults task settings, including the initial goal sampling path.
        self.task_config = {}
        self.max_episode_steps = 200
        self.step_count = 0
        self.training_num_timesteps = 0
        self.training_n_updates = 0

        self.observation_space = self._build_observation_space()

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.robot_dof,), dtype=np.float32
        )

        self._last_valid_observation = self._create_empty_observation()
        self._last_pointcloud_empty = False
        self._last_pointcloud_size = 0
        self.pointcloud_timing_log_interval = 1000
        self._pointcloud_timing_totals: Dict[str, float] = {}
        self._pointcloud_timing_count = 0

        # Build per-episode object pool for multi-object training.
        # Each entry: {"name": str, "xml_path": str, "object_spec": YCBObjectSpec}
        self._ycb_object_pool: Optional[List[Dict]] = None
        self._object_runtime_cache: Dict[int, Dict[str, Any]] = {}
        self._active_object_pool_index: int = 0
        self._fixed_object_pool_index: Optional[int] = None
        if requested_object_names and len(requested_object_names) > 1:
            pool: List[Dict] = []
            for index, name in enumerate(requested_object_names):
                obj_root = self.ycb_object_root.parent / name
                xml_path_p, obj_spec = ensure_single_object_ycb_scene(
                    obj_root,
                    scale=self.target_scale,
                    source=self.ycb_asset_source,
                )
                pool.append(
                    {
                        "name": name,
                        "object_root": obj_root,
                        "xml_path": xml_path_p.as_posix(),
                        "object_spec": obj_spec,
                    }
                )
            self._ycb_object_pool = pool
            self._active_object_pool_index = 0
            self._object_runtime_cache[0] = self._capture_current_runtime_bundle()

    # ------------------------------------------------------------------
    # Multi-object helpers
    # ------------------------------------------------------------------

    def _capture_current_runtime_bundle(self) -> Dict[str, Any]:
        """Capture the currently active MuJoCo runtime so it can be reused later."""
        return {
            "env": self.env,
            "camera": self.camera,
            "hand_body_id": self.hand_body_id,
            "attachment_site_id": self.attachment_site_id,
            "ctrl_min": self.ctrl_min,
            "ctrl_max": self.ctrl_max,
            "gripper_ctrl_min": self.gripper_ctrl_min,
            "gripper_ctrl_max": self.gripper_ctrl_max,
            "workspace_bounds": self.workspace_bounds,
            "workspace_info": self.workspace_info,
            "table_height": self.table_height,
            "table_body_name": self.table_body_name,
            "target_id": self.target_id,
            "table_body_id": self.table_body_id,
            "target_spec": self.target_spec,
            "target_rest_height": self.target_rest_height,
            "ycb_object_root": self.ycb_object_root,
            "xml_path": self.xml_path,
        }

    def _create_runtime_bundle(self, pool_index: int) -> Dict[str, Any]:
        """Create a reusable MuJoCo runtime bundle for one object in the pool."""
        entry = self._ycb_object_pool[pool_index]
        env = FrankaEnvironment(entry["xml_path"], rate=self.rate, frame_skip=self.frame_skip)
        env.set_viewer_marker_callback(self._get_viewer_debug_markers)
        camera = MujocoCamera(env, width=self.camera_width, height=self.camera_height)
        workspace_bounds, workspace_info = get_workspace_configuration(env.model)
        table_height = float(workspace_info["table_height"])
        table_body_name = str(workspace_info["table_body_name"])
        ctrl_min = env.model.actuator_ctrlrange[: self.robot_dof, 0]
        ctrl_max = env.model.actuator_ctrlrange[: self.robot_dof, 1]
        return {
            "env": env,
            "camera": camera,
            "hand_body_id": env.model.body("hand").id,
            "attachment_site_id": env.model.site("attachment_site").id,
            "ctrl_min": ctrl_min,
            "ctrl_max": ctrl_max,
            "gripper_ctrl_min": float(ctrl_min[7]),
            "gripper_ctrl_max": float(ctrl_max[7]),
            "workspace_bounds": workspace_bounds,
            "workspace_info": workspace_info,
            "table_height": table_height,
            "table_body_name": table_body_name,
            "target_id": int(env.model.body(self.target_body_name).id),
            "table_body_id": int(env.model.body(table_body_name).id),
            "target_spec": entry["object_spec"],
            "target_rest_height": table_height + float(entry["object_spec"].rest_offset_z),
            "ycb_object_root": entry["object_root"],
            "xml_path": entry["xml_path"],
        }

    def _activate_runtime_bundle(self, bundle: Dict[str, Any], pool_index: int) -> None:
        """Swap this wrapper onto a cached MuJoCo runtime bundle."""
        self.env = bundle["env"]
        self.camera = bundle["camera"]
        self.hand_body_id = bundle["hand_body_id"]
        self.attachment_site_id = bundle["attachment_site_id"]
        self.ctrl_min = bundle["ctrl_min"]
        self.ctrl_max = bundle["ctrl_max"]
        self.gripper_ctrl_min = bundle["gripper_ctrl_min"]
        self.gripper_ctrl_max = bundle["gripper_ctrl_max"]
        self.workspace_bounds = bundle["workspace_bounds"]
        self.workspace_info = bundle["workspace_info"]
        self.table_height = bundle["table_height"]
        self.table_body_name = bundle["table_body_name"]
        self.target_id = bundle["target_id"]
        self.table_body_id = bundle["table_body_id"]
        self.target_spec = bundle["target_spec"]
        self.target_rest_height = bundle["target_rest_height"]
        self.ycb_object_root = bundle["ycb_object_root"]
        self.xml_path = bundle["xml_path"]
        self._active_object_pool_index = pool_index

    def _close_runtime_bundle(self, bundle: Dict[str, Any]) -> None:
        """Release renderer and viewer resources held by a runtime bundle."""
        camera = bundle.get("camera")
        env = bundle.get("env")
        if camera is not None:
            camera.close()
        if env is not None:
            env.close()

    def _switch_runtime_for_object(self, pool_index: int) -> None:
        """Swap the active runtime to a cached object-specific MuJoCo bundle."""
        if self._ycb_object_pool is None:
            return
        bundle = self._object_runtime_cache.get(pool_index)
        if bundle is None:
            bundle = self._create_runtime_bundle(pool_index)
            self._object_runtime_cache[pool_index] = bundle
        self._activate_runtime_bundle(bundle, pool_index)

    def get_available_object_names(self) -> List[str]:
        """Return the object names available to this environment."""
        if self._ycb_object_pool is None:
            return [self.ycb_object_root.name]
        return [str(entry["name"]) for entry in self._ycb_object_pool]

    def get_active_object_name(self) -> str:
        """Return the currently loaded object's name."""
        return str(self.ycb_object_root.name)

    def set_fixed_object(self, object_name: Optional[str]) -> None:
        """Pin resets to a specific object name, or clear the override."""
        if object_name is None:
            self._fixed_object_pool_index = None
            return

        if self._ycb_object_pool is None:
            if object_name != self.ycb_object_root.name:
                raise ValueError(
                    f"Environment only has object '{self.ycb_object_root.name}', got '{object_name}'"
                )
            self._fixed_object_pool_index = None
            return

        for index, entry in enumerate(self._ycb_object_pool):
            if entry["name"] != object_name:
                continue
            self._fixed_object_pool_index = index
            if index != self._active_object_pool_index:
                self._switch_runtime_for_object(index)
            return

        available = ", ".join(self.get_available_object_names())
        raise ValueError(
            f"Unknown object '{object_name}'. Available objects: {available}"
        )

    def reset(self) -> Dict[str, np.ndarray]:
        """Reset environment and return initial observation."""
        # Pick a random object from the pool before resetting.
        if self._ycb_object_pool is not None:
            if self._fixed_object_pool_index is None:
                new_index = int(self._rng.integers(0, len(self._ycb_object_pool)))
            else:
                new_index = int(self._fixed_object_pool_index)
            if new_index != self._active_object_pool_index:
                self._switch_runtime_for_object(new_index)

        self.env.reset()
        self.env.clear_collision_exceptions()

        if self.randomize_target_pose:
            self._reset_target_pose()

        # Step a few times to settle simulation
        for _ in range(5):
            self.env.step()

        target_pos = self.get_target_position()
        self._reset_episode_goal_position(target_position=target_pos)

        self.step_count = 0

        # Clear frame buffer for new episode
        self._last_valid_observation = self._create_empty_observation()
        self._last_pointcloud_empty = False
        self._last_pointcloud_size = 0

        obs = self._get_observation()
        return obs

    def seed(self, seed: Optional[int] = None) -> List[Optional[int]]:
        """Seed the environment RNG used for object and goal sampling."""
        self._rng = np.random.default_rng(seed)
        return [seed]

    def _current_task_name(self) -> str:
        return str(self.task_config.get("task_name", self.task_name))

    def _task_uses_goal_position(self) -> bool:
        return self._current_task_name() in {"placing_v2", "placing_v3"}

    def _build_observation_space(self) -> spaces.Dict:
        observation_dict: Dict[str, spaces.Space] = {
            "pointcloud": spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.num_points, 3), dtype=np.float32
            ),
            "joint_state": spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.joint_dim,), dtype=np.float32
            ),
            "ee_position": spaces.Box(
                low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
            ),
            "target_position": spaces.Box(
                low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
            ),
        }
        if self._task_uses_goal_position():
            observation_dict["goal_position"] = spaces.Box(
                low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
            )
        return spaces.Dict(observation_dict)

    def _create_empty_observation(self) -> Dict[str, np.ndarray]:
        observation = {
            "pointcloud": np.zeros((self.num_points, 3), dtype=np.float32),
            "joint_state": np.zeros((self.joint_dim,), dtype=np.float32),
            "ee_position": np.zeros((3,), dtype=np.float32),
            "target_position": np.zeros((3,), dtype=np.float32),
        }
        if self._task_uses_goal_position():
            observation["goal_position"] = np.zeros((3,), dtype=np.float32)
        return observation

    def _reset_target_pose(self) -> None:
        bounds = self.workspace_bounds
        placement_margin = float(self.target_spec.placement_radius)

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
        z = self.table_height + float(self.target_spec.rest_offset_z) + drop_height
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

    def _reset_episode_goal_position(
        self, target_position: Optional[np.ndarray] = None
    ) -> None:
        if self._task_uses_goal_position():
            # Sample a random placing goal on the table inside the workspace.
            bounds = self.workspace_bounds
            placement_margin = float(self.target_spec.placement_radius)
            min_x = bounds["min_x"] + placement_margin
            max_x = bounds["max_x"] - placement_margin
            min_y = bounds["min_y"] + placement_margin
            max_y = bounds["max_y"] - placement_margin

            goal_z = self.table_height + float(self.target_spec.rest_offset_z)

            # Try to keep the goal at least 15 cm away from the object start
            # position so the policy has to actually move the object.
            min_separation = 0.15
            for _ in range(20):
                gx = float(self._rng.uniform(min_x, max_x))
                gy = float(self._rng.uniform(min_y, max_y))
                if target_position is None:
                    break
                if float(np.hypot(gx - target_position[0], gy - target_position[1])) >= min_separation:
                    break

            self.goal_position = np.array([gx, gy, goal_z], dtype=np.float32)
        else:
            self.goal_position = np.zeros(3, dtype=np.float32)

    def get_target_position(self) -> np.ndarray:
        # check if target_offset_site exists in the model, if so use its position as the target position, otherwise fall back to using the target body's position
        if self.env.model.site("target_offset_site") is not None:
            return (
                self.env.data.site_xpos[self.env.model.site("target_offset_site").id]
                .copy()
                .astype(np.float32)
            )
        else:
            return self.env.get_object_position(self.target_body_name)

    def _get_target_collision_half_extents(self) -> np.ndarray:
        collision_geom_type = str(self.target_spec.collision_geom_type)
        collision_size = np.asarray(self.target_spec.collision_size, dtype=np.float64)
        if collision_geom_type == "cylinder":
            radius = float(collision_size[0])
            half_height = float(collision_size[1])
            return np.array([radius, radius, half_height], dtype=np.float64)
        if collision_geom_type in {"box", "mesh"}:
            return collision_size.astype(np.float64)
        raise ValueError(
            f"Unsupported collision geometry type: {collision_geom_type}"
        )

    def get_target_bottom_height(self) -> float:
        target_pos, target_quat = self.env.get_object_pose(self.target_body_name)
        if target_pos is None or target_quat is None:
            target_pos = self.env.get_object_position(self.target_body_name)
            return float(target_pos[2] - self.target_spec.rest_offset_z)

        rotation = self._quat_to_rotation_matrix(target_quat)
        collision_center = np.asarray(self.target_spec.collision_pos, dtype=np.float64)
        half_extents = self._get_target_collision_half_extents()
        world_collision_center = np.asarray(target_pos, dtype=np.float64) + rotation @ collision_center
        vertical_radius = float(np.dot(np.abs(rotation[2, :]), half_extents))
        return float(world_collision_center[2] - vertical_radius)

    def get_target_lift_height(self) -> float:
        return float(max(self.get_target_bottom_height() - self.table_height, 0.0))

    def is_target_below_table(self, margin: float = 0.05) -> bool:
        return bool(self.get_target_bottom_height() < (self.table_height - margin))

    def _get_viewer_debug_markers(self) -> List[Dict[str, Any]]:
        target_pos = self.get_target_position().astype(np.float64)
        offset_target_pos = target_pos + np.array([0.0, 0.0, 0.01], dtype=np.float64)
        marker_radius = 0.008
        markers = [
            {
                "pos": target_pos,
                "size": np.array([marker_radius, marker_radius, marker_radius]),
                "rgba": np.array([0.95, 0.2, 0.2, 0.85]),
            },
            {
                "pos": offset_target_pos,
                "size": np.array([marker_radius, marker_radius, marker_radius]),
                "rgba": np.array([0.2, 0.85, 0.3, 0.85]),
            },
        ]
        if self._task_uses_goal_position():
            goal_pos = self.goal_position.astype(np.float64)
            markers.append(
                {
                    "pos": goal_pos,
                    "size": np.array([0.02, 0.02, 0.005]),
                    "rgba": np.array([0.1, 0.4, 1.0, 0.9]),
                }
            )
        return markers

    def get_target_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.env.get_object_pose(self.target_body_name)

    def get_finger_positions(self) -> Tuple[np.ndarray, np.ndarray]:
        left_finger_pos = self.env.get_object_position("left_finger").astype(np.float32)
        right_finger_pos = self.env.get_object_position("right_finger").astype(
            np.float32
        )
        return left_finger_pos, right_finger_pos

    def get_left_finger_position(self) -> np.ndarray:
        left, _ = self.get_finger_positions()
        return left

    def get_right_finger_position(self) -> np.ndarray:
        _, right = self.get_finger_positions()
        return right

    def get_finger_midpoint_position(self) -> np.ndarray:
        left_finger_pos, right_finger_pos = self.get_finger_positions()
        return (0.5 * (left_finger_pos + right_finger_pos)).astype(np.float32)

    def get_gripper_opening_width(self) -> float:
        left_finger_pos, right_finger_pos = self.get_finger_positions()
        return float(np.linalg.norm(right_finger_pos - left_finger_pos))

    def get_gripper_actuator_force(self) -> float:
        return float(self.env.data.actuator_force[7])

    def get_gripper_joint_position(self) -> float:
        return float(self.env.data.qpos[7])

    def get_gripper_open_fraction(self) -> float:
        ctrl_span = self.gripper_ctrl_max - self.gripper_ctrl_min
        if ctrl_span <= 1e-8:
            return 0.0
        joint_position = self.get_gripper_joint_position()
        return float(
            np.clip((joint_position - self.gripper_ctrl_min) / ctrl_span, 0.0, 1.0)
        )

    def get_end_effector_position(self) -> np.ndarray:
        return self.env.data.site_xpos[self.attachment_site_id].astype(np.float32)

    def get_end_effector_down_alignment(self) -> float:
        rotation = self.env.data.site_xmat[self.attachment_site_id].reshape(3, 3)
        approach_axis = rotation[:, 2]
        return float(np.clip(-approach_axis[2], 0.0, 1.0))

    def get_gripper_target_contact_score(self) -> float:
        target_body_id = self.env.model.body(self.target_body_name).id
        finger_body_ids = {
            self.env.model.body("left_finger").id,
            self.env.model.body("right_finger").id,
        }
        contacted_fingers = set()

        for index in range(self.env.data.ncon):
            contact = self.env.data.contact[index]
            body1 = int(self.env.model.geom_bodyid[contact.geom1])
            body2 = int(self.env.model.geom_bodyid[contact.geom2])

            if body1 == target_body_id and body2 in finger_body_ids:
                contacted_fingers.add(body2)
            elif body2 == target_body_id and body1 in finger_body_ids:
                contacted_fingers.add(body1)

        return float(len(contacted_fingers) / 2.0)

    def _check_failure_termination(self) -> Optional[Dict[str, Any]]:
        target_pos = self.get_target_position()
        terminate_on_target_drop = bool(
            self.task_config.get("terminate_on_target_drop", True)
        )
        terminate_on_target_escape = bool(
            self.task_config.get("terminate_on_target_escape", True)
        )
        xy_margin = float(
            self.task_config.get("failure_xy_margin", self.failure_xy_margin)
        )
        z_margin = float(
            self.task_config.get("failure_z_margin", self.failure_z_margin)
        )

        min_x = self.workspace_bounds["min_x"] - xy_margin
        max_x = self.workspace_bounds["max_x"] + xy_margin
        min_y = self.workspace_bounds["min_y"] - xy_margin
        max_y = self.workspace_bounds["max_y"] + xy_margin

        target_below_table = terminate_on_target_drop and bool(
            target_pos[2] < (self.table_height - z_margin)
        )
        target_out_of_workspace = terminate_on_target_escape and bool(
            target_pos[0] < min_x
            or target_pos[0] > max_x
            or target_pos[1] < min_y
            or target_pos[1] > max_y
        )

        if not target_below_table and not target_out_of_workspace:
            return None

        episode_failure_reason = (
            "target_below_table" if target_below_table else "target_out_of_workspace"
        )
        return {
            "episode_failure_reason": episode_failure_reason,
            "target_below_table": target_below_table,
            "target_out_of_workspace": target_out_of_workspace,
            "target_position_x": float(target_pos[0]),
            "target_position_y": float(target_pos[1]),
            "target_position_z": float(target_pos[2]),
        }

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
        self._last_action = action.copy()

        arm_action = action[:7]
        gripper_action = action[7]

        current_arm_qpos = self.env.data.qpos[:7].copy()

        rl_dt = self.env.model.opt.timestep * self.env.frame_skip

        target_arm_qpos = current_arm_qpos + (1.4 * arm_action * rl_dt)

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
            "gripper_joint_position": self.get_gripper_joint_position(),
            "gripper_open_fraction": self.get_gripper_open_fraction(),
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

        if not done:
            failure_info = self._check_failure_termination()
            if failure_info is not None:
                reward = float(
                    self.task_config.get("failure_penalty", self.failure_penalty)
                )
                done = True
                info.update(failure_info)
                info["failure_penalty"] = reward

        # Episode termination on max steps
        if self.step_count >= self.max_episode_steps:
            done = True
            info["step_limit_reached"] = True

        return obs, reward, done, info

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Extract point cloud and proprioceptive observations from all cameras."""
        timing_stats: Optional[Dict[str, float]] = None
        observation_start = time.perf_counter()
        if self.pointcloud_timing_log_interval > 0 and self.use_depth_only_pointcloud:
            timing_stats = {}

        if self.use_depth_only_pointcloud:
            merged_points = collect_fused_pointcloud_for_training(
                self.env,
                self.camera,
                camera_names=self.camera_names,
                num_points=self.num_points,
                camera_height=self.camera_height,
                camera_width=self.camera_width,
                workspace_bounds=self.workspace_bounds,
                table_height=self.table_height,
                timing_stats=timing_stats,
            )
        else:
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

            self._last_pointcloud_empty = True
            joint_state = self.env.data.qpos[: self.robot_dof].astype(np.float32)
            ee_position = self.get_end_effector_position()
            target_position = self.get_target_position()
            fallback_obs = {
                "pointcloud": self._last_valid_observation["pointcloud"].copy(),
                "joint_state": joint_state,
                "ee_position": ee_position,
                "target_position": target_position,
            }
            if self._task_uses_goal_position():
                fallback_obs["goal_position"] = self.goal_position.copy()
            if timing_stats is not None:
                timing_stats["obs_total_s"] = timing_stats.get("obs_total_s", 0.0) + (
                    time.perf_counter() - observation_start
                )
                timing_stats["obs_pointcloud_size"] = timing_stats.get(
                    "obs_pointcloud_size", 0.0
                ) + float(len(merged_points))
                timing_stats["obs_empty"] = timing_stats.get("obs_empty", 0.0) + 1.0
                self._record_pointcloud_timing(timing_stats)
            self._last_valid_observation = fallback_obs
            return fallback_obs

        sample_start = time.perf_counter()
        pointcloud = sample_pointcloud(merged_points, self.num_points)
        if timing_stats is not None:
            timing_stats["obs_sample_s"] = timing_stats.get("obs_sample_s", 0.0) + (
                time.perf_counter() - sample_start
            )
        self._last_pointcloud_empty = False

        # Extract joint state (positions only)
        joint_state = self.env.data.qpos[: self.robot_dof].astype(np.float32)

        # Extract end-effector position and stable goal position
        ee_position = self.get_end_effector_position()
        target_position = self.get_target_position()

        obs = {
            "pointcloud": pointcloud,
            "joint_state": joint_state,
            "ee_position": ee_position,
            "target_position": target_position,
        }
        if self._task_uses_goal_position():
            obs["goal_position"] = self.goal_position.copy()
        self._last_valid_observation = {
            "pointcloud": pointcloud.copy(),
            "joint_state": joint_state.copy(),
            "ee_position": ee_position.copy(),
            "target_position": target_position.copy(),
        }
        if self._task_uses_goal_position():
            self._last_valid_observation["goal_position"] = self.goal_position.copy()
        if timing_stats is not None:
            timing_stats["obs_total_s"] = timing_stats.get("obs_total_s", 0.0) + (
                time.perf_counter() - observation_start
            )
            timing_stats["obs_pointcloud_size"] = timing_stats.get(
                "obs_pointcloud_size", 0.0
            ) + float(len(merged_points))
            self._record_pointcloud_timing(timing_stats)
        return obs

    def _record_pointcloud_timing(self, timing_stats: Dict[str, float]) -> None:
        if self.pointcloud_timing_log_interval <= 0:
            return

        self._pointcloud_timing_count += 1
        for key, value in timing_stats.items():
            self._pointcloud_timing_totals[key] = (
                self._pointcloud_timing_totals.get(key, 0.0) + value
            )

        if self._pointcloud_timing_count % self.pointcloud_timing_log_interval != 0:
            return

        count = float(self._pointcloud_timing_count)
        totals = self._pointcloud_timing_totals

        def avg_ms(key: str) -> float:
            return 1000.0 * totals.get(key, 0.0) / count

        def avg_count(key: str) -> float:
            return totals.get(key, 0.0) / count

        print(
            "[PointcloudTiming] "
            f"pid={os.getpid()} obs={self._pointcloud_timing_count} "
            f"avg_ms(obs={avg_ms('obs_total_s'):.1f}, collect={avg_ms('collect_total_s'):.1f}, "
            f"camera_call={avg_ms('collect_camera_call_s'):.1f}, render={avg_ms('camera_render_depth_s'):.1f}, "
            f"setup={avg_ms('camera_setup_s'):.1f}, depth_filter={avg_ms('camera_depth_filter_s'):.1f}, "
            f"sample={avg_ms('camera_sample_indices_s'):.1f}, unproject={avg_ms('camera_unproject_s'):.1f}, "
            f"world_filter={avg_ms('collect_world_filter_s'):.1f}, concat={avg_ms('collect_concat_s'):.1f}, "
            f"obs_sample={avg_ms('obs_sample_s'):.1f}) "
            f"avg_counts(cameras={avg_count('camera_calls'):.1f}, pixels={avg_count('camera_pixels'):.0f}, "
            f"valid={avg_count('camera_valid_points'):.0f}, sampled={avg_count('camera_sampled_points'):.0f}, "
            f"filtered={avg_count('collect_points_after_filter'):.0f}, merged={avg_count('collect_merged_points'):.0f}, "
            f"pointcloud={avg_count('obs_pointcloud_size'):.0f}, empty={avg_count('obs_empty'):.3f})",
            flush=True,
        )

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
        self.task_name = task_config.get("task_name", self.task_name)
        self.max_episode_steps = task_config.get("max_episode_steps", 400)
        self.randomize_target_pose = task_config.get(
            "randomize_target_pose", self.randomize_target_pose
        )
        self.observation_space = self._build_observation_space()
        if not self._task_uses_goal_position():
            self.goal_position = np.zeros(3, dtype=np.float32)
        self._last_valid_observation = self._create_empty_observation()

    def set_task_reward(self, reward_fn):
        """Set a custom reward function."""
        self.task_config["reward_fn"] = reward_fn

    def set_training_progress(self, num_timesteps: int, n_updates: int) -> None:
        """Expose global learner progress to task rewards even when wrapped by Monitor."""
        self.training_num_timesteps = int(num_timesteps)
        self.training_n_updates = int(n_updates)
        self.task_config["training_num_timesteps"] = self.training_num_timesteps
        self.task_config["training_n_updates"] = self.training_n_updates

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
            camera_name = self.camera_names[0]

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
            samples_per_camera = get_pointcloud_samples_per_camera(
                self.num_points, [camera_name]
            )
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
        if self._object_runtime_cache:
            closed_runtime_ids = set()
            for bundle in self._object_runtime_cache.values():
                env = bundle.get("env")
                runtime_id = id(env)
                if runtime_id in closed_runtime_ids:
                    continue
                self._close_runtime_bundle(bundle)
                closed_runtime_ids.add(runtime_id)
            self._object_runtime_cache.clear()
            return

        if self.env.viewer is not None:
            self.env.viewer.close()

        if self.camera is not None:
            self.camera.close()
