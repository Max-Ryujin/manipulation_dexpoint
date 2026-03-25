"""Task definitions and configurations for DexPoint learning."""

import numpy as np
from typing import Dict, Tuple, Any, Callable
from pathlib import Path
import mujoco


class GraspingTask:
    """Object grasping task configuration."""

    NAME = "grasping"
    TARGET_OBJECT = "target_object"

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """Get default task configuration."""
        return {
            "max_episode_steps": 1000,
            "randomize_target_pose": True,
            "reward_fn": GraspingTask.reward_function,
            "task_name": GraspingTask.NAME,
            "target_body_name": GraspingTask.TARGET_OBJECT,
        }

    @staticmethod
    def reward_function(env) -> Tuple[float, bool, Dict[str, Any]]:
        """
        Compute reward for grasping task.

        Reward incentivizes:
        - Approach to target object
        - Gripper closure around object
        - Lifting object
        - Stable grasp (not dropping)

        Args:
            env: FrankaGymEnvironment instance

        Returns:
            reward: Scalar reward
            done: Episode termination flag
            info: Info dictionary with metrics
        """
        reward = 0.0
        done = False
        info = {
            "task": GraspingTask.NAME,
            "step": env.step_count,
        }

        ee_pos = env.env.data.xpos[env.env.model.site("attachment_site").id]

        target_pos = env.get_target_position()
        target_height = float(target_pos[2])
        distance_to_target = float(np.linalg.norm(ee_pos - target_pos))
        xy_distance = float(np.linalg.norm(ee_pos[:2] - target_pos[:2]))

        gripper_qpos = float(env.env.data.qpos[7])
        gripper_vel = float(env.env.data.qvel[7]) if len(env.env.data.qvel) > 7 else 0.0
        gripper_span = max(float(env.ctrl_max[7] - env.ctrl_min[7]), 1e-6)
        gripper_open_fraction = float(
            np.clip((gripper_qpos - env.ctrl_min[7]) / gripper_span, 0.0, 1.0)
        )
        gripper_closed_fraction = 1.0 - gripper_open_fraction

        lift_height = max(0.0, target_height - float(env.target_rest_height))
        near_can = distance_to_target < 0.06
        grasp_candidate = near_can and gripper_closed_fraction > 0.45
        object_lifted = lift_height > float(env.success_lift_height)

        reward += max(0.0, 0.50 - distance_to_target)
        reward += max(0.0, 0.20 - xy_distance)

        if near_can:
            reward += 0.6 * gripper_closed_fraction

        reward += 8.0 * lift_height

        if lift_height > 0.02:
            reward += 0.5
        if lift_height > 0.05:
            reward += 1.0
        if grasp_candidate:
            reward += 0.5

        if object_lifted and distance_to_target < 0.12:
            reward += 5.0
            done = True
            info["reason"] = "success"

        info.update(
            {
                "distance_to_target": distance_to_target,
                "xy_distance_to_target": xy_distance,
                "target_height": target_height,
                "lift_height": lift_height,
                "gripper_open_fraction": gripper_open_fraction,
                "gripper_closed_fraction": gripper_closed_fraction,
                "object_lifted": object_lifted,
                "step_reward": reward,
            }
        )

        return reward, done, info


def create_task_config(task_name: str, **kwargs) -> Dict[str, Any]:
    """
    Create a task configuration by name.

    Args:
        task_name: 'grasping'
        **kwargs: Additional parameters to override defaults

    Returns:
        Task configuration dictionary
    """
    if task_name == "grasping":
        config = GraspingTask.get_default_config()
    else:
        raise ValueError(f"Unknown task: {task_name}")

    # Override with kwargs
    config.update(kwargs)

    return config
