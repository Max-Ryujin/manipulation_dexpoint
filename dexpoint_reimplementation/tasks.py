"""Task definitions and configurations for DexPoint learning."""

import numpy as np
from typing import Any, Dict, Tuple


class GraspingTask:
    """Object grasping task configuration."""

    NAME = "grasping"
    TARGET_OBJECT = "target_object"
    REACH_REWARD_SCALE = 1.0
    GOAL_REWARD_SCALE = 0.5
    GOAL_REWARD_ACTIVATION_DISTANCE = 0.05
    SUCCESS_DISTANCE_THRESHOLD = 0.04
    SUCCESS_BONUS = 5.0
    TIME_PENALTY = 1e-3
    REACH_DISTANCE_OFFSET = 0.7
    GOAL_DISTANCE_OFFSET = 0.5

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """Get default task configuration."""
        return {
            "max_episode_steps": 1000,
            "randomize_target_pose": True,
            "reward_fn": GraspingTask.reward_function,
            "task_name": GraspingTask.NAME,
            "target_body_name": GraspingTask.TARGET_OBJECT,
            "reach_reward_scale": GraspingTask.REACH_REWARD_SCALE,
            "goal_reward_scale": GraspingTask.GOAL_REWARD_SCALE,
            "goal_reward_activation_distance": GraspingTask.GOAL_REWARD_ACTIVATION_DISTANCE,
            "success_distance_threshold": GraspingTask.SUCCESS_DISTANCE_THRESHOLD,
            "success_bonus": GraspingTask.SUCCESS_BONUS,
            "time_penalty": GraspingTask.TIME_PENALTY,
            "reach_distance_offset": GraspingTask.REACH_DISTANCE_OFFSET,
            "goal_distance_offset": GraspingTask.GOAL_DISTANCE_OFFSET,
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
        info = {"task": GraspingTask.NAME, "step": env.step_count}

        ee_pos = env.get_end_effector_position()
        target_pos = env.get_target_position()
        goal_pos = env.goal_position

        reach_distance = float(np.linalg.norm(ee_pos - target_pos))
        ee_target_xy_distance = float(np.linalg.norm(ee_pos[:2] - target_pos[:2]))
        ee_target_z_distance = float(abs(ee_pos[2] - target_pos[2]))
        goal_distance = float(np.linalg.norm(target_pos - goal_pos))
        goal_xy_distance = float(np.linalg.norm(target_pos[:2] - goal_pos[:2]))
        goal_z_distance = float(abs(target_pos[2] - goal_pos[2]))
        target_height_above_table = float(target_pos[2] - env.table_height)
        target_lift = float(target_pos[2] - env.target_rest_height)

        reach_reward_scale = float(
            env.task_config.get(
                "reach_reward_scale", GraspingTask.REACH_REWARD_SCALE
            )
        )
        goal_reward_scale = float(
            env.task_config.get("goal_reward_scale", GraspingTask.GOAL_REWARD_SCALE)
        )
        goal_reward_activation_distance = float(
            env.task_config.get(
                "goal_reward_activation_distance",
                GraspingTask.GOAL_REWARD_ACTIVATION_DISTANCE,
            )
        )
        success_distance_threshold = float(
            env.task_config.get(
                "success_distance_threshold",
                GraspingTask.SUCCESS_DISTANCE_THRESHOLD,
            )
        )
        success_bonus_value = float(
            env.task_config.get("success_bonus", GraspingTask.SUCCESS_BONUS)
        )
        time_penalty_magnitude = float(
            env.task_config.get("time_penalty", GraspingTask.TIME_PENALTY)
        )
        reach_distance_offset = float(
            env.task_config.get(
                "reach_distance_offset", GraspingTask.REACH_DISTANCE_OFFSET
            )
        )
        goal_distance_offset = float(
            env.task_config.get(
                "goal_distance_offset", GraspingTask.GOAL_DISTANCE_OFFSET
            )
        )

        reach_reward = reach_reward_scale * (reach_distance_offset - reach_distance)
        reward += reach_reward

        goal_reward = 0.0
        goal_reward_active = reach_distance <= goal_reward_activation_distance
        if goal_reward_active:
            goal_reward = goal_reward_scale * (goal_distance_offset - goal_distance)
            reward += goal_reward

        time_penalty = -time_penalty_magnitude
        reward += time_penalty

        is_success = goal_distance <= success_distance_threshold
        success_bonus = success_bonus_value if is_success else 0.0
        if is_success:
            reward += success_bonus
            done = True

        info.update(
            {
                "reach_distance": reach_distance,
                "ee_target_xy_distance": ee_target_xy_distance,
                "ee_target_z_distance": ee_target_z_distance,
                "goal_distance": goal_distance,
                "goal_xy_distance": goal_xy_distance,
                "goal_z_distance": goal_z_distance,
                "target_height_above_table": target_height_above_table,
                "target_lift": target_lift,
                "reach_reward": reach_reward,
                "goal_reward": goal_reward,
                "time_penalty": time_penalty,
                "success_bonus": success_bonus,
                "is_success": is_success,
                "goal_reward_active": goal_reward_active,
                "reward_total": reward,
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
