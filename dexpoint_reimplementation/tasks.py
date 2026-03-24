"""Task definitions and configurations for DexPoint learning."""

import numpy as np
from typing import Dict, Tuple, Any, Callable
from pathlib import Path
import mujoco


class GraspingTask:
    """Object grasping task configuration."""

    NAME = "grasping"
    TARGET_OBJECT = "target_object"  # Name of the object to grasp

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """Get default task configuration."""
        return {
            "max_episode_steps": 1000,
            "n_blocks_to_place": 1,
            "reward_fn": GraspingTask.reward_function,
            "task_name": GraspingTask.NAME,
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

        # Get collision status
        collision_free = env.env.check_collisions()
        if not collision_free:
            done = True
            reward = -3.0
            info["reason"] = "collision"
            return reward, done, info

        try:
            # Get end effector position
            ee_pos = env.env.data.xpos[env.env.model.site("attachment_site").id]

            # Get gripper state (finger opening)
            gripper_state = env.env.data.qpos[7]  # Joint 7 is gripper
            gripper_vel = env.env.data.qvel[7] if len(env.env.data.qvel) > 7 else 0.0

            # Try to get target object position
            object_lifted = False
            target_height = 0.0
            distance_to_target = 0.0

            try:
                target_pos = env.env.get_object_position(GraspingTask.TARGET_OBJECT)
                target_height = target_pos[2]

                # Distance from gripper to target
                dist_vec = ee_pos - target_pos
                distance_to_target = np.linalg.norm(dist_vec)

                # Phase 1: Approach (minimize distance)
                approach_reward = max(0, 0.3 - distance_to_target) * 0.4
                reward += approach_reward

                # Phase 2: Grasping (closing gripper)
                # Gripper state ranges from 0 (closed) to 1 (open)
                # We want the gripper to be closed (state < 0.1) when near object
                if distance_to_target < 0.05:  # Close to object
                    grasp_reward = (1.0 - gripper_state) * 0.5  # Reward closing gripper
                    reward += grasp_reward

                # Phase 3: Lifting (object above initial position, gripper closed)
                initial_object_height = 0.05  # Approximate initial height
                if distance_to_target < 0.1 and gripper_state < 0.3:
                    lift_height = max(0, target_height - initial_object_height)
                    lift_reward = lift_height * 0.8
                    reward += lift_reward

                    if lift_height > 0.1:
                        object_lifted = True

                # Success bonus for fully lifted grasp
                if object_lifted:
                    reward += 2.0

                info.update(
                    {
                        "distance_to_target": distance_to_target,
                        "target_height": target_height,
                        "gripper_state": gripper_state,
                        "object_lifted": object_lifted,
                    }
                )

            except (ValueError, AttributeError):
                # Target object not found in environment state
                # Use default reward structure based on gripper activity
                reward += max(0, 0.1 - abs(gripper_vel)) * 0.05
                info["error"] = "Target object not found in environment"

            # Penalize erratic gripper movements
            if abs(gripper_vel) > 1.0:
                reward -= 0.05

            info.update(
                {
                    "step_reward": reward,
                }
            )

        except Exception as e:
            info["error"] = str(e)
            reward = 0.0

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
