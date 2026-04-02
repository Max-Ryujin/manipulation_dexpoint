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
    SUCCESS_DISTANCE_THRESHOLD = 0.03
    SUCCESS_BONUS = 5.0
    TIME_PENALTY = 1e-3
    REACH_DISTANCE_OFFSET = 0.7
    GOAL_DISTANCE_OFFSET = 0.5
    FINGER_HEIGHT_ALIGNMENT_SCALE = 0.02
    FINGER_HEIGHT_ALIGNMENT_TOLERANCE = 0.02
    GRASP_REWARD_SCALE = 0.1
    GRASP_CENTERING_TOLERANCE = 0.03
    GRASP_ACTUATOR_FORCE_THRESHOLD = 2.0
    GRASP_ACTUATOR_FORCE_TOLERANCE = 10.0
    GRASP_BETWEEN_FINGERS_MARGIN = 0.01

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """Get default task configuration."""
        return {
            "max_episode_steps": 800,
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
            "grasp_reward_scale": GraspingTask.GRASP_REWARD_SCALE,
            "grasp_centering_tolerance": GraspingTask.GRASP_CENTERING_TOLERANCE,
            "grasp_actuator_force_threshold": GraspingTask.GRASP_ACTUATOR_FORCE_THRESHOLD,
            "grasp_actuator_force_tolerance": GraspingTask.GRASP_ACTUATOR_FORCE_TOLERANCE,
            "grasp_between_fingers_margin": GraspingTask.GRASP_BETWEEN_FINGERS_MARGIN,
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
        left_finger_pos, right_finger_pos = env.get_finger_positions()
        target_pos = env.get_target_position()
        goal_pos = env.goal_position
        finger_midpoint = 0.5 * (left_finger_pos + right_finger_pos)
        finger_span_vector = right_finger_pos - left_finger_pos
        finger_span = float(np.linalg.norm(finger_span_vector))
        target_to_finger_midpoint = float(np.linalg.norm(target_pos - finger_midpoint))
        gripper_opening_width = env.get_gripper_opening_width()
        gripper_actuator_force = float(abs(env.get_gripper_actuator_force()))

        reach_distance = float(np.linalg.norm(ee_pos - target_pos))
        ee_target_xy_distance = float(np.linalg.norm(ee_pos[:2] - target_pos[:2]))
        ee_target_z_distance = float(abs(ee_pos[2] - target_pos[2]))
        finger_height_difference = float(abs(left_finger_pos[2] - right_finger_pos[2]))
        goal_distance = float(np.linalg.norm(target_pos - goal_pos))
        goal_xy_distance = float(np.linalg.norm(target_pos[:2] - goal_pos[:2]))
        goal_z_distance = float(abs(target_pos[2] - goal_pos[2]))
        target_height_above_table = float(target_pos[2] - env.table_height)
        target_lift = float(target_pos[2] - env.target_rest_height)

        reach_reward_scale = float(
            env.task_config.get("reach_reward_scale", GraspingTask.REACH_REWARD_SCALE)
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
        finger_height_alignment_scale = float(
            env.task_config.get(
                "finger_height_alignment_scale",
                GraspingTask.FINGER_HEIGHT_ALIGNMENT_SCALE,
            )
        )
        finger_height_alignment_tolerance = float(
            env.task_config.get(
                "finger_height_alignment_tolerance",
                GraspingTask.FINGER_HEIGHT_ALIGNMENT_TOLERANCE,
            )
        )
        grasp_reward_scale = float(
            env.task_config.get("grasp_reward_scale", GraspingTask.GRASP_REWARD_SCALE)
        )
        grasp_centering_tolerance = float(
            env.task_config.get(
                "grasp_centering_tolerance",
                GraspingTask.GRASP_CENTERING_TOLERANCE,
            )
        )
        grasp_actuator_force_threshold = float(
            env.task_config.get(
                "grasp_actuator_force_threshold",
                GraspingTask.GRASP_ACTUATOR_FORCE_THRESHOLD,
            )
        )
        grasp_actuator_force_tolerance = float(
            env.task_config.get(
                "grasp_actuator_force_tolerance",
                GraspingTask.GRASP_ACTUATOR_FORCE_TOLERANCE,
            )
        )
        grasp_between_fingers_margin = float(
            env.task_config.get(
                "grasp_between_fingers_margin",
                GraspingTask.GRASP_BETWEEN_FINGERS_MARGIN,
            )
        )

        reach_reward = reach_reward_scale * (reach_distance_offset - reach_distance)
        reward += reach_reward

        finger_height_alignment_reward = finger_height_alignment_scale * max(
            0.0,
            1.0
            - (finger_height_difference / max(finger_height_alignment_tolerance, 1e-6)),
        )
        reward += finger_height_alignment_reward

        target_half_extent = float(np.max(env.target_spec.half_extents))
        target_projection_distance = 0.0
        target_lateral_distance = target_to_finger_midpoint
        between_fingers_score = 0.0
        target_between_fingers = False
        if finger_span > 1e-6:
            finger_span_axis = finger_span_vector / finger_span
            target_offset = target_pos - finger_midpoint
            target_projection = float(np.dot(target_offset, finger_span_axis))
            target_projection_distance = float(abs(target_projection))
            target_lateral_offset = target_offset - (
                target_projection * finger_span_axis
            )
            target_lateral_distance = float(np.linalg.norm(target_lateral_offset))

            max_projection_distance = (
                0.5 * finger_span + target_half_extent + grasp_between_fingers_margin
            )
            target_between_fingers = (
                target_projection_distance <= max_projection_distance
            )
            between_fingers_score = max(
                0.0,
                1.0
                - (
                    target_lateral_distance
                    / max(
                        grasp_centering_tolerance + target_half_extent,
                        1e-6,
                    )
                ),
            )
            between_fingers_score *= float(target_between_fingers)

        grasp_resistance_score = np.clip(
            (gripper_actuator_force - grasp_actuator_force_threshold)
            / max(grasp_actuator_force_tolerance, 1e-6),
            0.0,
            1.0,
        )
        grasp_detected = bool(target_between_fingers and grasp_resistance_score > 0.0)
        grasp_reward = (
            grasp_reward_scale * between_fingers_score * float(grasp_resistance_score)
        )
        reward += grasp_reward

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
                "finger_height_difference": finger_height_difference,
                "finger_span": finger_span,
                "target_to_finger_midpoint": target_to_finger_midpoint,
                "target_projection_distance": target_projection_distance,
                "target_lateral_distance": target_lateral_distance,
                "gripper_opening_width": gripper_opening_width,
                "gripper_actuator_force": gripper_actuator_force,
                "goal_distance": goal_distance,
                "goal_xy_distance": goal_xy_distance,
                "goal_z_distance": goal_z_distance,
                "target_height_above_table": target_height_above_table,
                "target_lift": target_lift,
                "reach_reward": reach_reward,
                "finger_height_alignment_reward": finger_height_alignment_reward,
                "between_fingers_score": between_fingers_score,
                "grasp_resistance_score": float(grasp_resistance_score),
                "grasp_reward": grasp_reward,
                "target_between_fingers": target_between_fingers,
                "grasp_detected": grasp_detected,
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
