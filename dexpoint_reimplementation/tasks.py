"""Task definitions and configurations for DexPoint learning."""

from typing import Any, Dict, Tuple

import numpy as np


class GraspingTask:
    """Object grasping task configuration."""

    NAME = "grasping"
    TARGET_OBJECT = "target_object"
    DISTANCE_REWARD_SCALE = 0.75    
    DISTANCE_SCALE = 0.25
    ORIENTATION_REWARD_SCALE = 0.2
    ORIENTATION_THRESHOLD = 0.90
    GRASP_REWARD_SCALE = 2.50
    CAGING_DISTANCE = 0.08
    GRASP_ACTUATOR_FORCE_THRESHOLD = 2.0
    GRASP_ACTUATOR_FORCE_SCALE = 8.0
    GRASP_WIDTH_TOLERANCE = 0.012
    LIFT_REWARD_SCALE = 2.0
    LIFT_CLEARANCE = 0.01
    LIFT_HEIGHT_THRESHOLD = 0.08
    GOAL_HEIGHT_REWARD_SCALE = 1.0
    GOAL_HEIGHT_TOLERANCE = 0.08
    SUCCESS_GOAL_HEIGHT_THRESHOLD = 0.02
    SUCCESS_BONUS = 0.5
    TIME_PENALTY = 1e-3
    FAILURE_PENALTY = -1.0
    FAILURE_XY_MARGIN = 0.05
    FAILURE_Z_MARGIN = 0.01
    GRASP_SHAPING_START_STEP = 0
    GRASP_SHAPING_FULL_STEP = 2_000_000
    GOAL_REWARD_START_STEP = 10_000_000
    GOAL_REWARD_FULL_STEP = 12_000_000
    GOAL_REWARD_START_UPDATE = 4_000
    GOAL_REWARD_FULL_UPDATE = 4_800

    @staticmethod
    def _safe_progress_ratio(current: float, start: float, end: float) -> float:
        if end <= start:
            return float(current >= end)
        return float(np.clip((current - start) / (end - start), 0.0, 1.0))

    @staticmethod
    def _get_training_progress(env) -> Tuple[float, float]:
        task_config = getattr(env, "task_config", {})
        num_timesteps = float(
            task_config.get(
                "training_num_timesteps",
                getattr(env, "training_num_timesteps", 0),
            )
        )
        n_updates = float(
            task_config.get(
                "training_n_updates",
                getattr(env, "training_n_updates", 0),
            )
        )
        return num_timesteps, n_updates

    @staticmethod
    def _compute_reward_features(env) -> Dict[str, Any]:
        target_pos = env.get_target_position() + np.array([0.0, 0.0, 0.01])
        goal_pos = env.goal_position
        ee_pos = env.get_end_effector_position()
        finger_midpoint = env.get_finger_midpoint_position()
        lfinger_pos = env.get_left_finger_position()
        rfinger_pos = env.get_right_finger_position()
        gripper_opening_width = env.get_gripper_opening_width()
        gripper_open_fraction = env.get_gripper_open_fraction()
        gripper_actuator_force = float(abs(env.get_gripper_actuator_force()))
        target_contact_score = env.get_gripper_target_contact_score()
        orientation_down_alignment = env.get_end_effector_down_alignment()

        distance_scale = float(
            env.task_config.get("distance_scale", GraspingTask.DISTANCE_SCALE)
        )
        orientation_threshold = float(
            env.task_config.get(
                "orientation_threshold", GraspingTask.ORIENTATION_THRESHOLD
            )
        )
        caging_distance = float(
            env.task_config.get("caging_distance", GraspingTask.CAGING_DISTANCE)
        )
        force_threshold = float(
            env.task_config.get(
                "grasp_actuator_force_threshold",
                GraspingTask.GRASP_ACTUATOR_FORCE_THRESHOLD,
            )
        )
        force_scale = float(
            env.task_config.get(
                "grasp_actuator_force_scale", GraspingTask.GRASP_ACTUATOR_FORCE_SCALE
            )
        )
        grasp_width_tolerance = float(
            env.task_config.get(
                "grasp_width_tolerance", GraspingTask.GRASP_WIDTH_TOLERANCE
            )
        )
        lift_clearance = float(
            env.task_config.get("lift_clearance", GraspingTask.LIFT_CLEARANCE)
        )
        lift_height_threshold = float(
            env.task_config.get(
                "lift_height_threshold", GraspingTask.LIFT_HEIGHT_THRESHOLD
            )
        )
        goal_height_tolerance = float(
            env.task_config.get(
                "goal_height_tolerance", GraspingTask.GOAL_HEIGHT_TOLERANCE
            )
        )
        success_goal_height_threshold = float(
            env.task_config.get(
                "success_goal_height_threshold",
                GraspingTask.SUCCESS_GOAL_HEIGHT_THRESHOLD,
            )
        )

        reach_distance = float(np.linalg.norm(target_pos - finger_midpoint))
        ee_object_distance = float(np.linalg.norm(target_pos - ee_pos))
        ee_target_xy_distance = float(np.linalg.norm(ee_pos[:2] - target_pos[:2]))
        ee_target_z_distance = float(abs(ee_pos[2] - target_pos[2]))
        goal_distance = float(np.linalg.norm(target_pos - goal_pos))
        goal_xy_distance = float(np.linalg.norm(target_pos[:2] - goal_pos[:2]))
        goal_z_distance = float(abs(target_pos[2] - goal_pos[2]))
        lfinger_dist = float(np.linalg.norm(lfinger_pos - target_pos))
        rfinger_dist = float(np.linalg.norm(rfinger_pos - target_pos))
        finger_dist = 0.5 * (lfinger_dist + rfinger_dist)
        target_height_above_table = float(target_pos[2] - env.table_height)
        target_lift = float(target_pos[2] - env.target_rest_height)

        distance_score = 1.0 - np.tanh(reach_distance / max(distance_scale, 1e-6))
        finger_score = 1.0 - np.tanh(finger_dist / max(distance_scale, 1e-6))
        orientation_score = np.clip(
            (orientation_down_alignment - orientation_threshold)
            / max(1.0 - orientation_threshold, 1e-6),
            0.0,
            1.0,
        )
        caging_score = max(
            1.0 - np.tanh(reach_distance / max(caging_distance, 1e-6)),
            1.0 - np.tanh(finger_dist / max(0.75 * caging_distance, 1e-6)),
        )
        force_score = np.clip(
            (gripper_actuator_force - force_threshold) / max(force_scale, 1e-6),
            0.0,
            1.0,
        )
        expected_grasp_width = 2.0 * float(np.max(env.target_spec.half_extents[:2]))
        width_match_score = np.clip(
            1.0
            - abs(gripper_opening_width - expected_grasp_width)
            / max(grasp_width_tolerance, 1e-6),
            0.0,
            1.0,
        )
        contact_grasp_score = np.clip(float(target_contact_score), 0.0, 1.0)
        force_grasp_score = force_score * max(width_match_score, 0.25)
        enclosure_score = np.clip(0.5 * (finger_score + caging_score), 0.0, 1.0)
        grasp_evidence_score = max(
            float(contact_grasp_score),
            float(force_grasp_score),
            float(enclosure_score * width_match_score),
        )
        grasp_score = np.clip(
            0.35 * enclosure_score
            + 0.25 * orientation_score
            + 0.20 * contact_grasp_score
            + 0.20 * force_grasp_score,
            0.0,
            1.0,
        )
        lift_progress = np.clip(
            (target_lift - lift_clearance)
            / max(lift_height_threshold - lift_clearance, 1e-6),
            0.0,
            1.0,
        )
        goal_height_score = np.clip(
            1.0 - goal_z_distance / max(goal_height_tolerance, 1e-6),
            0.0,
            1.0,
        )
        target_between_fingers = bool(
            enclosure_score > 0.55 and width_match_score > 0.15
        )
        grasp_detected = bool(
            grasp_evidence_score > 0.35
            and (caging_score > 0.35 or target_contact_score >= 1.0)
        )
        is_success = bool(
            target_lift >= lift_height_threshold
            and goal_z_distance <= success_goal_height_threshold
        )

        return {
            "reach_distance": reach_distance,
            "ee_object_distance": ee_object_distance,
            "ee_target_xy_distance": ee_target_xy_distance,
            "ee_target_z_distance": ee_target_z_distance,
            "goal_distance": goal_distance,
            "goal_xy_distance": goal_xy_distance,
            "goal_z_distance": goal_z_distance,
            "goal_height_distance": goal_z_distance,
            "lfinger_dist": lfinger_dist,
            "rfinger_dist": rfinger_dist,
            "finger_dist": finger_dist,
            "gripper_opening_width": gripper_opening_width,
            "gripper_open_fraction": gripper_open_fraction,
            "gripper_actuator_force": gripper_actuator_force,
            "target_contact_score": float(target_contact_score),
            "target_height_above_table": target_height_above_table,
            "target_lift": target_lift,
            "orientation_down_alignment": orientation_down_alignment,
            "distance_score": float(distance_score),
            "finger_score": float(finger_score),
            "orientation_score": float(orientation_score),
            "caging_score": float(caging_score),
            "force_score": float(force_score),
            "width_match_score": float(width_match_score),
            "contact_grasp_score": float(contact_grasp_score),
            "force_grasp_score": float(force_grasp_score),
            "enclosure_score": float(enclosure_score),
            "grasp_evidence_score": float(grasp_evidence_score),
            "grasp_score": float(grasp_score),
            "lift_progress": float(lift_progress),
            "goal_height_score": float(goal_height_score),
            "target_between_fingers": target_between_fingers,
            "grasp_detected": grasp_detected,
            "is_success": is_success,
        }

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """Get default task configuration."""
        return {
            "max_episode_steps": 800,
            "randomize_target_pose": True,
            "reward_fn": GraspingTask.reward_function_shaped,
            "reward_variant": "shaped",
            "task_name": GraspingTask.NAME,
            "target_body_name": GraspingTask.TARGET_OBJECT,
            "distance_reward_scale": GraspingTask.DISTANCE_REWARD_SCALE,
            "distance_scale": GraspingTask.DISTANCE_SCALE,
            "orientation_reward_scale": GraspingTask.ORIENTATION_REWARD_SCALE,
            "orientation_threshold": GraspingTask.ORIENTATION_THRESHOLD,
            "grasp_reward_scale": GraspingTask.GRASP_REWARD_SCALE,
            "caging_distance": GraspingTask.CAGING_DISTANCE,
            "grasp_actuator_force_threshold": GraspingTask.GRASP_ACTUATOR_FORCE_THRESHOLD,
            "grasp_actuator_force_scale": GraspingTask.GRASP_ACTUATOR_FORCE_SCALE,
            "grasp_width_tolerance": GraspingTask.GRASP_WIDTH_TOLERANCE,
            "lift_reward_scale": GraspingTask.LIFT_REWARD_SCALE,
            "lift_clearance": GraspingTask.LIFT_CLEARANCE,
            "lift_height_threshold": GraspingTask.LIFT_HEIGHT_THRESHOLD,
            "goal_height_reward_scale": GraspingTask.GOAL_HEIGHT_REWARD_SCALE,
            "goal_height_tolerance": GraspingTask.GOAL_HEIGHT_TOLERANCE,
            "success_goal_height_threshold": GraspingTask.SUCCESS_GOAL_HEIGHT_THRESHOLD,
            "success_bonus": GraspingTask.SUCCESS_BONUS,
            "time_penalty": GraspingTask.TIME_PENALTY,
            "failure_penalty": GraspingTask.FAILURE_PENALTY,
            "failure_xy_margin": GraspingTask.FAILURE_XY_MARGIN,
            "failure_z_margin": GraspingTask.FAILURE_Z_MARGIN,
            "grasp_shaping_start_step": GraspingTask.GRASP_SHAPING_START_STEP,
            "grasp_shaping_full_step": GraspingTask.GRASP_SHAPING_FULL_STEP,
            "goal_reward_start_step": GraspingTask.GOAL_REWARD_START_STEP,
            "goal_reward_full_step": GraspingTask.GOAL_REWARD_FULL_STEP,
            "goal_reward_start_update": GraspingTask.GOAL_REWARD_START_UPDATE,
            "goal_reward_full_update": GraspingTask.GOAL_REWARD_FULL_UPDATE,
        }

    # @staticmethod
    # def reward_function(env) -> Tuple[float, bool, Dict[str, Any]]:
    #     reward = 0.0
    #     done = False
    #     info = {"task": GraspingTask.NAME, "step": env.step_count}

    #     target_pos = env.get_target_position() + np.array(
    #         [0.0, 0.0, 0.01]
    #     )  # Slightly above target center for better grasping
    #     goal_pos = env.goal_position
    #     ee_pos = env.get_end_effector_position()
    #     finger_midpoint = env.get_finger_midpoint_position()
    #     gripper_opening_width = env.get_gripper_opening_width()
    #     gripper_open_fraction = env.get_gripper_open_fraction()
    #     gripper_actuator_force = float(abs(env.get_gripper_actuator_force()))
    #     target_contact_score = env.get_gripper_target_contact_score()
    #     reach_distance = float(np.linalg.norm(target_pos - finger_midpoint))
    #     ee_target_xy_distance = float(np.linalg.norm(ee_pos[:2] - target_pos[:2]))
    #     ee_target_z_distance = float(abs(ee_pos[2] - target_pos[2]))
    #     goal_distance = float(np.linalg.norm(target_pos - goal_pos))
    #     goal_xy_distance = float(np.linalg.norm(target_pos[:2] - goal_pos[:2]))
    #     goal_z_distance = float(abs(target_pos[2] - goal_pos[2]))
    #     target_height_above_table = float(target_pos[2] - env.table_height)
    #     target_lift = float(target_pos[2] - env.target_rest_height)
    #     orientation_down_alignment = env.get_end_effector_down_alignment()

    #     distance_scale = float(
    #         env.task_config.get("distance_scale", GraspingTask.DISTANCE_SCALE)
    #     )
    #     distance_reward_scale = float(
    #         env.task_config.get(
    #             "distance_reward_scale", GraspingTask.DISTANCE_REWARD_SCALE
    #         )
    #     )
    #     orientation_threshold = float(
    #         env.task_config.get(
    #             "orientation_threshold", GraspingTask.ORIENTATION_THRESHOLD
    #         )
    #     )
    #     orientation_reward_scale = float(
    #         env.task_config.get(
    #             "orientation_reward_scale", GraspingTask.ORIENTATION_REWARD_SCALE
    #         )
    #     )
    #     caging_distance = float(
    #         env.task_config.get("caging_distance", GraspingTask.CAGING_DISTANCE)
    #     )
    #     grasp_reward_scale = float(
    #         env.task_config.get("grasp_reward_scale", GraspingTask.GRASP_REWARD_SCALE)
    #     )
    #     force_threshold = float(
    #         env.task_config.get(
    #             "grasp_actuator_force_threshold",
    #             GraspingTask.GRASP_ACTUATOR_FORCE_THRESHOLD,
    #         )
    #     )
    #     force_scale = float(
    #         env.task_config.get(
    #             "grasp_actuator_force_scale", GraspingTask.GRASP_ACTUATOR_FORCE_SCALE
    #         )
    #     )
    #     grasp_width_tolerance = float(
    #         env.task_config.get(
    #             "grasp_width_tolerance", GraspingTask.GRASP_WIDTH_TOLERANCE
    #         )
    #     )
    #     lift_clearance = float(
    #         env.task_config.get("lift_clearance", GraspingTask.LIFT_CLEARANCE)
    #     )
    #     lift_height_threshold = float(
    #         env.task_config.get(
    #             "lift_height_threshold", GraspingTask.LIFT_HEIGHT_THRESHOLD
    #         )
    #     )
    #     lift_reward_scale = float(
    #         env.task_config.get("lift_reward_scale", GraspingTask.LIFT_REWARD_SCALE)
    #     )
    #     goal_height_tolerance = float(
    #         env.task_config.get(
    #             "goal_height_tolerance", GraspingTask.GOAL_HEIGHT_TOLERANCE
    #         )
    #     )
    #     goal_height_reward_scale = float(
    #         env.task_config.get(
    #             "goal_height_reward_scale", GraspingTask.GOAL_HEIGHT_REWARD_SCALE
    #         )
    #     )
    #     success_goal_height_threshold = float(
    #         env.task_config.get(
    #             "success_goal_height_threshold",
    #             GraspingTask.SUCCESS_GOAL_HEIGHT_THRESHOLD,
    #         )
    #     )
    #     success_bonus = float(
    #         env.task_config.get("success_bonus", GraspingTask.SUCCESS_BONUS)
    #     )
    #     time_penalty = -float(
    #         env.task_config.get("time_penalty", GraspingTask.TIME_PENALTY)
    #     )

    #     distance_score = 1.0 - np.tanh(reach_distance / max(distance_scale, 1e-6))
    #     orientation_score = np.clip(
    #         (orientation_down_alignment - orientation_threshold)
    #         / max(1.0 - orientation_threshold, 1e-6),
    #         0.0,
    #         1.0,
    #     )
    #     caging_score = np.clip(
    #         1.0 - reach_distance / max(caging_distance, 1e-6), 0.0, 1.0
    #     )
    #     force_score = np.clip(
    #         (gripper_actuator_force - force_threshold) / max(force_scale, 1e-6),
    #         0.0,
    #         1.0,
    #     )
    #     expected_grasp_width = 2.0 * float(np.max(env.target_spec.half_extents[:2]))
    #     width_match_score = np.clip(
    #         1.0
    #         - abs(gripper_opening_width - expected_grasp_width)
    #         / max(grasp_width_tolerance, 1e-6),
    #         0.0,
    #         1.0,
    #     )
    #     force_grasp_score = force_score * width_match_score
    #     contact_grasp_score = 1.0 if target_contact_score >= 1.0 else 0.0
    #     between_fingers_score = max(
    #         float(width_match_score * caging_score), float(contact_grasp_score)
    #     )
    #     grasp_evidence_score = max(float(force_grasp_score), float(contact_grasp_score))
    #     grasp_score = grasp_evidence_score * (0.5 + 0.5 * float(caging_score))
    #     lift_progress = np.clip(
    #         (target_lift - lift_clearance)
    #         / max(lift_height_threshold - lift_clearance, 1e-6),
    #         0.0,
    #         1.0,
    #     )
    #     goal_height_score = np.clip(
    #         0.5 - goal_z_distance / max(goal_height_tolerance, 1e-6), 0.0, 1.0
    #     )

    #     distance_reward = distance_reward_scale * float(distance_score)
    #     orientation_reward = orientation_reward_scale * float(orientation_score)
    #     grasp_reward = grasp_reward_scale * grasp_score
    #     lift_reward = lift_reward_scale * float(lift_progress)
    #     goal_height_reward = goal_height_reward_scale * float(goal_height_score)

    #     reward = (
    #         distance_reward
    #         + orientation_reward
    #         + grasp_reward
    #         + lift_reward
    #         + goal_height_reward
    #         + time_penalty
    #     )
    #     grasp_detected = bool(grasp_evidence_score > 0.25 and caging_score > 0.25)
    #     target_between_fingers = bool(
    #         between_fingers_score > 0.5 and caging_score > 0.25
    #     )
    #     is_success = bool(
    #         target_lift >= lift_height_threshold
    #         and goal_z_distance <= success_goal_height_threshold
    #     )
    #     if is_success:
    #         reward += success_bonus
    #         done = True

    #     info.update(
    #         {
    #             "reach_distance": reach_distance,
    #             "ee_target_xy_distance": ee_target_xy_distance,
    #             "ee_target_z_distance": ee_target_z_distance,
    #             "gripper_opening_width": gripper_opening_width,
    #             "gripper_open_fraction": gripper_open_fraction,
    #             "gripper_actuator_force": gripper_actuator_force,
    #             "goal_distance": goal_distance,
    #             "goal_xy_distance": goal_xy_distance,
    #             "goal_z_distance": goal_z_distance,
    #             "goal_height_distance": goal_z_distance,
    #             "target_height_above_table": target_height_above_table,
    #             "target_lift": target_lift,
    #             "orientation_down_alignment": orientation_down_alignment,
    #             "distance_score": float(distance_score),
    #             "orientation_score": float(orientation_score),
    #             "caging_score": float(caging_score),
    #             "force_score": float(force_score),
    #             "width_match_score": float(width_match_score),
    #             "target_contact_score": float(target_contact_score),
    #             "grasp_evidence_score": float(grasp_evidence_score),
    #             "between_fingers_score": float(between_fingers_score),
    #             "grasp_resistance_score": float(force_grasp_score),
    #             "lift_progress": float(lift_progress),
    #             "goal_height_score": float(goal_height_score),
    #             "distance_reward": distance_reward,
    #             "orientation_reward": orientation_reward,
    #             "grasp_reward": grasp_reward,
    #             "lift_reward": lift_reward,
    #             "goal_reward": goal_height_reward,
    #             "goal_height_reward": goal_height_reward,
    #             "target_between_fingers": target_between_fingers,
    #             "grasp_detected": grasp_detected,
    #             "time_penalty": time_penalty,
    #             "success_bonus": success_bonus if is_success else 0.0,
    #             "is_success": is_success,
    #             "goal_reward_active": bool(goal_height_score > 0.0),
    #             "reward_total": reward,
    #         }
    #     )
    #     return reward, done, info

    @staticmethod
    def reward_function(env) -> Tuple[float, bool, Dict[str, Any]]:
        reward = 0.0
        done = False
        info = {"task": GraspingTask.NAME, "step": env.step_count}

        target_pos = env.get_target_position() + np.array(
            [0.0, 0.0, 0.01]
        )  # Slightly above target center for better grasping
        ee_pos = env.get_end_effector_position()
        finger_midpoint = env.get_finger_midpoint_position()
        lfinger_pos = env.get_left_finger_position()
        rfinger_pos = env.get_right_finger_position()

        goal_pos = env.goal_position

        reach_distance = float(np.linalg.norm(target_pos - finger_midpoint))
        ee_object_distance = float(np.linalg.norm(target_pos - ee_pos))
        goal_z_distance = float(abs(target_pos[2] - goal_pos[2]))

        lfinger_dist = float(np.linalg.norm(lfinger_pos - target_pos))
        rfinger_dist = float(np.linalg.norm(rfinger_pos - target_pos))
        finger_dist = 0.5 * (lfinger_dist + rfinger_dist)

        target_lift = float(target_pos[2] - env.target_rest_height)

        distance_scale = float(env.task_config.get("distance_scale", 0.1))
        lift_height_threshold = float(env.task_config.get("lift_height_threshold", 0.1))

        reach_score = 1.0 - np.tanh(reach_distance / max(distance_scale, 1e-6))

        finger_score = 1.0 - np.tanh(finger_dist / max(distance_scale, 1e-6))

        caging_score = 1.0 - np.tanh(reach_distance / 0.05)

        lift_score = np.clip(target_lift / max(lift_height_threshold, 1e-6), 0.0, 1.0)

        w_reach = 1.0
        w_finger = 1.0
        w_caging = 0.5
        w_lift = 2.0

        reward = (
            w_reach * reach_score
            + w_finger * finger_score
            + w_caging * caging_score
            + w_lift * lift_score
        )

        is_success = bool(target_lift >= lift_height_threshold)
        if is_success:
            reward += 5.0
            done = True

        info.update(
            {
                "reach_distance": reach_distance,
                "finger_dist": finger_dist,
                "target_lift": target_lift,
                "reach_score": reach_score,
                "finger_score": finger_score,
                "caging_score": caging_score,
                "lift_score": lift_score,
                "is_success": is_success,
                "reward_total": reward,
            }
        )

        return reward, done, info

    @staticmethod
    def reward_function_shaped(env) -> Tuple[float, bool, Dict[str, Any]]:
        reward = 0.0
        done = False
        info = {"task": f"{GraspingTask.NAME}_shaped", "step": env.step_count}

        features = GraspingTask._compute_reward_features(env)
        num_timesteps, n_updates = GraspingTask._get_training_progress(env)

        grasp_weight = GraspingTask._safe_progress_ratio(
            num_timesteps,
            float(
                env.task_config.get(
                    "grasp_shaping_start_step", GraspingTask.GRASP_SHAPING_START_STEP
                )
            ),
            float(
                env.task_config.get(
                    "grasp_shaping_full_step", GraspingTask.GRASP_SHAPING_FULL_STEP
                )
            ),
        )
        goal_step_weight = GraspingTask._safe_progress_ratio(
            num_timesteps,
            float(
                env.task_config.get(
                    "goal_reward_start_step", GraspingTask.GOAL_REWARD_START_STEP
                )
            ),
            float(
                env.task_config.get(
                    "goal_reward_full_step", GraspingTask.GOAL_REWARD_FULL_STEP
                )
            ),
        )
        goal_update_weight = GraspingTask._safe_progress_ratio(
            n_updates,
            float(
                env.task_config.get(
                    "goal_reward_start_update",
                    GraspingTask.GOAL_REWARD_START_UPDATE,
                )
            ),
            float(
                env.task_config.get(
                    "goal_reward_full_update", GraspingTask.GOAL_REWARD_FULL_UPDATE
                )
            ),
        )
        goal_weight = min(goal_step_weight, goal_update_weight)

        distance_reward_scale = float(
            env.task_config.get(
                "distance_reward_scale", GraspingTask.DISTANCE_REWARD_SCALE
            )
        )
        orientation_reward_scale = float(
            env.task_config.get(
                "orientation_reward_scale", GraspingTask.ORIENTATION_REWARD_SCALE
            )
        )
        grasp_reward_scale = float(
            env.task_config.get("grasp_reward_scale", GraspingTask.GRASP_REWARD_SCALE)
        )
        lift_reward_scale = float(
            env.task_config.get("lift_reward_scale", GraspingTask.LIFT_REWARD_SCALE)
        )
        goal_height_reward_scale = float(
            env.task_config.get(
                "goal_height_reward_scale", GraspingTask.GOAL_HEIGHT_REWARD_SCALE
            )
        )
        success_bonus = float(
            env.task_config.get("success_bonus", GraspingTask.SUCCESS_BONUS)
        )
        time_penalty = -float(
            env.task_config.get("time_penalty", GraspingTask.TIME_PENALTY)
        )

        base_distance_reward = distance_reward_scale * features["distance_score"]
        base_orientation_reward = (
            orientation_reward_scale * features["orientation_score"]
        )
        base_grasp_reward = grasp_reward_scale * (
            0.60 * features["grasp_score"] + 0.40 * features["grasp_evidence_score"]
        )
        lift_reward = lift_reward_scale * features["lift_progress"]
        base_goal_reward = goal_height_reward_scale * features["goal_height_score"]

        reward = (
            base_distance_reward
            + base_orientation_reward
            + grasp_weight * base_grasp_reward
            + max(grasp_weight, 0.25) * lift_reward
            + goal_weight * base_goal_reward
            + time_penalty
        )

        is_success = bool(features["is_success"])
        if is_success:
            reward += success_bonus
            done = True

        info.update(
            {
                **features,
                "training_num_timesteps": num_timesteps,
                "training_n_updates": n_updates,
                "grasp_curriculum_weight": grasp_weight,
                "goal_curriculum_weight": goal_weight,
                "distance_reward": base_distance_reward,
                "orientation_reward": base_orientation_reward,
                "grasp_reward": grasp_weight * base_grasp_reward,
                "lift_reward": max(grasp_weight, 0.25) * lift_reward,
                "goal_reward": goal_weight * base_goal_reward,
                "goal_height_reward": goal_weight * base_goal_reward,
                "time_penalty": time_penalty,
                "success_bonus": success_bonus if is_success else 0.0,
                "goal_reward_active": bool(goal_weight > 0.0),
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

    reward_variant = kwargs.pop("reward_variant", config.get("reward_variant", "shaped"))
    if reward_variant == "legacy":
        config["reward_fn"] = GraspingTask.reward_function
    elif reward_variant == "shaped":
        config["reward_fn"] = GraspingTask.reward_function_shaped
    else:
        raise ValueError(f"Unknown reward variant: {reward_variant}")
    config["reward_variant"] = reward_variant

    # Override with kwargs
    config.update(kwargs)

    return config
