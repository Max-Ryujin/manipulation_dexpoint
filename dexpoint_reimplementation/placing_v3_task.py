"""Placing V3 task definition with smoother lift-to-place shaping."""

from typing import Any, Dict, Tuple

import numpy as np


def build_placing_v3_task(reaching_task_cls, lifting_task_cls, grasping_task_cls):
    """Build the PlacingV3Task class from existing task primitives."""

    class PlacingV3Task:
        """Placing task with smooth lift gating, settle shaping, and release logic."""

        NAME = "placing_v3"
        TARGET_OBJECT = lifting_task_cls.TARGET_OBJECT
        MAX_EPISODE_STEPS = 250
        TIME_PENALTY = lifting_task_cls.TIME_PENALTY
        PLACEMENT_GATE_MIN_HEIGHT = 0.06
        PLACEMENT_GATE_FULL_HEIGHT = 0.1
        XY_DISTANCE_PENALTY_SCALE = 8.0
        XY_DISTANCE_CAP = 0.25
        Z_DISTANCE_PENALTY_SCALE = 6.0
        Z_DISTANCE_CAP = 0.08
        XY_ALIGNMENT_THRESHOLD = 0.05
        SETTLE_XY_THRESHOLD = 0.04
        SETTLE_Z_THRESHOLD = 0.025
        SETTLE_REWARD = 0.5
        RELEASE_OPEN_THRESHOLD = 0.65
        RELEASE_EE_DISTANCE_THRESHOLD = 0.05
        RELEASE_REWARD_SCALE = 0.6
        GRASP_PENALTY_SCALE = 0.35
        SUCCESS_DISTANCE_THRESHOLD = 0.03
        SUCCESS_OPEN_FRACTION_THRESHOLD = 0.7
        SUCCESS_EE_DISTANCE_THRESHOLD = 0.06
        SUCCESS_HOLD_STEPS = 4
        SUCCESS_BONUS = 2.0
        OFF_TABLE_PENALTY_SCALE = 10.0
        CAN_FALL_THRESHOLD = 0.05
        GOAL_REWARD_RAMP_STEPS = 4_000_000
        MAX_GOAL_WEIGHT = 8.0

        @staticmethod
        def _smooth_progress(value: float, start: float, end: float) -> float:
            if end <= start:
                return float(value >= end)
            raw = float(np.clip((value - start) / (end - start), 0.0, 1.0))
            return raw * raw * (3.0 - 2.0 * raw)

        @staticmethod
        def _update_success_hold_count(env, is_stable_success: bool) -> int:
            return lifting_task_cls._update_hold_count(
                env,
                "_placing_v3_success_hold_count",
                is_stable_success,
            )

        @staticmethod
        def _compute_lifting_terms(env) -> Dict[str, Any]:
            shared_terms = lifting_task_cls._compute_shared_reward_terms(
                env,
                bonus_hold_attr="_placing_v3_bonus_hold_count",
                negative_bonus_hold_attr="_placing_v3_negative_bonus_hold_count",
            )
            table_distance_reward_scale = float(
                env.task_config.get(
                    "table_distance_reward_scale",
                    lifting_task_cls.TABLE_DISTANCE_REWARD_SCALE,
                )
            )
            time_penalty = -float(
                env.task_config.get("time_penalty", PlacingV3Task.TIME_PENALTY)
            )
            approach_reward = -shared_terms["gripper_can_distance"]
            table_distance_reward = (
                table_distance_reward_scale * shared_terms["target_table_distance"]
            )
            reward_total = (
                time_penalty
                + approach_reward
                + shared_terms["close_reward"]
                + shared_terms["bonus_reward"]
                + shared_terms["negative_bonus_reward"]
                + table_distance_reward
            )

            return {
                **shared_terms,
                "approach_reward": float(approach_reward),
                "table_distance_reward": float(table_distance_reward),
                "time_penalty": float(time_penalty),
                "reward_total": float(reward_total),
            }

        @staticmethod
        def _get_curriculum_weights(env, num_timesteps: float) -> Dict[str, float]:
            weights = dict(grasping_task_cls._get_curriculum_weights(env, num_timesteps))
            lifting_blend_full_step = float(
                env.task_config.get(
                    "lifting_blend_full_step",
                    grasping_task_cls.LIFTING_BLEND_FULL_STEP,
                )
            )
            goal_reward_start_step = float(
                env.task_config.get("goal_reward_start_step", lifting_blend_full_step)
            )
            goal_reward_full_step = float(
                env.task_config.get(
                    "goal_reward_full_step",
                    goal_reward_start_step + PlacingV3Task.GOAL_REWARD_RAMP_STEPS,
                )
            )
            max_goal_weight = float(
                env.task_config.get("max_goal_weight", PlacingV3Task.MAX_GOAL_WEIGHT)
            )
            goal_weight = max_goal_weight * grasping_task_cls._safe_progress_ratio(
                num_timesteps,
                goal_reward_start_step,
                goal_reward_full_step,
            )

            weights["bonus_weight"] = min(weights["bonus_weight"], 1.0)
            weights["table_distance_weight"] = min(
                weights["table_distance_weight"],
                1.0,
            )
            weights["goal_weight"] = float(goal_weight)
            weights["goal_reward_start_step"] = float(goal_reward_start_step)
            weights["goal_reward_full_step"] = float(goal_reward_full_step)
            return weights

        @staticmethod
        def _compute_goal_terms(
            env,
            *,
            lifting_terms: Dict[str, Any],
        ) -> Dict[str, Any]:
            target_pos = env.get_target_position()
            goal_position = np.asarray(env.goal_position, dtype=np.float32)
            goal_delta = target_pos - goal_position
            xy_distance = float(np.linalg.norm(goal_delta[:2]))
            z_distance = float(abs(goal_delta[2]))
            goal_distance = float(np.linalg.norm(goal_delta))

            placement_gate_min_height = float(
                env.task_config.get(
                    "placement_gate_min_height",
                    PlacingV3Task.PLACEMENT_GATE_MIN_HEIGHT,
                )
            )
            placement_gate_full_height = float(
                env.task_config.get(
                    "placement_gate_full_height",
                    PlacingV3Task.PLACEMENT_GATE_FULL_HEIGHT,
                )
            )
            xy_distance_penalty_scale = float(
                env.task_config.get(
                    "xy_distance_penalty_scale",
                    PlacingV3Task.XY_DISTANCE_PENALTY_SCALE,
                )
            )
            xy_distance_cap = float(
                env.task_config.get("xy_distance_cap", PlacingV3Task.XY_DISTANCE_CAP)
            )
            z_distance_penalty_scale = float(
                env.task_config.get(
                    "z_distance_penalty_scale",
                    PlacingV3Task.Z_DISTANCE_PENALTY_SCALE,
                )
            )
            z_distance_cap = float(
                env.task_config.get("z_distance_cap", PlacingV3Task.Z_DISTANCE_CAP)
            )
            xy_alignment_threshold = float(
                env.task_config.get(
                    "xy_alignment_threshold",
                    PlacingV3Task.XY_ALIGNMENT_THRESHOLD,
                )
            )
            settle_xy_threshold = float(
                env.task_config.get(
                    "settle_xy_threshold",
                    PlacingV3Task.SETTLE_XY_THRESHOLD,
                )
            )
            settle_z_threshold = float(
                env.task_config.get(
                    "settle_z_threshold",
                    PlacingV3Task.SETTLE_Z_THRESHOLD,
                )
            )
            settle_reward_value = float(
                env.task_config.get("settle_reward", PlacingV3Task.SETTLE_REWARD)
            )
            release_open_threshold = float(
                env.task_config.get(
                    "release_open_threshold",
                    PlacingV3Task.RELEASE_OPEN_THRESHOLD,
                )
            )
            release_ee_distance_threshold = float(
                env.task_config.get(
                    "release_ee_distance_threshold",
                    PlacingV3Task.RELEASE_EE_DISTANCE_THRESHOLD,
                )
            )
            release_reward_scale = float(
                env.task_config.get(
                    "release_reward_scale",
                    PlacingV3Task.RELEASE_REWARD_SCALE,
                )
            )
            grasp_penalty_scale = float(
                env.task_config.get(
                    "grasp_penalty_scale",
                    PlacingV3Task.GRASP_PENALTY_SCALE,
                )
            )

            placement_gate = PlacingV3Task._smooth_progress(
                lifting_terms["target_table_distance"],
                placement_gate_min_height,
                placement_gate_full_height,
            )
            xy_alignment_gate = PlacingV3Task._smooth_progress(
                xy_alignment_threshold - xy_distance,
                0.0,
                xy_alignment_threshold,
            )
            settle_gate = float(
                lifting_terms["target_table_contact"]
                and xy_distance <= settle_xy_threshold
                and z_distance <= settle_z_threshold
            )
            release_open_ratio = float(
                np.clip(
                    (lifting_terms["gripper_open_fraction"] - release_open_threshold)
                    / max(1e-6, 1.0 - release_open_threshold),
                    0.0,
                    1.0,
                )
            )
            release_clearance_ratio = float(
                np.clip(
                    lifting_terms["gripper_can_distance"]
                    / max(1e-6, release_ee_distance_threshold),
                    0.0,
                    1.0,
                )
            )

            xy_transport_penalty = -placement_gate * xy_distance_penalty_scale * min(
                xy_distance,
                xy_distance_cap,
            )
            z_descent_penalty = (
                -placement_gate
                * xy_alignment_gate
                * z_distance_penalty_scale
                * min(z_distance, z_distance_cap)
            )
            settle_reward = placement_gate * settle_gate * settle_reward_value
            release_reward = (
                placement_gate
                * settle_gate
                * release_reward_scale
                * release_open_ratio
                * release_clearance_ratio
            )
            grasp_after_contact_penalty = (
                -placement_gate
                * settle_gate
                * grasp_penalty_scale
                * (1.0 - release_open_ratio)
                * (1.0 - release_clearance_ratio)
            )

            return {
                "goal_distance": float(goal_distance),
                "goal_xy_distance": float(xy_distance),
                "goal_z_distance": float(z_distance),
                "placement_gate": float(placement_gate),
                "xy_alignment_gate": float(xy_alignment_gate),
                "settle_gate": float(settle_gate),
                "xy_transport_penalty": float(xy_transport_penalty),
                "z_descent_penalty": float(z_descent_penalty),
                "settle_reward": float(settle_reward),
                "release_reward": float(release_reward),
                "grasp_after_contact_penalty": float(grasp_after_contact_penalty),
                "release_open_ratio": float(release_open_ratio),
                "release_clearance_ratio": float(release_clearance_ratio),
            }

        @staticmethod
        def _blend_reward_terms(
            reaching_terms: Dict[str, Any],
            lifting_terms: Dict[str, Any],
            goal_terms: Dict[str, Any],
            *,
            weights: Dict[str, float],
        ) -> Dict[str, float]:
            reaching_weight = weights["reaching_weight"]
            lifting_weight = weights["lifting_weight"]
            placement_gate = goal_terms["placement_gate"]
            settle_gate = goal_terms["settle_gate"]

            reaching_reward = reaching_weight * reaching_terms["reward_total"]
            close_reward_component = (
                weights["close_weight"]
                * (1.0 - settle_gate)
                * lifting_terms["close_reward"]
            )
            approach_reward_component = (
                lifting_weight
                * (1.0 - settle_gate)
                * lifting_terms["approach_reward"]
            )
            bonus_reward_component = (
                weights["bonus_weight"]
                * (1.0 - placement_gate)
                * lifting_terms["bonus_reward"]
            )
            negative_bonus_reward_component = (
                lifting_weight * lifting_terms["negative_bonus_reward"]
            )
            target_table_distance_component = (
                weights["table_distance_weight"]
                * (1.0 - placement_gate)
                * lifting_terms["table_distance_reward"]
            )
            lifting_time_penalty_component = lifting_weight * lifting_terms["time_penalty"]
            goal_reward_component = weights["goal_weight"] * (
                goal_terms["xy_transport_penalty"]
                + goal_terms["z_descent_penalty"]
                + goal_terms["settle_reward"]
                + goal_terms["release_reward"]
                + goal_terms["grasp_after_contact_penalty"]
            )
            lifting_reward = (
                close_reward_component
                + approach_reward_component
                + bonus_reward_component
                + negative_bonus_reward_component
                + target_table_distance_component
                + lifting_time_penalty_component
                + goal_reward_component
            )

            return {
                "reaching_reward": float(reaching_reward),
                "lifting_reward": float(lifting_reward),
                "reaching_distance_reward": float(
                    reaching_weight * reaching_terms["distance_reward"]
                ),
                "reaching_open_reward": float(
                    reaching_weight * reaching_terms["open_reward"]
                ),
                "reaching_action_penalty": float(
                    reaching_weight * reaching_terms["action_penalty"]
                ),
                "reaching_time_penalty": float(
                    reaching_weight * reaching_terms["time_penalty"]
                ),
                "reaching_success_bonus": float(
                    reaching_weight * reaching_terms["success_bonus"]
                ),
                "close_reward_component": float(close_reward_component),
                "bonus_reward_component": float(bonus_reward_component),
                "negative_bonus_reward_component": float(
                    negative_bonus_reward_component
                ),
                "approach_reward_component": float(approach_reward_component),
                "target_table_distance_component": float(
                    target_table_distance_component
                ),
                "lifting_time_penalty_component": float(lifting_time_penalty_component),
                "goal_reward_component": float(goal_reward_component),
                "time_penalty": float(
                    reaching_weight * reaching_terms["time_penalty"]
                    + lifting_time_penalty_component
                ),
                "reward_total": float(reaching_reward + lifting_reward),
            }

        @staticmethod
        def get_default_config() -> Dict[str, Any]:
            return {
                "max_episode_steps": PlacingV3Task.MAX_EPISODE_STEPS,
                "randomize_target_pose": True,
                "reward_fn": PlacingV3Task.reward_function,
                "reward_variant": "default",
                "task_name": PlacingV3Task.NAME,
                "target_body_name": PlacingV3Task.TARGET_OBJECT,
                "close_reward_scale": lifting_task_cls.CLOSE_REWARD_SCALE,
                "close_distance_threshold": lifting_task_cls.CLOSE_DISTANCE_THRESHOLD,
                "table_distance_reward_scale": lifting_task_cls.TABLE_DISTANCE_REWARD_SCALE,
                "unsupported_air_height_threshold": lifting_task_cls.UNSUPPORTED_AIR_HEIGHT_THRESHOLD,
                "time_penalty": PlacingV3Task.TIME_PENALTY,
                "placement_gate_min_height": PlacingV3Task.PLACEMENT_GATE_MIN_HEIGHT,
                "placement_gate_full_height": PlacingV3Task.PLACEMENT_GATE_FULL_HEIGHT,
                "xy_distance_penalty_scale": PlacingV3Task.XY_DISTANCE_PENALTY_SCALE,
                "xy_distance_cap": PlacingV3Task.XY_DISTANCE_CAP,
                "z_distance_penalty_scale": PlacingV3Task.Z_DISTANCE_PENALTY_SCALE,
                "z_distance_cap": PlacingV3Task.Z_DISTANCE_CAP,
                "xy_alignment_threshold": PlacingV3Task.XY_ALIGNMENT_THRESHOLD,
                "settle_xy_threshold": PlacingV3Task.SETTLE_XY_THRESHOLD,
                "settle_z_threshold": PlacingV3Task.SETTLE_Z_THRESHOLD,
                "settle_reward": PlacingV3Task.SETTLE_REWARD,
                "release_open_threshold": PlacingV3Task.RELEASE_OPEN_THRESHOLD,
                "release_ee_distance_threshold": PlacingV3Task.RELEASE_EE_DISTANCE_THRESHOLD,
                "release_reward_scale": PlacingV3Task.RELEASE_REWARD_SCALE,
                "grasp_penalty_scale": PlacingV3Task.GRASP_PENALTY_SCALE,
                "success_distance_threshold": PlacingV3Task.SUCCESS_DISTANCE_THRESHOLD,
                "success_open_fraction_threshold": PlacingV3Task.SUCCESS_OPEN_FRACTION_THRESHOLD,
                "success_ee_distance_threshold": PlacingV3Task.SUCCESS_EE_DISTANCE_THRESHOLD,
                "success_hold_steps": PlacingV3Task.SUCCESS_HOLD_STEPS,
                "success_bonus": PlacingV3Task.SUCCESS_BONUS,
                "reaching_only_end_step": grasping_task_cls.REACHING_ONLY_END_STEP,
                "close_reward_full_step": grasping_task_cls.CLOSE_REWARD_FULL_STEP,
                "lifting_blend_full_step": grasping_task_cls.LIFTING_BLEND_FULL_STEP,
                "close_reward_decay_end_step": grasping_task_cls.CLOSE_REWARD_DECAY_END_STEP,
                "goal_reward_ramp_steps": PlacingV3Task.GOAL_REWARD_RAMP_STEPS,
                "max_goal_weight": PlacingV3Task.MAX_GOAL_WEIGHT,
                "off_table_penalty_scale": PlacingV3Task.OFF_TABLE_PENALTY_SCALE,
                "terminate_on_can_fall": True,
                "can_fall_threshold": PlacingV3Task.CAN_FALL_THRESHOLD,
                "terminate_on_target_escape": False,
            }

        @staticmethod
        def reward_function(env) -> Tuple[float, bool, Dict[str, Any]]:
            reaching_terms = grasping_task_cls._compute_reaching_terms(env)
            lifting_terms = PlacingV3Task._compute_lifting_terms(env)
            num_timesteps, n_updates = grasping_task_cls._get_training_progress(env)
            weights = PlacingV3Task._get_curriculum_weights(env, num_timesteps)
            goal_terms = PlacingV3Task._compute_goal_terms(
                env,
                lifting_terms=lifting_terms,
            )
            reward_terms = PlacingV3Task._blend_reward_terms(
                reaching_terms,
                lifting_terms,
                goal_terms,
                weights=weights,
            )

            success_distance_threshold = float(
                env.task_config.get(
                    "success_distance_threshold",
                    PlacingV3Task.SUCCESS_DISTANCE_THRESHOLD,
                )
            )
            success_open_fraction_threshold = float(
                env.task_config.get(
                    "success_open_fraction_threshold",
                    PlacingV3Task.SUCCESS_OPEN_FRACTION_THRESHOLD,
                )
            )
            success_ee_distance_threshold = float(
                env.task_config.get(
                    "success_ee_distance_threshold",
                    PlacingV3Task.SUCCESS_EE_DISTANCE_THRESHOLD,
                )
            )
            success_hold_steps = int(
                env.task_config.get(
                    "success_hold_steps",
                    PlacingV3Task.SUCCESS_HOLD_STEPS,
                )
            )
            success_bonus = float(
                env.task_config.get("success_bonus", PlacingV3Task.SUCCESS_BONUS)
            )
            terminate_on_can_fall = bool(
                env.task_config.get("terminate_on_can_fall", True)
            )
            can_fall_threshold = float(
                env.task_config.get(
                    "can_fall_threshold",
                    PlacingV3Task.CAN_FALL_THRESHOLD,
                )
            )
            off_table_penalty_scale = float(
                env.task_config.get(
                    "off_table_penalty_scale",
                    PlacingV3Task.OFF_TABLE_PENALTY_SCALE,
                )
            )

            stable_success_pose = bool(
                lifting_terms["target_table_contact"]
                and goal_terms["goal_distance"] <= success_distance_threshold
                and lifting_terms["gripper_open_fraction"]
                >= success_open_fraction_threshold
                and lifting_terms["gripper_can_distance"]
                >= success_ee_distance_threshold
                and not env.is_target_below_table(margin=can_fall_threshold)
            )
            success_hold_count = PlacingV3Task._update_success_hold_count(
                env,
                stable_success_pose,
            )
            is_success = bool(success_hold_count >= success_hold_steps)
            success_bonus_value = success_bonus if is_success else 0.0

            can_below_table = bool(env.is_target_below_table(margin=can_fall_threshold))
            off_table_penalty = -off_table_penalty_scale if can_below_table else 0.0
            done = bool(is_success or (terminate_on_can_fall and can_below_table))
            reward = reward_terms["reward_total"] + success_bonus_value + off_table_penalty

            info = {
                "task": PlacingV3Task.NAME,
                "step": env.step_count,
                "training_num_timesteps": num_timesteps,
                "training_n_updates": n_updates,
                **lifting_terms,
                **goal_terms,
                **reward_terms,
                "reward_total": float(reward),
                "curriculum_phase": weights["curriculum_phase"],
                "curriculum_phase_index": weights["curriculum_phase_index"],
                "reaching_weight": weights["reaching_weight"],
                "close_phase_weight": weights["close_phase_weight"],
                "lifting_weight": weights["lifting_weight"],
                "close_weight": weights["close_weight"],
                "bonus_weight": weights["bonus_weight"],
                "table_distance_weight": weights["table_distance_weight"],
                "goal_weight": weights["goal_weight"],
                "goal_reward_start_step": weights["goal_reward_start_step"],
                "goal_reward_full_step": weights["goal_reward_full_step"],
                "ee_distance": reaching_terms["ee_distance"],
                "gripper_open_fraction": lifting_terms["gripper_open_fraction"],
                "bonus_reward": lifting_terms["bonus_reward"],
                "close_reward": lifting_terms["close_reward"],
                "negative_bonus_reward": lifting_terms["negative_bonus_reward"],
                "approach_reward": lifting_terms["approach_reward"],
                "table_distance_reward": lifting_terms["table_distance_reward"],
                "stable_success_pose": stable_success_pose,
                "success_hold_count": success_hold_count,
                "success_hold_steps": success_hold_steps,
                "success_bonus": float(success_bonus_value),
                "is_success": is_success,
                "can_below_table": can_below_table,
                "off_table_penalty": float(off_table_penalty),
            }
            return float(reward), done, info

    return PlacingV3Task