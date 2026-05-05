"""Task definitions and configurations for DexPoint learning."""

from typing import Any, Dict, Tuple

import numpy as np


class ReachingTask:
    """End-effector reaching task configuration."""

    NAME = "reaching"
    TARGET_OBJECT = "target_object"
    MAX_EPISODE_STEPS = 150
    EE_DISTANCE_REWARD_SCALE = 1.0
    OPEN_REWARD_SCALE = 0.1
    ACTION_PENALTY_SCALE = 1e-3
    SUCCESS_DISTANCE_THRESHOLD = 0.02
    SUCCESS_OPEN_FRACTION_THRESHOLD = 0.5
    SUCCESS_HOLD_STEPS = 4
    SUCCESS_BONUS = 2.0
    TIME_PENALTY = 1e-3

    @staticmethod
    def _update_success_hold_count(env, in_success_pose: bool) -> int:
        if env.step_count <= 1:
            env._reaching_success_hold_count = 0

        current_hold_count = int(getattr(env, "_reaching_success_hold_count", 0))
        if in_success_pose:
            current_hold_count += 1
        else:
            current_hold_count = 0

        env._reaching_success_hold_count = current_hold_count
        return current_hold_count

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        return {
            "max_episode_steps": ReachingTask.MAX_EPISODE_STEPS,
            "randomize_target_pose": True,
            "reward_fn": ReachingTask.reward_function,
            "reward_variant": "default",
            "task_name": ReachingTask.NAME,
            "target_body_name": ReachingTask.TARGET_OBJECT,
            "ee_distance_reward_scale": ReachingTask.EE_DISTANCE_REWARD_SCALE,
            "open_reward_scale": ReachingTask.OPEN_REWARD_SCALE,
            "action_penalty_scale": ReachingTask.ACTION_PENALTY_SCALE,
            "success_distance_threshold": ReachingTask.SUCCESS_DISTANCE_THRESHOLD,
            "success_open_fraction_threshold": ReachingTask.SUCCESS_OPEN_FRACTION_THRESHOLD,
            "success_hold_steps": ReachingTask.SUCCESS_HOLD_STEPS,
            "success_bonus": ReachingTask.SUCCESS_BONUS,
            "time_penalty": ReachingTask.TIME_PENALTY,
            "terminate_on_target_escape": False,
        }

    @staticmethod
    def reward_function(env) -> Tuple[float, bool, Dict[str, Any]]:
        info = {"task": ReachingTask.NAME, "step": env.step_count}

        target_pos = env.get_target_position()
        ee_pos = env.get_end_effector_position()

        ee_distance = float(np.linalg.norm(target_pos - ee_pos))
        gripper_open_fraction = env.get_gripper_open_fraction()
        ee_distance_reward_scale = float(
            env.task_config.get(
                "ee_distance_reward_scale",
                ReachingTask.EE_DISTANCE_REWARD_SCALE,
            )
        )
        open_reward_scale = float(
            env.task_config.get("open_reward_scale", ReachingTask.OPEN_REWARD_SCALE)
        )
        action_penalty_scale = float(
            env.task_config.get(
                "action_penalty_scale", ReachingTask.ACTION_PENALTY_SCALE
            )
        )
        success_distance_threshold = float(
            env.task_config.get(
                "success_distance_threshold",
                ReachingTask.SUCCESS_DISTANCE_THRESHOLD,
            )
        )
        success_open_fraction_threshold = float(
            env.task_config.get(
                "success_open_fraction_threshold",
                ReachingTask.SUCCESS_OPEN_FRACTION_THRESHOLD,
            )
        )
        success_hold_steps = int(
            env.task_config.get("success_hold_steps", ReachingTask.SUCCESS_HOLD_STEPS)
        )
        success_bonus = float(
            env.task_config.get("success_bonus", ReachingTask.SUCCESS_BONUS)
        )
        time_penalty = -float(
            env.task_config.get("time_penalty", ReachingTask.TIME_PENALTY)
        )

        distance_reward = -ee_distance_reward_scale * ee_distance
        open_reward = open_reward_scale * float(gripper_open_fraction)
        action = np.asarray(
            getattr(env, "_last_action", np.zeros(getattr(env, "robot_dof", 8))),
            dtype=np.float32,
        )
        action_penalty = -action_penalty_scale * float(np.dot(action, action))

        reward = distance_reward + float(open_reward) + action_penalty + time_penalty
        in_success_pose = bool(
            ee_distance <= success_distance_threshold
            and gripper_open_fraction >= success_open_fraction_threshold
        )
        success_hold_count = ReachingTask._update_success_hold_count(
            env, in_success_pose
        )
        is_success = bool(in_success_pose and success_hold_count >= success_hold_steps)
        if is_success:
            reward += success_bonus

        info.update(
            {
                "ee_distance": ee_distance,
                "gripper_open_fraction": float(gripper_open_fraction),
                "in_success_pose": in_success_pose,
                "success_hold_count": success_hold_count,
                "success_hold_steps": success_hold_steps,
                "distance_reward": float(distance_reward),
                "open_reward": float(open_reward),
                "action_penalty": float(action_penalty),
                "success_bonus": success_bonus if is_success else 0.0,
                "is_success": is_success,
                "reward_total": float(reward),
            }
        )

        return float(reward), is_success, info

class LiftingTask:
    """Simple lifting task for fine-tuning from a reaching checkpoint."""

    NAME = "lifting"
    TARGET_OBJECT = "target_object"
    MAX_EPISODE_STEPS = 200
    TIME_PENALTY = 1e-3
    CLOSE_REWARD_SCALE = 0.8
    CLOSE_DISTANCE_THRESHOLD = 0.03
    TABLE_DISTANCE_REWARD_SCALE = 35.0
    UNSUPPORTED_AIR_HEIGHT_THRESHOLD = 0.01

    @staticmethod
    def _update_hold_count(env, attr_name: str, is_active: bool) -> int:
        if env.step_count <= 1:
            setattr(env, attr_name, 0)

        current_hold_count = int(getattr(env, attr_name, 0))
        if is_active:
            current_hold_count += 1
        else:
            current_hold_count = 0

        setattr(env, attr_name, current_hold_count)
        return current_hold_count

    @staticmethod
    def _update_bonus_hold_count(env, qualifies_for_bonus: bool) -> int:
        return LiftingTask._update_hold_count(
            env,
            "_lifting_bonus_hold_count",
            qualifies_for_bonus,
        )

    @staticmethod
    def _get_bonus_reward(hold_count: int) -> float:
        if hold_count <= 0:
            return 0.0
        if hold_count == 1:
            return 0.1
        if hold_count < 4:
            return 0.2
        return 0.4

    @staticmethod
    def _has_target_table_contact(env) -> bool:
        for i in range(env.env.data.ncon):
            contact = env.env.data.contact[i]
            body1 = env.env.model.geom_bodyid[contact.geom1]
            body2 = env.env.model.geom_bodyid[contact.geom2]
            if (body1 == env.target_id and body2 == env.table_body_id) or (
                body1 == env.table_body_id and body2 == env.target_id
            ):
                return True
        return False

    @staticmethod
    def _compute_shared_reward_terms(
        env,
        *,
        bonus_hold_attr: str,
        negative_bonus_hold_attr: str,
    ) -> Dict[str, Any]:
        close_reward_scale = float(
            env.task_config.get(
                "close_reward_scale",
                LiftingTask.CLOSE_REWARD_SCALE,
            )
        )
        close_distance_threshold = float(
            env.task_config.get(
                "close_distance_threshold",
                LiftingTask.CLOSE_DISTANCE_THRESHOLD,
            )
        )
        unsupported_air_height_threshold = float(
            env.task_config.get(
                "unsupported_air_height_threshold",
                LiftingTask.UNSUPPORTED_AIR_HEIGHT_THRESHOLD,
            )
        )

        target_pos = env.get_target_position()
        ee_pos = env.get_end_effector_position()
        gripper_can_distance = float(np.linalg.norm(target_pos - ee_pos))
        gripper_actuator_force = float(abs(env.get_gripper_actuator_force()))
        target_bottom_height = float(env.get_target_bottom_height())
        target_table_distance = float(env.get_target_lift_height())
        gripper_open_fraction = float(env.get_gripper_open_fraction())
        target_table_contact = LiftingTask._has_target_table_contact(env)

        if gripper_can_distance <= close_distance_threshold:
            close_reward = close_reward_scale * min(
                1.0 - gripper_open_fraction,
                0.5,
            ) + (0.02 * np.abs(gripper_actuator_force))
            bonus_reward_active = not target_table_contact
        else:
            close_reward = 0.01 * gripper_open_fraction
            bonus_reward_active = False

        bonus_hold_count = LiftingTask._update_hold_count(
            env,
            bonus_hold_attr,
            bonus_reward_active,
        )
        bonus_reward = LiftingTask._get_bonus_reward(bonus_hold_count)

        negative_bonus_reward_active = bool(
            target_table_distance > unsupported_air_height_threshold
            and gripper_can_distance > close_distance_threshold
            and not target_table_contact
        )
        negative_bonus_hold_count = LiftingTask._update_hold_count(
            env,
            negative_bonus_hold_attr,
            negative_bonus_reward_active,
        )
        negative_bonus_reward = -LiftingTask._get_bonus_reward(
            negative_bonus_hold_count
        )

        return {
            "target_table_distance": target_table_distance,
            "target_bottom_height": target_bottom_height,
            "target_table_contact": target_table_contact,
            "gripper_can_distance": gripper_can_distance,
            "gripper_actuator_force": gripper_actuator_force,
            "gripper_open_fraction": gripper_open_fraction,
            "close_distance_threshold": close_distance_threshold,
            "unsupported_air_height_threshold": unsupported_air_height_threshold,
            "close_reward": float(close_reward),
            "bonus_reward": float(bonus_reward),
            "bonus_reward_active": bonus_reward_active,
            "bonus_hold_count": bonus_hold_count,
            "negative_bonus_reward": float(negative_bonus_reward),
            "negative_bonus_reward_active": negative_bonus_reward_active,
            "negative_bonus_hold_count": negative_bonus_hold_count,
        }

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        return {
            "max_episode_steps": LiftingTask.MAX_EPISODE_STEPS,
            "randomize_target_pose": True,
            "reward_fn": LiftingTask.reward_function,
            "reward_variant": "default",
            "task_name": LiftingTask.NAME,
            "target_body_name": LiftingTask.TARGET_OBJECT,
            "close_reward_scale": LiftingTask.CLOSE_REWARD_SCALE,
            "close_distance_threshold": LiftingTask.CLOSE_DISTANCE_THRESHOLD,
            "table_distance_reward_scale": LiftingTask.TABLE_DISTANCE_REWARD_SCALE,
            "unsupported_air_height_threshold": LiftingTask.UNSUPPORTED_AIR_HEIGHT_THRESHOLD,
            "time_penalty": LiftingTask.TIME_PENALTY,
            "terminate_on_target_escape": False,
        }

    @staticmethod
    def reward_function(env) -> Tuple[float, bool, Dict[str, Any]]:
        time_penalty = -float(
            env.task_config.get("time_penalty", LiftingTask.TIME_PENALTY)
        )
        table_distance_reward_scale = float(
            env.task_config.get(
                "table_distance_reward_scale",
                LiftingTask.TABLE_DISTANCE_REWARD_SCALE,
            )
        )
        shared_terms = LiftingTask._compute_shared_reward_terms(
            env,
            bonus_hold_attr="_lifting_bonus_hold_count",
            negative_bonus_hold_attr="_lifting_negative_bonus_hold_count",
        )

        approach_reward = -shared_terms["gripper_can_distance"]
        table_distance_reward = (
            table_distance_reward_scale * shared_terms["target_table_distance"]
        )
        reward = (
            time_penalty
            + approach_reward
            + shared_terms["close_reward"]
            + shared_terms["bonus_reward"]
            + shared_terms["negative_bonus_reward"]
            + table_distance_reward
        )

        info = {
            "task": LiftingTask.NAME,
            "step": env.step_count,
            **shared_terms,
            "time_penalty": float(time_penalty),
            "approach_reward": float(approach_reward),
            "table_distance_reward": float(table_distance_reward),
            "is_success": False,
            "reward_total": float(reward),
        }
        return float(reward), False, info

class PlacingTask:
    """Lifting task that switches to placement shaping after a successful lift."""

    NAME = "placing"
    TARGET_OBJECT = LiftingTask.TARGET_OBJECT
    MAX_EPISODE_STEPS = LiftingTask.MAX_EPISODE_STEPS
    LIFT_PHASE_COMPLETE_HEIGHT = 0.10
    POST_LIFT_DISTANCE_PENALTY_SCALE = 4.0
    POST_LIFT_TABLE_CONTACT_REWARD = 0.05
    OFF_TABLE_PENALTY_SCALE = 10.0
    CAN_FALL_THRESHOLD = 0.03

    @staticmethod
    def _update_post_lift_phase(env, target_table_distance: float) -> bool:
        if env.step_count <= 1:
            env._placing_post_lift_phase = False

        has_reached_post_lift_phase = bool(
            getattr(env, "_placing_post_lift_phase", False)
        )
        lift_phase_complete_height = float(
            env.task_config.get(
                "lift_phase_complete_height",
                PlacingTask.LIFT_PHASE_COMPLETE_HEIGHT,
            )
        )
        if target_table_distance >= lift_phase_complete_height:
            has_reached_post_lift_phase = True

        env._placing_post_lift_phase = has_reached_post_lift_phase
        return has_reached_post_lift_phase

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        return {
            "max_episode_steps": PlacingTask.MAX_EPISODE_STEPS,
            "randomize_target_pose": True,
            "reward_fn": PlacingTask.reward_function,
            "reward_variant": "default",
            "task_name": PlacingTask.NAME,
            "target_body_name": PlacingTask.TARGET_OBJECT,
            "close_reward_scale": LiftingTask.CLOSE_REWARD_SCALE,
            "close_distance_threshold": LiftingTask.CLOSE_DISTANCE_THRESHOLD,
            "table_distance_reward_scale": LiftingTask.TABLE_DISTANCE_REWARD_SCALE,
            "unsupported_air_height_threshold": LiftingTask.UNSUPPORTED_AIR_HEIGHT_THRESHOLD,
            "time_penalty": LiftingTask.TIME_PENALTY,
            "lift_phase_complete_height": PlacingTask.LIFT_PHASE_COMPLETE_HEIGHT,
            "post_lift_distance_penalty_scale": PlacingTask.POST_LIFT_DISTANCE_PENALTY_SCALE,
            "post_lift_table_contact_reward": PlacingTask.POST_LIFT_TABLE_CONTACT_REWARD,
            "terminate_on_target_escape": False,
            "off_table_penalty_scale": PlacingTask.OFF_TABLE_PENALTY_SCALE,
            "terminate_on_can_fall": True,
            "can_fall_threshold": PlacingTask.CAN_FALL_THRESHOLD,
        }

    @staticmethod
    def reward_function(env) -> Tuple[float, bool, Dict[str, Any]]:
        time_penalty = -float(
            env.task_config.get("time_penalty", LiftingTask.TIME_PENALTY)
        )
        table_distance_reward_scale = float(
            env.task_config.get(
                "table_distance_reward_scale",
                LiftingTask.TABLE_DISTANCE_REWARD_SCALE,
            )
        )
        post_lift_distance_penalty_scale = float(
            env.task_config.get(
                "post_lift_distance_penalty_scale",
                PlacingTask.POST_LIFT_DISTANCE_PENALTY_SCALE,
            )
        )
        post_lift_table_contact_reward = float(
            env.task_config.get(
                "post_lift_table_contact_reward",
                PlacingTask.POST_LIFT_TABLE_CONTACT_REWARD,
            )
        )
        off_table_penalty_scale = float(
            env.task_config.get(
                "off_table_penalty_scale",
                PlacingTask.OFF_TABLE_PENALTY_SCALE,
            )
        )
        terminate_on_can_fall = bool(env.task_config.get("terminate_on_can_fall", True))
        can_fall_threshold = float(
            env.task_config.get("can_fall_threshold", PlacingTask.CAN_FALL_THRESHOLD)
        )
        shared_terms = LiftingTask._compute_shared_reward_terms(
            env,
            bonus_hold_attr="_placing_bonus_hold_count",
            negative_bonus_hold_attr="_placing_negative_bonus_hold_count",
        )

        post_lift_phase_active = PlacingTask._update_post_lift_phase(
            env,
            shared_terms["target_table_distance"],
        )
        approach_reward = -shared_terms["gripper_can_distance"]

        if post_lift_phase_active:
            bonus_reward = 0.0
            bonus_reward_active = False
            table_distance_reward = 0.0
            post_lift_distance_penalty = (
                -post_lift_distance_penalty_scale
                * shared_terms["target_table_distance"]
            )
            table_contact_reward = (
                post_lift_table_contact_reward
                if shared_terms["target_table_contact"]
                else 0.0
            )
        else:
            bonus_reward = shared_terms["bonus_reward"]
            bonus_reward_active = shared_terms["bonus_reward_active"]
            table_distance_reward = (
                table_distance_reward_scale * shared_terms["target_table_distance"]
            )
            post_lift_distance_penalty = 0.0
            table_contact_reward = 0.0

        # Penalise (and optionally terminate) when the can has fallen below the
        # table surface.  target_table_distance clips at 0 so it cannot detect
        # this case; we must read target_pos directly.
        can_below_table = bool(
            env.is_target_below_table(margin=can_fall_threshold)
        )
        off_table_penalty = -off_table_penalty_scale if can_below_table else 0.0
        done = terminate_on_can_fall and can_below_table

        reward = (
            time_penalty
            + approach_reward
            + shared_terms["close_reward"]
            + bonus_reward
            + shared_terms["negative_bonus_reward"]
            + table_distance_reward
            + post_lift_distance_penalty
            + table_contact_reward
            + off_table_penalty
        )

        info = {
            "task": PlacingTask.NAME,
            "step": env.step_count,
            **shared_terms,
            "bonus_reward": float(bonus_reward),
            "bonus_reward_active": bonus_reward_active,
            "time_penalty": float(time_penalty),
            "approach_reward": float(approach_reward),
            "table_distance_reward": float(table_distance_reward),
            "post_lift_phase_active": post_lift_phase_active,
            "lift_phase_complete_height": float(
                env.task_config.get(
                    "lift_phase_complete_height",
                    PlacingTask.LIFT_PHASE_COMPLETE_HEIGHT,
                )
            ),
            "post_lift_distance_penalty": float(post_lift_distance_penalty),
            "table_contact_reward": float(table_contact_reward),
            "can_below_table": can_below_table,
            "off_table_penalty": float(off_table_penalty),
            "is_success": False,
            "reward_total": float(reward),
        }
        return float(reward), done, info

class LiftingOnlyTask:
    """Pure lifting task with no reach or grasp shaping rewards."""

    NAME = "lifting_only"
    TARGET_OBJECT = LiftingTask.TARGET_OBJECT
    MAX_EPISODE_STEPS = LiftingTask.MAX_EPISODE_STEPS

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        return {
            "max_episode_steps": LiftingOnlyTask.MAX_EPISODE_STEPS,
            "randomize_target_pose": True,
            "reward_fn": LiftingOnlyTask.reward_function,
            "reward_variant": "default",
            "task_name": LiftingOnlyTask.NAME,
            "target_body_name": LiftingOnlyTask.TARGET_OBJECT,
            "terminate_on_target_escape": False,
        }

    @staticmethod
    def reward_function(env) -> Tuple[float, bool, Dict[str, Any]]:
        time_penalty = -1e-3
        close_distance_threshold = 0.03

        target_pos = env.get_target_position()
        ee_pos = env.get_end_effector_position()
        gripper_can_distance = float(np.linalg.norm(target_pos - ee_pos))
        gripper_actuator_force = float(abs(env.get_gripper_actuator_force()))
        target_bottom_height = float(env.get_target_bottom_height())
        target_table_distance = float(env.get_target_lift_height())
        gripper_open_fraction = float(env.get_gripper_open_fraction())

        bonus_reward_active = False
        if gripper_can_distance <= close_distance_threshold:
            bonus_reward_active = True
            for i in range(env.env.data.ncon):
                contact = env.env.data.contact[i]
                body1 = env.env.model.geom_bodyid[contact.geom1]
                body2 = env.env.model.geom_bodyid[contact.geom2]
                if (body1 == env.target_id and body2 == env.table_body_id) or (
                    body1 == env.table_body_id and body2 == env.target_id
                ):
                    bonus_reward_active = False
                    break

        bonus_hold_count = LiftingTask._update_bonus_hold_count(
            env, bonus_reward_active
        )
        bonus_reward = LiftingTask._get_bonus_reward(bonus_hold_count)

        reward = time_penalty + (400 * (target_table_distance**2)) + bonus_reward

        info = {
            "task": LiftingOnlyTask.NAME,
            "step": env.step_count,
            "target_table_distance": target_table_distance,
            "target_bottom_height": target_bottom_height,
            "gripper_can_distance": gripper_can_distance,
            "gripper_actuator_force": gripper_actuator_force,
            "gripper_open_fraction": gripper_open_fraction,
            "bonus_reward": float(bonus_reward),
            "bonus_reward_active": bonus_reward_active,
            "bonus_hold_count": bonus_hold_count,
            "close_reward": 0.0,
            "time_penalty": time_penalty,
            "reward_total": float(reward),
        }
        return float(reward), False, info

class GraspingTask:
    """Object grasping task configuration."""

    NAME = "grasping"
    TARGET_OBJECT = "target_object"
    SUCCESS_BONUS = 0.5
    TIME_PENALTY = 1e-3
    FAILURE_PENALTY = -1.0
    FAILURE_XY_MARGIN = 0.05
    FAILURE_Z_MARGIN = 0.01
    REACHING_ONLY_END_STEP = 500_000
    CLOSE_REWARD_FULL_STEP = 2_200_000
    LIFTING_BLEND_FULL_STEP = 3_500_000
    CLOSE_REWARD_DECAY_END_STEP = 12_000_000
    BONUS_REWARD_FULL_BOOST_STEP = 20_000_000
    TABLE_DISTANCE_FULL_BOOST_STEP = 20_000_000
    MAX_BONUS_WEIGHT = 2.0
    MAX_TABLE_DISTANCE_WEIGHT = 2.0

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
    def _compute_reaching_terms(env) -> Dict[str, Any]:
        target_pos = env.get_target_position()
        ee_pos = env.get_end_effector_position()

        ee_distance = float(np.linalg.norm(target_pos - ee_pos))
        gripper_open_fraction = float(env.get_gripper_open_fraction())
        ee_distance_reward_scale = float(
            env.task_config.get(
                "ee_distance_reward_scale",
                ReachingTask.EE_DISTANCE_REWARD_SCALE,
            )
        )
        open_reward_scale = float(
            env.task_config.get("open_reward_scale", ReachingTask.OPEN_REWARD_SCALE)
        )
        action_penalty_scale = float(
            env.task_config.get(
                "action_penalty_scale", ReachingTask.ACTION_PENALTY_SCALE
            )
        )
        success_distance_threshold = float(
            env.task_config.get(
                "success_distance_threshold",
                ReachingTask.SUCCESS_DISTANCE_THRESHOLD,
            )
        )
        success_open_fraction_threshold = float(
            env.task_config.get(
                "success_open_fraction_threshold",
                ReachingTask.SUCCESS_OPEN_FRACTION_THRESHOLD,
            )
        )
        success_hold_steps = int(
            env.task_config.get("success_hold_steps", ReachingTask.SUCCESS_HOLD_STEPS)
        )
        success_bonus = float(
            env.task_config.get("reaching_success_bonus", ReachingTask.SUCCESS_BONUS)
        )
        time_penalty = -float(
            env.task_config.get("time_penalty", ReachingTask.TIME_PENALTY)
        )

        distance_reward = -ee_distance_reward_scale * ee_distance
        open_reward = open_reward_scale * gripper_open_fraction
        action = np.asarray(
            getattr(env, "_last_action", np.zeros(getattr(env, "robot_dof", 8))),
            dtype=np.float32,
        )
        action_penalty = -action_penalty_scale * float(np.dot(action, action))
        in_success_pose = bool(
            ee_distance <= success_distance_threshold
            and gripper_open_fraction >= success_open_fraction_threshold
        )
        success_hold_count = ReachingTask._update_success_hold_count(
            env, in_success_pose
        )
        is_success = bool(in_success_pose and success_hold_count >= success_hold_steps)
        success_bonus_value = success_bonus if is_success else 0.0
        reward_total = (
            distance_reward
            + open_reward
            + action_penalty
            + time_penalty
            + success_bonus_value
        )

        return {
            "ee_distance": ee_distance,
            "gripper_open_fraction": gripper_open_fraction,
            "in_success_pose": in_success_pose,
            "success_hold_count": success_hold_count,
            "success_hold_steps": success_hold_steps,
            "distance_reward": float(distance_reward),
            "open_reward": float(open_reward),
            "action_penalty": float(action_penalty),
            "time_penalty": float(time_penalty),
            "success_bonus": float(success_bonus_value),
            "is_success": is_success,
            "reward_total": float(reward_total),
        }

    @staticmethod
    def _compute_lifting_terms(env) -> Dict[str, Any]:
        target_pos = env.get_target_position()
        ee_pos = env.get_end_effector_position()
        gripper_can_distance = float(np.linalg.norm(target_pos - ee_pos))
        gripper_actuator_force = float(abs(env.get_gripper_actuator_force()))
        target_bottom_height = float(env.get_target_bottom_height())
        target_table_distance = float(env.get_target_lift_height())
        gripper_open_fraction = float(env.get_gripper_open_fraction())

        close_distance_threshold = float(
            env.task_config.get("close_distance_threshold", 0.03)
        )
        close_reward_scale = float(env.task_config.get("close_reward_scale", 1.0))
        approach_reward_scale = float(env.task_config.get("approach_reward_scale", 1.0))
        table_distance_reward_scale = float(
            env.task_config.get("table_distance_reward_scale", 40.0)
        )

        time_penalty = -float(
            env.task_config.get("lifting_time_penalty", GraspingTask.TIME_PENALTY)
        )

        bonus_reward_active = False
        if gripper_can_distance <= close_distance_threshold:
            close_reward = close_reward_scale * min(
                1.0 - gripper_open_fraction, 0.5
            ) + (0.02 * np.abs(gripper_actuator_force))
            bonus_reward_active = True
            for i in range(env.env.data.ncon):
                contact = env.env.data.contact[i]
                body1 = env.env.model.geom_bodyid[contact.geom1]
                body2 = env.env.model.geom_bodyid[contact.geom2]
                if (body1 == env.target_id and body2 == env.table_body_id) or (
                    body1 == env.table_body_id and body2 == env.target_id
                ):
                    bonus_reward_active = False
                    break
        else:
            close_reward = 0.01 * gripper_open_fraction

        bonus_hold_count = LiftingTask._update_bonus_hold_count(
            env, bonus_reward_active
        )
        bonus_reward = LiftingTask._get_bonus_reward(bonus_hold_count)
        approach_reward = -approach_reward_scale * gripper_can_distance
        table_distance_reward = table_distance_reward_scale * target_table_distance
        reward_total = (
            time_penalty
            + approach_reward
            + float(close_reward)
            + float(bonus_reward)
            + table_distance_reward
        )

        return {
            "gripper_can_distance": gripper_can_distance,
            "gripper_actuator_force": gripper_actuator_force,
            "gripper_open_fraction": gripper_open_fraction,
            "target_table_distance": target_table_distance,
            "target_bottom_height": target_bottom_height,
            "bonus_reward_active": bonus_reward_active,
            "bonus_hold_count": bonus_hold_count,
            "close_reward": float(close_reward),
            "bonus_reward": float(bonus_reward),
            "approach_reward": float(approach_reward),
            "table_distance_reward": float(table_distance_reward),
            "time_penalty": float(time_penalty),
            "reward_total": float(reward_total),
        }

    @staticmethod
    def _get_curriculum_weights(env, num_timesteps: float) -> Dict[str, float]:
        reaching_only_end_step = float(
            env.task_config.get(
                "reaching_only_end_step", GraspingTask.REACHING_ONLY_END_STEP
            )
        )
        close_reward_full_step = float(
            env.task_config.get(
                "close_reward_full_step", GraspingTask.CLOSE_REWARD_FULL_STEP
            )
        )
        lifting_blend_full_step = float(
            env.task_config.get(
                "lifting_blend_full_step", GraspingTask.LIFTING_BLEND_FULL_STEP
            )
        )
        close_reward_decay_end_step = float(
            env.task_config.get(
                "close_reward_decay_end_step",
                GraspingTask.CLOSE_REWARD_DECAY_END_STEP,
            )
        )
        bonus_reward_full_boost_step = float(
            env.task_config.get(
                "bonus_reward_full_boost_step",
                GraspingTask.BONUS_REWARD_FULL_BOOST_STEP,
            )
        )
        table_distance_full_boost_step = float(
            env.task_config.get(
                "table_distance_full_boost_step",
                GraspingTask.TABLE_DISTANCE_FULL_BOOST_STEP,
            )
        )
        max_bonus_weight = float(
            env.task_config.get("max_bonus_weight", GraspingTask.MAX_BONUS_WEIGHT)
        )
        max_table_distance_weight = float(
            env.task_config.get(
                "max_table_distance_weight",
                GraspingTask.MAX_TABLE_DISTANCE_WEIGHT,
            )
        )

        close_phase_weight = GraspingTask._safe_progress_ratio(
            num_timesteps,
            reaching_only_end_step,
            close_reward_full_step,
        )
        lifting_weight = GraspingTask._safe_progress_ratio(
            num_timesteps,
            close_reward_full_step,
            lifting_blend_full_step,
        )

        if num_timesteps < close_reward_full_step:
            reaching_weight = 1.0
            close_weight = close_phase_weight
        elif num_timesteps < lifting_blend_full_step:
            reaching_weight = 1.0 - lifting_weight
            close_weight = 1.0
        else:
            reaching_weight = 0.0
            close_weight = 1.0 - GraspingTask._safe_progress_ratio(
                num_timesteps,
                lifting_blend_full_step,
                close_reward_decay_end_step,
            )

        if num_timesteps < lifting_blend_full_step:
            bonus_weight = lifting_weight
            table_distance_weight = lifting_weight
        else:
            bonus_weight = 1.0 + (
                max_bonus_weight - 1.0
            ) * GraspingTask._safe_progress_ratio(
                num_timesteps,
                lifting_blend_full_step,
                bonus_reward_full_boost_step,
            )
            table_distance_weight = 1.0 + (
                max_table_distance_weight - 1.0
            ) * GraspingTask._safe_progress_ratio(
                num_timesteps,
                lifting_blend_full_step,
                table_distance_full_boost_step,
            )

        if num_timesteps < reaching_only_end_step:
            curriculum_phase = "reaching"
            curriculum_phase_index = 0.0
        elif num_timesteps < close_reward_full_step:
            curriculum_phase = "reach_plus_close"
            curriculum_phase_index = 1.0
        elif num_timesteps < lifting_blend_full_step:
            curriculum_phase = "blend_to_lifting"
            curriculum_phase_index = 2.0
        else:
            curriculum_phase = "lift_dominant"
            curriculum_phase_index = 3.0

        return {
            "curriculum_phase": curriculum_phase,
            "curriculum_phase_index": float(curriculum_phase_index),
            "reaching_weight": float(reaching_weight),
            "close_phase_weight": float(close_phase_weight),
            "lifting_weight": float(lifting_weight),
            "close_weight": float(close_weight),
            "bonus_weight": float(bonus_weight),
            "table_distance_weight": float(table_distance_weight),
        }

    @staticmethod
    def _blend_reward_terms(
        reaching_terms: Dict[str, Any],
        lifting_terms: Dict[str, Any],
        weights: Dict[str, float],
    ) -> Dict[str, float]:
        reaching_weight = weights["reaching_weight"]
        lifting_weight = weights["lifting_weight"]

        reaching_reward = reaching_weight * reaching_terms["reward_total"]
        close_reward_component = weights["close_weight"] *  lifting_terms["close_reward"]
        approach_reward_component = lifting_weight * lifting_terms["approach_reward"]
        bonus_reward_component = weights["bonus_weight"] * lifting_terms["bonus_reward"]
        target_table_distance_component = (
            weights["table_distance_weight"] * lifting_terms["table_distance_reward"]
        )
        lifting_time_penalty_component = lifting_weight * lifting_terms["time_penalty"]
        lifting_reward = (
            close_reward_component
            + approach_reward_component
            + bonus_reward_component
            + target_table_distance_component
            + lifting_time_penalty_component
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
            "approach_reward_component": float(approach_reward_component),
            "target_table_distance_component": float(target_table_distance_component),
            "lifting_time_penalty_component": float(lifting_time_penalty_component),
            "time_penalty": float(
                reaching_weight * reaching_terms["time_penalty"]
                + lifting_time_penalty_component
            ),
            "reward_total": float(reaching_reward + lifting_reward),
        }

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """Get default task configuration."""
        return {
            "max_episode_steps": 200,
            "randomize_target_pose": True,
            "reward_fn": GraspingTask.reward_function_shaped,
            "reward_variant": "shaped",
            "task_name": GraspingTask.NAME,
            "target_body_name": GraspingTask.TARGET_OBJECT,
            "success_bonus": 0.0,
            "time_penalty": GraspingTask.TIME_PENALTY,
            "failure_penalty": GraspingTask.FAILURE_PENALTY,
            "failure_xy_margin": GraspingTask.FAILURE_XY_MARGIN,
            "failure_z_margin": GraspingTask.FAILURE_Z_MARGIN,
            "reaching_only_end_step": GraspingTask.REACHING_ONLY_END_STEP,
            "close_reward_full_step": GraspingTask.CLOSE_REWARD_FULL_STEP,
            "lifting_blend_full_step": GraspingTask.LIFTING_BLEND_FULL_STEP,
            "close_reward_decay_end_step": GraspingTask.CLOSE_REWARD_DECAY_END_STEP,
            "bonus_reward_full_boost_step": GraspingTask.BONUS_REWARD_FULL_BOOST_STEP,
            "table_distance_full_boost_step": GraspingTask.TABLE_DISTANCE_FULL_BOOST_STEP,
            "max_bonus_weight": GraspingTask.MAX_BONUS_WEIGHT,
            "max_table_distance_weight": GraspingTask.MAX_TABLE_DISTANCE_WEIGHT,
            "close_distance_threshold": 0.03,
            "close_reward_scale": 1.0,
            "approach_reward_scale": 1.0,
            "table_distance_reward_scale": 40.0,
            "lifting_time_penalty": GraspingTask.TIME_PENALTY,
        }

    @staticmethod
    def reward_function(env) -> Tuple[float, bool, Dict[str, Any]]:
        return GraspingTask.reward_function_shaped(env)

    @staticmethod
    def reward_function_shaped(env) -> Tuple[float, bool, Dict[str, Any]]:
        done = False
        info = {"task": f"{GraspingTask.NAME}_shaped", "step": env.step_count}

        reaching_terms = GraspingTask._compute_reaching_terms(env)
        lifting_terms = GraspingTask._compute_lifting_terms(env)
        num_timesteps, n_updates = GraspingTask._get_training_progress(env)
        weights = GraspingTask._get_curriculum_weights(env, num_timesteps)
        reward_terms = GraspingTask._blend_reward_terms(
            reaching_terms, lifting_terms, weights
        )

        success_bonus = float(env.task_config.get("success_bonus", 0.0))
        # is_success = bool(lifting_terms["is_success"])
        # success_bonus_value = success_bonus if is_success else 0.0
        reward = reward_terms["reward_total"]
        # if is_success:
        #    reward += success_bonus_value
        #    done = True

        info.update(
            {
                "training_num_timesteps": num_timesteps,
                "training_n_updates": n_updates,
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
                "ee_distance": reaching_terms["ee_distance"],
                "gripper_can_distance": lifting_terms["gripper_can_distance"],
                "target_table_distance": lifting_terms["target_table_distance"],
                "gripper_open_fraction": lifting_terms["gripper_open_fraction"],
                "gripper_actuator_force": lifting_terms["gripper_actuator_force"],
                "close_reward": lifting_terms["close_reward"],
                "bonus_reward": lifting_terms["bonus_reward"],
                "approach_reward": lifting_terms["approach_reward"],
                "table_distance_reward": lifting_terms["table_distance_reward"],
                "in_success_pose": reaching_terms["in_success_pose"],
                "success_hold_count": reaching_terms["success_hold_count"],
                "success_hold_steps": reaching_terms["success_hold_steps"],
                "bonus_reward_active": lifting_terms["bonus_reward_active"],
                "bonus_hold_count": lifting_terms["bonus_hold_count"],
                # "success_bonus": float(success_bonus_value),
                #  "is_success": is_success,
                "goal_reward_active": False,
            }
        )

        return float(reward), done, info

class PlacingV2Task:
    """Staged placing task with a late goal-reaching curriculum."""

    NAME = "placing_v2"
    TARGET_OBJECT = LiftingTask.TARGET_OBJECT
    MAX_EPISODE_STEPS = 250
    TIME_PENALTY = LiftingTask.TIME_PENALTY
    LIFT_PHASE_COMPLETE_HEIGHT = 0.1
    PLACE_DISTANCE_REWARD_SCALE = 20.0
    SUCCESS_DISTANCE_THRESHOLD = 0.03
    SUCCESS_BONUS = 2.0
    GOAL_REWARD_RAMP_STEPS = 4_000_000
    MAX_GOAL_WEIGHT = 10.0

    @staticmethod
    def _update_post_lift_phase(env, target_table_distance: float) -> bool:
        if env.step_count <= 1:
            env._placing_v2_post_lift_phase = False

        has_reached = bool(getattr(env, "_placing_v2_post_lift_phase", False))
        lift_threshold = float(
            env.task_config.get(
                "lift_phase_complete_height",
                PlacingV2Task.LIFT_PHASE_COMPLETE_HEIGHT,
            )
        )
        if target_table_distance >= lift_threshold:
            has_reached = True

        env._placing_v2_post_lift_phase = has_reached
        return has_reached

    @staticmethod
    def _compute_lifting_terms(env) -> Dict[str, Any]:
        shared_terms = LiftingTask._compute_shared_reward_terms(
            env,
            bonus_hold_attr="_placing_v2_bonus_hold_count",
            negative_bonus_hold_attr="_placing_v2_negative_bonus_hold_count",
        )
        table_distance_reward_scale = float(
            env.task_config.get(
                "table_distance_reward_scale",
                LiftingTask.TABLE_DISTANCE_REWARD_SCALE,
            )
        )
        time_penalty = -float(
            env.task_config.get("time_penalty", PlacingV2Task.TIME_PENALTY)
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
        weights = dict(GraspingTask._get_curriculum_weights(env, num_timesteps))
        lifting_blend_full_step = float(
            env.task_config.get(
                "lifting_blend_full_step",
                GraspingTask.LIFTING_BLEND_FULL_STEP,
            )
        )
        goal_reward_start_step = float(
            env.task_config.get("goal_reward_start_step", lifting_blend_full_step)
        )
        goal_reward_full_step = float(
            env.task_config.get(
                "goal_reward_full_step",
                goal_reward_start_step + PlacingV2Task.GOAL_REWARD_RAMP_STEPS,
            )
        )
        max_goal_weight = float(
            env.task_config.get("max_goal_weight", PlacingV2Task.MAX_GOAL_WEIGHT)
        )
        goal_weight = max_goal_weight * GraspingTask._safe_progress_ratio(
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
    def _blend_reward_terms(
        reaching_terms: Dict[str, Any],
        lifting_terms: Dict[str, Any],
        *,
        goal_reward: float,
        weights: Dict[str, float],
    ) -> Dict[str, float]:
        reaching_weight = weights["reaching_weight"]
        lifting_weight = weights["lifting_weight"]

        reaching_reward = reaching_weight * reaching_terms["reward_total"]
        close_reward_component = weights["close_weight"] * lifting_terms["close_reward"]
        approach_reward_component = lifting_weight * lifting_terms["approach_reward"]
        bonus_reward_component = weights["bonus_weight"] * lifting_terms["bonus_reward"]
        negative_bonus_reward_component = (
            lifting_weight * lifting_terms["negative_bonus_reward"]
        )
        target_table_distance_component = (
            weights["table_distance_weight"] * lifting_terms["table_distance_reward"]
        )
        lifting_time_penalty_component = lifting_weight * lifting_terms["time_penalty"]
        goal_reward_component = weights["goal_weight"] * goal_reward
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
            "max_episode_steps": PlacingV2Task.MAX_EPISODE_STEPS,
            "randomize_target_pose": True,
            "reward_fn": PlacingV2Task.reward_function,
            "reward_variant": "default",
            "task_name": PlacingV2Task.NAME,
            "target_body_name": PlacingV2Task.TARGET_OBJECT,
            "close_reward_scale": LiftingTask.CLOSE_REWARD_SCALE,
            "close_distance_threshold": LiftingTask.CLOSE_DISTANCE_THRESHOLD,
            "table_distance_reward_scale": LiftingTask.TABLE_DISTANCE_REWARD_SCALE,
            "unsupported_air_height_threshold": LiftingTask.UNSUPPORTED_AIR_HEIGHT_THRESHOLD,
            "time_penalty": PlacingV2Task.TIME_PENALTY,
            "lift_phase_complete_height": PlacingV2Task.LIFT_PHASE_COMPLETE_HEIGHT,
            "place_distance_reward_scale": PlacingV2Task.PLACE_DISTANCE_REWARD_SCALE,
            "success_distance_threshold": PlacingV2Task.SUCCESS_DISTANCE_THRESHOLD,
            "success_bonus": PlacingV2Task.SUCCESS_BONUS,
            "reaching_only_end_step": GraspingTask.REACHING_ONLY_END_STEP,
            "close_reward_full_step": GraspingTask.CLOSE_REWARD_FULL_STEP,
            "lifting_blend_full_step": GraspingTask.LIFTING_BLEND_FULL_STEP,
            "close_reward_decay_end_step": GraspingTask.CLOSE_REWARD_DECAY_END_STEP,
            "goal_reward_ramp_steps": PlacingV2Task.GOAL_REWARD_RAMP_STEPS,
            "max_goal_weight": PlacingV2Task.MAX_GOAL_WEIGHT,
            "terminate_on_target_escape": False,
        }

    @staticmethod
    def reward_function(env) -> Tuple[float, bool, Dict[str, Any]]:
        reaching_terms = GraspingTask._compute_reaching_terms(env)
        lifting_terms = PlacingV2Task._compute_lifting_terms(env)
        num_timesteps, n_updates = GraspingTask._get_training_progress(env)
        weights = PlacingV2Task._get_curriculum_weights(env, num_timesteps)
        place_distance_reward_scale = float(
            env.task_config.get(
                "place_distance_reward_scale",
                PlacingV2Task.PLACE_DISTANCE_REWARD_SCALE,
            )
        )
        success_distance_threshold = float(
            env.task_config.get(
                "success_distance_threshold",
                PlacingV2Task.SUCCESS_DISTANCE_THRESHOLD,
            )
        )
        success_bonus = float(
            env.task_config.get("success_bonus", PlacingV2Task.SUCCESS_BONUS)
        )

        post_lift_phase_active = PlacingV2Task._update_post_lift_phase(
            env,
            lifting_terms["target_table_distance"],
        )

        target_pos = env.get_target_position()
        goal_position = np.asarray(env.goal_position, dtype=np.float32)
        goal_distance = float(np.linalg.norm(target_pos - goal_position))

        place_reward = -place_distance_reward_scale * goal_distance
        staged_place_reward = place_reward if post_lift_phase_active else 0.0
        reward_terms = PlacingV2Task._blend_reward_terms(
            reaching_terms,
            lifting_terms,
            goal_reward=staged_place_reward,
            weights=weights,
        )

        is_success = bool(
            post_lift_phase_active
            and lifting_terms["target_table_contact"]
            and goal_distance <= success_distance_threshold
        )
        success_bonus_value = success_bonus if is_success else 0.0
        reward = reward_terms["reward_total"] + success_bonus_value

        info = {
            "task": PlacingV2Task.NAME,
            "step": env.step_count,
            "training_num_timesteps": num_timesteps,
            "training_n_updates": n_updates,
            **lifting_terms,
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
            "place_reward": float(place_reward),
            "staged_place_reward": float(staged_place_reward),
            "goal_distance": float(goal_distance),
            "post_lift_phase_active": post_lift_phase_active,
            "lift_phase_complete_height": float(
                env.task_config.get(
                    "lift_phase_complete_height",
                    PlacingV2Task.LIFT_PHASE_COMPLETE_HEIGHT,
                )
            ),
            "is_success": is_success,
            "success_bonus": float(success_bonus_value),
        }
        return float(reward), is_success, info


def create_task_config(task_name: str, **kwargs) -> Dict[str, Any]:
    """
    Create a task configuration by name.

    Args:
        task_name: 'grasping', 'reaching', 'lifting', 'lifting_only', or 'placing'
        **kwargs: Additional parameters to override defaults

    Returns:
        Task configuration dictionary
    """
    if task_name == "grasping":
        config = GraspingTask.get_default_config()
    elif task_name == "reaching":
        config = ReachingTask.get_default_config()
    elif task_name == "lifting":
        config = LiftingTask.get_default_config()
    elif task_name == "lifting_only":
        config = LiftingOnlyTask.get_default_config()
    elif task_name == "placing":
        config = PlacingTask.get_default_config()
    elif task_name == "placing_v2":
        config = PlacingV2Task.get_default_config()
    else:
        raise ValueError(f"Unknown task: {task_name}")

    reward_variant = kwargs.pop(
        "reward_variant", config.get("reward_variant", "default")
    )
    if task_name == "grasping":
        if reward_variant == "legacy":
            config["reward_fn"] = GraspingTask.reward_function
        elif reward_variant == "shaped":
            config["reward_fn"] = GraspingTask.reward_function_shaped
        else:
            raise ValueError(f"Unknown reward variant: {reward_variant}")
    elif task_name == "lifting":
        if reward_variant in {"default", "lifting"}:
            config["reward_fn"] = LiftingTask.reward_function
        else:
            raise ValueError(f"Unknown reward variant: {reward_variant}")
    elif task_name == "lifting_only":
        if reward_variant in {"default", "lifting_only"}:
            config["reward_fn"] = LiftingOnlyTask.reward_function
        else:
            raise ValueError(f"Unknown reward variant: {reward_variant}")
    elif task_name == "placing":
        if reward_variant in {"default", "placing"}:
            config["reward_fn"] = PlacingTask.reward_function
        else:
            raise ValueError(f"Unknown reward variant: {reward_variant}")
    elif task_name == "placing_v2":
        if reward_variant in {"default", "placing_v2"}:
            config["reward_fn"] = PlacingV2Task.reward_function
        else:
            raise ValueError(f"Unknown reward variant: {reward_variant}")
    elif reward_variant in {"default", "reaching"}:
        config["reward_fn"] = ReachingTask.reward_function
    else:
        raise ValueError(f"Unknown reward variant: {reward_variant}")
    config["reward_variant"] = reward_variant

    # Override with kwargs
    config.update(kwargs)

    return config
