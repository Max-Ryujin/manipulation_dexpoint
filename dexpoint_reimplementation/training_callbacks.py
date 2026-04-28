"""Training callbacks for richer task and reward logging."""

from collections import Counter, defaultdict
from numbers import Number
from typing import TYPE_CHECKING

import numpy as np
import wandb

if TYPE_CHECKING:
    from dexart_baselines.stable_baselines3.common.callbacks import BaseCallback
else:
    from stable_baselines3.common.callbacks import BaseCallback


class TaskInfoLoggingCallback(BaseCallback):
    """Aggregate scalar task metrics from info and report them per rollout."""

    WANDB_TASK_METRIC_WHITELISTS = {
        "grasping": {
            "rollout/task/training_num_timesteps",
            "rollout/task/reward_total",
            "rollout/task/reaching_reward",
            "rollout/task/lifting_reward",
            "rollout/task/reaching_distance_reward",
            "rollout/task/reaching_open_reward",
            "rollout/task/reaching_action_penalty",
            "rollout/task/reaching_time_penalty",
            "rollout/task/reaching_success_bonus",
            "rollout/task/close_reward_component",
            "rollout/task/approach_reward_component",
            "rollout/task/bonus_reward_component",
            "rollout/task/target_table_distance_component",
            "rollout/task/lifting_time_penalty_component",
            "rollout/task/reaching_weight",
            "rollout/task/close_phase_weight",
            "rollout/task/lifting_weight",
            "rollout/task/close_weight",
            "rollout/task/bonus_weight",
            "rollout/task/table_distance_weight",
            "rollout/task/curriculum_phase_index",
            "rollout/task/ee_distance",
            "rollout/task/gripper_can_distance",
            "rollout/task/target_table_distance",
            "rollout/task/gripper_open_fraction",
            "rollout/task/gripper_actuator_force",
            "rollout/task/close_reward",
            "rollout/task/bonus_reward",
            "rollout/task/approach_reward",
            "rollout/task/table_distance_reward",
            "rollout/task/in_success_pose",
            "rollout/task/bonus_reward_active",
            "rollout/task/is_success",
            "episode/reward_total",
            "episode/reaching_reward",
            "episode/lifting_reward",
            "episode/close_reward_component",
            "episode/approach_reward_component",
            "episode/bonus_reward_component",
            "episode/target_table_distance_component",
            "episode/reaching_weight",
            "episode/close_weight",
            "episode/bonus_weight",
            "episode/table_distance_weight",
            "episode/target_table_distance",
            "episode/bonus_reward",
            "episode/close_reward",
            "episode/is_success",
            "episode/reward",
            "episode/length",
        },
        "lifting": {
            "rollout/task/target_table_distance",
            "rollout/task/target_table_contact",
            "rollout/task/gripper_can_distance",
            "rollout/task/gripper_open_fraction",
            "rollout/task/close_reward",
            "rollout/task/bonus_reward",
            "rollout/task/negative_bonus_reward",
            "rollout/task/negative_bonus_reward_active",
            "rollout/task/time_penalty",
            "rollout/task/is_success",
            "rollout/task/reward_total",
            "episode/is_success",
            "episode/reward_total",
            "episode/reward",
            "episode/length",
            "episode/target_table_distance",
            "episode/gripper_can_distance",
            "episode/gripper_open_fraction",
            "episode/bonus_reward",
            "episode/negative_bonus_reward",
            "episode/gripper_actuator_force",
            "episode/close_reward",
        },
        "placing": {
            "rollout/task/target_table_distance",
            "rollout/task/target_table_contact",
            "rollout/task/gripper_can_distance",
            "rollout/task/gripper_open_fraction",
            "rollout/task/close_reward",
            "rollout/task/bonus_reward",
            "rollout/task/negative_bonus_reward",
            "rollout/task/negative_bonus_reward_active",
            "rollout/task/post_lift_phase_active",
            "rollout/task/post_lift_distance_penalty",
            "rollout/task/table_contact_reward",
            "rollout/task/table_distance_reward",
            "rollout/task/time_penalty",
            "rollout/task/is_success",
            "rollout/task/reward_total",
            "episode/is_success",
            "episode/reward_total",
            "episode/reward",
            "episode/length",
            "episode/target_table_distance",
            "episode/target_table_contact",
            "episode/gripper_can_distance",
            "episode/gripper_open_fraction",
            "episode/bonus_reward",
            "episode/negative_bonus_reward",
            "episode/post_lift_phase_active",
            "episode/post_lift_distance_penalty",
            "episode/table_contact_reward",
            "episode/gripper_actuator_force",
            "episode/close_reward",
        },
        "lifting_only": {
            "rollout/task/target_table_distance",
            "rollout/task/gripper_can_distance",
            "rollout/task/gripper_open_fraction",
            "rollout/task/close_reward",
            "rollout/task/time_penalty",
            "rollout/task/is_success",
            "rollout/task/reward_total",
            "episode/is_success",
            "episode/reward_total",
            "episode/reward",
            "episode/length",
            "episode/target_table_distance",
            "episode/gripper_can_distance",
            "episode/gripper_open_fraction",
            "episode/bonus_reward",
            "episode/gripper_actuator_force",
            "episode/close_reward",
        },
    }

    EPISODE_REASON_ALIASES = {
        "success": "success",
        "step_limit": "step_limit",
        "empty_pointcloud": "empty_pc",
        "target_below_table": "below_tbl",
        "target_out_of_workspace": "out_ws",
        "other": "other",
    }

    def __init__(self, use_wandb: bool = False, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.use_wandb = use_wandb
        self.rollout_metrics = defaultdict(list)
        self.episode_metrics = defaultdict(list)
        self.episode_reason_counts = Counter()
        self.completed_episodes = 0

    @staticmethod
    def _is_scalar(value) -> bool:
        return isinstance(value, (Number, np.bool_))

    @classmethod
    def _episode_reason_metric_prefix(cls, reason: str) -> str:
        alias = cls.EPISODE_REASON_ALIASES.get(reason, reason)
        return f"rollout/ep_end/{alias}"

    def _on_rollout_start(self) -> None:
        self.rollout_metrics.clear()
        self.episode_metrics.clear()
        self.episode_reason_counts.clear()
        self.completed_episodes = 0

        if self.training_env is not None:
            num_timesteps = int(getattr(self.model, "num_timesteps", 0))
            n_updates = int(getattr(self.model, "_n_updates", 0))
            self.training_env.env_method(
                "set_training_progress",
                num_timesteps,
                n_updates,
            )

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for index, info in enumerate(infos):
            episode_info = info.get("episode")
            if episode_info is not None:
                self.episode_metrics["episode/reward"].append(float(episode_info["r"]))
                self.episode_metrics["episode/length"].append(float(episode_info["l"]))

            for key, value in info.items():
                if key in {"episode", "terminal_observation"}:
                    continue
                if self._is_scalar(value):
                    self.rollout_metrics[f"task/{key}"].append(float(value))
                    if index < len(dones) and bool(dones[index]):
                        self.episode_metrics[f"episode/{key}"].append(float(value))

            if index < len(dones) and bool(dones[index]):
                self.completed_episodes += 1
                failure_reason = info.get("episode_failure_reason")
                if bool(info.get("is_success", False)):
                    self.episode_reason_counts["success"] += 1
                elif isinstance(failure_reason, str) and failure_reason:
                    self.episode_reason_counts[failure_reason] += 1
                elif bool(info.get("step_limit_reached", False)):
                    self.episode_reason_counts["step_limit"] += 1
                else:
                    self.episode_reason_counts["other"] += 1

        return True

    def _on_rollout_end(self) -> None:
        aggregated_metrics = {}

        for metric_name, values in self.rollout_metrics.items():
            if values:
                aggregated_metrics[f"rollout/{metric_name}"] = float(np.mean(values))

        for metric_name, values in self.episode_metrics.items():
            if values:
                aggregated_metrics[metric_name] = float(np.mean(values))

        if self.completed_episodes > 0:
            for reason, count in self.episode_reason_counts.items():
                metric_prefix = self._episode_reason_metric_prefix(reason)
                aggregated_metrics[f"{metric_prefix}/rate"] = float(
                    count / self.completed_episodes
                )
                aggregated_metrics[f"{metric_prefix}/count"] = float(count)

        for metric_name, value in aggregated_metrics.items():
            self.logger.record(metric_name, value)

        if self.use_wandb and wandb.run is not None and aggregated_metrics:
            wandb_metrics = dict(aggregated_metrics)
            whitelist = self.WANDB_TASK_METRIC_WHITELISTS.get(
                self.training_env.get_attr("task_name")[0]
                if self.training_env is not None
                else None
            )
            if whitelist is not None:
                wandb_metrics = {
                    metric_name: value
                    for metric_name, value in aggregated_metrics.items()
                    if metric_name in whitelist
                }
            if wandb_metrics:
                wandb.log(
                    wandb_metrics,
                    step=int(getattr(self.model, "num_timesteps", 0)),
                )
