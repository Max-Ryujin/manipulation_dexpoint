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
            self.training_env.set_attr("training_num_timesteps", num_timesteps)
            self.training_env.set_attr("training_n_updates", n_updates)

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
            wandb.log(aggregated_metrics)
