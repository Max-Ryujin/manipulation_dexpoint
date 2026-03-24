"""Training callbacks for richer task and reward logging."""

from collections import defaultdict
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

    def __init__(self, use_wandb: bool = False, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.use_wandb = use_wandb
        self.rollout_metrics = defaultdict(list)
        self.episode_metrics = defaultdict(list)

    @staticmethod
    def _is_scalar(value) -> bool:
        return isinstance(value, (Number, np.bool_))

    def _on_rollout_start(self) -> None:
        self.rollout_metrics.clear()
        self.episode_metrics.clear()

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

        return True

    def _on_rollout_end(self) -> None:
        aggregated_metrics = {}

        for metric_name, values in self.rollout_metrics.items():
            if values:
                aggregated_metrics[f"rollout/{metric_name}"] = float(np.mean(values))

        for metric_name, values in self.episode_metrics.items():
            if values:
                aggregated_metrics[metric_name] = float(np.mean(values))

        for metric_name, value in aggregated_metrics.items():
            self.logger.record(metric_name, value)

        if self.use_wandb and wandb.run is not None and aggregated_metrics:
            wandb.log(aggregated_metrics)
