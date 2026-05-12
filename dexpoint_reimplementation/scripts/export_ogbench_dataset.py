from __future__ import annotations

"""Export DexPoint checkpoint rollouts as OGBench-compatible offline datasets."""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from _script_bootstrap import DEXPOINT_ROOT, REPO_ROOT, ensure_script_imports

ensure_script_imports()


_OUTPUT_DIR = DEXPOINT_ROOT / "offline_datasets"
_YCB_SIM_ROOT = REPO_ROOT / "YCB_sim"
DEFAULT_YCB_OBJECT_ROOT = (
    REPO_ROOT
    / "dexart_baselines"
    / "pretrain"
    / "data"
    / "ycb_raw"
    / "005_tomato_soup_can"
)
YCB_ASSET_SOURCE_RAW = "raw"
YCB_ASSET_SOURCE_YCB_SIM = "ycb_sim"
_DEFAULT_TRAIN_CAMERA_WIDTH = 576
_DEFAULT_TRAIN_CAMERA_HEIGHT = 432


def to_jsonable(value: Any) -> Any:
    """Convert nested metadata to JSON-safe Python primitives."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    return repr(value)


def add_batch_dimension(obs: Dict[str, Any]) -> Dict[str, Any]:
    """Add a leading batch dimension to every numpy observation field."""
    batched_obs: Dict[str, Any] = {}
    for key, value in obs.items():
        if isinstance(value, np.ndarray):
            batched_obs[key] = np.expand_dims(value, axis=0)
        else:
            batched_obs[key] = value
    return batched_obs


def load_run_config(checkpoint_path: Path) -> Dict[str, Any]:
    """Load the sibling training config when present."""
    config_path = checkpoint_path.parent / "config.json"
    if not config_path.exists():
        return {}

    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def create_output_dir(checkpoint_path: Path, output_dir: Optional[str]) -> Path:
    """Create the export directory for the generated dataset."""
    if output_dir is not None:
        run_dir = Path(output_dir).expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = _OUTPUT_DIR / f"{checkpoint_path.stem}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def get_pretrained_policy_kwargs(
    checkpoint_path: Path,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load policy kwargs from a saved RL checkpoint for safe restoration."""
    from dexart_baselines.stable_baselines3.common.save_util import load_from_zip_file

    data, _, _ = load_from_zip_file(str(checkpoint_path), device="cpu")
    policy_kwargs = dict(data.get("policy_kwargs", {}))

    # Full policy checkpoints already contain PointNet weights.
    policy_kwargs["pointnet_checkpoint_path"] = None
    return policy_kwargs, data


def _resolve_existing_object_root(
    candidate_roots: Sequence[Path],
    *,
    object_name: Optional[str],
    asset_source: str,
) -> Path:
    for candidate in candidate_roots:
        resolved = candidate.expanduser()
        if resolved.exists():
            return resolved.resolve()

    if object_name is not None:
        if asset_source == YCB_ASSET_SOURCE_YCB_SIM:
            fallback = (_YCB_SIM_ROOT / object_name).resolve()
            if fallback.exists():
                return fallback
        raw_parent = DEFAULT_YCB_OBJECT_ROOT.parent / object_name
        if raw_parent.exists():
            return raw_parent.resolve()

    default_root = DEFAULT_YCB_OBJECT_ROOT.resolve()
    if default_root.exists():
        return default_root

    return Path(candidate_roots[0]).expanduser().resolve()


def resolve_ycb_selection(
    run_config: Dict[str, Any],
    *,
    ycb_object_root_override: Optional[str],
    ycb_object_name_override: Optional[str],
    ycb_asset_source_override: Optional[str],
    target_scale_override: Optional[float],
) -> Dict[str, Any]:
    """Resolve which YCB object/assets to use for dataset collection."""
    if ycb_object_root_override is not None and ycb_object_name_override is not None:
        raise ValueError(
            "Pass either --ycb-object-root or --ycb-object-name, not both."
        )

    env_config = dict(run_config.get("env_config", {}))
    configured_object_names = list(run_config.get("ycb_object_names") or [])

    ycb_asset_source = str(
        ycb_asset_source_override
        or env_config.get("ycb_asset_source")
        or YCB_ASSET_SOURCE_YCB_SIM
    ).lower()
    if ycb_asset_source not in {YCB_ASSET_SOURCE_RAW, YCB_ASSET_SOURCE_YCB_SIM}:
        raise ValueError(
            f"Unsupported YCB asset source '{ycb_asset_source}'. "
            f"Expected '{YCB_ASSET_SOURCE_RAW}' or '{YCB_ASSET_SOURCE_YCB_SIM}'."
        )

    configured_root = Path(
        env_config.get("ycb_object_root", DEFAULT_YCB_OBJECT_ROOT.as_posix())
    ).expanduser()
    fallback_object_name = configured_object_names[0] if configured_object_names else configured_root.name
    object_name = ycb_object_name_override or fallback_object_name

    if ycb_object_root_override is not None:
        chosen_root = Path(ycb_object_root_override).expanduser()
    elif ycb_object_name_override is not None:
        if ycb_asset_source == YCB_ASSET_SOURCE_YCB_SIM:
            chosen_root = _YCB_SIM_ROOT / ycb_object_name_override
        else:
            chosen_root = DEFAULT_YCB_OBJECT_ROOT.parent / ycb_object_name_override
    else:
        chosen_root = configured_root

    resolved_root = _resolve_existing_object_root(
        [chosen_root, configured_root],
        object_name=object_name,
        asset_source=ycb_asset_source,
    )

    target_scale_value = (
        target_scale_override
        if target_scale_override is not None
        else env_config.get("target_scale", 1.0)
    )
    target_scale = float(target_scale_value)
    if target_scale <= 0:
        raise ValueError(f"Target scale must be positive, got {target_scale}")

    return {
        "ycb_object_root": resolved_root,
        "ycb_object_name": resolved_root.name,
        "ycb_asset_source": ycb_asset_source,
        "target_scale": target_scale,
        "ycb_object_names": configured_object_names,
    }


def build_environment(
    task_name: str,
    run_config: Dict[str, Any],
    *,
    ycb_selection: Dict[str, Any],
    seed: Optional[int],
    fixed_object_name: Optional[str],
) -> Any:
    """Create a data-collection environment compatible with training."""
    from franka_gym_env import FrankaGymEnvironment
    from tasks import create_task_config

    env_config = dict(run_config.get("env_config", {}))
    reward_variant = run_config.get("reward_variant") or env_config.get(
        "reward_variant"
    )
    if reward_variant is None:
        reward_variant = "shaped" if task_name == "grasping" else "default"

    env = FrankaGymEnvironment(
        xml_path=None,
        task_name=task_name,
        ycb_object_root=ycb_selection["ycb_object_root"].as_posix(),
        ycb_asset_source=str(ycb_selection["ycb_asset_source"]),
        target_scale=float(ycb_selection["target_scale"]),
        num_points=int(env_config.get("num_points", 512)),
        camera_height=int(
            env_config.get("training_camera_height", _DEFAULT_TRAIN_CAMERA_HEIGHT)
        ),
        camera_width=int(
            env_config.get("training_camera_width", _DEFAULT_TRAIN_CAMERA_WIDTH)
        ),
        camera_names=env_config.get("camera_names"),
        rate=200.0,
        frame_skip=10,
        visualize_pointclouds=False,
        use_depth_only_pointcloud=bool(
            env_config.get("use_depth_only_pointcloud", True)
        ),
        ycb_object_names=ycb_selection.get("ycb_object_names"),
    )
    task_config = create_task_config(
        task_name,
        target_body_name=env_config.get("target_body_name", env.target_body_name),
        reward_variant=reward_variant,
    )
    env.configure_task(task_config)
    if seed is not None:
        env.seed(seed)
    if fixed_object_name is not None:
        env.set_fixed_object(fixed_object_name)
    return env


def load_agent(
    checkpoint_path: Path,
    env: Any,
    agent_name: str,
) -> Tuple[str, Any, Dict[str, Any], Dict[str, Any]]:
    """Load a PPO or A2C checkpoint, optionally auto-detecting the algorithm."""
    from dexart_baselines.stable_baselines3.a2c import A2C
    from dexart_baselines.stable_baselines3.ppo import PPO

    policy_kwargs, checkpoint_data = get_pretrained_policy_kwargs(checkpoint_path)
    custom_objects = {"policy_kwargs": policy_kwargs}
    candidates = [agent_name] if agent_name != "auto" else ["ppo", "a2c"]
    load_errors: Dict[str, str] = {}

    for candidate in candidates:
        algorithm_cls = PPO if candidate == "ppo" else A2C
        try:
            model = algorithm_cls.load(
                str(checkpoint_path),
                env=env,
                device="auto",
                custom_objects=custom_objects,
            )
            return candidate, model, policy_kwargs, checkpoint_data
        except Exception as exc:  # pragma: no cover - depends on checkpoint contents
            load_errors[candidate] = str(exc)

    raise RuntimeError(
        "Failed to load checkpoint. Tried algorithms: "
        + ", ".join(f"{name} ({error})" for name, error in load_errors.items())
    )


def get_observation_layout(observation: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Create a stable flattened observation layout from the first observation."""
    preferred_order = [
        "pointcloud",
        "joint_state",
        "ee_position",
        "target_position",
        "goal_position",
    ]
    field_order = [key for key in preferred_order if key in observation]
    field_order.extend(key for key in observation.keys() if key not in field_order)

    fields: List[Dict[str, Any]] = []
    offset = 0
    for key in field_order:
        value = np.asarray(observation[key], dtype=np.float32)
        size = int(value.size)
        fields.append(
            {
                "key": key,
                "shape": list(value.shape),
                "size": size,
                "start": offset,
                "end": offset + size,
            }
        )
        offset += size

    return {"fields": fields, "dim": offset}


def flatten_observation(
    observation: Dict[str, np.ndarray],
    observation_layout: Dict[str, Any],
) -> np.ndarray:
    """Flatten a dict observation into one float32 vector using the saved layout."""
    flattened_fields: List[np.ndarray] = []
    for field in observation_layout["fields"]:
        value = np.asarray(observation[field["key"]], dtype=np.float32)
        if list(value.shape) != field["shape"]:
            raise ValueError(
                f"Observation field '{field['key']}' changed shape from "
                f"{field['shape']} to {list(value.shape)}"
            )
        flattened_fields.append(value.reshape(-1))

    return np.concatenate(flattened_fields, axis=0).astype(np.float32, copy=False)


def finalize_dataset(dataset_buffers: Dict[str, List[Any]]) -> Dict[str, np.ndarray]:
    """Convert collected trajectory lists into numpy arrays with stable dtypes."""
    dtype_overrides = {
        "observations": np.float32,
        "actions": np.float32,
        "rewards": np.float32,
        "masks": np.float32,
        "qpos": np.float32,
        "qvel": np.float32,
        "terminals": bool,
        "timeouts": bool,
        "successes": bool,
        "object_ids": np.int32,
    }

    finalized: Dict[str, np.ndarray] = {}
    for key, values in dataset_buffers.items():
        finalized[key] = np.asarray(values, dtype=dtype_overrides.get(key, np.float32))
    return finalized


def summarize_episodes(episode_stats: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-episode statistics for metadata."""
    if not episode_stats:
        return {
            "num_episodes": 0,
            "num_steps": 0,
            "mean_return": 0.0,
            "mean_length": 0.0,
            "success_rate": 0.0,
        }

    returns = np.asarray([item["episode_return"] for item in episode_stats], dtype=np.float32)
    lengths = np.asarray([item["episode_length"] for item in episode_stats], dtype=np.float32)
    successes = np.asarray([item["success"] for item in episode_stats], dtype=np.float32)
    return {
        "num_episodes": int(len(episode_stats)),
        "num_steps": int(lengths.sum()),
        "mean_return": float(returns.mean()),
        "mean_length": float(lengths.mean()),
        "success_rate": float(successes.mean()),
    }


def collect_dataset_split(
    model: Any,
    env: Any,
    *,
    num_episodes: int,
    deterministic: bool,
    object_name_to_id: Dict[str, int],
    observation_layout: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any], List[Dict[str, Any]]]:
    """Roll out one dataset split and return arrays plus observation metadata."""
    dataset_buffers: Dict[str, List[Any]] = defaultdict(list)
    episode_stats: List[Dict[str, Any]] = []
    resolved_layout = observation_layout

    for episode_index in range(num_episodes):
        observation = env.reset()
        active_object_name = env.get_active_object_name()
        if active_object_name not in object_name_to_id:
            raise KeyError(f"Unknown active object '{active_object_name}'")

        if resolved_layout is None:
            resolved_layout = get_observation_layout(observation)

        done = False
        episode_return = 0.0
        episode_length = 0
        episode_success = False
        last_info: Dict[str, Any] = {}

        while not done:
            flat_observation = flatten_observation(observation, resolved_layout)
            qpos = env.env.data.qpos.copy().astype(np.float32)
            qvel = env.env.data.qvel.copy().astype(np.float32)
            batched_obs = add_batch_dimension(observation)
            action, _ = model.predict(batched_obs, deterministic=deterministic)

            if isinstance(action, np.ndarray) and action.ndim > 1:
                action = action[0]
            action = np.asarray(action, dtype=np.float32).reshape(-1)

            next_observation, reward, done, info = env.step(action)
            timeout = bool(info.get("step_limit_reached", False))
            success = bool(info.get("is_success", False))

            dataset_buffers["observations"].append(flat_observation)
            dataset_buffers["actions"].append(action)
            dataset_buffers["rewards"].append(float(reward))
            dataset_buffers["terminals"].append(bool(done))
            dataset_buffers["timeouts"].append(timeout)
            dataset_buffers["masks"].append(0.0 if done else 1.0)
            dataset_buffers["qpos"].append(qpos)
            dataset_buffers["qvel"].append(qvel)
            dataset_buffers["successes"].append(success)
            dataset_buffers["object_ids"].append(object_name_to_id[active_object_name])

            episode_return += float(reward)
            episode_length += 1
            episode_success = episode_success or success
            last_info = info
            observation = next_observation

        episode_stats.append(
            {
                "episode_index": episode_index,
                "object_name": active_object_name,
                "episode_return": float(episode_return),
                "episode_length": int(episode_length),
                "success": bool(episode_success),
                "end_reason": str(last_info.get("episode_failure_reason", "done")),
                "timeout": bool(last_info.get("step_limit_reached", False)),
            }
        )

    if resolved_layout is None:
        raise RuntimeError("No episodes were collected; observation layout is undefined.")

    return finalize_dataset(dataset_buffers), resolved_layout, episode_stats


def save_dataset(dataset: Dict[str, np.ndarray], path: Path) -> None:
    """Save one split as a compressed NPZ file."""
    np.savez_compressed(path, **dataset)


def export_dataset(
    checkpoint_path: Path,
    *,
    agent_name: str,
    task_name: Optional[str],
    output_dir: Optional[str],
    train_episodes: int,
    val_episodes: int,
    deterministic: bool,
    seed: Optional[int],
    fixed_object_name: Optional[str],
    ycb_object_root_override: Optional[str],
    ycb_object_name_override: Optional[str],
    ycb_asset_source_override: Optional[str],
    target_scale_override: Optional[float],
) -> Path:
    """Export train/val rollouts from a saved checkpoint."""
    import torch
    import dexart_baselines.stable_baselines3  # noqa: F401

    run_config = load_run_config(checkpoint_path)
    resolved_task_name = task_name or run_config.get("task")
    if resolved_task_name is None:
        raise ValueError(
            "Could not infer task from checkpoint directory. Pass --task explicitly."
        )

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    ycb_selection = resolve_ycb_selection(
        run_config,
        ycb_object_root_override=ycb_object_root_override,
        ycb_object_name_override=ycb_object_name_override,
        ycb_asset_source_override=ycb_asset_source_override,
        target_scale_override=target_scale_override,
    )
    run_dir = create_output_dir(checkpoint_path, output_dir)
    env = build_environment(
        resolved_task_name,
        run_config,
        ycb_selection=ycb_selection,
        seed=seed,
        fixed_object_name=fixed_object_name,
    )

    try:
        resolved_agent_name, model, policy_kwargs, checkpoint_data = load_agent(
            checkpoint_path,
            env,
            agent_name,
        )
        available_object_names = env.get_available_object_names()
        object_name_to_id = {
            object_name: index for index, object_name in enumerate(available_object_names)
        }

        print("DexPoint Offline Dataset Export")
        print("=" * 70)
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Task: {resolved_task_name}")
        print(f"Algorithm: {resolved_agent_name.upper()}")
        print(f"Output directory: {run_dir}")
        print(f"Train episodes: {train_episodes}")
        print(f"Validation episodes: {val_episodes}")
        print(f"Deterministic actions: {deterministic}")
        print(f"Available objects: {available_object_names}")
        if fixed_object_name is not None:
            print(f"Fixed object: {fixed_object_name}")
        print(f"Resolved YCB root: {ycb_selection['ycb_object_root']}")
        print(f"YCB asset source: {ycb_selection['ycb_asset_source']}")
        print(f"Target scale: {ycb_selection['target_scale']}")

        train_dataset, observation_layout, train_episode_stats = collect_dataset_split(
            model,
            env,
            num_episodes=train_episodes,
            deterministic=deterministic,
            object_name_to_id=object_name_to_id,
        )
        val_dataset, _, val_episode_stats = collect_dataset_split(
            model,
            env,
            num_episodes=val_episodes,
            deterministic=deterministic,
            object_name_to_id=object_name_to_id,
            observation_layout=observation_layout,
        )

        train_path = run_dir / "train_dataset.npz"
        val_path = run_dir / "val_dataset.npz"
        save_dataset(train_dataset, train_path)
        save_dataset(val_dataset, val_path)

        metadata = {
            "checkpoint_path": checkpoint_path.as_posix(),
            "task": resolved_task_name,
            "algorithm": resolved_agent_name,
            "deterministic": bool(deterministic),
            "seed": seed,
            "fixed_object_name": fixed_object_name,
            "observation_layout": observation_layout,
            "observation_dim": int(observation_layout["dim"]),
            "action_dim": int(train_dataset["actions"].shape[-1]),
            "qpos_dim": int(train_dataset["qpos"].shape[-1]),
            "qvel_dim": int(train_dataset["qvel"].shape[-1]),
            "object_id_to_name": {
                str(index): name for name, index in object_name_to_id.items()
            },
            "train_summary": summarize_episodes(train_episode_stats),
            "val_summary": summarize_episodes(val_episode_stats),
            "run_config": run_config,
            "resolved_ycb_selection": {
                "ycb_object_root": ycb_selection["ycb_object_root"].as_posix(),
                "ycb_object_name": ycb_selection["ycb_object_name"],
                "ycb_asset_source": ycb_selection["ycb_asset_source"],
                "target_scale": float(ycb_selection["target_scale"]),
                "ycb_object_names": ycb_selection.get("ycb_object_names") or [],
            },
            "checkpoint_data_keys": sorted(checkpoint_data.keys()),
            "policy_kwargs": policy_kwargs,
        }
        metadata_path = run_dir / "metadata.json"
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(to_jsonable(metadata), handle, indent=2)

        print(f"Saved train dataset: {train_path}")
        print(f"Saved validation dataset: {val_path}")
        print(f"Saved metadata: {metadata_path}")
        return run_dir
    finally:
        env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export DexPoint checkpoint rollouts as OGBench-style offline datasets"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to a saved PPO or A2C checkpoint (.zip).",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Task name override. Uses the checkpoint's config when omitted.",
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="auto",
        choices=["auto", "ppo", "a2c"],
        help="Algorithm used for the checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory where train_dataset.npz and val_dataset.npz are written.",
    )
    parser.add_argument(
        "--train-episodes",
        type=int,
        default=1000,
        help="Number of training episodes to collect.",
    )
    parser.add_argument(
        "--val-episodes",
        type=int,
        default=100,
        help="Number of validation episodes to collect.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic actions instead of sampling from the policy.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for environment resets and stochastic policy sampling.",
    )
    parser.add_argument(
        "--fixed-object",
        type=str,
        default=None,
        help="Optional object name override for multi-object checkpoints.",
    )
    parser.add_argument(
        "--ycb-object-root",
        type=str,
        default=None,
        help="Explicit YCB object root override.",
    )
    parser.add_argument(
        "--ycb-object-name",
        type=str,
        default=None,
        help="YCB object folder name override.",
    )
    parser.add_argument(
        "--ycb-asset-source",
        type=str,
        default=None,
        choices=[YCB_ASSET_SOURCE_RAW, YCB_ASSET_SOURCE_YCB_SIM],
        help="YCB asset source override.",
    )
    parser.add_argument(
        "--target-scale",
        type=float,
        default=None,
        help="Optional target-object scale override.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if checkpoint_path.suffix != ".zip":
        raise ValueError(f"Checkpoint must be a .zip file: {checkpoint_path}")
    if args.train_episodes < 1:
        raise ValueError(f"--train-episodes must be >= 1, got {args.train_episodes}")
    if args.val_episodes < 1:
        raise ValueError(f"--val-episodes must be >= 1, got {args.val_episodes}")

    export_dataset(
        checkpoint_path,
        agent_name=args.agent,
        task_name=args.task,
        output_dir=args.output_dir,
        train_episodes=args.train_episodes,
        val_episodes=args.val_episodes,
        deterministic=args.deterministic,
        seed=args.seed,
        fixed_object_name=args.fixed_object,
        ycb_object_root_override=args.ycb_object_root,
        ycb_object_name_override=args.ycb_object_name,
        ycb_asset_source_override=args.ycb_asset_source,
        target_scale_override=args.target_scale,
    )


if __name__ == "__main__":
    main()