"""Run a saved DexPoint checkpoint for multiple evaluation episodes."""

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import imageio
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _script_bootstrap import DEXPOINT_ROOT, REPO_ROOT, ensure_script_imports

ensure_script_imports()

# Import the local fork first so it registers itself as stable_baselines3.
import dexart_baselines.stable_baselines3

from dexart_baselines.stable_baselines3.common.save_util import load_from_zip_file
from dexart_baselines.stable_baselines3.a2c import A2C
from dexart_baselines.stable_baselines3.ppo import PPO
from franka_gym_env import FrankaGymEnvironment
from tasks import create_task_config
from ycb_scene import (
    DEFAULT_YCB_OBJECT_ROOT,
    YCB_ASSET_SOURCE_RAW,
    YCB_ASSET_SOURCE_YCB_SIM,
)


_OUTPUT_DIR = DEXPOINT_ROOT / "evaluation_runs"
_YCB_SIM_ROOT = REPO_ROOT / "YCB_sim"
DEFAULT_CAMERA_WIDTH = 640
DEFAULT_CAMERA_HEIGHT = 480


def add_batch_dimension(obs: Dict[str, Any]) -> Dict[str, Any]:
    """Add a batch dimension to numpy observations for SB3 prediction."""
    batched_obs: Dict[str, Any] = {}
    for key, value in obs.items():
        if isinstance(value, np.ndarray):
            batched_obs[key] = np.expand_dims(value, axis=0)
        else:
            batched_obs[key] = value
    return batched_obs


def load_run_config(checkpoint_path: Path) -> Dict[str, Any]:
    """Load the sibling training config when available."""
    config_path = checkpoint_path.parent / "config.json"
    if not config_path.exists():
        return {}

    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def create_output_dir(checkpoint_path: Path, output_dir: Optional[str]) -> Path:
    """Create a timestamped evaluation output directory."""
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
    data, _, _ = load_from_zip_file(str(checkpoint_path), device="cpu")
    policy_kwargs = dict(data.get("policy_kwargs", {}))

    # Full policy checkpoints already contain PointNet weights.
    policy_kwargs["pointnet_checkpoint_path"] = None
    return policy_kwargs, data


def resolve_ycb_selection(
    run_config: Dict[str, Any],
    *,
    ycb_object_root_override: Optional[str],
    ycb_object_name_override: Optional[str],
    ycb_asset_source_override: Optional[str],
    target_scale_override: Optional[float],
) -> Dict[str, Any]:
    """Resolve which YCB object/assets to use for evaluation."""
    if ycb_object_root_override is not None and ycb_object_name_override is not None:
        raise ValueError(
            "Pass either --ycb-object-root or --ycb-object-name, not both."
        )

    env_config = dict(run_config.get("env_config", {}))
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

    default_root = Path(
        env_config.get("ycb_object_root", DEFAULT_YCB_OBJECT_ROOT.as_posix())
    ).expanduser()
    if ycb_object_root_override is not None:
        ycb_object_root = Path(ycb_object_root_override).expanduser()
    elif ycb_object_name_override is not None:
        if ycb_asset_source == YCB_ASSET_SOURCE_YCB_SIM:
            ycb_object_root = _YCB_SIM_ROOT / ycb_object_name_override
        else:
            ycb_object_root = default_root.parent / ycb_object_name_override
    else:
        ycb_object_root = default_root

    target_scale_value = (
        target_scale_override
        if target_scale_override is not None
        else env_config.get("target_scale", 1.0)
    )
    target_scale = float(target_scale_value)
    if target_scale <= 0:
        raise ValueError(f"Target scale must be positive, got {target_scale}")

    resolved_root = ycb_object_root.resolve()
    return {
        "ycb_object_root": resolved_root,
        "ycb_object_name": resolved_root.name,
        "ycb_asset_source": ycb_asset_source,
        "target_scale": target_scale,
    }


def build_environment(
    task_name: str,
    run_config: Dict[str, Any],
    *,
    ycb_selection: Dict[str, Any],
    visualize_pointclouds: bool,
    pointcloud_point_size: int,
    pointcloud_alpha: float,
    use_depth_only_pointcloud: bool,
    camera_height: int,
    camera_width: int,
    seed: Optional[int],
) -> FrankaGymEnvironment:
    """Create an evaluation environment compatible with training."""
    env_config = dict(run_config.get("env_config", {}))
    reward_variant = run_config.get("reward_variant") or env_config.get(
        "reward_variant"
    )
    if reward_variant is None:
        reward_variant = "shaped" if task_name == "grasping" else "default"

    num_points = int(env_config.get("num_points", 512))
    camera_names = env_config.get("camera_names")

    env = FrankaGymEnvironment(
        xml_path=None,
        task_name=task_name,
        ycb_object_root=ycb_selection["ycb_object_root"].as_posix(),
        ycb_asset_source=str(ycb_selection["ycb_asset_source"]),
        target_scale=float(ycb_selection["target_scale"]),
        num_points=num_points,
        camera_height=camera_height,
        camera_width=camera_width,
        camera_names=camera_names,
        rate=200.0,
        frame_skip=10,
        visualize_pointclouds=visualize_pointclouds,
        pointcloud_point_size=pointcloud_point_size,
        pointcloud_alpha=pointcloud_alpha,
        use_depth_only_pointcloud=use_depth_only_pointcloud,
    )
    task_config = create_task_config(
        task_name,
        target_body_name=env_config.get("target_body_name", env.target_body_name),
        reward_variant=reward_variant,
    )
    env.configure_task(task_config)
    if seed is not None:
        env.seed(seed)
    return env


def load_agent(
    checkpoint_path: Path,
    env: FrankaGymEnvironment,
    agent_name: str,
) -> Tuple[str, Any, Dict[str, Any], Dict[str, Any]]:
    """Load a PPO or A2C checkpoint, optionally auto-detecting the algorithm."""
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


def get_target_table_contact(env: FrankaGymEnvironment) -> bool:
    """Return whether the target object is touching the table."""
    for index in range(env.env.data.ncon):
        contact = env.env.data.contact[index]
        body1 = int(env.env.model.geom_bodyid[contact.geom1])
        body2 = int(env.env.model.geom_bodyid[contact.geom2])
        if (body1 == env.target_id and body2 == env.table_body_id) or (
            body1 == env.table_body_id and body2 == env.target_id
        ):
            return True
    return False


def get_target_table_distance(env: FrankaGymEnvironment) -> float:
    """Return the current target clearance above the table."""
    return float(max(env.get_target_position()[2] - env.target_rest_height, 0.0))


def get_episode_end_reason(info: Dict[str, Any]) -> str:
    """Map the final info dict to a human-readable ending reason."""
    if bool(info.get("is_success", False)):
        return "success"
    if bool(info.get("pointcloud_empty", False)):
        return "empty_pointcloud"
    if bool(info.get("step_limit_reached", False)):
        return "step_limit"
    if isinstance(info.get("episode_failure_reason"), str):
        return str(info["episode_failure_reason"])
    return "done"


def save_episode_timeseries_csv(
    episode_rows: Sequence[Dict[str, Any]], csv_path: Path
) -> None:
    """Write the full stepwise episode trace to CSV."""
    if not episode_rows:
        return

    fieldnames = list(episode_rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(episode_rows)


def plot_episode_metrics(
    episode_rows: Sequence[Dict[str, Any]], plot_path: Path, title: str
) -> None:
    """Plot the requested per-step metrics for a single episode."""
    steps = [int(row["step"]) for row in episode_rows]
    gripper_force = [float(row["gripper_actuator_force"]) for row in episode_rows]
    target_table_distance = [
        float(row["target_table_distance"]) for row in episode_rows
    ]
    table_can_contact = [float(row["table_can_contact"]) for row in episode_rows]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(steps, gripper_force, color="#1f77b4", linewidth=1.8)
    axes[0].set_ylabel("Force")
    axes[0].set_title("Gripper actuator force")
    axes[0].grid(alpha=0.3)

    axes[1].plot(steps, target_table_distance, color="#ff7f0e", linewidth=1.8)
    axes[1].set_ylabel("Distance (m)")
    axes[1].set_title("Target-table distance")
    axes[1].grid(alpha=0.3)

    axes[2].step(steps, table_can_contact, where="post", color="#2ca02c", linewidth=1.8)
    axes[2].set_ylabel("Contact")
    axes[2].set_xlabel("Step")
    axes[2].set_yticks([0.0, 1.0])
    axes[2].set_title("Table-can contact")
    axes[2].grid(alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_all_episodes(
    all_episode_rows: Sequence[Sequence[Dict[str, Any]]], plot_path: Path
) -> None:
    """Plot the requested metrics across all episodes on shared axes."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=False)

    for episode_index, episode_rows in enumerate(all_episode_rows, start=1):
        if not episode_rows:
            continue
        steps = [int(row["step"]) for row in episode_rows]
        label = f"episode_{episode_index:03d}"
        axes[0].plot(
            steps,
            [float(row["gripper_actuator_force"]) for row in episode_rows],
            linewidth=1.5,
            alpha=0.85,
            label=label,
        )
        axes[1].plot(
            steps,
            [float(row["target_table_distance"]) for row in episode_rows],
            linewidth=1.5,
            alpha=0.85,
            label=label,
        )
        axes[2].step(
            steps,
            [float(row["table_can_contact"]) for row in episode_rows],
            where="post",
            linewidth=1.2,
            alpha=0.85,
            label=label,
        )

    axes[0].set_title("Gripper actuator force")
    axes[0].set_ylabel("Force")
    axes[0].grid(alpha=0.3)

    axes[1].set_title("Target-table distance")
    axes[1].set_ylabel("Distance (m)")
    axes[1].grid(alpha=0.3)

    axes[2].set_title("Table-can contact")
    axes[2].set_ylabel("Contact")
    axes[2].set_xlabel("Step")
    axes[2].set_yticks([0.0, 1.0])
    axes[2].grid(alpha=0.3)

    if len(all_episode_rows) <= 12:
        axes[0].legend(loc="upper right", fontsize=8)

    fig.suptitle("Checkpoint evaluation metrics")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_summary_csv(summary_rows: Sequence[Dict[str, Any]], csv_path: Path) -> None:
    """Write a compact episode-level summary CSV."""
    if not summary_rows:
        return

    fieldnames = list(summary_rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def evaluate_checkpoint(
    checkpoint_path: Path,
    *,
    task_name: str,
    agent_name: str,
    ycb_object_root: Optional[str],
    ycb_object_name: Optional[str],
    ycb_asset_source: Optional[str],
    target_scale: Optional[float],
    num_episodes: int,
    output_dir: Optional[str],
    deterministic: bool,
    video_fps: int,
    camera_height: int,
    camera_width: int,
    max_steps: Optional[int],
    visualize_pointclouds: bool,
    pointcloud_point_size: int,
    pointcloud_alpha: float,
    use_depth_only_pointcloud: bool,
    seed: Optional[int],
) -> Path:
    """Run a saved checkpoint for N episodes and save videos and plots."""
    run_config = load_run_config(checkpoint_path)
    effective_task_name = task_name or run_config.get("task")
    if not effective_task_name:
        raise ValueError(
            "Could not infer task from checkpoint directory. Pass --task explicitly."
        )

    effective_agent_name = agent_name
    if effective_agent_name == "auto" and isinstance(run_config.get("agent"), str):
        effective_agent_name = str(run_config["agent"])

    ycb_selection = resolve_ycb_selection(
        run_config,
        ycb_object_root_override=ycb_object_root,
        ycb_object_name_override=ycb_object_name,
        ycb_asset_source_override=ycb_asset_source,
        target_scale_override=target_scale,
    )

    run_dir = create_output_dir(checkpoint_path, output_dir)
    videos_dir = run_dir / "videos"
    plots_dir = run_dir / "plots"
    traces_dir = run_dir / "traces"
    for directory in (videos_dir, plots_dir, traces_dir):
        directory.mkdir(parents=True, exist_ok=True)

    env = build_environment(
        effective_task_name,
        run_config,
        ycb_selection=ycb_selection,
        visualize_pointclouds=visualize_pointclouds,
        pointcloud_point_size=pointcloud_point_size,
        pointcloud_alpha=pointcloud_alpha,
        use_depth_only_pointcloud=use_depth_only_pointcloud,
        camera_height=camera_height,
        camera_width=camera_width,
        seed=seed,
    )

    resolved_agent_name, model, policy_kwargs, checkpoint_data = load_agent(
        checkpoint_path,
        env,
        effective_agent_name,
    )

    all_episode_rows: List[List[Dict[str, Any]]] = []
    summary_rows: List[Dict[str, Any]] = []

    print("\n" + "=" * 70)
    print("DexPoint Checkpoint Evaluation")
    print("=" * 70)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Task: {effective_task_name}")
    print(f"Agent: {resolved_agent_name.upper()}")
    print(f"Episodes: {num_episodes}")
    print(
        "YCB object: "
        f"{ycb_selection['ycb_object_name']} "
        f"(source={ycb_selection['ycb_asset_source']}, scale={ycb_selection['target_scale']:.3f})"
    )
    print(f"YCB object root: {ycb_selection['ycb_object_root']}")
    print(f"Output directory: {run_dir}")

    try:
        for episode_index in range(1, num_episodes + 1):
            episode_seed = None if seed is None else seed + episode_index - 1
            if episode_seed is not None:
                env.seed(episode_seed)

            obs = env.reset()
            episode_rows: List[Dict[str, Any]] = []
            frames: List[np.ndarray] = []
            episode_reward = 0.0
            final_info: Dict[str, Any] = {}
            success = False
            done = False
            step_limit = max_steps or env.max_episode_steps

            initial_frame = env.render_with_pointcloud(mode="rgb_array")
            if initial_frame is not None:
                frames.append(initial_frame)

            print(f"\nEpisode {episode_index}/{num_episodes}")

            for step_index in range(1, step_limit + 1):
                batched_obs = add_batch_dimension(obs)
                action, _ = model.predict(batched_obs, deterministic=deterministic)
                if isinstance(action, np.ndarray) and action.ndim > 1:
                    action = action[0]

                obs, reward, done, info = env.step(action)
                final_info = info
                success = bool(info.get("is_success", False))
                episode_reward += float(reward)

                row = {
                    "step": step_index,
                    "reward": float(reward),
                    "reward_total": float(info.get("reward_total", reward)),
                    "gripper_actuator_force": float(
                        info.get(
                            "gripper_actuator_force", env.get_gripper_actuator_force()
                        )
                    ),
                    "target_table_distance": float(
                        info.get(
                            "target_table_distance", get_target_table_distance(env)
                        )
                    ),
                    "table_can_contact": float(get_target_table_contact(env)),
                    "contact_count": int(info.get("contact_count", env.env.data.ncon)),
                    "is_success": float(success),
                    "pointcloud_empty": float(
                        bool(info.get("pointcloud_empty", False))
                    ),
                }
                episode_rows.append(row)

                frame = env.render_with_pointcloud(mode="rgb_array")
                if frame is not None:
                    frames.append(frame)

                if done:
                    break

            video_path = videos_dir / f"episode_{episode_index:03d}.mp4"
            if frames:
                imageio.mimwrite(video_path, frames, fps=video_fps)

            trace_csv_path = traces_dir / f"episode_{episode_index:03d}_timeseries.csv"
            plot_path = plots_dir / f"episode_{episode_index:03d}_metrics.png"
            save_episode_timeseries_csv(episode_rows, trace_csv_path)
            plot_episode_metrics(
                episode_rows,
                plot_path,
                title=f"Episode {episode_index:03d} metrics",
            )

            end_reason = get_episode_end_reason(final_info)
            summary_row = {
                "episode": episode_index,
                "seed": episode_seed,
                "num_steps": len(episode_rows),
                "episode_reward": float(episode_reward),
                "is_success": bool(success),
                "end_reason": end_reason,
                "max_gripper_force": float(
                    max(
                        (row["gripper_actuator_force"] for row in episode_rows),
                        default=0.0,
                    )
                ),
                "max_target_table_distance": float(
                    max(
                        (row["target_table_distance"] for row in episode_rows),
                        default=0.0,
                    )
                ),
                "table_contact_fraction": (
                    float(np.mean([row["table_can_contact"] for row in episode_rows]))
                    if episode_rows
                    else 0.0
                ),
                "video_path": video_path.as_posix(),
                "trace_csv_path": trace_csv_path.as_posix(),
                "plot_path": plot_path.as_posix(),
            }
            summary_rows.append(summary_row)
            all_episode_rows.append(episode_rows)

            print(
                f"  steps={summary_row['num_steps']} reward={summary_row['episode_reward']:.3f} "
                f"success={summary_row['is_success']} end_reason={end_reason}"
            )

        combined_plot_path = plots_dir / "all_episodes_metrics.png"
        plot_all_episodes(all_episode_rows, combined_plot_path)

        summary_csv_path = run_dir / "episode_summary.csv"
        summary_json_path = run_dir / "summary.json"
        save_summary_csv(summary_rows, summary_csv_path)
        with summary_json_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "checkpoint_path": checkpoint_path.as_posix(),
                    "task": effective_task_name,
                    "agent": resolved_agent_name,
                    "num_episodes": num_episodes,
                    "deterministic": deterministic,
                    "camera_height": camera_height,
                    "camera_width": camera_width,
                    "visualize_pointclouds": visualize_pointclouds,
                    "use_depth_only_pointcloud": use_depth_only_pointcloud,
                    "ycb_object_root": ycb_selection["ycb_object_root"].as_posix(),
                    "ycb_object_name": ycb_selection["ycb_object_name"],
                    "ycb_asset_source": ycb_selection["ycb_asset_source"],
                    "target_scale": ycb_selection["target_scale"],
                    "policy_kwargs": policy_kwargs,
                    "checkpoint_data_keys": sorted(checkpoint_data.keys()),
                    "episodes": summary_rows,
                    "combined_plot_path": combined_plot_path.as_posix(),
                },
                handle,
                indent=2,
                default=str,
            )

        print("\nEvaluation complete.")
        print(f"Summary CSV: {summary_csv_path}")
        print(f"Summary JSON: {summary_json_path}")
        print(f"Combined plot: {combined_plot_path}")
        return run_dir
    finally:
        env.close()


def parse_args() -> argparse.Namespace:
    """Parse evaluation CLI arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a saved DexPoint checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to a saved PPO or A2C checkpoint (.zip).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes to run.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        choices=["grasping", "reaching", "lifting", "placing"],
        help="Task name. If omitted, the script tries to read it from config.json.",
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="ppo",
        choices=["auto", "ppo", "a2c"],
        help="Algorithm used for the checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional explicit output directory.",
    )
    parser.add_argument(
        "--ycb-object-root",
        type=str,
        default=None,
        help="Explicit YCB object root. Use a raw object directory or any path whose basename matches a YCB_sim object name.",
    )
    parser.add_argument(
        "--ycb-object-name",
        type=str,
        default=None,
        help="Override only the object name. For YCB_sim this resolves under the repo's YCB_sim directory and uses the include-defined collision geom.",
    )
    parser.add_argument(
        "--ycb-asset-source",
        type=str,
        default=None,
        choices=[YCB_ASSET_SOURCE_RAW, YCB_ASSET_SOURCE_YCB_SIM],
        help="Which YCB asset source to use. Defaults to the checkpoint config when present, otherwise ycb_sim.",
    )
    parser.add_argument(
        "--target-scale",
        type=float,
        default=None,
        help="Optional target-object scale override. Defaults to the checkpoint config when present, otherwise 1.0.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Optional hard cap on evaluation steps per episode.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second for saved videos.",
    )
    parser.add_argument(
        "--camera-width",
        type=int,
        default=DEFAULT_CAMERA_WIDTH,
        help="Evaluation camera width.",
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=DEFAULT_CAMERA_HEIGHT,
        help="Evaluation camera height.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional base seed. Each episode increments it by one.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic actions instead of deterministic policy evaluation.",
    )
    parser.add_argument(
        "--no-pointcloud-overlay",
        action="store_false",
        dest="visualize_pointclouds",
        help="Disable pointcloud overlay in rendered videos.",
    )
    parser.add_argument(
        "--pointcloud-point-size",
        type=int,
        default=1,
        help="Point size for the rendered pointcloud overlay.",
    )
    parser.add_argument(
        "--pointcloud-alpha",
        type=float,
        default=0.7,
        help="Pointcloud overlay alpha.",
    )
    parser.add_argument(
        "--use-depth-only-pointcloud",
        action="store_true",
        help="Use the faster depth-only pointcloud observation path during evaluation.",
    )
    parser.set_defaults(visualize_pointclouds=True)
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if checkpoint_path.suffix != ".zip":
        raise ValueError(f"Checkpoint must be a .zip file: {checkpoint_path}")

    evaluate_checkpoint(
        checkpoint_path,
        task_name=args.task,
        agent_name=args.agent,
        ycb_object_root=args.ycb_object_root,
        ycb_object_name=args.ycb_object_name,
        ycb_asset_source=args.ycb_asset_source,
        target_scale=args.target_scale,
        num_episodes=args.episodes,
        output_dir=args.output_dir,
        deterministic=not args.stochastic,
        video_fps=args.fps,
        camera_height=args.camera_height,
        camera_width=args.camera_width,
        max_steps=args.max_steps,
        visualize_pointclouds=args.visualize_pointclouds,
        pointcloud_point_size=args.pointcloud_point_size,
        pointcloud_alpha=args.pointcloud_alpha,
        use_depth_only_pointcloud=args.use_depth_only_pointcloud,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
