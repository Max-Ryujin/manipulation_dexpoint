"""Training script for DexPoint on Franka robots."""

import imageio
import numpy as np
import argparse
import re
from pathlib import Path
import json
from datetime import datetime
import sys
from typing import TYPE_CHECKING, Callable, List, Optional
import matplotlib
import torch
import wandb

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add dexart to path
root_path = Path(__file__).parent.parent
sys.path.insert(0, str(root_path))

# Import the local fork first so it registers itself as stable_baselines3.
import dexart_baselines.stable_baselines3  # noqa: F401

from franka_gym_env import FrankaGymEnvironment
from tasks import create_task_config
from training_callbacks import TaskInfoLoggingCallback
from ycb_scene import DEFAULT_YCB_OBJECT_ROOT

if TYPE_CHECKING:
    from dexart_baselines.stable_baselines3.ppo import PPO
    from dexart_baselines.stable_baselines3.a2c import A2C
    from dexart_baselines.stable_baselines3.common.policies import (
        MultiInputActorCriticPolicy,
    )
else:
    from stable_baselines3.ppo import PPO
    from stable_baselines3.a2c import A2C
    from stable_baselines3.common.policies import (
        MultiInputActorCriticPolicy,
    )
from dexart_baselines.stable_baselines3.common.vec_env import SubprocVecEnv
from dexart_baselines.stable_baselines3.common.save_util import load_from_zip_file
from dexpoint_policy import DexPointPolicy

# from dexart_baselines.stable_baselines3.common.policies import MlpPolicy


def add_batch_dimension(obs):
    """
    Add batch dimension to observation dict if not already present.

    Args:
        obs: Dict of numpy arrays from env.reset() or env.step()

    Returns:
        batched_obs: Dict with batch dimension added to each observation
    """
    if isinstance(obs, dict):
        batched_obs = {}
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                batched_obs[key] = np.expand_dims(value, axis=0)
            else:
                batched_obs[key] = value
        return batched_obs
    return obs


def sanitize_metric_key(value: str) -> str:
    """Normalize object names so they are safe to use in W&B metric keys."""
    return re.sub(r"[^0-9A-Za-z_]+", "_", value).strip("_") or "object"


def save_validation_reward_plot(
    reward_trace: List[float],
    plot_path: Path,
    *,
    object_name: str,
    eval_try_index: int,
    success: bool,
) -> None:
    """Persist a per-step validation reward plot for one object/try pair."""
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    rewards = np.asarray(reward_trace, dtype=np.float32)
    cumulative_rewards = np.cumsum(rewards)
    step_axis = np.arange(1, len(rewards) + 1)

    axes[0].plot(step_axis, rewards, color="#1f77b4", linewidth=1.5)
    axes[0].set_ylabel("Reward")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(step_axis, cumulative_rewards, color="#d62728", linewidth=1.5)
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Cumulative")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(
        f"Validation rewards: {object_name} | try {eval_try_index:02d} | success={int(success)}"
    )
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


_HERE = Path(__file__).parent
_DEFAULT_SIMSIAM_POINTNET_CHECKPOINT = (
    _HERE
    / ".."
    / "log"
    / "simsiam"
    / "ycb"
    / "ycb_medium_simsiam"
    / "simsiam_pn_30.pth"
)
_DEFAULT_RECONSTRUCTION_POINTNET_CHECKPOINT = (
    _HERE
    / ".."
    / "log"
    / "reconstruction"
    / "ycb"
    / "ycb_medium_reconstruction"
    / "complete_pn_50.pth"
)
_OUTPUT_DIR = _HERE / "training_runs"
TRAIN_CAMERA_WIDTH = 576
TRAIN_CAMERA_HEIGHT = 432
EVAL_CAMERA_WIDTH = 640
EVAL_CAMERA_HEIGHT = 480


def infer_pointnet_variant(checkpoint_path: Optional[str]) -> str:
    """Infer the PointNet architecture variant from checkpoint keys."""
    if checkpoint_path is None:
        return "medium"

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = (
        checkpoint.get("state_dict", checkpoint)
        if isinstance(checkpoint, dict)
        else checkpoint
    )
    keys = set(state_dict.keys())

    if {"local_mlp.0.weight", "local_mlp.2.weight"}.issubset(keys):
        if "local_mlp.10.weight" in keys:
            return "large"
        if "local_mlp.6.weight" in keys:
            return "medium"
        return "small"

    raise RuntimeError(
        f"Could not infer PointNet variant from checkpoint: {checkpoint_path}"
    )


def get_default_pointnet_checkpoint() -> Optional[str]:
    """Return the default checkpoint path that matches the RL medium encoder."""

    if _DEFAULT_RECONSTRUCTION_POINTNET_CHECKPOINT.exists():
        return _DEFAULT_RECONSTRUCTION_POINTNET_CHECKPOINT.as_posix()
    if _DEFAULT_SIMSIAM_POINTNET_CHECKPOINT.exists():
        return _DEFAULT_SIMSIAM_POINTNET_CHECKPOINT.as_posix()
    return None


def get_pretrained_policy_kwargs(
    checkpoint_path: str,
    *,
    freeze_pointnet: bool = False,
) -> dict:
    """Load policy kwargs from a saved RL checkpoint for compatible warm starts."""
    data, _, _ = load_from_zip_file(str(checkpoint_path), device="cpu")
    policy_kwargs = dict(data.get("policy_kwargs", {}))

    # Full policy checkpoints already contain PointNet weights, so do not require
    # the original external encoder checkpoint to be present during restore.
    policy_kwargs["pointnet_checkpoint_path"] = None
    if freeze_pointnet:
        policy_kwargs["freeze_pointnet"] = True

    return policy_kwargs


def initialize_agent_from_checkpoint(agent, checkpoint_path: str) -> None:
    """Warm-start a fresh agent from a saved RL checkpoint without restoring trainer state."""
    agent.set_parameters(str(checkpoint_path), exact_match=True, device=agent.device)


def create_output_dir():
    """Create output directory for training results."""
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Create timestamped subdirectory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = _OUTPUT_DIR / f"dexpoint_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    return run_dir


def create_configured_environment(
    task_name: str,
    *,
    visualize_pointclouds: bool,
    pointcloud_point_size: int = 1,
    pointcloud_alpha: float = 0.7,
    use_depth_only_pointcloud: bool = False,
    camera_height: int = EVAL_CAMERA_HEIGHT,
    camera_width: int = EVAL_CAMERA_WIDTH,
    ycb_object_names: Optional[List[str]] = None,
) -> FrankaGymEnvironment:
    """Build and configure a FrankaGymEnvironment for training or evaluation."""
    env = FrankaGymEnvironment(
        xml_path=None,
        task_name=task_name,
        ycb_object_root=DEFAULT_YCB_OBJECT_ROOT.as_posix(),
        num_points=512,
        camera_height=camera_height,
        camera_width=camera_width,
        rate=200.0,
        frame_skip=10,
        visualize_pointclouds=visualize_pointclouds,
        pointcloud_point_size=pointcloud_point_size,
        pointcloud_alpha=pointcloud_alpha,
        use_depth_only_pointcloud=use_depth_only_pointcloud,
        ycb_object_names=ycb_object_names,
    )
    task_config = create_task_config(
        task_name,
        target_body_name=env.target_body_name,
    )
    env.configure_task(task_config)
    return env


def make_environment_factory(
    task_name: str,
    *,
    visualize_pointclouds: bool,
    pointcloud_point_size: int = 1,
    pointcloud_alpha: float = 0.7,
    use_depth_only_pointcloud: bool = False,
    camera_height: int = EVAL_CAMERA_HEIGHT,
    camera_width: int = EVAL_CAMERA_WIDTH,
    seed: Optional[int] = None,
    ycb_object_names: Optional[List[str]] = None,
) -> Callable[[], FrankaGymEnvironment]:
    """Create a picklable environment factory for vectorized rollouts."""

    def _make_env() -> FrankaGymEnvironment:
        env = create_configured_environment(
            task_name,
            visualize_pointclouds=visualize_pointclouds,
            pointcloud_point_size=pointcloud_point_size,
            pointcloud_alpha=pointcloud_alpha,
            use_depth_only_pointcloud=use_depth_only_pointcloud,
            camera_height=camera_height,
            camera_width=camera_width,
            ycb_object_names=ycb_object_names,
        )
        if seed is not None:
            env.seed(seed)
        return env

    return _make_env


def get_validation_object_names(
    eval_env: FrankaGymEnvironment, ycb_object_names: Optional[List[str]]
) -> List[str]:
    """Return the deterministic object list to use during validation."""
    if ycb_object_names:
        return list(ycb_object_names)
    return eval_env.get_available_object_names()


def train_dexpoint(
    task_name: str = "grasping",
    agent_name: str = "ppo",
    num_envs: int = 1,
    total_timesteps: int = 100000,
    learning_rate: float = 3e-4,
    batch_size: int = 64,
    n_epochs: int = 6,
    n_steps: int = 10000,
    save_interval: int = 100000,
    verbose: int = 1,
    use_wandb: bool = False,
    record_video: bool = True,
    video_interval: int = 100000,
    pointnet_checkpoint_path: Optional[str] = None,
    pretrained_model_checkpoint_path: Optional[str] = None,
    pointnet_variant: str = "auto",
    freeze_pointnet: bool = False,
    wandb_run_name: Optional[str] = None,
    resume_training_state: bool = False,
    ycb_object_names: Optional[List[str]] = None,
    eval_tries: int = 1,
):
    """
    Train a DexPoint policy using PPO or A2C.

    Args:
        task_name: Task to train on
        agent_name: RL algorithm to train with
        num_envs: Number of parallel training environments
        total_timesteps: Total training timesteps
        learning_rate: Learning rate for Adam optimizer
        batch_size: Batch size for PPO updates
        n_epochs: Number of PPO policy update epochs per update
        n_steps: Number of steps to collect per rollout before each policy update
        save_interval: Save model every N timesteps
        verbose: Verbosity level (0=silent, 1=verbose)
        use_wandb: Enable Weights & Biases logging
        record_video: Record training videos
        video_interval: Record video every N timesteps
        pointnet_checkpoint_path: Optional PointNet encoder checkpoint
        pretrained_model_checkpoint_path: Optional RL checkpoint used to initialize
            policy weights for a fresh run, unless resume_training_state=True
        pointnet_variant: PointNet architecture variant or "auto" to infer it
        freeze_pointnet: Whether to keep PointNet frozen during RL training
        wandb_run_name: Optional explicit Weights & Biases run name
        resume_training_state: Whether to fully resume optimizer/timestep state from
            pretrained_model_checkpoint_path instead of only copying policy weights
        ycb_object_names: Optional list of YCB object folder names to sample per episode
            (e.g. ["005_tomato_soup_can", "006_mustard_bottle"]).  When provided, every
            environment reloads a randomly chosen object at the start of each episode.
        eval_tries: Number of validation episodes to run per object whenever evaluation triggers.
    """
    print("\n" + "=" * 70)
    print("DexPoint Training - Franka Manipulation")
    print("=" * 70)

    resolved_wandb_run_name = wandb_run_name or (
        f"{task_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    pretrained_policy_kwargs = None
    if pretrained_model_checkpoint_path is not None:
        pretrained_policy_kwargs = get_pretrained_policy_kwargs(
            pretrained_model_checkpoint_path,
            freeze_pointnet=freeze_pointnet,
        )
    resolved_pointnet_variant = (
        str(pretrained_policy_kwargs.get("pointnet_variant", "unknown"))
        if pretrained_policy_kwargs is not None
        else (
            infer_pointnet_variant(pointnet_checkpoint_path)
            if pointnet_variant == "auto"
            else pointnet_variant
        )
    )
    initialization_mode = (
        "pretrained_model_resume"
        if pretrained_model_checkpoint_path is not None and resume_training_state
        else (
            "pretrained_model_weights"
            if pretrained_model_checkpoint_path is not None
            else "pointnet_only"
        )
    )

    if num_envs < 1:
        raise ValueError(f"num_envs must be >= 1, got {num_envs}")
    if eval_tries < 1:
        raise ValueError(f"eval_tries must be >= 1, got {eval_tries}")

    if ycb_object_names and len(ycb_object_names) > 0:
        print(f"  - Multi-object pool: {ycb_object_names}")

    if use_wandb:
        project_name = ""
        if task_name == "reaching":
            project_name = "dexpoint-franka_reaching"
        elif task_name == "lifting":
            project_name = "dexpoint-franka_lifting"
        elif task_name == "lifting_only":
            project_name = "dexpoint-franka_lifting"
        elif task_name == "placing":
            project_name = "dexpoint-franka_placing"
        elif task_name == "placing_v2":
            project_name = "dexpoint-franka_placing"
        elif task_name == "placing_v3":
            project_name = "dexpoint-franka_placing"
        else:
            project_name = "dexpoint-franka"
        wandb.init(
            project=(project_name if project_name else None),
            name=resolved_wandb_run_name,
            config={
                "task": task_name,
                "agent": agent_name,
                "total_timesteps": total_timesteps,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "n_epochs": n_epochs,
                "n_steps": n_steps,
                "save_interval": save_interval,
                "video_interval": video_interval,
                "pointcloud_points": 512,
                "pointnet_checkpoint_path": pointnet_checkpoint_path,
                "pretrained_model_checkpoint_path": pretrained_model_checkpoint_path,
                "pointnet_variant": resolved_pointnet_variant,
                "freeze_pointnet": freeze_pointnet,
                "initialization_mode": initialization_mode,
                "resume_training_state": resume_training_state,
                "wandb_run_name": resolved_wandb_run_name,
                "ycb_object_names": ycb_object_names,
                "eval_tries": eval_tries,
            },
        )
        wandb.define_metric("validation/timesteps")
        wandb.define_metric("validation/*", step_metric="validation/timesteps")

    # Create output directory
    run_dir = create_output_dir()
    print(f"\nOutput directory: {run_dir}")

    # Create environment
    print(f"\n Creating environment...")
    reference_env = create_configured_environment(
        task_name,
        visualize_pointclouds=False,
        use_depth_only_pointcloud=True,
        camera_height=TRAIN_CAMERA_HEIGHT,
        camera_width=TRAIN_CAMERA_WIDTH,
        ycb_object_names=ycb_object_names,
    )
    env = reference_env
    if num_envs > 1:
        env.close()
        env = SubprocVecEnv(
            [
                make_environment_factory(
                    task_name,
                    visualize_pointclouds=False,
                    use_depth_only_pointcloud=True,
                    camera_height=TRAIN_CAMERA_HEIGHT,
                    camera_width=TRAIN_CAMERA_WIDTH,
                    seed=env_index,
                    ycb_object_names=ycb_object_names,
                )
                for env_index in range(num_envs)
            ]
        )
    eval_env = (
        create_configured_environment(
            task_name,
            visualize_pointclouds=True,
            use_depth_only_pointcloud=False,
            camera_height=EVAL_CAMERA_HEIGHT,
            camera_width=EVAL_CAMERA_WIDTH,
            ycb_object_names=ycb_object_names,
        )
        if record_video
        else None
    )
    print(f"✓ Environment ready")
    print(f"  - Task: {task_name}")
    print(f"  - Agent: {agent_name.upper()}")
    print(f"  - Parallel envs: {num_envs}")
    print(f"  - PointNet variant: {resolved_pointnet_variant}")
    print(f"  - Initialization mode: {initialization_mode}")
    if pretrained_model_checkpoint_path is not None:
        checkpoint_mode = (
            "full training-state resume"
            if resume_training_state
            else "policy-weight warm start"
        )
        print(
            f"  - Pretrained RL checkpoint ({checkpoint_mode}): "
            f"{pretrained_model_checkpoint_path}"
        )
    elif pointnet_checkpoint_path is not None:
        print(f"  - Pretrained PointNet checkpoint: {pointnet_checkpoint_path}")
    print(f"  - Observation space: {env.observation_space}")
    print(f"  - Action space: {env.action_space}")

    # Create training agent
    print(f"\n Creating {agent_name.upper()} agent...")

    agent = None

    if agent_name == "ppo":
        policy_kwargs = pretrained_policy_kwargs or {
            "net_arch": [dict(pi=[128, 128], vf=[128, 128])],
            "activation_fn": __import__("torch.nn", fromlist=["ReLU"]).ReLU,
            "pointnet_variant": resolved_pointnet_variant,
            "pointnet_checkpoint_path": pointnet_checkpoint_path,
            "freeze_pointnet": freeze_pointnet,
        }
        if pretrained_model_checkpoint_path is not None and resume_training_state:
            agent = PPO.load(
                pretrained_model_checkpoint_path,
                env=env,
                device="auto",
                custom_objects={"policy_kwargs": pretrained_policy_kwargs},
                learning_rate=learning_rate,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                verbose=verbose,
                wandb_project="dexpoint-franka" if use_wandb else None,
                wandb_run_name=resolved_wandb_run_name if use_wandb else None,
            )
            print(f"|Resumed PPO agent from {pretrained_model_checkpoint_path}")
        else:
            agent = PPO(
                policy=DexPointPolicy,
                env=env,
                learning_rate=learning_rate,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                gamma=0.992,
                gae_lambda=0.95,
                clip_range=0.2,
                clip_range_vf=None,
                ent_coef=0.0,
                vf_coef=0.5,
                max_grad_norm=0.5,
                use_sde=False,
                sde_sample_freq=-1,
                target_kl=None,
                create_eval_env=False,
                policy_kwargs=policy_kwargs,
                verbose=verbose,
                wandb_project="dexpoint-franka" if use_wandb else None,
                wandb_run_name=resolved_wandb_run_name if use_wandb else None,
            )
            if pretrained_model_checkpoint_path is not None:
                initialize_agent_from_checkpoint(
                    agent,
                    pretrained_model_checkpoint_path,
                )
                print(
                    f"|Initialized PPO agent weights from {pretrained_model_checkpoint_path}"
                )
            else:
                print(f"|PPO agent created")
    elif agent_name == "a2c":
        policy_kwargs = pretrained_policy_kwargs or {
            "activation_fn": __import__("torch.nn", fromlist=["ReLU"]).ReLU,
            "pointnet_variant": resolved_pointnet_variant,
            "pointnet_checkpoint_path": pointnet_checkpoint_path,
            "freeze_pointnet": freeze_pointnet,
        }
        if pretrained_model_checkpoint_path is not None and resume_training_state:
            agent = A2C.load(
                pretrained_model_checkpoint_path,
                env=env,
                device="auto",
                custom_objects={"policy_kwargs": pretrained_policy_kwargs},
                learning_rate=learning_rate,
                n_steps=n_steps // 2,
                verbose=verbose,
            )
            print(f"|Resumed A2C agent from {pretrained_model_checkpoint_path}")
        else:
            agent = A2C(
                policy=DexPointPolicy,
                env=env,
                learning_rate=learning_rate,
                n_steps=n_steps // 2,  # A2C typically uses shorter rollouts
                gamma=0.992,
                gae_lambda=0.95,
                max_grad_norm=0.5,
                use_sde=False,
                sde_sample_freq=-1,
                create_eval_env=False,
                policy_kwargs=policy_kwargs,
                verbose=verbose,
            )
            if pretrained_model_checkpoint_path is not None:
                initialize_agent_from_checkpoint(
                    agent,
                    pretrained_model_checkpoint_path,
                )
                print(
                    f"|Initialized A2C agent weights from {pretrained_model_checkpoint_path}"
                )
            else:
                print(f"|A2C agent created")

    if agent is None:
        print(f"Unsupported agent: {agent_name}")
        env.close()
        if eval_env is not None:
            eval_env.close()
        return

    # Training configuration
    print(f"\n Training configuration:")
    print(f"  - Agent: {agent_name.upper()}")
    print(f"  - Device: {agent.device}")
    print(f"  - Parallel envs: {num_envs}")
    print(f"  - Rollout size: {n_steps * num_envs}")
    print(f"  - Total timesteps: {total_timesteps}")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Epochs per update: {n_epochs}")
    print(f"  - Steps per update: {n_steps}")
    print(f"  - Save checkpoint every: {save_interval} steps")
    print(f"  - Validation tries per object: {eval_tries}")

    # Save training config
    config = {
        "task": task_name,
        "agent": agent_name,
        "num_envs": num_envs,
        "total_timesteps": total_timesteps,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "n_steps": n_steps,
        "rollout_size": n_steps * num_envs,
        "timestamp": datetime.now().isoformat(),
        "use_wandb": use_wandb,
        "record_video": record_video,
        "video_interval": video_interval,
        "wandb_run_name": resolved_wandb_run_name if use_wandb else None,
        "initialization_mode": initialization_mode,
        "pretrained_model_checkpoint_path": pretrained_model_checkpoint_path,
        "resume_training_state": resume_training_state,
        "eval_tries": eval_tries,
        "pointnet_variant": resolved_pointnet_variant,
        "env_config": {
            "num_points": 512,
            "camera_names": reference_env.camera_names,
            "use_depth_only_pointcloud": True,
            "training_camera_height": TRAIN_CAMERA_HEIGHT,
            "training_camera_width": TRAIN_CAMERA_WIDTH,
            "eval_camera_height": EVAL_CAMERA_HEIGHT,
            "eval_camera_width": EVAL_CAMERA_WIDTH,
            "target_body_name": reference_env.target_body_name,
            "ycb_object_root": reference_env.ycb_object_root.as_posix(),
            "ycb_asset_source": reference_env.ycb_asset_source,
            "target_scale": reference_env.target_scale,
        },
        "pointnet_checkpoint_path": pointnet_checkpoint_path,
        "freeze_pointnet": freeze_pointnet,
        "ycb_object_names": ycb_object_names,
    }

    config_path = run_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  - Config saved to: {config_path}")

    print(f"  Starting training loop...")

    try:
        n_save_steps = save_interval
        steps_so_far = 0
        step_count = 0
        next_video_step = video_interval
        video_recording_enabled = record_video and eval_env is not None
        video_frames = []
        training_callback = TaskInfoLoggingCallback(use_wandb=use_wandb)

        while steps_so_far < total_timesteps:
            remaining = total_timesteps - steps_so_far
            train_steps = min(n_save_steps, remaining)
            batch_start = steps_so_far

            print(
                f"\n  Training batch {step_count + 1}: target {train_steps} steps ({steps_so_far}/{total_timesteps} total)"
            )

            # Train for this batch
            agent.learn(
                total_timesteps=train_steps,
                callback=training_callback,
                reset_num_timesteps=(step_count == 0),
            )
            steps_so_far = int(agent.num_timesteps)
            step_count += 1
            actual_batch_steps = steps_so_far - batch_start
            print(f"    Collected {actual_batch_steps} environment steps")

            # Record video if needed
            if video_recording_enabled and steps_so_far >= next_video_step:
                validation_object_names = get_validation_object_names(
                    eval_env, ycb_object_names
                )
                validation_scalar_logs = {}
                validation_media_logs = {}
                all_validation_rewards = []
                all_validation_successes = []
                all_validation_episode_lengths = []
                representative_plot_path = None
                representative_video_path = None
                print(
                    "    Recording validation episodes "
                    f"({len(validation_object_names)} object(s) x {eval_tries} try/tries)..."
                )
                max_episode_steps_video = 200

                for object_name in validation_object_names:
                    object_metric_key = sanitize_metric_key(object_name)
                    object_rewards = []
                    object_successes = []
                    object_episode_lengths = []
                    eval_env.set_fixed_object(object_name)
                    for eval_try_index in range(1, eval_tries + 1):
                        video_frames = []
                        obs = eval_env.reset()
                        episode_reward = 0.0
                        episode_steps = 0
                        success = False
                        bonus_reward = 0.0
                        gripper_actuator_force = 0.0
                        reward_trace = []

                        while episode_steps < max_episode_steps_video:
                            frame = eval_env.render_with_pointcloud(mode="rgb_array")
                            if frame is not None:
                                video_frames.append(frame)

                            batched_obs = add_batch_dimension(obs)
                            action, _ = agent.predict(batched_obs, deterministic=True)

                            if isinstance(action, np.ndarray) and action.ndim > 1:
                                action = action[0]

                            obs, reward, done, info = eval_env.step(action)
                            episode_reward += reward
                            reward_trace.append(float(reward))
                            episode_steps += 1

                            success = bool(info.get("is_success", False))
                            bonus_reward = float(info.get("bonus_reward", 0.0))
                            gripper_actuator_force = float(
                                info.get("gripper_actuator_force", 0.0)
                            )

                            if done:
                                if success:
                                    print(
                                        "    Validation success "
                                        f"[{object_name} try {eval_try_index}] "
                                        f"reward={episode_reward:.2f} steps={episode_steps}"
                                    )
                                break

                        video_path = (
                            run_dir
                            / f"video_step_{steps_so_far}_{object_name}_try_{eval_try_index:02d}.mp4"
                        )
                        plot_path = (
                            run_dir
                            / f"plot_step_{steps_so_far}_{object_name}_try_{eval_try_index:02d}.png"
                        )
                        save_validation_reward_plot(
                            reward_trace,
                            plot_path,
                            object_name=object_name,
                            eval_try_index=eval_try_index,
                            success=success,
                        )
                        if representative_plot_path is None:
                            representative_plot_path = plot_path

                        object_rewards.append(float(episode_reward))
                        object_successes.append(float(success))
                        object_episode_lengths.append(float(episode_steps))
                        all_validation_rewards.append(float(episode_reward))
                        all_validation_successes.append(float(success))
                        all_validation_episode_lengths.append(float(episode_steps))

                        validation_scalar_logs[
                            f"validation/objects/{object_metric_key}/try_{eval_try_index:02d}/episode_reward"
                        ] = float(episode_reward)
                        validation_scalar_logs[
                            f"validation/objects/{object_metric_key}/try_{eval_try_index:02d}/episode_steps"
                        ] = float(episode_steps)
                        validation_scalar_logs[
                            f"validation/objects/{object_metric_key}/try_{eval_try_index:02d}/episode_success"
                        ] = float(success)
                        validation_scalar_logs[
                            f"validation/objects/{object_metric_key}/try_{eval_try_index:02d}/bonus_reward"
                        ] = float(bonus_reward)
                        validation_scalar_logs[
                            f"validation/objects/{object_metric_key}/try_{eval_try_index:02d}/gripper_actuator_force"
                        ] = float(gripper_actuator_force)
                        validation_scalar_logs[
                            f"validation/{object_metric_key}/episode_reward"
                        ] = float(episode_reward)
                        validation_scalar_logs[
                            f"validation/{object_metric_key}/episode_steps"
                        ] = float(episode_steps)
                        validation_scalar_logs[
                            f"validation/{object_metric_key}/episode_success"
                        ] = float(success)

                        if video_frames:
                            try:
                                imageio.mimwrite(video_path, video_frames, fps=30)
                                print(f"    Video saved: {video_path}")
                                if representative_video_path is None:
                                    representative_video_path = video_path
                                if use_wandb and wandb.run is not None:
                                    validation_media_logs[
                                        f"validation/videos/{object_metric_key}/try_{eval_try_index:02d}"
                                    ] = wandb.Video(str(video_path), format="mp4")
                            except (FileNotFoundError, OSError) as exc:
                                video_recording_enabled = False
                                print(
                                    "    Video recording disabled for the rest of this run: "
                                    f"{exc}"
                                )
                                if use_wandb and wandb.run is not None:
                                    validation_scalar_logs[
                                        f"validation/objects/{object_metric_key}/video_error"
                                    ] = str(exc)
                                break

                    if not video_recording_enabled:
                        break

                    if object_rewards:
                        validation_scalar_logs[
                            f"validation/objects/{object_metric_key}/mean_episode_reward"
                        ] = float(np.mean(object_rewards))
                        validation_scalar_logs[
                            f"validation/objects/{object_metric_key}/success_rate"
                        ] = float(np.mean(object_successes))
                        validation_scalar_logs[
                            f"validation/objects/{object_metric_key}/mean_episode_steps"
                        ] = float(np.mean(object_episode_lengths))
                        validation_scalar_logs[
                            f"validation/{object_metric_key}/mean_episode_reward"
                        ] = float(np.mean(object_rewards))
                        validation_scalar_logs[
                            f"validation/{object_metric_key}/success_rate"
                        ] = float(np.mean(object_successes))
                        validation_scalar_logs[
                            f"validation/{object_metric_key}/mean_episode_steps"
                        ] = float(np.mean(object_episode_lengths))

                eval_env.set_fixed_object(None)
                if all_validation_rewards:
                    validation_scalar_logs["validation/timesteps"] = float(
                        steps_so_far
                    )
                    validation_scalar_logs["validation/episode_reward"] = float(
                        np.mean(all_validation_rewards)
                    )
                    validation_scalar_logs["validation/episode_steps"] = float(
                        np.mean(all_validation_episode_lengths)
                    )
                    validation_scalar_logs["validation/episode_success"] = float(
                        np.mean(all_validation_successes)
                    )
                    validation_scalar_logs["validation/mean_episode_reward"] = float(
                        np.mean(all_validation_rewards)
                    )
                    validation_scalar_logs["validation/mean_episode_steps"] = float(
                        np.mean(all_validation_episode_lengths)
                    )
                    validation_scalar_logs["validation/success_rate"] = float(
                        np.mean(all_validation_successes)
                    )
                    validation_scalar_logs["validation/object_count"] = float(
                        len(validation_object_names)
                    )

                if use_wandb and wandb.run is not None:
                    if representative_plot_path is not None:
                        validation_media_logs["validation/reward_plot"] = wandb.Image(
                            str(representative_plot_path)
                        )
                    if representative_video_path is not None:
                        validation_media_logs["validation/video"] = wandb.Video(
                            str(representative_video_path), format="mp4"
                        )

                    combined_validation_logs = {
                        key: value
                        for key, value in {
                            **validation_scalar_logs,
                            **validation_media_logs,
                        }.items()
                        if value is not None
                    }
                    if combined_validation_logs:
                        wandb.log(combined_validation_logs)
                while next_video_step <= steps_so_far:
                    next_video_step += video_interval

            # Save checkpoint
            checkpoint_path = run_dir / f"model_checkpoint_{steps_so_far}.zip"
            agent.save(str(checkpoint_path))
            print(f"    Checkpoint saved: {checkpoint_path}")

        # Save final model
        final_model_path = run_dir / "model_final.zip"
        agent.save(str(final_model_path))
        print(f"\n✓ Training complete!")
        print(f"  - Final model saved: {final_model_path}")
        print(f"  - Training run: {run_dir}")

        if use_wandb:
            wandb.log(
                {
                    "training/final_timesteps": steps_so_far,
                    "training/total_batches": step_count,
                },
                step=steps_so_far,
            )
            wandb.finish()

    except KeyboardInterrupt:
        print(f"\n✓ Training interrupted by user")
        final_model_path = run_dir / "model_interrupted.zip"
        agent.save(str(final_model_path))
        print(f"  - Checkpoint saved: {final_model_path}")
        if use_wandb:
            wandb.finish()

    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback

        traceback.print_exc()
        if use_wandb:
            wandb.finish()

    finally:
        env.close()
        if eval_env is not None:
            eval_env.close()
        print(f"\nEnvironment closed.")


def main():
    """Main training entry point."""

    parser = argparse.ArgumentParser(description="DexPoint training script")
    parser.add_argument(
        "--task",
        type=str,
        default="grasping",
        choices=["grasping", "reaching", "lifting", "lifting_only", "placing", "placing_v2", "placing_v3"],
        help="Task to train on",
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="ppo",
        choices=["ppo", "a2c"],
        help="RL algorithm to train with",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Number of parallel training environments.",
    )
    parser.add_argument("--steps", type=int, default=10000, help="Total training steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Batch size for PPO updates"
    )
    parser.add_argument("--epochs", type=int, default=6, help="Epochs per PPO update")
    parser.add_argument(
        "--verbose", type=int, default=1, choices=[0, 1], help="Verbosity level"
    )
    parser.add_argument(
        "--wandb", action="store_true", help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Explicit Weights & Biases run name.",
    )
    parser.add_argument(
        "--record-video",
        action="store_true",
        default=True,
        help="Record training videos (default: True)",
    )
    parser.add_argument(
        "--no-video",
        action="store_false",
        dest="record_video",
        help="Disable video recording",
    )
    parser.add_argument(
        "--video-interval",
        type=int,
        default=20000,
        help="Record video every N timesteps",
    )
    parser.add_argument(
        "--pointnet-checkpoint",
        type=str,
        default=get_default_pointnet_checkpoint(),
        help="Path to a pretrained PointNet encoder checkpoint.",
    )
    parser.add_argument(
        "--pretrained-model-checkpoint",
        type=str,
        default=None,
        help=(
            "Path to a pretrained RL checkpoint (.zip) whose policy weights are used "
            "to initialize a fresh training run. When provided, this takes precedence "
            "over PointNet-only initialization."
        ),
    )
    parser.add_argument(
        "--resume-training-state",
        action="store_true",
        help=(
            "Fully resume optimizer, timestep, and scheduler state from "
            "--pretrained-model-checkpoint instead of using it only for weight initialization."
        ),
    )
    parser.add_argument(
        "--pointnet-variant",
        type=str,
        default="auto",
        choices=["auto", "small", "medium", "large"],
        help="PointNet architecture to use. 'auto' infers the variant from the checkpoint.",
    )
    parser.add_argument(
        "--freeze-pointnet",
        action="store_true",
        help="Freeze the pretrained PointNet encoder during RL training.",
    )
    parser.add_argument(
        "--eval-tries",
        type=int,
        default=1,
        help=(
            "Number of validation episodes to run per object whenever evaluation is triggered. "
            "If a single object is configured, it runs this many times; if multiple objects are "
            "configured, it runs this many times for each object."
        ),
    )
    parser.add_argument(
        "--ycb-object-names",
        type=str,
        nargs="+",
        default=None,
        metavar="OBJECT_NAME",
        help=(
            "List of YCB object folder names to randomly sample one from per episode "
            "(e.g. 005_tomato_soup_can 006_mustard_bottle).  Requires pre-generated "
            "YCB_sim scene files for each name."
        ),
    )
    parser.add_argument(
        "--rollout-size",
        type=int,
        default=10000,
        help="Number of steps per rollout.",
    )

    args = parser.parse_args()

    train_dexpoint(
        task_name=args.task,
        agent_name=args.agent,
        num_envs=args.num_envs,
        total_timesteps=args.steps,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        n_steps=args.rollout_size,
        verbose=args.verbose,
        use_wandb=args.wandb,
        record_video=args.record_video,
        video_interval=args.video_interval,
        pointnet_checkpoint_path=args.pointnet_checkpoint,
        pretrained_model_checkpoint_path=args.pretrained_model_checkpoint,
        pointnet_variant=args.pointnet_variant,
        freeze_pointnet=args.freeze_pointnet,
        wandb_run_name=args.wandb_run_name,
        resume_training_state=args.resume_training_state,
        ycb_object_names=args.ycb_object_names,
        eval_tries=args.eval_tries,
    )


if __name__ == "__main__":
    main()
