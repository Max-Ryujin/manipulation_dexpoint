"""Training script for DexPoint on Franka robots."""

import imageio
import numpy as np
import argparse
from pathlib import Path
import json
from datetime import datetime
import sys
from typing import TYPE_CHECKING, Callable, Optional
import torch
import wandb

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
    """Load policy kwargs from a saved RL checkpoint for safe restoration."""
    data, _, _ = load_from_zip_file(str(checkpoint_path), device="cpu")
    policy_kwargs = dict(data.get("policy_kwargs", {}))

    # Full policy checkpoints already contain PointNet weights, so do not require
    # the original external encoder checkpoint to be present during restore.
    policy_kwargs["pointnet_checkpoint_path"] = None
    if freeze_pointnet:
        policy_kwargs["freeze_pointnet"] = True

    return policy_kwargs


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
        )
        if seed is not None:
            env.seed(seed)
        return env

    return _make_env


def train_dexpoint(
    task_name: str = "grasping",
    agent_name: str = "ppo",
    num_envs: int = 1,
    total_timesteps: int = 100000,
    learning_rate: float = 3e-4,
    batch_size: int = 64,
    n_epochs: int = 6,
    n_steps: int = 2400,
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
        pretrained_model_checkpoint_path: Optional RL checkpoint to continue training from
        pointnet_variant: PointNet architecture variant or "auto" to infer it
        freeze_pointnet: Whether to keep PointNet frozen during RL training
        wandb_run_name: Optional explicit Weights & Biases run name
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
        "pretrained_model"
        if pretrained_model_checkpoint_path is not None
        else "pointnet_only"
    )

    if num_envs < 1:
        raise ValueError(f"num_envs must be >= 1, got {num_envs}")

    if use_wandb:
        wandb.init(
            project=(
                "dexpoint-franka_reaching"
                if task_name == "reaching"
                else "dexpoint-franka"
            ),
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
                "wandb_run_name": resolved_wandb_run_name,
            },
        )

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
        print(f"  - Pretrained RL checkpoint: {pretrained_model_checkpoint_path}")
    elif pointnet_checkpoint_path is not None:
        print(f"  - Pretrained PointNet checkpoint: {pointnet_checkpoint_path}")
    print(f"  - Observation space: {env.observation_space}")
    print(f"  - Action space: {env.action_space}")

    # Create training agent
    print(f"\n Creating {agent_name.upper()} agent...")

    agent = None

    if agent_name == "ppo":
        if pretrained_model_checkpoint_path is not None:
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
            print(f"|Loaded PPO agent from {pretrained_model_checkpoint_path}")
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
                policy_kwargs={
                    "net_arch": [dict(pi=[128, 128], vf=[128, 128])],
                    "activation_fn": __import__("torch.nn", fromlist=["ReLU"]).ReLU,
                    "pointnet_variant": resolved_pointnet_variant,
                    "pointnet_checkpoint_path": pointnet_checkpoint_path,
                    "freeze_pointnet": freeze_pointnet,
                },
                verbose=verbose,
                wandb_project="dexpoint-franka" if use_wandb else None,
                wandb_run_name=resolved_wandb_run_name if use_wandb else None,
            )
            print(f"|PPO agent created")
    elif agent_name == "a2c":
        if pretrained_model_checkpoint_path is not None:
            agent = A2C.load(
                pretrained_model_checkpoint_path,
                env=env,
                device="auto",
                custom_objects={"policy_kwargs": pretrained_policy_kwargs},
                learning_rate=learning_rate,
                n_steps=n_steps // 2,
                verbose=verbose,
            )
            print(f"|Loaded A2C agent from {pretrained_model_checkpoint_path}")
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
                policy_kwargs={
                    "activation_fn": __import__("torch.nn", fromlist=["ReLU"]).ReLU,
                    "pointnet_variant": resolved_pointnet_variant,
                    "pointnet_checkpoint_path": pointnet_checkpoint_path,
                    "freeze_pointnet": freeze_pointnet,
                },
                verbose=verbose,
            )
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
        },
        "pointnet_checkpoint_path": pointnet_checkpoint_path,
        "freeze_pointnet": freeze_pointnet,
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
                print(f"    Recording validation video...")
                video_frames = []
                obs = eval_env.reset()
                episode_reward = 0.0
                episode_steps = 0
                max_episode_steps_video = 1000

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
                    episode_steps += 1

                    success = info.get("is_success", False)

                    if done:
                        # check for success
                        if success:
                            print(
                                f"    Episode success! Reward: {episode_reward:.2f}, Steps: {episode_steps}"
                            )

                        break

                # Save video
                if video_frames:
                    video_path = run_dir / f"video_step_{steps_so_far}.mp4"
                    try:
                        imageio.mimwrite(video_path, video_frames, fps=30)
                        print(f"    Video saved: {video_path}")
                        if use_wandb and wandb.run is not None:
                            wandb.log(
                                {
                                    "validation/episode_reward": episode_reward,
                                    "validation/episode_steps": episode_steps,
                                    "validation/video": wandb.Video(
                                        str(video_path), format="mp4"
                                    ),
                                    "validation/episode_success": success,
                                }
                            )
                    except (FileNotFoundError, OSError) as exc:
                        video_recording_enabled = False
                        print(
                            "    Video recording disabled for the rest of this run: "
                            f"{exc}"
                        )
                        if use_wandb and wandb.run is not None:
                            wandb.log(
                                {
                                    "validation/episode_reward": episode_reward,
                                    "validation/episode_steps": episode_steps,
                                    "validation/episode_success": success,
                                    "validation/video_error": str(exc),
                                }
                            )
                elif use_wandb and wandb.run is not None:
                    wandb.log(
                        {
                            "validation/episode_reward": episode_reward,
                            "validation/episode_steps": episode_steps,
                        }
                    )
                while next_video_step <= steps_so_far:
                    next_video_step += video_interval

            # Save checkpoint
            checkpoint_path = run_dir / f"model_checkpoint_{steps_so_far}.zip"
            agent.save(str(checkpoint_path))
            print(f"    Checkpoint saved: {checkpoint_path}")

            # Log to W&B
            if use_wandb:
                wandb.log(
                    {
                        "training/timesteps": steps_so_far,
                        "training/batch": step_count,
                    }
                )

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
                }
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
        choices=["grasping", "reaching"],
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
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size for PPO updates"
    )
    parser.add_argument("--epochs", type=int, default=10, help="Epochs per PPO update")
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
            "Path to a pretrained RL checkpoint (.zip) to continue training from. "
            "When provided, this takes precedence over PointNet-only initialization."
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

    args = parser.parse_args()

    train_dexpoint(
        task_name=args.task,
        agent_name=args.agent,
        num_envs=args.num_envs,
        total_timesteps=args.steps,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        verbose=args.verbose,
        use_wandb=args.wandb,
        record_video=args.record_video,
        video_interval=args.video_interval,
        pointnet_checkpoint_path=args.pointnet_checkpoint,
        pretrained_model_checkpoint_path=args.pretrained_model_checkpoint,
        pointnet_variant=args.pointnet_variant,
        freeze_pointnet=args.freeze_pointnet,
        wandb_run_name=args.wandb_run_name,
    )


if __name__ == "__main__":
    main()
