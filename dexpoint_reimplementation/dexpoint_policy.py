import numpy as np
import torch as th
import torch.nn as nn
from typing import Dict, List, Optional, Any, Type, Union
import gym
from gym import spaces

# Import from dexart_baselines
import sys
from pathlib import Path

root_path = Path(__file__).parent.parent
sys.path.insert(0, str(root_path))

from dexart_baselines.stable_baselines3.networks.pretrain_nets import (
    PointNet,
    PointNetMedium,
    PointNetLarge,
)
from dexart_baselines.stable_baselines3.common.policies import (
    ActorCriticPolicy,
    BaseFeaturesExtractor,
    MlpExtractor,
)


class PointNetExtractor(nn.Module):
    """PointNet feature extractor wrapper for point clouds."""

    def __init__(
        self,
        pointnet_variant: str = "medium",
        output_dim: int = 256,
        checkpoint_path: Optional[str] = None,
        freeze: bool = False,
    ):
        """
        Initialize PointNet extractor.

        Args:
            pointnet_variant: "small", "medium", or "large"
            output_dim: Output feature dimension
        """
        super().__init__()

        if pointnet_variant == "small":
            self.pointnet = PointNet(point_channel=3, output_dim=output_dim)
        elif pointnet_variant == "medium":
            self.pointnet = PointNetMedium(point_channel=3, output_dim=output_dim)
        elif pointnet_variant == "large":
            self.pointnet = PointNetLarge(point_channel=3, output_dim=output_dim)
        else:
            raise ValueError(f"Unknown PointNet variant: {pointnet_variant}")

        self.output_dim = output_dim

        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path)

        if freeze:
            for parameter in self.pointnet.parameters():
                parameter.requires_grad = False

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = th.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        expected_keys = set(self.pointnet.state_dict().keys())

        candidate_state_dicts = [state_dict]
        for prefix in ("module.", "encoder.", "pointnet.", "pointnet_extractor.pointnet."):
            stripped = {}
            changed = False
            for key, value in state_dict.items():
                if key.startswith(prefix):
                    stripped[key[len(prefix) :]] = value
                    changed = True
                else:
                    stripped[key] = value
            if changed:
                candidate_state_dicts.append(stripped)

        for candidate in candidate_state_dicts:
            if expected_keys.issubset(candidate.keys()):
                filtered_candidate = {key: candidate[key] for key in expected_keys}
                self.pointnet.load_state_dict(filtered_candidate, strict=True)
                print(f"Loaded pretrained PointNet weights from {checkpoint_path}")
                return

        raise RuntimeError(
            f"Checkpoint at {checkpoint_path} does not match PointNet {type(self.pointnet).__name__}"
        )

    def forward(self, pointcloud: th.Tensor) -> th.Tensor:
        """
        Extract features from point cloud.

        Args:
            pointcloud: Shape [B, N, 3] - batch of point clouds

        Returns:
            features: Shape [B, output_dim] - extracted features
        """
        return self.pointnet(pointcloud)


class ProprioceptiveExtractor(nn.Module):
    """Simple MLP extractor for proprioceptive (joint) state."""

    def __init__(self, input_dim: int, output_dim: int = 64):
        """
        Initialize proprioceptive extractor.

        Args:
            input_dim: Input dimension
            output_dim: Output feature dimension
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )
        self.output_dim = output_dim

    def forward(self, joint_state: th.Tensor) -> th.Tensor:
        """
        Extract features from joint state.

        Args:
            joint_state: Shape [B, input_dim]

        Returns:
            features: Shape [B, output_dim]
        """
        return self.mlp(joint_state)


class DexPointFeaturesExtractor(BaseFeaturesExtractor):
    """
    Multi-modal feature extractor combining point clouds and proprioception.

    Combines PointNet for point cloud encoding with a simple MLP for joint state,
    then concatenates both features for downstream processing.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        pointnet_variant: str = "medium",
        pointnet_output_dim: int = 256,
        proprioceptive_output_dim: int = 64,
        pointnet_checkpoint_path: Optional[str] = None,
        freeze_pointnet: bool = False,
    ):
        """
        Initialize DexPoint features extractor.

        Args:
            observation_space: Gym Dict observation space with 'pointcloud' and 'joint_state'
            pointnet_variant: PointNet architecture variant ("small", "medium", "large")
            pointnet_output_dim: PointNet output dimension (keep at 256 for pretrained weights)
            proprioceptive_output_dim: Proprioceptive encoder output dimension
        """
        # Extract subspace shapes BEFORE calling super().__init__
        pointcloud_space = observation_space.spaces["pointcloud"]
        joint_state_space = observation_space.spaces["joint_state"]
        ee_position_space = observation_space.spaces.get("ee_position")
        goal_position_space = observation_space.spaces.get("goal_position")

        assert isinstance(pointcloud_space, spaces.Box), "pointcloud must be Box space"
        assert len(pointcloud_space.shape) == 2, "pointcloud must be [N, 3]"
        assert pointcloud_space.shape[1] == 3, "pointcloud must have 3 channels (xyz)"

        assert isinstance(
            joint_state_space, spaces.Box
        ), "joint_state must be Box space"
        assert len(joint_state_space.shape) == 1, "joint_state must be 1D"

        # Proprioceptive dim = joint_state + ee_position (3) + goal_position (3)
        proprio_input_dim = joint_state_space.shape[0]
        if ee_position_space is not None:
            proprio_input_dim += ee_position_space.shape[0]
        if goal_position_space is not None:
            proprio_input_dim += goal_position_space.shape[0]
        self._has_ee_position = ee_position_space is not None
        self._has_goal_position = goal_position_space is not None

        # Calculate total features dimension
        total_features_dim = pointnet_output_dim + proprioceptive_output_dim

        # Call parent init with features_dim
        super().__init__(observation_space, features_dim=total_features_dim)

        # Initialize PointNet for point clouds
        self.pointnet_extractor = PointNetExtractor(
            pointnet_variant=pointnet_variant,
            output_dim=pointnet_output_dim,
            checkpoint_path=pointnet_checkpoint_path,
            freeze=freeze_pointnet,
        )

        # Initialize MLP for proprioceptive state (joints + ee_position + goal_position)
        self.proprioceptive_extractor = ProprioceptiveExtractor(
            input_dim=proprio_input_dim,
            output_dim=proprioceptive_output_dim,
        )

    def forward(self, observations: Dict[str, th.Tensor]) -> th.Tensor:
        """
        Extract and combine features from all modalities.

        Args:
            observations: Dict with keys:
                - 'pointcloud': [B, N, 3]
                - 'joint_state': [B, joint_dim]
                - 'ee_position': [B, 3]  (optional)
                - 'goal_position': [B, 3]  (optional)

        Returns:
            combined_features: [B, features_dim]
        """
        # Extract point cloud features
        pointcloud = observations["pointcloud"]
        pc_features = self.pointnet_extractor(pointcloud)

        # Build proprioceptive vector: joints [+ ee_position] [+ goal_position]
        proprio_parts = [observations["joint_state"]]
        if self._has_ee_position:
            proprio_parts.append(observations["ee_position"])
        if self._has_goal_position:
            proprio_parts.append(observations["goal_position"])
        proprio_vec = th.cat(proprio_parts, dim=1)
        proprio_features = self.proprioceptive_extractor(proprio_vec)

        # Concatenate all features
        combined_features = th.cat([pc_features, proprio_features], dim=1)

        return combined_features


class DexPointPolicy(ActorCriticPolicy):
    """
    Actor-Critic policy for DexPoint using point clouds and proprioception.

    Combines PointNet-based point cloud encoding with proprioceptive state
    for learning dexterous manipulation tasks via PPO.

    See dexart_baselines/stable_baselines3/common/policies.py for base class.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = False,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        pointnet_variant: str = "medium",
        pointnet_output_dim: int = 256,
        proprioceptive_output_dim: int = 64,
        pointnet_checkpoint_path: Optional[str] = None,
        freeze_pointnet: bool = False,
    ):
        """
        Initialize DexPoint policy.

        Args:
            observation_space: Gym Dict space with 'pointcloud' and 'joint_state'
            action_space: Gym Box space for continuous actions
            lr_schedule: Learning rate schedule
            net_arch: Network architecture dict with 'pi' and 'vf' keys
            activation_fn: Activation function
            ortho_init: Whether to use orthogonal initialization
            use_sde: Whether to use State Dependent Exploration
            log_std_init: Initial log standard deviation
            features_extractor_kwargs: Additional kwargs for DexPointFeaturesExtractor
            normalize_images: Whether to normalize images
            optimizer_class: Optimizer class
            optimizer_kwargs: Optimizer kwargs
            pointnet_variant: PointNet architecture ("small", "medium", "large")
            pointnet_output_dim: PointNet output dimension
            proprioceptive_output_dim: Proprioceptive encoder output dimension
        """
        # Default network architecture if not provided
        if net_arch is None:
            net_arch = [dict(pi=[64, 64], vf=[64, 64])]

        if features_extractor_kwargs is None:
            features_extractor_kwargs = {}

        # Add DexPoint-specific kwargs to features extractor
        features_extractor_kwargs.update(
            {
                "pointnet_variant": pointnet_variant,
                "pointnet_output_dim": pointnet_output_dim,
                "proprioceptive_output_dim": proprioceptive_output_dim,
                "pointnet_checkpoint_path": pointnet_checkpoint_path,
                "freeze_pointnet": freeze_pointnet,
            }
        )

        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
            use_sde=use_sde,
            log_std_init=log_std_init,
            full_std=full_std,
            use_expln=use_expln,
            squash_output=squash_output,
            features_extractor_class=DexPointFeaturesExtractor,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )

        self.pointnet_variant = pointnet_variant


class MultiInputActorCriticPolicy(ActorCriticPolicy):
    """
    See dexart_baselines/stable_baselines3/ppo/policies.py
    """

    pass


def create_dexpoint_policy(
    observation_space: gym.spaces.Dict,
    action_space: gym.spaces.Box,
    lr_schedule,
    pointnet_variant: str = "medium",
    net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
    pointnet_checkpoint_path: Optional[str] = None,
    freeze_pointnet: bool = False,
    **kwargs,
) -> DexPointPolicy:
    """
    Create a DexPoint policy.

    Args:
        observation_space: Dict space with 'pointcloud' and 'joint_state'
        action_space: Continuous action space
        lr_schedule: Learning rate schedule
        pointnet_variant: PointNet variant ("small", "medium", "large")
        net_arch: Network architecture
        **kwargs: Additional kwargs for DexPointPolicy

    Returns:
        DexPointPolicy instance
    """
    return DexPointPolicy(
        observation_space=observation_space,
        action_space=action_space,
        lr_schedule=lr_schedule,
        net_arch=net_arch,
        pointnet_variant=pointnet_variant,
        pointnet_checkpoint_path=pointnet_checkpoint_path,
        freeze_pointnet=freeze_pointnet,
        **kwargs,
    )
