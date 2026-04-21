#!/usr/bin/env python3

"""Analyze how trained DexPoint PPO checkpoints use observation groups."""

import argparse
import json
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch as th
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont

from _script_bootstrap import ensure_script_imports

ensure_script_imports()

import dexpoint_policy  # noqa: F401
from dexart_baselines.stable_baselines3.common.save_util import load_from_zip_file
from dexart_baselines.stable_baselines3.ppo import PPO


OBSERVATION_GROUP_KEYS = ["pointcloud", "joint_state", "ee_position", "goal_position"]
BRANCH_KEYS = ["pointcloud", "proprio"]


def _linear_layers(module: nn.Module) -> List[nn.Linear]:
    return [
        submodule for submodule in module.modules() if isinstance(submodule, nn.Linear)
    ]


def _effective_linear_map(modules: Iterable[nn.Module], input_dim: int) -> th.Tensor:
    effective: Optional[th.Tensor] = None
    current_input_dim = input_dim

    for module in modules:
        for linear in _linear_layers(module):
            weight = linear.weight.detach().cpu()
            effective = weight if effective is None else weight @ effective
            current_input_dim = linear.in_features

    if effective is None:
        return th.eye(current_input_dim, dtype=th.float32)

    return effective


def _sum_abs_columns(matrix: th.Tensor, column_slice: slice) -> float:
    if column_slice.start == column_slice.stop:
        return 0.0
    return float(matrix[:, column_slice].abs().sum().item())


def _normalized_shares(scores: Mapping[str, float]) -> OrderedDict:
    total = float(sum(scores.values()))
    if total <= 0.0:
        return OrderedDict((key, 0.0) for key in scores)
    return OrderedDict((key, 100.0 * value / total) for key, value in scores.items())


def _size_normalized_scores(
    scores: Mapping[str, float], group_dims: Mapping[str, int]
) -> OrderedDict:
    normalized = OrderedDict()
    for key, value in scores.items():
        dim_count = int(group_dims.get(key, 0))
        normalized[key] = 0.0 if dim_count <= 0 else float(value) / float(dim_count)
    return normalized


def _module_summary(module: nn.Module) -> Dict[str, float]:
    params = [parameter.detach().cpu().reshape(-1) for parameter in module.parameters()]
    if not params:
        return {"parameter_count": 0, "l1_sum": 0.0, "l2_norm": 0.0}

    flat = th.cat(params)
    return {
        "parameter_count": int(flat.numel()),
        "l1_sum": float(flat.abs().sum().item()),
        "l2_norm": float(th.linalg.vector_norm(flat, ord=2).item()),
    }


def _make_observation_layout(policy: nn.Module) -> Dict[str, object]:
    observation_space = policy.observation_space
    features_extractor = policy.features_extractor

    pointcloud_shape = tuple(
        int(dim) for dim in observation_space.spaces["pointcloud"].shape
    )
    joint_dim = int(observation_space.spaces["joint_state"].shape[0])
    ee_dim = (
        int(observation_space.spaces["ee_position"].shape[0])
        if "ee_position" in observation_space.spaces
        else 0
    )
    goal_dim = (
        int(observation_space.spaces["goal_position"].shape[0])
        if "goal_position" in observation_space.spaces
        else 0
    )

    pointcloud_feature_dim = int(features_extractor.pointnet_extractor.output_dim)
    proprio_feature_dim = int(features_extractor.proprioceptive_extractor.output_dim)

    proprio_slices = OrderedDict()
    start = 0
    proprio_slices["joint_state"] = slice(start, start + joint_dim)
    start += joint_dim
    if ee_dim > 0:
        proprio_slices["ee_position"] = slice(start, start + ee_dim)
        start += ee_dim
    if goal_dim > 0:
        proprio_slices["goal_position"] = slice(start, start + goal_dim)

    feature_slices = OrderedDict(
        pointcloud=slice(0, pointcloud_feature_dim),
        proprio=slice(
            pointcloud_feature_dim, pointcloud_feature_dim + proprio_feature_dim
        ),
    )

    return {
        "pointcloud_shape": pointcloud_shape,
        "raw_dims": OrderedDict(
            pointcloud=int(np.prod(pointcloud_shape)),
            joint_state=joint_dim,
            ee_position=ee_dim,
            goal_position=goal_dim,
        ),
        "pointcloud_feature_dim": pointcloud_feature_dim,
        "proprio_feature_dim": proprio_feature_dim,
        "proprio_input_dim": start,
        "group_analysis_dims": OrderedDict(
            pointcloud=pointcloud_feature_dim,
            joint_state=joint_dim,
            ee_position=ee_dim,
            goal_position=goal_dim,
        ),
        "branch_analysis_dims": OrderedDict(
            pointcloud=pointcloud_feature_dim,
            proprio=proprio_feature_dim,
        ),
        "proprio_slices": proprio_slices,
        "feature_slices": feature_slices,
    }


def _compute_policy_usage(
    policy: nn.Module, layout: Mapping[str, object]
) -> Dict[str, object]:
    pointcloud_feature_dim = int(layout["pointcloud_feature_dim"])
    proprio_feature_dim = int(layout["proprio_feature_dim"])
    group_analysis_dims = layout["group_analysis_dims"]
    branch_analysis_dims = layout["branch_analysis_dims"]
    proprio_slices = layout["proprio_slices"]
    feature_slices = layout["feature_slices"]
    proprio_extractor = policy.features_extractor.proprioceptive_extractor

    actor_effective = _effective_linear_map(
        [
            policy.mlp_extractor.shared_net,
            policy.mlp_extractor.policy_net,
            policy.action_net,
        ],
        input_dim=policy.features_dim,
    )
    critic_effective = _effective_linear_map(
        [
            policy.mlp_extractor.shared_net,
            policy.mlp_extractor.value_net,
            policy.value_net,
        ],
        input_dim=policy.features_dim,
    )
    proprio_effective = _effective_linear_map(
        [proprio_extractor.mlp], input_dim=int(layout["proprio_input_dim"])
    )

    actor_proprio = actor_effective[:, feature_slices["proprio"]]
    critic_proprio = critic_effective[:, feature_slices["proprio"]]

    actor_raw_proprio = actor_proprio @ proprio_effective
    critic_raw_proprio = critic_proprio @ proprio_effective

    actor_group_scores = OrderedDict()
    actor_group_scores["pointcloud"] = _sum_abs_columns(
        actor_effective, feature_slices["pointcloud"]
    )
    actor_group_scores["joint_state"] = _sum_abs_columns(
        actor_raw_proprio, proprio_slices["joint_state"]
    )
    actor_group_scores["ee_position"] = _sum_abs_columns(
        actor_raw_proprio, proprio_slices.get("ee_position", slice(0, 0))
    )
    actor_group_scores["goal_position"] = _sum_abs_columns(
        actor_raw_proprio, proprio_slices.get("goal_position", slice(0, 0))
    )

    critic_group_scores = OrderedDict()
    critic_group_scores["pointcloud"] = _sum_abs_columns(
        critic_effective, feature_slices["pointcloud"]
    )
    critic_group_scores["joint_state"] = _sum_abs_columns(
        critic_raw_proprio, proprio_slices["joint_state"]
    )
    critic_group_scores["ee_position"] = _sum_abs_columns(
        critic_raw_proprio, proprio_slices.get("ee_position", slice(0, 0))
    )
    critic_group_scores["goal_position"] = _sum_abs_columns(
        critic_raw_proprio, proprio_slices.get("goal_position", slice(0, 0))
    )

    actor_branch_scores = OrderedDict(
        pointcloud=_sum_abs_columns(actor_effective, feature_slices["pointcloud"]),
        proprio=_sum_abs_columns(actor_effective, feature_slices["proprio"]),
    )
    critic_branch_scores = OrderedDict(
        pointcloud=_sum_abs_columns(critic_effective, feature_slices["pointcloud"]),
        proprio=_sum_abs_columns(critic_effective, feature_slices["proprio"]),
    )

    combined_group_shares = OrderedDict(
        (
            key,
            0.5
            * (
                _normalized_shares(actor_group_scores)[key]
                + _normalized_shares(critic_group_scores)[key]
            ),
        )
        for key in actor_group_scores
    )
    actor_group_relative_scores = _size_normalized_scores(
        actor_group_scores, group_analysis_dims
    )
    critic_group_relative_scores = _size_normalized_scores(
        critic_group_scores, group_analysis_dims
    )
    combined_group_relative_shares = OrderedDict(
        (
            key,
            0.5
            * (
                _normalized_shares(actor_group_relative_scores)[key]
                + _normalized_shares(critic_group_relative_scores)[key]
            ),
        )
        for key in actor_group_relative_scores
    )
    actor_branch_relative_scores = _size_normalized_scores(
        actor_branch_scores, branch_analysis_dims
    )
    critic_branch_relative_scores = _size_normalized_scores(
        critic_branch_scores, branch_analysis_dims
    )

    first_proprio_linear = _linear_layers(proprio_extractor.mlp)[0]
    proprio_first_layer_scores = OrderedDict(
        (
            key,
            float(
                first_proprio_linear.weight.detach()
                .cpu()[:, column_slice]
                .abs()
                .sum()
                .item()
            ),
        )
        for key, column_slice in proprio_slices.items()
    )
    proprio_first_layer_relative_scores = _size_normalized_scores(
        proprio_first_layer_scores, group_analysis_dims
    )

    pointnet_summary = _module_summary(policy.features_extractor.pointnet_extractor)
    proprio_summary = _module_summary(proprio_extractor)

    return {
        "actor_effective_shape": list(actor_effective.shape),
        "critic_effective_shape": list(critic_effective.shape),
        "actor_group_scores": actor_group_scores,
        "critic_group_scores": critic_group_scores,
        "actor_group_shares": _normalized_shares(actor_group_scores),
        "critic_group_shares": _normalized_shares(critic_group_scores),
        "combined_group_shares": combined_group_shares,
        "actor_group_relative_scores": actor_group_relative_scores,
        "critic_group_relative_scores": critic_group_relative_scores,
        "actor_group_relative_shares": _normalized_shares(actor_group_relative_scores),
        "critic_group_relative_shares": _normalized_shares(
            critic_group_relative_scores
        ),
        "combined_group_relative_shares": combined_group_relative_shares,
        "actor_branch_scores": actor_branch_scores,
        "critic_branch_scores": critic_branch_scores,
        "actor_branch_shares": _normalized_shares(actor_branch_scores),
        "critic_branch_shares": _normalized_shares(critic_branch_scores),
        "actor_branch_relative_scores": actor_branch_relative_scores,
        "critic_branch_relative_scores": critic_branch_relative_scores,
        "actor_branch_relative_shares": _normalized_shares(
            actor_branch_relative_scores
        ),
        "critic_branch_relative_shares": _normalized_shares(
            critic_branch_relative_scores
        ),
        "proprio_first_layer_scores": proprio_first_layer_scores,
        "proprio_first_layer_shares": _normalized_shares(proprio_first_layer_scores),
        "proprio_first_layer_relative_scores": proprio_first_layer_relative_scores,
        "proprio_first_layer_relative_shares": _normalized_shares(
            proprio_first_layer_relative_scores
        ),
        "pointnet_summary": pointnet_summary,
        "proprio_summary": proprio_summary,
        "pointcloud_feature_dim": pointcloud_feature_dim,
        "proprio_feature_dim": proprio_feature_dim,
    }


def _format_score_lines(title: str, scores: Mapping[str, float]) -> List[str]:
    lines = [title]
    for key, value in scores.items():
        lines.append(f"  - {key}: {value:.2f}%")
    return lines


def _build_report(
    checkpoint_path: Path,
    output_dir: Path,
    layout: Mapping[str, object],
    usage: Mapping[str, object],
) -> str:
    raw_dims = layout["raw_dims"]
    lines = [
        "DexPoint Observation Usage Analysis",
        "",
        f"Checkpoint: {checkpoint_path}",
        f"Output directory: {output_dir}",
        "",
        "Method:",
        "  - Loads the trained PPO policy and inspects trained weights only.",
        "  - Pointcloud usage is measured at the learned pointcloud feature branch.",
        "  - joint_state / ee_position / goal_position usage is traced through the proprio MLP into actor and critic readouts.",
        "  - Absolute shares are based on effective linear weight mass, so they are an approximation of reliance, not a causal attribution.",
        "  - Relative importance shares divide each group's score by its analysis dimensionality before normalizing across groups.",
        "  - For pointcloud this uses learned pointcloud feature width, because the point encoder is nonlinear and the script does not trace attribution back to raw points.",
        "",
        "Observation layout:",
        f"  - pointcloud: shape={layout['pointcloud_shape']} raw_dims={raw_dims['pointcloud']}",
        f"  - joint_state: dims={raw_dims['joint_state']}",
        f"  - ee_position: dims={raw_dims['ee_position']}",
        f"  - goal_position: dims={raw_dims['goal_position']}",
        f"  - pointcloud feature dim: {layout['pointcloud_feature_dim']}",
        f"  - proprio feature dim: {layout['proprio_feature_dim']}",
        "  - size-normalized group dims: "
        + ", ".join(
            f"{key}={value}" for key, value in layout["group_analysis_dims"].items()
        ),
        "",
    ]

    lines.extend(
        _format_score_lines(
            "Approximate actor usage shares:", usage["actor_group_shares"]
        )
    )
    lines.append("")
    lines.extend(
        _format_score_lines(
            "Approximate critic usage shares:", usage["critic_group_shares"]
        )
    )
    lines.append("")
    lines.extend(
        _format_score_lines(
            "Average actor/critic usage shares:", usage["combined_group_shares"]
        )
    )
    lines.append("")
    lines.extend(
        _format_score_lines(
            "Size-normalized actor relative importance shares:",
            usage["actor_group_relative_shares"],
        )
    )
    lines.append("")
    lines.extend(
        _format_score_lines(
            "Size-normalized critic relative importance shares:",
            usage["critic_group_relative_shares"],
        )
    )
    lines.append("")
    lines.extend(
        _format_score_lines(
            "Size-normalized average actor/critic relative importance shares:",
            usage["combined_group_relative_shares"],
        )
    )
    lines.append("")
    lines.extend(
        _format_score_lines(
            "Actor branch shares at fused feature interface:",
            usage["actor_branch_shares"],
        )
    )
    lines.append("")
    lines.extend(
        _format_score_lines(
            "Critic branch shares at fused feature interface:",
            usage["critic_branch_shares"],
        )
    )
    lines.append("")
    lines.extend(
        _format_score_lines(
            "Size-normalized fused actor branch relative importance:",
            usage["actor_branch_relative_shares"],
        )
    )
    lines.append("")
    lines.extend(
        _format_score_lines(
            "Size-normalized fused critic branch relative importance:",
            usage["critic_branch_relative_shares"],
        )
    )
    lines.append("")
    lines.extend(
        _format_score_lines(
            "Proprio first-layer input shares:", usage["proprio_first_layer_shares"]
        )
    )
    lines.append("")
    lines.extend(
        _format_score_lines(
            "Size-normalized proprio first-layer relative importance:",
            usage["proprio_first_layer_relative_shares"],
        )
    )
    lines.append("")
    lines.append("Encoder parameter summaries:")
    lines.append(
        "  - pointnet encoder: "
        f"params={usage['pointnet_summary']['parameter_count']} "
        f"l1={usage['pointnet_summary']['l1_sum']:.2f} "
        f"l2={usage['pointnet_summary']['l2_norm']:.2f}"
    )
    lines.append(
        "  - proprio encoder: "
        f"params={usage['proprio_summary']['parameter_count']} "
        f"l1={usage['proprio_summary']['l1_sum']:.2f} "
        f"l2={usage['proprio_summary']['l2_norm']:.2f}"
    )
    lines.append("")
    lines.append("Files:")
    lines.append("  - observation_usage_summary.json")
    lines.append("  - observation_usage_report.txt")
    lines.append("  - actor_usage.png")
    lines.append("  - critic_usage.png")
    lines.append("  - combined_usage.png")
    lines.append("  - actor_relative_usage.png")
    lines.append("  - critic_relative_usage.png")
    lines.append("  - combined_relative_usage.png")
    lines.append("  - proprio_first_layer.png")
    lines.append("  - proprio_first_layer_relative.png")
    lines.append("  - branch_usage.png")
    lines.append("  - branch_relative_usage.png")
    return "\n".join(lines)


def _draw_bar_chart(
    title: str,
    scores: Mapping[str, float],
    output_path: Path,
    subtitle: Optional[str] = None,
) -> None:
    width = 1100
    top_margin = 110
    row_height = 92
    left_margin = 270
    right_margin = 120
    bar_height = 42
    bottom_margin = 70
    height = top_margin + row_height * len(scores) + bottom_margin

    image = Image.new("RGB", (width, height), color=(248, 245, 238))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    title_color = (32, 37, 43)
    text_color = (55, 60, 68)
    axis_color = (190, 183, 171)
    bar_color = (44, 110, 158)
    accent_color = (201, 115, 57)

    draw.text((50, 28), title, fill=title_color, font=font)
    if subtitle:
        draw.text((50, 55), subtitle, fill=text_color, font=font)

    bar_start_x = left_margin
    bar_end_x = width - right_margin
    bar_width = bar_end_x - bar_start_x

    for tick_index in range(6):
        tick_share = tick_index / 5.0
        x = int(bar_start_x + tick_share * bar_width)
        draw.line(
            [(x, top_margin - 12), (x, height - bottom_margin + 18)],
            fill=axis_color,
            width=1,
        )
        draw.text(
            (x - 10, height - bottom_margin + 24),
            f"{int(100 * tick_share)}",
            fill=text_color,
            font=font,
        )

    for index, (label, value) in enumerate(scores.items()):
        y = top_margin + index * row_height
        normalized = max(0.0, min(100.0, float(value)))
        fill_width = int(bar_width * (normalized / 100.0))
        bar_top = y + 18
        bar_bottom = bar_top + bar_height

        draw.text((50, y + 12), label, fill=text_color, font=font)
        draw.rounded_rectangle(
            [(bar_start_x, bar_top), (bar_end_x, bar_bottom)],
            radius=8,
            fill=(231, 225, 214),
        )
        if fill_width > 0:
            draw.rounded_rectangle(
                [(bar_start_x, bar_top), (bar_start_x + fill_width, bar_bottom)],
                radius=8,
                fill=bar_color if index % 2 == 0 else accent_color,
            )
        draw.text(
            (bar_end_x + 16, y + 28), f"{normalized:.2f}%", fill=title_color, font=font
        )

    image.save(output_path)


def _write_outputs(
    checkpoint_path: Path,
    output_dir: Path,
    layout: Mapping[str, object],
    usage: Mapping[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "checkpoint_path": str(checkpoint_path),
        "method": {
            "type": "static_weight_analysis",
            "grouping": OBSERVATION_GROUP_KEYS,
            "notes": [
                "pointcloud scores reflect downstream reliance on the learned pointcloud feature branch",
                "joint_state, ee_position, and goal_position are traced through the proprio MLP into actor and critic readouts",
                "shares are normalized absolute effective weight magnitudes",
            ],
        },
        "observation_layout": {
            "pointcloud_shape": list(layout["pointcloud_shape"]),
            "raw_dims": dict(layout["raw_dims"]),
            "pointcloud_feature_dim": int(layout["pointcloud_feature_dim"]),
            "proprio_feature_dim": int(layout["proprio_feature_dim"]),
            "proprio_input_dim": int(layout["proprio_input_dim"]),
            "group_analysis_dims": dict(layout["group_analysis_dims"]),
            "branch_analysis_dims": dict(layout["branch_analysis_dims"]),
        },
        "usage": json.loads(json.dumps(usage)),
    }

    report_text = _build_report(checkpoint_path, output_dir, layout, usage)

    (output_dir / "observation_usage_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    (output_dir / "observation_usage_report.txt").write_text(
        report_text + "\n", encoding="utf-8"
    )

    _draw_bar_chart(
        title="Actor Observation Usage",
        scores=usage["actor_group_shares"],
        output_path=output_dir / "actor_usage.png",
        subtitle="Approximate normalized weight-based reliance by observation group",
    )
    _draw_bar_chart(
        title="Critic Observation Usage",
        scores=usage["critic_group_shares"],
        output_path=output_dir / "critic_usage.png",
        subtitle="Approximate normalized weight-based reliance by observation group",
    )
    _draw_bar_chart(
        title="Average Actor/Critic Usage",
        scores=usage["combined_group_shares"],
        output_path=output_dir / "combined_usage.png",
        subtitle="Mean of normalized actor and critic shares",
    )
    _draw_bar_chart(
        title="Actor Relative Importance",
        scores=usage["actor_group_relative_shares"],
        output_path=output_dir / "actor_relative_usage.png",
        subtitle="Per-dimension normalized actor importance by observation group",
    )
    _draw_bar_chart(
        title="Critic Relative Importance",
        scores=usage["critic_group_relative_shares"],
        output_path=output_dir / "critic_relative_usage.png",
        subtitle="Per-dimension normalized critic importance by observation group",
    )
    _draw_bar_chart(
        title="Average Actor/Critic Relative Importance",
        scores=usage["combined_group_relative_shares"],
        output_path=output_dir / "combined_relative_usage.png",
        subtitle="Mean of size-normalized actor and critic shares",
    )
    _draw_bar_chart(
        title="Proprio First-Layer Input Usage",
        scores=usage["proprio_first_layer_shares"],
        output_path=output_dir / "proprio_first_layer.png",
        subtitle="How the first proprio MLP layer allocates weight across proprio inputs",
    )
    _draw_bar_chart(
        title="Proprio First-Layer Relative Importance",
        scores=usage["proprio_first_layer_relative_shares"],
        output_path=output_dir / "proprio_first_layer_relative.png",
        subtitle="First proprio MLP layer importance normalized by input-group size",
    )
    branch_average = OrderedDict(
        pointcloud=0.5
        * (
            usage["actor_branch_shares"]["pointcloud"]
            + usage["critic_branch_shares"]["pointcloud"]
        ),
        proprio=0.5
        * (
            usage["actor_branch_shares"]["proprio"]
            + usage["critic_branch_shares"]["proprio"]
        ),
    )
    branch_relative_average = OrderedDict(
        pointcloud=0.5
        * (
            usage["actor_branch_relative_shares"]["pointcloud"]
            + usage["critic_branch_relative_shares"]["pointcloud"]
        ),
        proprio=0.5
        * (
            usage["actor_branch_relative_shares"]["proprio"]
            + usage["critic_branch_relative_shares"]["proprio"]
        ),
    )
    _draw_bar_chart(
        title="Fused Branch Usage",
        scores=branch_average,
        output_path=output_dir / "branch_usage.png",
        subtitle="Average actor/critic reliance on pointcloud vs proprio learned features",
    )
    _draw_bar_chart(
        title="Fused Branch Relative Importance",
        scores=branch_relative_average,
        output_path=output_dir / "branch_relative_usage.png",
        subtitle="Average actor/critic branch importance normalized by feature width",
    )


def _checkpoint_custom_objects(checkpoint_path: Path, device: str) -> Dict[str, object]:
    data, _, _ = load_from_zip_file(str(checkpoint_path), device=device)
    policy_kwargs = dict(data.get("policy_kwargs", {}))
    if "pointnet_checkpoint_path" in policy_kwargs:
        policy_kwargs["pointnet_checkpoint_path"] = None
    return {"policy_kwargs": policy_kwargs}


def analyze_checkpoint(
    checkpoint_path: Path, output_dir: Path, device: str
) -> Tuple[Mapping[str, object], Mapping[str, object]]:
    model = PPO.load(
        str(checkpoint_path),
        env=None,
        device=device,
        custom_objects=_checkpoint_custom_objects(checkpoint_path, device),
    )
    model.policy.eval()

    layout = _make_observation_layout(model.policy)
    usage = _compute_policy_usage(model.policy, layout)
    _write_outputs(checkpoint_path, output_dir, layout, usage)
    return layout, usage


def _extract_checkpoint_step(checkpoint_path: Path) -> int:
    match = re.search(r"(\d+)$", checkpoint_path.stem)
    if match is None:
        raise ValueError(
            f"Could not extract training step from checkpoint name: {checkpoint_path.name}"
        )
    return int(match.group(1))


def _find_checkpoints(input_dir: Path) -> List[Path]:
    checkpoints = sorted(
        input_dir.glob("model_checkpoint_*.zip"),
        key=_extract_checkpoint_step,
    )
    if not checkpoints:
        raise FileNotFoundError(f"No model_checkpoint_*.zip files found in {input_dir}")
    return checkpoints


def _format_step(step: int) -> str:
    return f"{step:,}"


def _series_min_max(series_values: Sequence[float]) -> Tuple[float, float]:
    if not series_values:
        return 0.0, 1.0
    minimum = min(series_values)
    maximum = max(series_values)
    if abs(maximum - minimum) < 1e-9:
        padding = 1.0 if maximum == 0.0 else 0.05 * abs(maximum)
        return minimum - padding, maximum + padding
    padding = 0.08 * (maximum - minimum)
    return minimum - padding, maximum + padding


def _draw_trend_chart(
    title: str,
    steps: Sequence[int],
    series_by_label: Mapping[str, Sequence[float]],
    output_path: Path,
    subtitle: Optional[str] = None,
    y_label: str = "share (%)",
) -> None:
    width = 1300
    height = 760
    left_margin = 130
    right_margin = 180
    top_margin = 110
    bottom_margin = 110
    chart_left = left_margin
    chart_right = width - right_margin
    chart_top = top_margin
    chart_bottom = height - bottom_margin
    chart_width = chart_right - chart_left
    chart_height = chart_bottom - chart_top

    image = Image.new("RGB", (width, height), color=(248, 245, 238))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    title_color = (32, 37, 43)
    text_color = (55, 60, 68)
    grid_color = (205, 197, 184)
    axis_color = (120, 116, 108)
    palette = [
        (44, 110, 158),
        (201, 115, 57),
        (74, 133, 92),
        (154, 98, 151),
        (87, 87, 167),
        (186, 88, 106),
    ]

    draw.text((50, 28), title, fill=title_color, font=font)
    if subtitle:
        draw.text((50, 55), subtitle, fill=text_color, font=font)

    all_values: List[float] = []
    for values in series_by_label.values():
        all_values.extend(float(value) for value in values)
    y_min, y_max = _series_min_max(all_values)

    for tick_index in range(6):
        tick_ratio = tick_index / 5.0
        y_value = y_max - tick_ratio * (y_max - y_min)
        y = int(chart_top + tick_ratio * chart_height)
        draw.line([(chart_left, y), (chart_right, y)], fill=grid_color, width=1)
        draw.text((40, y - 6), f"{y_value:.1f}", fill=text_color, font=font)

    if len(steps) == 1:
        x_positions = [chart_left + chart_width // 2]
    else:
        x_positions = [
            int(chart_left + index * chart_width / (len(steps) - 1))
            for index in range(len(steps))
        ]

    for x, step in zip(x_positions, steps):
        draw.line([(x, chart_top), (x, chart_bottom)], fill=grid_color, width=1)
        draw.text(
            (x - 18, chart_bottom + 18), _format_step(step), fill=text_color, font=font
        )

    draw.line(
        [(chart_left, chart_bottom), (chart_right, chart_bottom)],
        fill=axis_color,
        width=2,
    )
    draw.line(
        [(chart_left, chart_top), (chart_left, chart_bottom)], fill=axis_color, width=2
    )
    draw.text((50, chart_top - 22), y_label, fill=text_color, font=font)
    draw.text((chart_right - 45, chart_bottom + 45), "step", fill=text_color, font=font)

    legend_x = chart_right + 30
    legend_y = chart_top + 10
    for color_index, (label, values) in enumerate(series_by_label.items()):
        color = palette[color_index % len(palette)]
        points = []
        for x, value in zip(x_positions, values):
            y_ratio = (
                0.5 if y_max == y_min else (float(value) - y_min) / (y_max - y_min)
            )
            y = int(chart_bottom - y_ratio * chart_height)
            points.append((x, y))

        if len(points) >= 2:
            draw.line(points, fill=color, width=4)
        for point_x, point_y in points:
            radius = 6
            draw.ellipse(
                [
                    (point_x - radius, point_y - radius),
                    (point_x + radius, point_y + radius),
                ],
                fill=color,
                outline=(248, 245, 238),
                width=1,
            )

        legend_row_y = legend_y + color_index * 28
        draw.line(
            [(legend_x, legend_row_y + 7), (legend_x + 22, legend_row_y + 7)],
            fill=color,
            width=4,
        )
        draw.text((legend_x + 30, legend_row_y), label, fill=text_color, font=font)

    image.save(output_path)


def _checkpoint_table_lines(results: Sequence[Mapping[str, object]]) -> List[str]:
    lines = []
    header = "step        pointcloud  joint_state  ee_position  goal_position"
    lines.append(header)
    lines.append("-" * len(header))
    for result in results:
        shares = result["usage"]["combined_group_shares"]
        lines.append(
            f"{_format_step(result['step']):>10}  "
            f"{shares['pointcloud']:10.2f}  "
            f"{shares['joint_state']:11.2f}  "
            f"{shares['ee_position']:11.2f}  "
            f"{shares['goal_position']:13.2f}"
        )
    return lines


def _trend_delta_lines(results: Sequence[Mapping[str, object]]) -> List[str]:
    if len(results) < 2:
        return ["Only one checkpoint found, so no trend delta is available."]

    first_shares = results[0]["usage"]["combined_group_shares"]
    last_shares = results[-1]["usage"]["combined_group_shares"]
    lines = []
    for key in OBSERVATION_GROUP_KEYS:
        delta = float(last_shares[key]) - float(first_shares[key])
        lines.append(f"  - {key}: {delta:+.2f} percentage points")
    return lines


def _write_run_outputs(
    input_dir: Path,
    output_dir: Path,
    results: Sequence[Mapping[str, object]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    steps = [int(result["step"]) for result in results]
    combined_series = OrderedDict(
        (
            key,
            [
                float(result["usage"]["combined_group_shares"][key])
                for result in results
            ],
        )
        for key in OBSERVATION_GROUP_KEYS
    )
    actor_series = OrderedDict(
        (key, [float(result["usage"]["actor_group_shares"][key]) for result in results])
        for key in OBSERVATION_GROUP_KEYS
    )
    critic_series = OrderedDict(
        (
            key,
            [float(result["usage"]["critic_group_shares"][key]) for result in results],
        )
        for key in OBSERVATION_GROUP_KEYS
    )
    branch_series = OrderedDict(
        (
            key,
            [
                0.5
                * (
                    float(result["usage"]["actor_branch_shares"][key])
                    + float(result["usage"]["critic_branch_shares"][key])
                )
                for result in results
            ],
        )
        for key in BRANCH_KEYS
    )
    combined_relative_series = OrderedDict(
        (
            key,
            [
                float(result["usage"]["combined_group_relative_shares"][key])
                for result in results
            ],
        )
        for key in OBSERVATION_GROUP_KEYS
    )
    actor_relative_series = OrderedDict(
        (
            key,
            [
                float(result["usage"]["actor_group_relative_shares"][key])
                for result in results
            ],
        )
        for key in OBSERVATION_GROUP_KEYS
    )
    critic_relative_series = OrderedDict(
        (
            key,
            [
                float(result["usage"]["critic_group_relative_shares"][key])
                for result in results
            ],
        )
        for key in OBSERVATION_GROUP_KEYS
    )
    branch_relative_series = OrderedDict(
        (
            key,
            [
                0.5
                * (
                    float(result["usage"]["actor_branch_relative_shares"][key])
                    + float(result["usage"]["critic_branch_relative_shares"][key])
                )
                for result in results
            ],
        )
        for key in BRANCH_KEYS
    )

    summary = {
        "input_dir": str(input_dir),
        "checkpoint_count": len(results),
        "checkpoints": [
            {
                "step": int(result["step"]),
                "checkpoint_path": str(result["checkpoint_path"]),
                "analysis_dir": str(result["analysis_dir"]),
                "combined_group_shares": result["usage"]["combined_group_shares"],
                "combined_group_relative_shares": result["usage"][
                    "combined_group_relative_shares"
                ],
                "actor_group_shares": result["usage"]["actor_group_shares"],
                "critic_group_shares": result["usage"]["critic_group_shares"],
                "actor_group_relative_shares": result["usage"][
                    "actor_group_relative_shares"
                ],
                "critic_group_relative_shares": result["usage"][
                    "critic_group_relative_shares"
                ],
            }
            for result in results
        ],
        "trend_series": {
            "steps": steps,
            "combined_group_shares": combined_series,
            "combined_group_relative_shares": combined_relative_series,
            "actor_group_shares": actor_series,
            "actor_group_relative_shares": actor_relative_series,
            "critic_group_shares": critic_series,
            "critic_group_relative_shares": critic_relative_series,
            "branch_average_shares": branch_series,
            "branch_average_relative_shares": branch_relative_series,
        },
    }

    report_lines = [
        "DexPoint Observation Usage Trends",
        "",
        f"Run directory: {input_dir}",
        f"Output directory: {output_dir}",
        f"Checkpoint count: {len(results)}",
        "",
        "Method:",
        "  - Each checkpoint is analyzed with the same static weight-based observation usage method as single-checkpoint mode.",
        "  - Trend plots show how normalized observation-group reliance changes over training steps.",
        "",
        "Combined usage by checkpoint:",
    ]
    report_lines.extend(_checkpoint_table_lines(results))
    report_lines.append("")
    report_lines.append("Change from first to last checkpoint:")
    report_lines.extend(_trend_delta_lines(results))
    report_lines.append("")
    report_lines.append("Files:")
    report_lines.append("  - observation_usage_trends.json")
    report_lines.append("  - observation_usage_trends_report.txt")
    report_lines.append("  - combined_usage_trend.png")
    report_lines.append("  - actor_usage_trend.png")
    report_lines.append("  - critic_usage_trend.png")
    report_lines.append("  - branch_usage_trend.png")
    report_lines.append("  - combined_relative_usage_trend.png")
    report_lines.append("  - actor_relative_usage_trend.png")
    report_lines.append("  - critic_relative_usage_trend.png")
    report_lines.append("  - branch_relative_usage_trend.png")

    (output_dir / "observation_usage_trends.json").write_text(
        json.dumps(json.loads(json.dumps(summary)), indent=2),
        encoding="utf-8",
    )
    (output_dir / "observation_usage_trends_report.txt").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    _draw_trend_chart(
        title="Average Actor/Critic Observation Usage Over Training",
        steps=steps,
        series_by_label=combined_series,
        output_path=output_dir / "combined_usage_trend.png",
        subtitle="Normalized combined observation-group shares for each checkpoint",
    )
    _draw_trend_chart(
        title="Actor Observation Usage Over Training",
        steps=steps,
        series_by_label=actor_series,
        output_path=output_dir / "actor_usage_trend.png",
        subtitle="Normalized actor observation-group shares for each checkpoint",
    )
    _draw_trend_chart(
        title="Critic Observation Usage Over Training",
        steps=steps,
        series_by_label=critic_series,
        output_path=output_dir / "critic_usage_trend.png",
        subtitle="Normalized critic observation-group shares for each checkpoint",
    )
    _draw_trend_chart(
        title="Fused Branch Usage Over Training",
        steps=steps,
        series_by_label=branch_series,
        output_path=output_dir / "branch_usage_trend.png",
        subtitle="Average actor/critic reliance on pointcloud and proprio learned features",
    )
    _draw_trend_chart(
        title="Average Actor/Critic Relative Importance Over Training",
        steps=steps,
        series_by_label=combined_relative_series,
        output_path=output_dir / "combined_relative_usage_trend.png",
        subtitle="Size-normalized combined observation-group shares for each checkpoint",
    )
    _draw_trend_chart(
        title="Actor Relative Importance Over Training",
        steps=steps,
        series_by_label=actor_relative_series,
        output_path=output_dir / "actor_relative_usage_trend.png",
        subtitle="Size-normalized actor observation-group shares for each checkpoint",
    )
    _draw_trend_chart(
        title="Critic Relative Importance Over Training",
        steps=steps,
        series_by_label=critic_relative_series,
        output_path=output_dir / "critic_relative_usage_trend.png",
        subtitle="Size-normalized critic observation-group shares for each checkpoint",
    )
    _draw_trend_chart(
        title="Fused Branch Relative Importance Over Training",
        steps=steps,
        series_by_label=branch_relative_series,
        output_path=output_dir / "branch_relative_usage_trend.png",
        subtitle="Average actor/critic branch importance normalized by feature width",
    )


def analyze_run_directory(
    input_dir: Path, output_dir: Path, device: str
) -> List[Mapping[str, object]]:
    checkpoints = _find_checkpoints(input_dir)
    results = []
    for checkpoint_path in checkpoints:
        checkpoint_output_dir = output_dir / checkpoint_path.stem
        layout, usage = analyze_checkpoint(
            checkpoint_path, checkpoint_output_dir, device
        )
        results.append(
            {
                "step": _extract_checkpoint_step(checkpoint_path),
                "checkpoint_path": checkpoint_path,
                "analysis_dir": checkpoint_output_dir,
                "layout": layout,
                "usage": usage,
            }
        )

    _write_run_outputs(input_dir, output_dir, results)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to a PPO checkpoint .zip file or a run directory containing model_checkpoint_*.zip files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for report, JSON summary, and PNG visualizations",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device for loading the checkpoint (default: cpu)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.input_path.expanduser().absolute()
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if args.output_dir is None:
        if input_path.is_dir():
            output_dir = input_path / "observation_usage_trends"
        else:
            output_dir = input_path.parent / f"{input_path.stem}_observation_usage"
    else:
        output_dir = args.output_dir.expanduser().absolute()

    if input_path.is_dir():
        results = analyze_run_directory(input_path, output_dir, args.device)
        print(f"Run directory: {input_path}")
        print(f"Output directory: {output_dir}")
        print(f"Checkpoints analyzed: {len(results)}")
        print()
        print("Average actor/critic usage by checkpoint:")
        for result in results:
            shares = result["usage"]["combined_group_shares"]
            print(
                f"  step {_format_step(result['step']):>10}: "
                f"pointcloud={shares['pointcloud']:.2f}% "
                f"joint_state={shares['joint_state']:.2f}% "
                f"ee_position={shares['ee_position']:.2f}% "
                f"goal_position={shares['goal_position']:.2f}%"
            )
    else:
        layout, usage = analyze_checkpoint(input_path, output_dir, args.device)

        print(f"Checkpoint: {input_path}")
        print(f"Output directory: {output_dir}")
        print()
        print("Average actor/critic usage shares:")
        for key, value in usage["combined_group_shares"].items():
            print(f"  {key:>14}: {value:6.2f}%")
        print()
        print("Actor usage shares:")
        for key, value in usage["actor_group_shares"].items():
            print(f"  {key:>14}: {value:6.2f}%")
        print()
        print("Critic usage shares:")
        for key, value in usage["critic_group_shares"].items():
            print(f"  {key:>14}: {value:6.2f}%")
        print()
        print("Observation layout:")
        print(f"  pointcloud shape: {layout['pointcloud_shape']}")
        for key, value in layout["raw_dims"].items():
            print(f"  {key:>14}: {value}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
