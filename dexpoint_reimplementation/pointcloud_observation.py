"""Utilities for RL-style point cloud observations."""

from __future__ import annotations

import time
from typing import Iterable, Optional, Sequence

import mujoco
import numpy as np


DEFAULT_CAMERA_NAMES = ("top_camera", "side_camera", "front_camera")
POINTCLOUD_OVERSAMPLE_FACTOR = 40


def get_default_camera_names() -> list[str]:
    return list(DEFAULT_CAMERA_NAMES)


def get_workspace_configuration(
    model,
    working_area: tuple[float, float] = (0.45, 0.45),
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    table_body_name: str = "simple_table",
    table_geom_name: str = "table_surface",
) -> tuple[dict[str, float], dict[str, object]]:
    temp_data = mujoco.MjData(model)
    mujoco.mj_forward(model, temp_data)

    table_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, table_geom_name)
    if table_geom_id < 0:
        raise ValueError(f"Could not find table geom '{table_geom_name}' in model")

    geom_xpos = temp_data.geom_xpos[table_geom_id]
    geom_size = model.geom_size[table_geom_id]

    center_x = float(geom_xpos[0] + offset_x)
    center_y = float(geom_xpos[1] - geom_size[1] + working_area[1] / 2.0 + offset_y)
    table_height = float(geom_xpos[2] + geom_size[2])

    half_width_x = working_area[0] / 2.0
    half_width_y = working_area[1] / 2.0

    bounds = {
        "min_x": center_x - half_width_x,
        "max_x": center_x + half_width_x,
        "min_y": center_y - half_width_y,
        "max_y": center_y + half_width_y,
        "table_height": table_height,
    }
    info = {
        "type": "tabletop",
        "working_area": working_area,
        "offset_x": offset_x,
        "offset_y": offset_y,
        "table_body_name": table_body_name,
        "table_geom_name": table_geom_name,
        "table_bounds": bounds,
        "table_height": table_height,
    }
    return bounds, info


def collect_fused_pointcloud(
    env,
    camera,
    *,
    camera_names: Sequence[str],
    num_points: int,
    camera_height: int,
    camera_width: int,
    workspace_bounds: dict[str, float],
    table_height: float,
    hand_body_name: str = "hand",
    min_depth: float = 0.1,
    max_depth: float = 3.0,
    max_height_above_table: float = 0.2,
    min_height_above_table: float = 0.01,
    hand_sphere_radius: float = 0.12,
) -> np.ndarray:
    all_points: list[np.ndarray] = []
    samples_per_camera = POINTCLOUD_OVERSAMPLE_FACTOR * num_points

    min_x = workspace_bounds["min_x"]
    max_x = workspace_bounds["max_x"]
    min_y = workspace_bounds["min_y"]
    max_y = workspace_bounds["max_y"]
    min_z = table_height
    min_allowed_z = min_z + min_height_above_table
    max_allowed_z = min_z + max_height_above_table
    hand_sphere_radius_sq = hand_sphere_radius * hand_sphere_radius

    hand_id = env.model.body(hand_body_name).id
    hand_pos = env.data.xpos[hand_id]

    for camera_name in camera_names:
        points, _colors, _pixels, _depths = camera.get_pointcloud(
            camera_name,
            width=camera_width,
            height=camera_height,
            num_samples=samples_per_camera,
            min_depth=min_depth,
            max_depth=max_depth,
        )

        if len(points) == 0:
            continue

        workspace_mask = (
            (points[:, 0] >= min_x)
            & (points[:, 0] <= max_x)
            & (points[:, 1] >= min_y)
            & (points[:, 1] <= max_y)
            & (points[:, 2] > min_allowed_z)
            & (points[:, 2] < max_allowed_z)
        )

        hand_deltas = points - hand_pos
        hand_mask = np.einsum("ij,ij->i", hand_deltas, hand_deltas) < hand_sphere_radius_sq
        np.logical_or(workspace_mask, hand_mask, out=workspace_mask)
        filtered_points = points[workspace_mask]

        if len(filtered_points) > 0:
            all_points.append(filtered_points)

    if not all_points:
        return np.zeros((0, 3), dtype=np.float32)

    return np.concatenate(all_points, axis=0).astype(np.float32, copy=False)


def collect_fused_pointcloud_for_training(
    env,
    camera,
    *,
    camera_names: Sequence[str],
    num_points: int,
    camera_height: int,
    camera_width: int,
    workspace_bounds: dict[str, float],
    table_height: float,
    hand_body_name: str = "hand",
    min_depth: float = 0.1,
    max_depth: float = 3.0,
    max_height_above_table: float = 0.2,
    min_height_above_table: float = 0.01,
    hand_sphere_radius: float = 0.12,
    timing_stats: Optional[dict[str, float]] = None,
) -> np.ndarray:
    """Collect a fused point cloud for training using depth only."""
    collection_start = time.perf_counter()
    all_points: list[np.ndarray] = []
    samples_per_camera = POINTCLOUD_OVERSAMPLE_FACTOR * num_points

    min_x = workspace_bounds["min_x"]
    max_x = workspace_bounds["max_x"]
    min_y = workspace_bounds["min_y"]
    max_y = workspace_bounds["max_y"]
    min_z = table_height
    min_allowed_z = min_z + min_height_above_table
    max_allowed_z = min_z + max_height_above_table
    hand_sphere_radius_sq = hand_sphere_radius * hand_sphere_radius

    hand_id = env.model.body(hand_body_name).id
    hand_pos = env.data.xpos[hand_id]

    for camera_name in camera_names:
        camera_call_start = time.perf_counter()
        points, _pixels, _depths = camera.get_pointcloud_depth_only(
            camera_name,
            width=camera_width,
            height=camera_height,
            num_samples=samples_per_camera,
            min_depth=min_depth,
            max_depth=max_depth,
            timing_stats=timing_stats,
        )
        if timing_stats is not None:
            timing_stats["collect_camera_call_s"] = timing_stats.get(
                "collect_camera_call_s", 0.0
            ) + (time.perf_counter() - camera_call_start)

        if len(points) == 0:
            continue

        filter_start = time.perf_counter()
        workspace_mask = (
            (points[:, 0] >= min_x)
            & (points[:, 0] <= max_x)
            & (points[:, 1] >= min_y)
            & (points[:, 1] <= max_y)
            & (points[:, 2] > min_allowed_z)
            & (points[:, 2] < max_allowed_z)
        )

        hand_deltas = points - hand_pos
        hand_mask = np.einsum("ij,ij->i", hand_deltas, hand_deltas) < hand_sphere_radius_sq
        np.logical_or(workspace_mask, hand_mask, out=workspace_mask)
        filtered_points = points[workspace_mask]
        if timing_stats is not None:
            timing_stats["collect_world_filter_s"] = timing_stats.get(
                "collect_world_filter_s", 0.0
            ) + (time.perf_counter() - filter_start)
            timing_stats["collect_points_after_filter"] = timing_stats.get(
                "collect_points_after_filter", 0.0
            ) + float(len(filtered_points))

        if len(filtered_points) > 0:
            all_points.append(filtered_points)

    concat_start = time.perf_counter()
    if not all_points:
        if timing_stats is not None:
            timing_stats["collect_concat_s"] = timing_stats.get(
                "collect_concat_s", 0.0
            ) + (time.perf_counter() - concat_start)
            timing_stats["collect_total_s"] = timing_stats.get(
                "collect_total_s", 0.0
            ) + (time.perf_counter() - collection_start)
            timing_stats["collect_merged_points"] = timing_stats.get(
                "collect_merged_points", 0.0
            )
        return np.zeros((0, 3), dtype=np.float32)

    merged_points = np.concatenate(all_points, axis=0).astype(np.float32, copy=False)
    if timing_stats is not None:
        timing_stats["collect_concat_s"] = timing_stats.get("collect_concat_s", 0.0) + (
            time.perf_counter() - concat_start
        )
        timing_stats["collect_total_s"] = timing_stats.get("collect_total_s", 0.0) + (
            time.perf_counter() - collection_start
        )
        timing_stats["collect_merged_points"] = timing_stats.get(
            "collect_merged_points", 0.0
        ) + float(len(merged_points))
    return merged_points


def sample_pointcloud(
    merged_points: np.ndarray,
    num_points: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    if len(merged_points) == 0:
        return np.zeros((num_points, 3), dtype=np.float32)

    pointcloud = merged_points.astype(np.float32, copy=False)

    sampler = rng if rng is not None else np.random
    if len(pointcloud) > num_points:
        indices = sampler.choice(len(pointcloud), size=num_points, replace=False)
        return pointcloud[indices]

    if len(pointcloud) < num_points:
        pad_size = num_points - len(pointcloud)
        padding = np.zeros((pad_size, 3), dtype=np.float32)
        pointcloud = np.vstack([pointcloud, padding])

    return pointcloud.astype(np.float32, copy=False)


def center_and_sample_pointcloud(
    merged_points: np.ndarray,
    num_points: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    # just in case
    return sample_pointcloud(merged_points, num_points, rng=rng)
