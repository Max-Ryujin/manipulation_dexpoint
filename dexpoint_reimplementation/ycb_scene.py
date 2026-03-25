"""Helpers for building single-object YCB MuJoCo scenes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Optional, Union
import xml.etree.ElementTree as ET

import numpy as np


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dexart_baselines.pretrain.ycb.rl_pretrain_support import (
    clean_points,
    find_source_mesh,
    find_source_ply,
    load_ply_vertices,
)

_ASSET_ROOT = (
    _REPO_ROOT / "manipulation" / "environments" / "assets" / "franka_emika_panda"
)
_GENERATED_SCENE_DIR = _ASSET_ROOT
_DEFAULT_SCENE_TEMPLATE = _ASSET_ROOT / "scene_005_tomato_soup_can.xml"

_YCB_COLLISION_CONTYPE = "2"
_YCB_COLLISION_CONAFFINITY = "1"
_YCB_COLLISION_CONDIM = "3"
_YCB_COLLISION_SOLIMP = "0.99 0.995 0.01"
_YCB_COLLISION_SOLREF = "0.01 1"
_YCB_COLLISION_FRICTION = "1 0.005 0.0001"

DEFAULT_YCB_OBJECT_ROOT = (
    _REPO_ROOT
    / "dexart_baselines"
    / "pretrain"
    / "data"
    / "ycb_raw"
    / "005_tomato_soup_can"
)


@dataclass(frozen=True)
class YCBObjectSpec:
    """Geometry and placement metadata for a single YCB object."""

    name: str
    object_root: Path
    mesh_path: Path
    pointcloud_path: Path
    mesh_offset: np.ndarray
    half_extents: np.ndarray
    placement_radius: float
    body_name: str = "target_object"


def _stable_box_inertia(half_extents: np.ndarray, mass: float) -> np.ndarray:
    full_extents = 2.0 * half_extents
    return (mass / 12.0) * np.array(
        [
            full_extents[1] ** 2 + full_extents[2] ** 2,
            full_extents[0] ** 2 + full_extents[2] ** 2,
            full_extents[0] ** 2 + full_extents[1] ** 2,
        ],
        dtype=np.float64,
    )


def _format_vector(values: np.ndarray, precision: int = 6) -> str:
    return " ".join(f"{float(value):.{precision}f}" for value in values)


def _require_element(root: ET.Element, query: str) -> ET.Element:
    element = root.find(query)
    if element is None:
        raise ValueError(f"Scene template is missing required element: {query}")
    return element


def _find_dynamic_material(root: ET.Element) -> ET.Element:
    asset = _require_element(root, ".//asset")
    for material in asset.findall("material"):
        if material.get("name") != "groundplane":
            return material
    raise ValueError("Scene template is missing the object material entry")


def load_ycb_object_spec(object_root: Union[str, Path]) -> YCBObjectSpec:
    """Load mesh and geometric bounds for a YCB object directory."""
    resolved_root = Path(object_root).resolve()
    mesh_path = find_source_mesh(resolved_root)
    pointcloud_path = find_source_ply(resolved_root)
    if mesh_path is None or pointcloud_path is None:
        raise FileNotFoundError(
            f"Could not find mesh and point cloud data under {resolved_root}"
        )

    points = clean_points(load_ply_vertices(pointcloud_path))
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    half_extents = 0.5 * (maxs - mins)
    if np.any(half_extents <= 0):
        raise ValueError(f"Degenerate YCB object bounds for {resolved_root}")

    center = 0.5 * (mins + maxs)
    return YCBObjectSpec(
        name=resolved_root.name,
        object_root=resolved_root,
        mesh_path=mesh_path.resolve(),
        pointcloud_path=pointcloud_path.resolve(),
        mesh_offset=-center.astype(np.float32),
        half_extents=half_extents.astype(np.float32),
        placement_radius=float(np.linalg.norm(half_extents[:2])),
    )


def create_single_object_ycb_scene(
    object_spec: YCBObjectSpec,
    scene_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Create or update a MuJoCo scene from the checked-in XML template."""
    output_dir = _GENERATED_SCENE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if scene_path is None:
        scene_path = output_dir / f"scene_{object_spec.name}.xml"
    else:
        scene_path = Path(scene_path)
        scene_path.parent.mkdir(parents=True, exist_ok=True)

    template_path = scene_path if scene_path.exists() else _DEFAULT_SCENE_TEMPLATE
    tree = ET.parse(template_path)
    root = tree.getroot()

    mesh_name = f"mesh_{object_spec.name}"
    material_name = f"material_{object_spec.name}"
    inertia = _stable_box_inertia(object_spec.half_extents, mass=0.15)

    mesh = _require_element(root, ".//asset/mesh")
    material = _find_dynamic_material(root)
    target_body = _require_element(root, ".//worldbody/body[@name='target_object']")
    target_joint = _require_element(
        root, ".//worldbody/body[@name='target_object']/joint"
    )
    inertial = _require_element(
        root, ".//worldbody/body[@name='target_object']/inertial"
    )
    visual_geom = _require_element(
        root, ".//worldbody/body[@name='target_object']/geom[@type='mesh']"
    )
    collision_geom = _require_element(
        root, ".//worldbody/body[@name='target_object']/geom[@type='box']"
    )

    mesh.set("name", mesh_name)
    mesh.set("file", object_spec.mesh_path.as_posix())

    material.set("name", material_name)
    material.set("rgba", "0.82 0.28 0.24 1")

    target_body.set("name", object_spec.body_name)
    target_body.set("pos", f"100 0 {float(object_spec.half_extents[2]):.6f}")

    target_joint.set("name", f"{object_spec.body_name}_freejoint")

    inertial.set("mass", "0.150000")
    inertial.set("pos", "0 0 0")
    inertial.set("diaginertia", _format_vector(inertia, precision=8))

    visual_geom.set("name", f"{object_spec.body_name}_visual")
    visual_geom.set("mesh", mesh_name)
    visual_geom.set("pos", _format_vector(object_spec.mesh_offset))
    visual_geom.set("material", material_name)
    visual_geom.set("contype", "0")
    visual_geom.set("conaffinity", "0")

    collision_geom.set("name", f"{object_spec.body_name}_collision")
    collision_geom.set("size", _format_vector(object_spec.half_extents))
    collision_geom.set("pos", "0 0 0")
    collision_geom.set("rgba", "0 0 0 0")
    collision_geom.set("contype", _YCB_COLLISION_CONTYPE)
    collision_geom.set("conaffinity", _YCB_COLLISION_CONAFFINITY)
    collision_geom.set("condim", _YCB_COLLISION_CONDIM)
    collision_geom.set("solimp", _YCB_COLLISION_SOLIMP)
    collision_geom.set("solref", _YCB_COLLISION_SOLREF)
    collision_geom.set("friction", _YCB_COLLISION_FRICTION)

    if hasattr(ET, "indent"):
        ET.indent(tree, space="  ")
    tree.write(scene_path, encoding="utf-8", xml_declaration=False)
    return scene_path


def ensure_single_object_ycb_scene(
    object_root: Union[str, Path] = DEFAULT_YCB_OBJECT_ROOT,
    scene_path: Optional[Union[str, Path]] = None,
) -> tuple[Path, YCBObjectSpec]:
    """Create or refresh the single-object scene for the requested YCB object."""
    object_spec = load_ycb_object_spec(object_root)
    xml_path = create_single_object_ycb_scene(object_spec, scene_path=scene_path)
    return xml_path, object_spec
