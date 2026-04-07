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

_TEXTURED_MESH_CANDIDATES = (
    "poisson/textured.obj",
    "tsdf/textured.obj",
)

_ASSET_ROOT = (
    _REPO_ROOT / "manipulation" / "environments" / "assets" / "franka_emika_panda"
)
_GENERATED_SCENE_DIR = _ASSET_ROOT
_DEFAULT_SCENE_TEMPLATE = _ASSET_ROOT / "scene_005_tomato_soup_can.xml"
_YCB_SIM_ROOT = _REPO_ROOT / "YCB_sim"
_YCB_SIM_INCLUDE_DIR = _YCB_SIM_ROOT / "includes"

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

YCB_ASSET_SOURCE_RAW = "raw"
YCB_ASSET_SOURCE_YCB_SIM = "ycb_sim"


@dataclass(frozen=True)
class YCBObjectSpec:
    """Geometry and placement metadata for a single YCB object."""

    name: str
    object_root: Path
    mesh_path: Path
    texture_path: Optional[Path]
    pointcloud_path: Optional[Path]
    mesh_offset: np.ndarray
    half_extents: np.ndarray
    placement_radius: float
    collision_size: np.ndarray
    collision_pos: np.ndarray
    rest_offset_z: float
    scale: float = 1.0
    body_name: str = "target_object"
    mass: float = 0.15
    collision_geom_type: str = "box"
    source: str = YCB_ASSET_SOURCE_RAW


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


def _stable_cylinder_inertia(radius: float, half_height: float, mass: float) -> np.ndarray:
    full_height = 2.0 * half_height
    axial = 0.5 * mass * (radius**2)
    radial = (mass / 12.0) * (3.0 * (radius**2) + full_height**2)
    return np.array([radial, radial, axial], dtype=np.float64)


def _parse_vector(text: Optional[str], expected_length: int) -> np.ndarray:
    if text is None:
        return np.zeros(expected_length, dtype=np.float32)
    values = np.fromstring(text, sep=" ", dtype=np.float32)
    if len(values) != expected_length:
        raise ValueError(
            f"Expected {expected_length} values in vector '{text}', got {len(values)}"
        )
    return values


def _ycb_sim_include_path(prefix: str, object_name: str) -> Path:
    return _YCB_SIM_INCLUDE_DIR / f"{prefix}_{object_name}.xml"


def _resolve_ycb_sim_asset_path(base_dir: Path, relative_path: str) -> Path:
    candidate = Path(relative_path)
    if candidate.is_absolute() and candidate.exists():
        return candidate

    direct_candidate = (base_dir / candidate).resolve()
    if direct_candidate.exists():
        return direct_candidate

    parts = candidate.parts
    if len(parts) >= 2 and parts[0] == ".." and parts[1] == "YCB_sim":
        adjusted_candidate = (base_dir.parent / Path(*parts[2:])).resolve()
        if adjusted_candidate.exists():
            return adjusted_candidate

    if "YCB_sim" in parts:
        marker = parts.index("YCB_sim")
        repo_relative = Path(*parts[marker + 1 :])
        repo_candidate = (_YCB_SIM_ROOT / repo_relative).resolve()
        if repo_candidate.exists():
            return repo_candidate

    return direct_candidate


def _collision_half_extents(
    collision_geom_type: str, collision_size: np.ndarray
) -> np.ndarray:
    if collision_geom_type == "box":
        if len(collision_size) != 3:
            raise ValueError("Box collision size must have 3 values")
        return collision_size.astype(np.float32)
    if collision_geom_type == "cylinder":
        if len(collision_size) != 2:
            raise ValueError("Cylinder collision size must have 2 values")
        radius, half_height = collision_size
        return np.array([radius, radius, half_height], dtype=np.float32)
    raise ValueError(f"Unsupported collision geometry type: {collision_geom_type}")


def _compute_rest_offset_z(
    collision_geom_type: str, collision_size: np.ndarray, collision_pos: np.ndarray
) -> float:
    if collision_geom_type == "box":
        z_half_extent = float(collision_size[2])
    elif collision_geom_type == "cylinder":
        z_half_extent = float(collision_size[1])
    else:
        raise ValueError(f"Unsupported collision geometry type: {collision_geom_type}")
    min_z = float(collision_pos[2] - z_half_extent)
    return max(0.0, -min_z)


def _stable_inertia(object_spec: YCBObjectSpec) -> np.ndarray:
    if object_spec.collision_geom_type == "cylinder":
        radius = float(object_spec.collision_size[0])
        half_height = float(object_spec.collision_size[1])
        return _stable_cylinder_inertia(radius, half_height, object_spec.mass)
    return _stable_box_inertia(object_spec.half_extents, mass=object_spec.mass)


def _format_vector(values: np.ndarray, precision: int = 6) -> str:
    return " ".join(f"{float(value):.{precision}f}" for value in values)


def _require_element(root: ET.Element, query: str) -> ET.Element:
    element = root.find(query)
    if element is None:
        raise ValueError(f"Scene template is missing required element: {query}")
    return element


def _find_visual_mesh(object_root: Path) -> Path | None:
    for relative_path in _TEXTURED_MESH_CANDIDATES:
        candidate = object_root / relative_path
        if candidate.exists():
            return candidate
    return find_source_mesh(object_root)


def _find_obj_texture(mesh_path: Path) -> Optional[Path]:
    if mesh_path.suffix.lower() != ".obj":
        return None

    mtllib_path: Optional[Path] = None
    with mesh_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.startswith("mtllib "):
                _, relative_mtl = stripped.split(None, 1)
                mtllib_path = (mesh_path.parent / relative_mtl).resolve()
                break

    if mtllib_path is None or not mtllib_path.exists():
        return None

    with mtllib_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.startswith("map_Kd "):
                _, relative_texture = stripped.split(None, 1)
                texture_path = (mtllib_path.parent / relative_texture).resolve()
                if texture_path.exists():
                    return texture_path
                break

    return None


def _load_obj_vertices(mesh_path: Path) -> np.ndarray:
    vertices = []
    with mesh_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("v "):
                continue
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])

    if not vertices:
        raise ValueError(f"OBJ mesh does not contain any vertices: {mesh_path}")

    return np.asarray(vertices, dtype=np.float32)


def _load_visual_geometry_points(mesh_path: Path, pointcloud_path: Path) -> np.ndarray:
    if mesh_path.suffix.lower() == ".obj":
        return _load_obj_vertices(mesh_path)
    return clean_points(load_ply_vertices(pointcloud_path))


def _find_dynamic_material(root: ET.Element) -> ET.Element:
    asset = _require_element(root, ".//asset")
    for material in asset.findall("material"):
        if material.get("name") != "groundplane":
            return material
    raise ValueError("Scene template is missing the object material entry")


def _find_named_asset(
    asset_root: ET.Element, element_type: str, name: str
) -> Optional[ET.Element]:
    for element in asset_root.findall(element_type):
        if element.get("name") == name:
            return element
    return None


def _default_scene_path(object_spec: YCBObjectSpec) -> Path:
    if object_spec.source == YCB_ASSET_SOURCE_YCB_SIM:
        return _GENERATED_SCENE_DIR / f"scene_{object_spec.name}_{object_spec.source}.xml"
    return _GENERATED_SCENE_DIR / f"scene_{object_spec.name}.xml"


def _load_raw_ycb_object_spec(
    object_root: Path, scale: float = 1.0
) -> YCBObjectSpec:
    mesh_path = _find_visual_mesh(object_root)
    pointcloud_path = find_source_ply(object_root)
    if mesh_path is None or pointcloud_path is None:
        raise FileNotFoundError(
            f"Could not find mesh and point cloud data under {object_root}"
        )
    texture_path = _find_obj_texture(mesh_path.resolve())

    points = _load_visual_geometry_points(mesh_path, pointcloud_path)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    half_extents = 0.5 * (maxs - mins)
    if np.any(half_extents <= 0):
        raise ValueError(f"Degenerate YCB object bounds for {object_root}")

    center = 0.5 * (mins + maxs)
    scaled_half_extents = (scale * half_extents).astype(np.float32)
    collision_pos = np.zeros(3, dtype=np.float32)
    return YCBObjectSpec(
        name=object_root.name,
        object_root=object_root,
        mesh_path=mesh_path.resolve(),
        texture_path=texture_path,
        pointcloud_path=pointcloud_path.resolve(),
        mesh_offset=(-scale * center).astype(np.float32),
        half_extents=scaled_half_extents,
        placement_radius=float(np.linalg.norm(scaled_half_extents[:2])),
        collision_size=scaled_half_extents.copy(),
        collision_pos=collision_pos,
        rest_offset_z=float(scaled_half_extents[2]),
        scale=scale,
    )


def _load_ycb_sim_object_spec(
    object_root: Path, scale: float = 1.0
) -> YCBObjectSpec:
    object_name = object_root.name
    assets_include_path = _ycb_sim_include_path("assets", object_name)
    body_include_path = _ycb_sim_include_path("body", object_name)

    if not assets_include_path.exists() or not body_include_path.exists():
        raise FileNotFoundError(
            f"Could not find YCB_sim includes for object '{object_name}' under {_YCB_SIM_INCLUDE_DIR}"
        )

    assets_root = ET.parse(assets_include_path).getroot()
    mesh_element = _require_element(assets_root, ".//asset/mesh")
    mesh_path = _resolve_ycb_sim_asset_path(
        assets_include_path.parent, mesh_element.get("file", "")
    )
    texture_element = assets_root.find(".//asset/texture")
    texture_path = None
    if texture_element is not None and texture_element.get("file") is not None:
        texture_path = _resolve_ycb_sim_asset_path(
            assets_include_path.parent, texture_element.get("file", "")
        )

    body_root = ET.parse(body_include_path).getroot()
    visual_geom = None
    collision_geom = None
    for geom in body_root.findall("geom"):
        geom_class = geom.get("class")
        if geom_class == "ycb_viz":
            visual_geom = geom
        elif geom_class == "ycb_col":
            collision_geom = geom

    if visual_geom is None or collision_geom is None:
        raise ValueError(
            f"YCB_sim body include for '{object_name}' is missing visual or collision geom"
        )

    visual_pos = scale * _parse_vector(visual_geom.get("pos"), 3)
    collision_geom_type = collision_geom.get("type", "box")
    size_length = 3 if collision_geom_type == "box" else 2 if collision_geom_type == "cylinder" else 0
    if size_length == 0:
        raise ValueError(
            f"Unsupported YCB_sim collision type '{collision_geom_type}' for '{object_name}'"
        )
    collision_size = scale * _parse_vector(collision_geom.get("size"), size_length)
    collision_pos = scale * _parse_vector(collision_geom.get("pos"), 3)
    half_extents = _collision_half_extents(collision_geom_type, collision_size)
    mesh_offset = visual_pos - collision_pos
    collision_pos = np.zeros(3, dtype=np.float32)
    rest_offset_z = float(half_extents[2])
    mass = float(collision_geom.get("mass", "0.15"))

    return YCBObjectSpec(
        name=object_name,
        object_root=object_root,
        mesh_path=mesh_path,
        texture_path=texture_path,
        pointcloud_path=None,
        mesh_offset=mesh_offset.astype(np.float32),
        half_extents=half_extents.astype(np.float32),
        placement_radius=float(np.linalg.norm(half_extents[:2])),
        collision_size=collision_size.astype(np.float32),
        collision_pos=collision_pos.astype(np.float32),
        rest_offset_z=rest_offset_z,
        scale=scale,
        mass=mass,
        collision_geom_type=collision_geom_type,
        source=YCB_ASSET_SOURCE_YCB_SIM,
    )


def load_ycb_object_spec(
    object_root: Union[str, Path],
    scale: float = 1.0,
    source: str = YCB_ASSET_SOURCE_RAW,
) -> YCBObjectSpec:
    """Load mesh and geometric bounds for a YCB object directory."""
    resolved_root = Path(object_root).resolve()
    scale = float(scale)
    if scale <= 0:
        raise ValueError(f"Object scale must be positive, got {scale}")

    normalized_source = source.lower()
    if normalized_source == YCB_ASSET_SOURCE_YCB_SIM:
        return _load_ycb_sim_object_spec(resolved_root, scale=scale)
    if normalized_source == YCB_ASSET_SOURCE_RAW:
        return _load_raw_ycb_object_spec(resolved_root, scale=scale)
    raise ValueError(
        f"Unsupported YCB object source '{source}'. Expected '{YCB_ASSET_SOURCE_RAW}' or '{YCB_ASSET_SOURCE_YCB_SIM}'."
    )


def create_single_object_ycb_scene(
    object_spec: YCBObjectSpec,
    scene_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Create a MuJoCo scene from the checked-in XML template if needed."""
    output_dir = _GENERATED_SCENE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if scene_path is None:
        scene_path = _default_scene_path(object_spec)
    else:
        scene_path = Path(scene_path)
        scene_path.parent.mkdir(parents=True, exist_ok=True)

    if scene_path.exists():
        return scene_path

    template_path = _DEFAULT_SCENE_TEMPLATE
    tree = ET.parse(template_path)
    root = tree.getroot()

    mesh_name = f"mesh_{object_spec.name}"
    material_name = f"material_{object_spec.name}"
    texture_name = f"texture_{object_spec.name}"
    inertia = _stable_inertia(object_spec)

    asset = _require_element(root, ".//asset")
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
    mesh.set(
        "scale",
        _format_vector(np.full(3, object_spec.scale, dtype=np.float32)),
    )

    material.set("name", material_name)
    texture = _find_named_asset(asset, "texture", texture_name)
    if object_spec.texture_path is not None:
        if texture is None:
            texture = ET.SubElement(asset, "texture")
        texture.attrib.clear()
        texture.set("name", texture_name)
        texture.set("type", "2d")
        texture.set("file", object_spec.texture_path.as_posix())

        material.attrib.pop("rgba", None)
        material.set("texture", texture_name)
    else:
        if texture is not None:
            asset.remove(texture)
        material.attrib.pop("texture", None)
        material.set("rgba", "0.82 0.28 0.24 1")

    target_body.set("name", object_spec.body_name)
    target_body.set("pos", f"100 0 {float(object_spec.rest_offset_z):.6f}")

    target_joint.set("name", f"{object_spec.body_name}_freejoint")

    inertial.set("mass", f"{object_spec.mass:.6f}")
    inertial.set("pos", "0 0 0")
    inertial.set("diaginertia", _format_vector(inertia, precision=8))

    visual_geom.set("name", f"{object_spec.body_name}_visual")
    visual_geom.set("mesh", mesh_name)
    visual_geom.set("pos", _format_vector(object_spec.mesh_offset))
    visual_geom.set("material", material_name)
    visual_geom.set("contype", "0")
    visual_geom.set("conaffinity", "0")

    collision_geom.set("name", f"{object_spec.body_name}_collision")
    collision_geom.set("type", object_spec.collision_geom_type)
    collision_geom.set("size", _format_vector(object_spec.collision_size))
    collision_geom.set("pos", _format_vector(object_spec.collision_pos))
    collision_geom.set("rgba", "0 0 0 0")
    collision_geom.set("contype", _YCB_COLLISION_CONTYPE)
    collision_geom.set("conaffinity", _YCB_COLLISION_CONAFFINITY)
    collision_geom.set("condim", _YCB_COLLISION_CONDIM)
    collision_geom.set("solimp", _YCB_COLLISION_SOLIMP)
    collision_geom.set("solref", _YCB_COLLISION_SOLREF)
    collision_geom.set("friction", _YCB_COLLISION_FRICTION)
    collision_geom.attrib.pop("mass", None)
    collision_geom.attrib.pop("density", None)

    if hasattr(ET, "indent"):
        ET.indent(tree, space="  ")
    tree.write(scene_path, encoding="utf-8", xml_declaration=False)
    return scene_path


def ensure_single_object_ycb_scene(
    object_root: Union[str, Path] = DEFAULT_YCB_OBJECT_ROOT,
    scene_path: Optional[Union[str, Path]] = None,
    scale: float = 1.0,
    source: str = YCB_ASSET_SOURCE_RAW,
) -> tuple[Path, YCBObjectSpec]:
    """Create the single-object scene for the requested YCB object if missing."""
    object_spec = load_ycb_object_spec(object_root, scale=scale, source=source)
    xml_path = create_single_object_ycb_scene(object_spec, scene_path=scene_path)
    return xml_path, object_spec
