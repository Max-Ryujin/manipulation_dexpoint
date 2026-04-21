#!/usr/bin/env python3

"""Compare PointNet encoder weights between two DexPoint RL checkpoints."""

import argparse
from collections import OrderedDict
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

import torch as th

from _script_bootstrap import ensure_script_imports

ensure_script_imports()

import dexpoint_policy  # noqa: F401
from dexart_baselines.stable_baselines3.a2c import A2C
from dexart_baselines.stable_baselines3.common.save_util import load_from_zip_file
from dexart_baselines.stable_baselines3.ppo import PPO


def _checkpoint_custom_objects(checkpoint_path: Path, device: str) -> Dict[str, object]:
    data, _, _ = load_from_zip_file(str(checkpoint_path), device=device)
    policy_kwargs = dict(data.get("policy_kwargs", {}))
    if "pointnet_checkpoint_path" in policy_kwargs:
        policy_kwargs["pointnet_checkpoint_path"] = None
    return {"policy_kwargs": policy_kwargs}


def _load_model(checkpoint_path: Path, algorithm: str, device: str):
    custom_objects = _checkpoint_custom_objects(checkpoint_path, device)
    candidates = [algorithm] if algorithm != "auto" else ["ppo", "a2c"]
    errors: Dict[str, str] = {}

    for candidate in candidates:
        algorithm_cls = PPO if candidate == "ppo" else A2C
        try:
            model = algorithm_cls.load(
                str(checkpoint_path),
                env=None,
                device=device,
                custom_objects=custom_objects,
            )
            model.policy.eval()
            return candidate, model
        except Exception as exc:  # pragma: no cover - depends on checkpoint type
            errors[candidate] = str(exc)

    raise RuntimeError(
        "Failed to load checkpoint. Tried algorithms: "
        + ", ".join(f"{name} ({error})" for name, error in errors.items())
    )


def _flatten_pointnet_state_dict(state_dict: Mapping[str, th.Tensor]) -> th.Tensor:
    flat_parts: List[th.Tensor] = []
    for key in sorted(state_dict.keys()):
        tensor = state_dict[key].detach().cpu().float().reshape(-1)
        flat_parts.append(tensor)
    if not flat_parts:
        return th.zeros(0, dtype=th.float32)
    return th.cat(flat_parts)


def _tensor_stats(
    delta: th.Tensor, reference: th.Tensor, other: th.Tensor
) -> Dict[str, float]:
    reference_norm = float(th.linalg.vector_norm(reference, ord=2).item())
    other_norm = float(th.linalg.vector_norm(other, ord=2).item())
    delta_norm = float(th.linalg.vector_norm(delta, ord=2).item())
    denom = max(reference_norm, 1e-12)
    cosine_similarity = float(
        th.nn.functional.cosine_similarity(
            reference.unsqueeze(0), other.unsqueeze(0)
        ).item()
    )
    return {
        "reference_l2_norm": reference_norm,
        "other_l2_norm": other_norm,
        "delta_l2_norm": delta_norm,
        "relative_l2_change": delta_norm / denom,
        "mean_abs_delta": float(delta.abs().mean().item()),
        "max_abs_delta": float(delta.abs().max().item()),
        "cosine_similarity": cosine_similarity,
    }


def _format_stats(title: str, stats: Mapping[str, float]) -> List[str]:
    return [
        title,
        f"  - reference_l2_norm: {stats['reference_l2_norm']:.6f}",
        f"  - other_l2_norm: {stats['other_l2_norm']:.6f}",
        f"  - delta_l2_norm: {stats['delta_l2_norm']:.6f}",
        f"  - relative_l2_change: {100.0 * stats['relative_l2_change']:.4f}%",
        f"  - mean_abs_delta: {stats['mean_abs_delta']:.8f}",
        f"  - max_abs_delta: {stats['max_abs_delta']:.8f}",
        f"  - cosine_similarity: {stats['cosine_similarity']:.8f}",
    ]


def _compare_state_dicts(
    reference_state_dict: Mapping[str, th.Tensor],
    other_state_dict: Mapping[str, th.Tensor],
) -> Tuple[Dict[str, object], OrderedDict[str, Dict[str, float]]]:
    reference_keys = set(reference_state_dict.keys())
    other_keys = set(other_state_dict.keys())
    if reference_keys != other_keys:
        missing = sorted(reference_keys - other_keys)
        extra = sorted(other_keys - reference_keys)
        raise ValueError(
            "PointNet parameter keys do not match between checkpoints: "
            f"missing={missing}, extra={extra}"
        )

    per_tensor = OrderedDict()
    for key in sorted(reference_state_dict.keys()):
        reference_tensor = reference_state_dict[key].detach().cpu().float()
        other_tensor = other_state_dict[key].detach().cpu().float()
        if reference_tensor.shape != other_tensor.shape:
            raise ValueError(
                f"PointNet tensor shape mismatch for {key}: "
                f"{tuple(reference_tensor.shape)} vs {tuple(other_tensor.shape)}"
            )
        delta_tensor = other_tensor - reference_tensor
        per_tensor[key] = {
            "numel": int(reference_tensor.numel()),
            **_tensor_stats(
                delta_tensor.reshape(-1),
                reference_tensor.reshape(-1),
                other_tensor.reshape(-1),
            ),
        }

    reference_flat = _flatten_pointnet_state_dict(reference_state_dict)
    other_flat = _flatten_pointnet_state_dict(other_state_dict)
    delta_flat = other_flat - reference_flat
    summary = {
        "parameter_count": int(reference_flat.numel()),
        **_tensor_stats(delta_flat, reference_flat, other_flat),
    }
    return summary, per_tensor


def _default_output_dir(checkpoint_a: Path, checkpoint_b: Path) -> Path:
    common_parent = checkpoint_a.parent
    if checkpoint_b.parent == checkpoint_a.parent:
        base_dir = common_parent
    else:
        base_dir = checkpoint_b.parent
    return base_dir / f"{checkpoint_a.stem}_vs_{checkpoint_b.stem}_pointnet_compare"


def compare_checkpoints(
    checkpoint_a: Path,
    checkpoint_b: Path,
    *,
    output_dir: Path,
    algorithm: str,
    device: str,
    top_k: int,
) -> Path:
    algorithm_a, model_a = _load_model(checkpoint_a, algorithm, device)
    algorithm_b, model_b = _load_model(checkpoint_b, algorithm, device)

    state_dict_a = (
        model_a.policy.features_extractor.pointnet_extractor.pointnet.state_dict()
    )
    state_dict_b = (
        model_b.policy.features_extractor.pointnet_extractor.pointnet.state_dict()
    )
    summary, per_tensor = _compare_state_dicts(state_dict_a, state_dict_b)

    output_dir.mkdir(parents=True, exist_ok=True)
    sorted_tensor_changes = sorted(
        per_tensor.items(),
        key=lambda item: item[1]["delta_l2_norm"],
        reverse=True,
    )
    top_tensor_changes = OrderedDict(sorted_tensor_changes[: max(top_k, 0)])

    payload = {
        "checkpoint_a": checkpoint_a.as_posix(),
        "checkpoint_b": checkpoint_b.as_posix(),
        "algorithm_a": algorithm_a,
        "algorithm_b": algorithm_b,
        "device": device,
        "summary": summary,
        "top_tensor_changes_by_delta_l2": top_tensor_changes,
        "per_tensor": per_tensor,
    }

    report_lines = [
        "DexPoint PointNet Checkpoint Comparison",
        "",
        f"Checkpoint A: {checkpoint_a}",
        f"Checkpoint B: {checkpoint_b}",
        f"Algorithm A: {algorithm_a}",
        f"Algorithm B: {algorithm_b}",
        f"Output directory: {output_dir}",
        "",
    ]
    report_lines.extend(_format_stats("Overall PointNet parameter drift:", summary))
    report_lines.append("")
    report_lines.append(
        f"Top {min(top_k, len(sorted_tensor_changes))} tensor changes by delta_l2_norm:"
    )
    for key, stats in top_tensor_changes.items():
        report_lines.append(f"  - {key} (numel={stats['numel']})")
        report_lines.append(f"    delta_l2_norm={stats['delta_l2_norm']:.6f}")
        report_lines.append(
            f"    relative_l2_change={100.0 * stats['relative_l2_change']:.4f}%"
        )
        report_lines.append(f"    mean_abs_delta={stats['mean_abs_delta']:.8f}")
        report_lines.append(f"    max_abs_delta={stats['max_abs_delta']:.8f}")
        report_lines.append(f"    cosine_similarity={stats['cosine_similarity']:.8f}")

    (output_dir / "pointnet_checkpoint_comparison.json").write_text(
        json.dumps(json.loads(json.dumps(payload)), indent=2),
        encoding="utf-8",
    )
    (output_dir / "pointnet_checkpoint_comparison.txt").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print(f"Checkpoint A: {checkpoint_a}")
    print(f"Checkpoint B: {checkpoint_b}")
    print(f"Algorithm A: {algorithm_a}")
    print(f"Algorithm B: {algorithm_b}")
    print(f"Output directory: {output_dir}")
    print()
    for line in _format_stats("Overall PointNet parameter drift:", summary):
        print(line)
    print()
    for key, stats in top_tensor_changes.items():
        print(
            f"{key}: delta_l2_norm={stats['delta_l2_norm']:.6f} "
            f"relative_l2_change={100.0 * stats['relative_l2_change']:.4f}% "
            f"max_abs_delta={stats['max_abs_delta']:.8f}"
        )

    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint_a", type=Path, help="Reference RL checkpoint (.zip)"
    )
    parser.add_argument(
        "checkpoint_b", type=Path, help="Comparison RL checkpoint (.zip)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for the JSON and text comparison outputs",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="auto",
        choices=["auto", "ppo", "a2c"],
        help="Checkpoint algorithm to load",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device for checkpoint loading",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="How many PointNet tensors to highlight in the summary",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint_a = args.checkpoint_a.expanduser().resolve()
    checkpoint_b = args.checkpoint_b.expanduser().resolve()
    for checkpoint_path in (checkpoint_a, checkpoint_b):
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        if checkpoint_path.suffix != ".zip":
            raise ValueError(f"Checkpoint must be a .zip file: {checkpoint_path}")

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else _default_output_dir(checkpoint_a, checkpoint_b)
    )
    compare_checkpoints(
        checkpoint_a,
        checkpoint_b,
        output_dir=output_dir,
        algorithm=args.algorithm,
        device=args.device,
        top_k=args.top_k,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
