#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw


REQUIRED_KEYS = ("depth_map", "depth_conf")
QUANTILES = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)
NON_CANONICAL_LABEL = (
    "NON-CANONICAL: regenerated with public VGGT; not an official paper reproduction"
)


def _to_map(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if not torch.is_tensor(tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    if tensor.ndim != 2:
        raise ValueError(f"{name} must have shape [H,W] or [1,H,W], got {tuple(tensor.shape)}")
    return tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()


def validate_payload(payload: Any) -> dict[str, torch.Tensor]:
    if not isinstance(payload, dict):
        raise ValueError("payload must be a dict")
    for key in REQUIRED_KEYS:
        if key not in payload:
            raise ValueError(f"payload is missing {key}")
        if not torch.is_tensor(payload[key]):
            raise ValueError(f"{key} must be a torch.Tensor")
    depth = payload["depth_map"]
    confidence = payload["depth_conf"]
    if depth.shape != confidence.shape:
        raise ValueError("depth_map and depth_conf must have the same shape")
    if depth.ndim != 2:
        raise ValueError("depth_map and depth_conf must be two-dimensional")
    if not torch.isfinite(depth).all() or not torch.isfinite(confidence).all():
        raise ValueError("depth_map and depth_conf must be finite")
    if not (confidence >= 0).all():
        raise ValueError("depth_conf must be nonnegative")
    return payload


def validate_payload_quality(payload: Any) -> dict[str, torch.Tensor]:
    payload = validate_payload(payload)
    if payload["depth_map"].to(torch.float64).std(unbiased=False).item() == 0:
        raise ValueError("constant depth map")
    if payload["depth_conf"].to(torch.float64).std(unbiased=False).item() == 0:
        raise ValueError("constant confidence map")
    return payload


def uniform_indices(count: int, sample_count: int = 5) -> list[int]:
    if count <= 0 or sample_count <= 0:
        return []
    if count <= sample_count:
        return list(range(count))
    return [round(index * (count - 1) / (sample_count - 1)) for index in range(sample_count)]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_payload(depth_map: torch.Tensor, depth_conf: torch.Tensor) -> dict[str, torch.Tensor]:
    payload = {
        "depth_map": _to_map(depth_map, "depth_map"),
        "depth_conf": _to_map(depth_conf, "depth_conf"),
    }
    return validate_payload(payload)


def tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    values = tensor.detach().to(device="cpu", dtype=torch.float64).flatten()
    if values.numel() == 0:
        raise ValueError("cannot summarize an empty tensor")
    quantile_values = torch.quantile(values, torch.tensor(QUANTILES, dtype=torch.float64))
    return {
        "count": values.numel(),
        "min": values.min().item(),
        "max": values.max().item(),
        "mean": values.mean().item(),
        "std": values.std(unbiased=False).item(),
        "zero_ratio": (values == 0).to(torch.float64).mean().item(),
        "quantiles": {
            f"p{int(round(q * 100)):02d}": value.item()
            for q, value in zip(QUANTILES, quantile_values)
        },
    }


class DistributionAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.total_squared = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf
        self.zero_count = 0
        self.samples: list[torch.Tensor] = []

    def update(self, tensor: torch.Tensor) -> None:
        values = tensor.detach().to(device="cpu", dtype=torch.float64).flatten()
        self.count += values.numel()
        self.total += values.sum().item()
        self.total_squared += (values * values).sum().item()
        self.minimum = min(self.minimum, values.min().item())
        self.maximum = max(self.maximum, values.max().item())
        self.zero_count += (values == 0).sum().item()
        stride = max(1, values.numel() // 4096)
        self.samples.append(values[::stride][:4096].clone())

    def finalize(self) -> dict[str, Any]:
        sample = torch.cat(self.samples)
        quantiles = torch.quantile(sample, torch.tensor(QUANTILES, dtype=torch.float64))
        mean = self.total / self.count
        variance = max(0.0, self.total_squared / self.count - mean * mean)
        return {
            "count": self.count,
            "min": self.minimum,
            "max": self.maximum,
            "mean": mean,
            "std": math.sqrt(variance),
            "zero_ratio": self.zero_count / self.count,
            "quantiles": {
                f"p{int(round(q * 100)):02d}": value.item()
                for q, value in zip(QUANTILES, quantiles)
            },
            "quantile_method": "deterministic uniform sample up to 4096 pixels per view",
            "quantile_sample_count": sample.numel(),
        }


def _records_for_split(manifest: dict[str, Any], split: str) -> list[dict[str, Any]]:
    records = [record for batch in manifest[f"{split}_batches"] for record in batch]
    return sorted(records, key=lambda item: int(item["frame_index"]))


def _colorize(values: torch.Tensor, *, log_scale: bool = False) -> Image.Image:
    array = values.detach().to(device="cpu", dtype=torch.float32).numpy()
    if log_scale:
        array = np.log1p(np.maximum(array, 0))
    low, high = np.quantile(array, [0.02, 0.98])
    normalized = np.clip((array - low) / max(high - low, 1e-12), 0, 1)
    # Small dependency-free blue/cyan/yellow diagnostic colour ramp.
    red = np.clip(2.0 * normalized - 0.25, 0, 1)
    green = np.clip(2.0 - np.abs(4.0 * normalized - 2.0), 0, 1)
    blue = np.clip(1.25 - 2.0 * normalized, 0, 1)
    rgb = np.stack([red, green, blue], axis=-1)
    return Image.fromarray((rgb * 255).astype(np.uint8), mode="RGB")


def _preprocess_rgb(path: Path) -> Image.Image:
    image = Image.open(path)
    if image.mode == "RGBA":
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, image)
    image = image.convert("RGB")
    width, height = image.size
    new_width = 518
    new_height = round(height * (new_width / width) / 14) * 14
    image = image.resize((new_width, new_height), Image.Resampling.BICUBIC)
    if new_height > 518:
        start_y = (new_height - 518) // 2
        image = image.crop((0, start_y, 518, start_y + 518))
    return image


def _write_visualization(
    source_path: Path,
    split: str,
    image_name: str,
    payload: dict[str, torch.Tensor],
    output_path: Path,
) -> None:
    rgb = _preprocess_rgb(source_path).resize((518, 518), Image.Resampling.BICUBIC)
    depth = _colorize(payload["depth_map"]).resize((518, 518), Image.Resampling.NEAREST)
    confidence = _colorize(payload["depth_conf"], log_scale=True).resize(
        (518, 518), Image.Resampling.NEAREST
    )
    panel = Image.new("RGB", (1554, 570), "white")
    panel.paste(rgb, (0, 30))
    panel.paste(depth, (518, 30))
    panel.paste(confidence, (1036, 30))
    draw = ImageDraw.Draw(panel)
    draw.text((8, 8), f"{split}/{image_name}: RGB", fill="black")
    draw.text((526, 8), "VGGT depth (p02-p98)", fill="black")
    draw.text((1044, 8), "VGGT confidence log1p (p02-p98)", fill="black")
    draw.text((8, 552), NON_CANONICAL_LABEL, fill="black")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(output_path)


def validate_tree(args: argparse.Namespace) -> dict[str, Any]:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    result: dict[str, Any] = {
        "label": NON_CANONICAL_LABEL,
        "dataset": manifest["dataset"],
        "scene": manifest["scene"],
        "status": "pass",
        "splits": {},
        "file_sha256_manifest": [],
    }
    all_depth = DistributionAccumulator()
    all_confidence = DistributionAccumulator()
    all_records = [
        record
        for split in ("train", "test")
        for record in _records_for_split(manifest, split)
    ]
    visualization_targets = set(uniform_indices(len(all_records), 5))
    global_record_index = 0
    normal_file_count = 0

    for split in ("train", "test"):
        expected_count = int(manifest["split_counts"][split])
        records = _records_for_split(manifest, split)
        transform = json.loads((args.scene_root / f"transforms_{split}.json").read_text(encoding="utf-8"))
        transform_names = [Path(frame["file_path"]).name for frame in transform["frames"]]
        depth_accumulator = DistributionAccumulator()
        confidence_accumulator = DistributionAccumulator()
        per_view = []
        shapes = set()
        image_sizes = set()
        previous_median = None
        adjacent_relative_jumps = []

        if len(records) != expected_count:
            raise ValueError(f"expected {expected_count} manifest records for {split}, got {len(records)}")
        for record in records:
            image_name = record["image_name"]
            prior_path = args.prior_root / record["prior_relpath"]
            payload = torch.load(prior_path, map_location="cpu", weights_only=True)
            validate_payload_quality(payload)
            depth = payload["depth_map"]
            confidence = payload["depth_conf"]
            shapes.add(tuple(depth.shape))
            depth_accumulator.update(depth)
            confidence_accumulator.update(confidence)
            all_depth.update(depth)
            all_confidence.update(confidence)
            image_path = args.scene_root / record["source_relpath"]
            normal_path = args.official_prior_root / record["normal_relpath"]
            if not image_path.is_file():
                raise FileNotFoundError(f"missing RGB for {split}/{image_name}: {image_path}")
            if not normal_path.is_file():
                raise FileNotFoundError(f"missing official normal for {split}/{image_name}: {normal_path}")
            with Image.open(image_path) as image:
                image_sizes.add(image.size)
                image.verify()
            with Image.open(normal_path) as normal:
                normal.verify()
            normal_file_count += 1
            if image_name != transform_names[int(record["frame_index"])]:
                raise ValueError(
                    f"camera/prior index mismatch for {split} frame {record['frame_index']}: "
                    f"{image_name} != {transform_names[int(record['frame_index'])]}"
                )
            depth_std = depth.to(torch.float64).std(unbiased=False).item()
            confidence_std = confidence.to(torch.float64).std(unbiased=False).item()
            median = depth.median().item()
            if previous_median is not None:
                adjacent_relative_jumps.append(abs(median - previous_median) / max(abs(previous_median), 1e-12))
            previous_median = median
            per_view.append(
                {
                    "image_name": image_name,
                    "frame_index": int(record["frame_index"]),
                    "source_relpath": record["source_relpath"],
                    "prior_relpath": record["prior_relpath"],
                    "normal_relpath": record["normal_relpath"],
                    "depth_min": depth.min().item(),
                    "depth_max": depth.max().item(),
                    "depth_mean": depth.mean().item(),
                    "depth_std": depth_std,
                    "confidence_min": confidence.min().item(),
                    "confidence_max": confidence.max().item(),
                    "confidence_mean": confidence.mean().item(),
                    "confidence_std": confidence_std,
                    "confidence_zero_ratio": (confidence == 0).to(torch.float32).mean().item(),
                }
            )
            result["file_sha256_manifest"].append(
                {
                    "split": split,
                    "frame_index": int(record["frame_index"]),
                    "image_name": image_name,
                    "prior_relpath": record["prior_relpath"],
                    "sha256": sha256_file(prior_path),
                    "size_bytes": prior_path.stat().st_size,
                }
            )
            if global_record_index in visualization_targets:
                _write_visualization(
                    image_path,
                    split,
                    image_name,
                    payload,
                    args.visualization_root
                    / f"view_{global_record_index:04d}_{split}_{image_name}.png",
                )
            global_record_index += 1

        expected_shape = (int(args.expected_height), int(args.expected_width))
        if shapes != {expected_shape}:
            raise ValueError(f"unexpected prior shapes for {split}: {shapes}")
        if len(transform_names) != expected_count:
            raise ValueError(
                f"transform/manifest count mismatch for {split}: {len(transform_names)} != {expected_count}"
            )
        result["splits"][split] = {
            "file_count": len(records),
            "schema": {
                "keys": list(REQUIRED_KEYS),
                "shape": list(expected_shape),
                "dtype": "torch.float32",
            },
            "source_image_sizes": [list(size) for size in sorted(image_sizes)],
            "transform_frame_count": len(transform["frames"]),
            "depth": depth_accumulator.finalize(),
            "confidence": confidence_accumulator.finalize(),
            "adjacent_frame_median_depth_relative_jump": (
                tensor_stats(torch.tensor(adjacent_relative_jumps))
                if adjacent_relative_jumps
                else {"count": 0}
            ),
            "per_view": per_view,
        }

    expected_paths = {args.prior_root / record["prior_relpath"] for record in all_records}
    actual_paths: set[Path] = set()
    for parent in {path.parent for path in expected_paths}:
        actual_paths.update(parent.glob("*.pth"))
    if actual_paths != expected_paths:
        missing = sorted(str(path) for path in expected_paths - actual_paths)
        extra = sorted(str(path) for path in actual_paths - expected_paths)
        raise ValueError(f"scene prior/camera bijection mismatch; missing={missing}, extra={extra}")

    result["overall"] = {
        "file_count": len(all_records),
        "normal_file_count": normal_file_count,
        "depth": all_depth.finalize(),
        "confidence": all_confidence.finalize(),
        "visualization_count": len(list(args.visualization_root.glob("*.png"))),
        "camera_prior_filename_index_bijection": True,
        "all_source_images_readable": True,
        "all_official_normals_readable": True,
        "manual_visual_checks_required": [
            "horizontal_or_vertical_flip",
            "incorrect_crop",
            "index_misalignment",
            "obvious_background_leakage",
        ],
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prior-root", type=Path, required=True)
    parser.add_argument("--official-prior-root", type=Path, required=True)
    parser.add_argument("--visualization-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-height", type=int, default=518)
    parser.add_argument("--expected-width", type=int, default=518)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = validate_tree(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "file_count": result["overall"]["file_count"]}))


if __name__ == "__main__":
    main()
