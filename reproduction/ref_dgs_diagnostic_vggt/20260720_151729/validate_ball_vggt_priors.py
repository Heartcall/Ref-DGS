#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
    return sorted(records, key=lambda item: int(item["frame_number"]))


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


def _write_visualization(
    scene_root: Path,
    split: str,
    image_name: str,
    payload: dict[str, torch.Tensor],
    output_path: Path,
) -> None:
    rgb = Image.open(scene_root / split / f"{image_name}.png").convert("RGB")
    rgb = rgb.resize((518, 518), Image.Resampling.BICUBIC)
    depth = _colorize(payload["depth_map"])
    confidence = _colorize(payload["depth_conf"], log_scale=True)
    panel = Image.new("RGB", (1554, 548), "white")
    panel.paste(rgb, (0, 30))
    panel.paste(depth, (518, 30))
    panel.paste(confidence, (1036, 30))
    draw = ImageDraw.Draw(panel)
    draw.text((8, 8), f"{split}/{image_name}: RGB", fill="black")
    draw.text((526, 8), "VGGT depth (p02-p98)", fill="black")
    draw.text((1044, 8), "VGGT confidence log1p (p02-p98)", fill="black")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(output_path)


def validate_tree(args: argparse.Namespace) -> dict[str, Any]:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    result: dict[str, Any] = {"label": NON_CANONICAL_LABEL, "status": "pass", "splits": {}}
    all_depth = DistributionAccumulator()
    all_confidence = DistributionAccumulator()
    visualization_targets = {"train": {0, 25, 50}, "test": {0, 100}}

    for split, expected_count in (("train", 100), ("test", 200)):
        records = _records_for_split(manifest, split)
        transform = json.loads((args.scene_root / f"transforms_{split}.json").read_text(encoding="utf-8"))
        transform_names = {Path(frame["file_path"]).name for frame in transform["frames"]}
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
            prior_path = args.prior_root / split / "depth" / f"{image_name}.pth"
            payload = torch.load(prior_path, map_location="cpu", weights_only=True)
            validate_payload(payload)
            depth = payload["depth_map"]
            confidence = payload["depth_conf"]
            shapes.add(tuple(depth.shape))
            depth_accumulator.update(depth)
            confidence_accumulator.update(confidence)
            all_depth.update(depth)
            all_confidence.update(confidence)
            image_path = args.scene_root / split / f"{image_name}.png"
            alpha_path = args.scene_root / split / f"{image_name}_alpha.png"
            if not image_path.is_file() or not alpha_path.is_file():
                raise FileNotFoundError(f"missing RGB/alpha for {split}/{image_name}")
            with Image.open(image_path) as image:
                image_sizes.add(image.size)
            if image_name not in transform_names:
                raise ValueError(f"{split}/{image_name} is absent from transforms_{split}.json")
            depth_std = depth.to(torch.float64).std(unbiased=False).item()
            confidence_std = confidence.to(torch.float64).std(unbiased=False).item()
            if depth_std == 0 or confidence_std == 0:
                raise ValueError(f"constant prior map detected at {split}/{image_name}")
            median = depth.median().item()
            if previous_median is not None:
                adjacent_relative_jumps.append(abs(median - previous_median) / max(abs(previous_median), 1e-12))
            previous_median = median
            per_view.append(
                {
                    "image_name": image_name,
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
            if int(record["frame_number"]) in visualization_targets[split]:
                _write_visualization(
                    args.scene_root,
                    split,
                    image_name,
                    payload,
                    args.visualization_root / f"{split}_{image_name}.png",
                )

        if shapes != {(518, 518)}:
            raise ValueError(f"unexpected prior shapes for {split}: {shapes}")
        result["splits"][split] = {
            "file_count": len(records),
            "schema": {"keys": list(REQUIRED_KEYS), "shape": [518, 518], "dtype": "torch.float32"},
            "source_image_sizes": [list(size) for size in sorted(image_sizes)],
            "transform_frame_count": len(transform["frames"]),
            "depth": depth_accumulator.finalize(),
            "confidence": confidence_accumulator.finalize(),
            "adjacent_frame_median_depth_relative_jump": tensor_stats(torch.tensor(adjacent_relative_jumps)),
            "per_view": per_view,
        }

    result["overall"] = {
        "file_count": 300,
        "depth": all_depth.finalize(),
        "confidence": all_confidence.finalize(),
        "visualization_count": len(list(args.visualization_root.glob("*.png"))),
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prior-root", type=Path, required=True)
    parser.add_argument("--visualization-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = validate_tree(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "file_count": result["overall"]["file_count"]}))


if __name__ == "__main__":
    main()
