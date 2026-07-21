#!/usr/bin/env python3
"""Generate NON-CANONICAL Ref-DGS priors with the public VGGT checkpoint.

The manifest preserves the exact 20-view groups and slot ordering recovered from
the intact pickle metadata of the author-published (but payload-corrupt) priors.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

from validate_ball_vggt_priors import prepare_payload


NON_CANONICAL_LABEL = (
    "NON-CANONICAL: regenerated with public VGGT; not an official paper reproduction"
)


def batches_for_split(manifest: dict[str, Any], split: str) -> list[list[dict[str, Any]]]:
    key = f"{split}_batches"
    if key not in manifest:
        raise ValueError(f"manifest is missing {key}")
    return manifest[key]


def image_paths_for_batch(
    batch: list[dict[str, Any]],
    scene_root: Path,
    split: str,
    *,
    require_exists: bool = True,
) -> list[Path]:
    ordered = sorted(batch, key=lambda item: int(item["slot"]))
    paths = [scene_root / split / f"{item['image_name']}.png" for item in ordered]
    if require_exists:
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing source RGB files: {missing}")
    return paths


def prediction_payloads(
    depth: torch.Tensor,
    confidence: torch.Tensor,
    *,
    expected_views: int,
) -> list[dict[str, torch.Tensor]]:
    if depth.ndim != 5 or depth.shape[0] != 1 or depth.shape[1] != expected_views:
        found = depth.shape[1] if depth.ndim >= 2 else 0
        raise ValueError(
            f"expected {expected_views} views in depth output, found {found}: "
            f"{tuple(depth.shape)}"
        )
    if depth.shape[2] == 1:
        depth_maps = depth
    elif depth.shape[-1] == 1:
        depth_maps = depth.permute(0, 1, 4, 2, 3)
    else:
        raise ValueError(
            "depth/confidence must have one singleton channel in either "
            f"[B,S,1,H,W] or [B,S,H,W,1], got {tuple(depth.shape)}"
        )
    if confidence.ndim == 4:
        confidence_maps = confidence.unsqueeze(2)
    elif confidence.ndim == 5 and confidence.shape[2] == 1:
        confidence_maps = confidence
    elif confidence.ndim == 5 and confidence.shape[-1] == 1:
        confidence_maps = confidence.permute(0, 1, 4, 2, 3)
    else:
        raise ValueError(
            "confidence must be [B,S,H,W], [B,S,1,H,W], or [B,S,H,W,1], got "
            f"{tuple(confidence.shape)}"
        )
    if confidence_maps.shape != depth_maps.shape:
        raise ValueError(
            f"normalized depth/confidence shape mismatch: {tuple(depth_maps.shape)} vs "
            f"{tuple(confidence_maps.shape)}"
        )
    return [
        prepare_payload(depth_maps[0, view_index], confidence_maps[0, view_index])
        for view_index in range(expected_views)
    ]


def _load_model(vggt_root: Path, checkpoint: Path, device: torch.device):
    sys.path.insert(0, str(vggt_root))
    from vggt.models.vggt import VGGT

    model = VGGT()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    # These branches are not used to create Ref-DGS depth priors. Remove them
    # only after strict checkpoint validation, before transferring to the GPU.
    model.camera_head = None
    model.point_head = None
    model.track_head = None
    return model.eval().to(device)


def _load_images(vggt_root: Path, paths: list[Path], mode: str) -> torch.Tensor:
    if str(vggt_root) not in sys.path:
        sys.path.insert(0, str(vggt_root))
    from vggt.utils.load_fn import load_and_preprocess_images

    return load_and_preprocess_images([str(path) for path in paths], mode=mode)


def _write_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def generate(args: argparse.Namespace) -> None:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    splits = [args.split] if args.split != "all" else ["train", "test"]
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("VGGT diagnostic generation requires an available CUDA GPU")

    dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.batch_log.parent.mkdir(parents=True, exist_ok=True)
    model = _load_model(args.vggt_root, args.checkpoint, device)

    run_metadata = {
        "label": NON_CANONICAL_LABEL,
        "checkpoint": str(args.checkpoint),
        "manifest": str(args.manifest),
        "scene_root": str(args.scene_root),
        "preprocess_mode": args.preprocess_mode,
        "inference_dtype": str(dtype),
        "device": str(device),
        "splits": splits,
        "probe_first_batch_only": args.probe_first_batch_only,
    }
    (args.output_root / "generation_metadata.json").write_text(
        json.dumps(run_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    for split in splits:
        batches = batches_for_split(manifest, split)
        if args.probe_first_batch_only:
            batches = batches[:1]
        depth_dir = args.output_root / split / "depth"
        depth_dir.mkdir(parents=True, exist_ok=True)
        for batch_index, batch in enumerate(batches):
            paths = image_paths_for_batch(batch, args.scene_root, split)
            images = _load_images(args.vggt_root, paths, args.preprocess_mode)
            if tuple(images.shape) != (len(batch), 3, 518, 518):
                raise ValueError(f"unexpected preprocessed image shape: {tuple(images.shape)}")
            images = images.to(device, non_blocking=True)
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            with torch.inference_mode():
                with torch.cuda.amp.autocast(dtype=dtype):
                    predictions = model(images)
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            payloads = prediction_payloads(
                predictions["depth"],
                predictions["depth_conf"],
                expected_views=len(batch),
            )
            for item, payload in zip(sorted(batch, key=lambda row: int(row["slot"])), payloads):
                torch.save(payload, depth_dir / f"{item['image_name']}.pth")
            record = {
                "label": NON_CANONICAL_LABEL,
                "split": split,
                "batch_index": batch_index,
                "view_count": len(batch),
                "image_names": [item["image_name"] for item in sorted(batch, key=lambda row: int(row["slot"]))],
                "input_shape": list(images.shape),
                "depth_shape": list(predictions["depth"].shape),
                "depth_dtype": str(predictions["depth"].dtype),
                "confidence_dtype": str(predictions["depth_conf"].dtype),
                "elapsed_seconds": elapsed,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            }
            _write_jsonl(args.batch_log, record)
            print(json.dumps(record, sort_keys=True), flush=True)
            del predictions, payloads, images
            torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vggt-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-log", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test", "all"), default="all")
    parser.add_argument("--preprocess-mode", choices=("crop", "pad"), default="crop")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--probe-first-batch-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    generate(parse_args())
