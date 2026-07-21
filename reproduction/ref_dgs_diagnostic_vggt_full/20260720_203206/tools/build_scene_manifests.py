#!/usr/bin/env python3
"""Build deterministic manifests for the non-canonical full VGGT diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


NON_CANONICAL_LABEL = (
    "NON-CANONICAL VGGT-REGENERATED PRIORS — NOT A STRICT PAPER REPRODUCTION"
)
GROUPING_POLICY = "transform_file_order_consecutive_chunks"


def make_batches(records: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    batches: list[list[dict[str, Any]]] = []
    for start in range(0, len(records), batch_size):
        batch = []
        for slot, record in enumerate(records[start : start + batch_size]):
            item = dict(record)
            item["slot"] = slot
            item["batch_index"] = len(batches)
            batch.append(item)
        batches.append(batch)
    return batches


def _load_records(
    dataset: str,
    scene: str,
    scene_root: Path,
    split: str,
) -> list[dict[str, Any]]:
    transform_path = scene_root / f"transforms_{split}.json"
    contents = json.loads(transform_path.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for frame_index, frame in enumerate(contents["frames"]):
        file_path = Path(frame["file_path"])
        image_name = file_path.name
        if image_name in seen:
            raise ValueError(f"duplicate {split} image name: {image_name}")
        seen.add(image_name)
        if dataset == "ShinySynthetic":
            source_relpath = f"{split}/{image_name}.png"
            prior_relpath = f"Ref-NeRF/refnerf/{scene}/{split}/depth/{image_name}.pth"
            normal_relpath = f"Ref-NeRF/refnerf/{scene}/{split}/normal/{image_name}.png"
        elif dataset == "GlossySynthetic":
            source_relpath = str(file_path.with_suffix(".png"))
            scene_key = f"{scene}_blender"
            prior_relpath = f"Glossy/GlossySynthetic/{scene_key}/depth/{image_name}.pth"
            normal_relpath = f"Glossy/GlossySynthetic/{scene_key}/normal/{image_name}.png"
        else:
            raise ValueError(f"unsupported diagnostic dataset: {dataset}")
        records.append(
            {
                "split": split,
                "frame_index": frame_index,
                "image_name": image_name,
                "source_relpath": source_relpath,
                "prior_relpath": prior_relpath,
                "normal_relpath": normal_relpath,
            }
        )
    return records


def build_manifest(
    dataset: str,
    scene: str,
    scene_root: Path,
    *,
    batch_size: int = 20,
) -> dict[str, Any]:
    train_records = _load_records(dataset, scene, scene_root, "train")
    test_records = _load_records(dataset, scene, scene_root, "test")
    train_names = {item["image_name"] for item in train_records}
    test_names = {item["image_name"] for item in test_records}
    overlap = sorted(train_names & test_names)
    if dataset == "GlossySynthetic" and overlap:
        raise ValueError(f"train/test image-name overlap would overwrite priors: {overlap}")
    return {
        "label": NON_CANONICAL_LABEL,
        "dataset": dataset,
        "scene": scene,
        "scene_root": str(scene_root.resolve()),
        "split_counts": {"train": len(train_records), "test": len(test_records)},
        "unique_view_count": len(train_records) + len(test_records),
        "grouping": {
            "policy": GROUPING_POLICY,
            "batch_size": batch_size,
            "final_partial_batch_allowed": True,
            "claimed_author_equivalent": False,
            "reason": "author multi-view grouping is unavailable for these scenes",
        },
        "train_batches": make_batches(train_records, batch_size),
        "test_batches": make_batches(test_records, batch_size),
    }


def apply_recovered_grouping(
    manifest: dict[str, Any], recovered: dict[str, Any]
) -> dict[str, Any]:
    if manifest["dataset"] != "ShinySynthetic" or manifest["scene"] != "ball":
        raise ValueError("recovered grouping is only valid for ShinySynthetic/ball")
    result = json.loads(json.dumps(manifest))
    for split in ("train", "test"):
        records = {
            item["image_name"]: item
            for batch in manifest[f"{split}_batches"]
            for item in batch
        }
        recovered_names = [
            item["image_name"]
            for batch in recovered[f"{split}_batches"]
            for item in batch
        ]
        if set(recovered_names) != set(records) or len(recovered_names) != len(records):
            raise ValueError(f"recovered {split} grouping does not match transform cameras")
        rebuilt = []
        for batch_index, recovered_batch in enumerate(recovered[f"{split}_batches"]):
            batch = []
            for slot, recovered_item in enumerate(recovered_batch):
                item = dict(records[recovered_item["image_name"]])
                item["slot"] = slot
                item["batch_index"] = batch_index
                batch.append(item)
            rebuilt.append(batch)
        result[f"{split}_batches"] = rebuilt
    result["grouping"] = {
        "policy": recovered["protocol"],
        "batch_size": int(recovered.get("views_per_batch", 20)),
        "final_partial_batch_allowed": False,
        "claimed_author_equivalent": False,
        "reason": (
            "ball batch membership and order were recovered from intact storage metadata "
            "in the corrupt official archive; the unpublished author generation environment "
            "and numeric payload remain unavailable"
        ),
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("ShinySynthetic", "GlossySynthetic"), required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--recovered-ball-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_manifest(args.dataset, args.scene, args.scene_root, batch_size=args.batch_size)
    if args.recovered_ball_manifest is not None:
        recovered = json.loads(args.recovered_ball_manifest.read_text(encoding="utf-8"))
        manifest = apply_recovered_grouping(manifest, recovered)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "views": manifest["unique_view_count"]}))


if __name__ == "__main__":
    main()
