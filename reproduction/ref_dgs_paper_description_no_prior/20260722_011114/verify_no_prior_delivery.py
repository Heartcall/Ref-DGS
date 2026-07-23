#!/usr/bin/env python3
"""Fresh, read-only validation of the completed no-prior experiment tree."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


LOSS_PREFIX = "[NO_GEOMETRY_PRIOR_LOSS] "
REQUIRED_CHECKPOINTS = (
    "gaussians_point_cloud.ply",
    "ref_gaussians_point_cloud.ply",
    "light_mlp.pt",
    "dir_encoding.pt",
)


def count_png(path: Path) -> int:
    return sum(1 for item in path.glob("*.png") if item.is_file())


def parse_loss_records(log_path: Path) -> list[dict]:
    records = []
    for line in log_path.read_text(errors="replace").splitlines():
        if not line.startswith(LOSS_PREFIX):
            continue
        json_text = line[len(LOSS_PREFIX):]
        json_text = re.sub(r" \[\d{2}/\d{2} .*$", "", json_text)
        records.append(json.loads(json_text))
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root.resolve()
    results = json.loads((root / "results_no_prior.json").read_text())
    checks = []
    exit_codes = {}
    errors = []

    for row in results["rows"]:
        dataset, scene, iterations = row["dataset"], row["scene"], int(row["iterations"])
        key = f"{dataset}/{scene}"
        attempt = root / "full" / dataset / scene / "attempt_001"
        model = attempt / "model"
        log_dir = root / "logs" / "full" / dataset / scene / "attempt_001"
        train_rc = int((log_dir / "train_exit_code.txt").read_text().strip())
        render_rc = int((log_dir / "render_exit_code.txt").read_text().strip())
        exit_codes[key] = {"train": train_rc, "render_eval": render_rc}

        expected_train, expected_test = ((100, 200) if dataset == "ShinySynthetic" else (112, 16))
        test_dir = model / "test" / f"ours_{iterations}"
        train_dir = model / "train" / f"ours_{iterations}"
        checkpoint = model / "point_cloud" / f"iteration_{iterations}"
        records = parse_loss_records(log_dir / "train.log")
        first100 = [record for record in records if 1 <= int(record["iteration"]) <= 100]
        finite_fields = ("loss_total", "loss_pbr", "alpha_loss", "normal_consistency_loss")
        loss_ok = (
            len(first100) == 100
            and all(record["geometry_prior_loaded"] is False for record in records)
            and all(record["vggt_depth_loss"] is None and record["vggt_normal_loss"] is None for record in records)
            and all(record["vggt_weight"] == 0.0 and record["vggt_until_iter"] == 0 for record in records)
            and all(math.isfinite(float(record[field])) for record in records for field in finite_fields)
        )
        numeric_fields = ("psnr", "ssim", "lpips", "normal_mae", "fps", "train_minutes", "render_eval_minutes")
        metric_ok = all(math.isfinite(float(row[field])) for field in numeric_fields)
        if dataset == "GlossySynthetic":
            metric_ok = metric_ok and math.isfinite(float(row["cd_x100"]))
        else:
            metric_ok = metric_ok and row["cd_x100"] is None

        scene_check = {
            "dataset": dataset,
            "scene": scene,
            "iterations": iterations,
            "train_exit_code": train_rc,
            "render_eval_exit_code": render_rc,
            "completion_marker": (attempt / "scene_complete.ok").is_file(),
            "checkpoint_artifacts": {name: (checkpoint / name).is_file() and (checkpoint / name).stat().st_size > 0 for name in REQUIRED_CHECKPOINTS},
            "loss_record_count": len(records),
            "first100_loss_record_count": len(first100),
            "loss_audit_valid": loss_ok,
            "test_render_count": count_png(test_dir / "renders"),
            "train_render_count": count_png(train_dir / "renders"),
            "test_normal_count": count_png(test_dir / "vis" / "normal"),
            "train_normal_count": count_png(train_dir / "vis" / "normal"),
            "expected_test_count": expected_test,
            "expected_train_count": expected_train,
            "test_metric_exists": (test_dir / "metric.txt").is_file(),
            "train_metric_exists": (train_dir / "metric.txt").is_file(),
            "fuse_mesh_exists": (train_dir / "fuse.ply").is_file(),
            "fuse_post_mesh_exists": (train_dir / "fuse_post.ply").is_file(),
            "mesh_log_rule_valid": ((train_dir / "mesh.log").is_file() if dataset == "GlossySynthetic" else not (train_dir / "mesh.log").exists()),
            "aggregated_metrics_finite": metric_ok,
        }
        scene_check["valid"] = (
            train_rc == 0
            and render_rc == 0
            and scene_check["completion_marker"]
            and all(scene_check["checkpoint_artifacts"].values())
            and loss_ok
            and scene_check["test_render_count"] == expected_test
            and scene_check["train_render_count"] == expected_train
            and scene_check["test_normal_count"] == expected_test
            and scene_check["train_normal_count"] == expected_train
            and scene_check["test_metric_exists"]
            and scene_check["train_metric_exists"]
            and scene_check["fuse_mesh_exists"]
            and scene_check["fuse_post_mesh_exists"]
            and scene_check["mesh_log_rule_valid"]
            and metric_ok
        )
        if not scene_check["valid"]:
            errors.append(key)
        checks.append(scene_check)

    output = {
        "classification": "PAPER-DESCRIPTION NO-GEOMETRY-PRIOR EXPERIMENT — NOT UNMODIFIED OFFICIAL CODE",
        "scene_count": len(checks),
        "valid_scene_count": sum(check["valid"] for check in checks),
        "all_valid": not errors and len(checks) == 14,
        "invalid_scenes": errors,
        "checks": checks,
    }
    (root / "exit_codes.json").write_text(json.dumps(exit_codes, indent=2, sort_keys=True) + "\n")
    (root / "verification_results.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: output[key] for key in ("scene_count", "valid_scene_count", "all_valid", "invalid_scenes")}, indent=2))
    if not output["all_valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
