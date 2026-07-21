#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


LABEL = "NON-CANONICAL VGGT-REGENERATED PRIORS — NOT A STRICT PAPER REPRODUCTION"
REQUIRED_ARTIFACTS = (
    "gaussians_point_cloud.ply",
    "ref_gaussians_point_cloud.ply",
    "light_mlp.pt",
    "dir_encoding.pt",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--exit-code", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = []
    decoder = json.JSONDecoder()
    for line in args.log.read_text(encoding="utf-8", errors="replace").splitlines():
        if "[DIAGNOSTIC_PRIOR] " in line:
            payload_text = line.split("[DIAGNOSTIC_PRIOR] ", 1)[1].lstrip()
            record, _ = decoder.raw_decode(payload_text)
            records.append(record)
    errors = []
    if args.exit_code != 0:
        errors.append(f"training exit code {args.exit_code}")
    if [record.get("iteration") for record in records] != [1, 2]:
        errors.append(f"expected diagnostic iterations [1, 2], got {[r.get('iteration') for r in records]}")
    for record in records:
        if record.get("prior_enabled") is not True:
            errors.append(f"prior not enabled at iteration {record.get('iteration')}")
        for key in ("loss_total", "vggt_depth_loss", "vggt_normal_loss"):
            value = record.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                errors.append(f"{key} is not finite at iteration {record.get('iteration')}: {value}")
    artifact_root = args.model_path / "point_cloud" / "iteration_2"
    artifacts = {name: (artifact_root / name).is_file() for name in REQUIRED_ARTIFACTS}
    for name, present in artifacts.items():
        if not present:
            errors.append(f"missing saved artifact: {name}")
    if "CUDA out of memory" in args.log.read_text(encoding="utf-8", errors="replace"):
        errors.append("CUDA OOM in log")

    result = {
        "label": LABEL,
        "dataset": args.dataset,
        "scene": args.scene,
        "status": "pass" if not errors else "failed",
        "exit_code": args.exit_code,
        "diagnostic_records": records,
        "artifacts": artifacts,
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "scene": args.scene}))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
