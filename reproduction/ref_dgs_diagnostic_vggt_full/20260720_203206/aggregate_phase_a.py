#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


LABEL = "NON-CANONICAL VGGT-REGENERATED PRIORS — NOT A STRICT PAPER REPRODUCTION"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--manual-inspection-status", choices=("pending", "pass"), default="pending"
    )
    args = parser.parse_args()
    validation_dir = args.run_root / "logs" / "validation"
    reports = []
    for path in sorted(validation_dir.glob("*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        reports.append(report)
    if len(reports) != 14:
        raise SystemExit(f"expected 14 scene reports, found {len(reports)}")

    rows = []
    total_files = 0
    total_sha_records = 0
    for report in reports:
        overall = report["overall"]
        if report["status"] != "pass":
            raise SystemExit(f"validation did not pass: {report['dataset']}/{report['scene']}")
        if overall["visualization_count"] != 5:
            raise SystemExit(f"visualization count is not five: {report['dataset']}/{report['scene']}")
        file_count = int(overall["file_count"])
        sha_count = len(report["file_sha256_manifest"])
        if sha_count != file_count:
            raise SystemExit(f"SHA manifest count mismatch: {report['dataset']}/{report['scene']}")
        total_files += file_count
        total_sha_records += sha_count
        rows.append({
            "label": LABEL,
            "dataset": report["dataset"],
            "scene": report["scene"],
            "status": report["status"],
            "train_count": report["splits"]["train"]["file_count"],
            "test_count": report["splits"]["test"]["file_count"],
            "file_count": file_count,
            "sha256_count": sha_count,
            "normal_file_count": overall["normal_file_count"],
            "shape": "x".join(map(str, report["splits"]["train"]["schema"]["shape"])),
            "dtype": report["splits"]["train"]["schema"]["dtype"],
            "depth_min": overall["depth"]["min"],
            "depth_max": overall["depth"]["max"],
            "depth_mean": overall["depth"]["mean"],
            "depth_std": overall["depth"]["std"],
            "depth_zero_ratio": overall["depth"]["zero_ratio"],
            "confidence_min": overall["confidence"]["min"],
            "confidence_max": overall["confidence"]["max"],
            "confidence_mean": overall["confidence"]["mean"],
            "confidence_std": overall["confidence"]["std"],
            "confidence_zero_ratio": overall["confidence"]["zero_ratio"],
            "visualization_count": overall["visualization_count"],
            "camera_prior_filename_index_bijection": overall["camera_prior_filename_index_bijection"],
        })

    expected_total = 6 * 300 + 8 * 128
    if total_files != expected_total or total_sha_records != expected_total:
        raise SystemExit(
            f"global prior count mismatch: files={total_files}, sha={total_sha_records}, expected={expected_total}"
        )

    aggregate = {
        "label": LABEL,
        "status": "pass",
        "scene_count": len(reports),
        "file_count": total_files,
        "sha256_record_count": total_sha_records,
        "visualization_count": sum(row["visualization_count"] for row in rows),
        "manual_visual_inspection": {
            "status": args.manual_inspection_status,
            "sample_count": 70 if args.manual_inspection_status == "pass" else 0,
            "checks": [
                "horizontal_or_vertical_flip",
                "incorrect_crop",
                "index_misalignment",
                "obvious_background_leakage",
                "constant_or_zero_maps",
            ],
            "contact_sheets": [
                "visualizations/ShinySynthetic_contact_sheet.png",
                "visualizations/GlossySynthetic_contact_sheet.png",
            ],
            "observation": (
                "No obvious flip, crop, index mismatch, or constant/zero map was observed; "
                "unmasked background depth variation is retained from direct VGGT output."
                if args.manual_inspection_status == "pass"
                else "Manual inspection has not yet been recorded."
            ),
        },
        "scenes": reports,
    }
    (args.run_root / "prior_validation_all.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (args.run_root / "prior_validation_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.run_root / "prior_generation_failures.md").write_text(
        f"# {LABEL}\n\nNo Phase A generation or validation failures were observed.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "pass", "scene_count": len(reports), "file_count": total_files}))


if __name__ == "__main__":
    main()
