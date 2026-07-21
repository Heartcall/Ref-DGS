#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


LABEL = "NON-CANONICAL VGGT-REGENERATED PRIORS — NOT A STRICT PAPER REPRODUCTION"


def time_value(log_text: str, key: str) -> str:
    match = re.search(rf"^\s*{re.escape(key)}:\s*(.+)$", log_text, flags=re.MULTILINE)
    return match.group(1).strip() if match else ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    result_paths = sorted((args.run_root / "logs" / "smoke").glob("*/*/result.json"))
    if len(result_paths) != 14:
        raise SystemExit(f"expected 14 smoke results, found {len(result_paths)}")
    rows = []
    exit_codes = {"label": LABEL, "scenes": {}}
    for path in result_paths:
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["status"] != "pass":
            raise SystemExit(f"failed smoke result: {result['dataset']}/{result['scene']}")
        records = result["diagnostic_records"]
        scene_log_dir = path.parent
        gpu_values = []
        with (scene_log_dir / "gpu_memory.csv").open(encoding="utf-8") as handle:
            next(handle)
            for line in handle:
                fields = [field.strip() for field in line.split(",")]
                if len(fields) >= 2:
                    gpu_values.append(int(fields[1]))
        log_text = (scene_log_dir / "train_i2.log").read_text(encoding="utf-8", errors="replace")
        row = {
            "label": LABEL,
            "dataset": result["dataset"],
            "scene": result["scene"],
            "status": result["status"],
            "exit_code": result["exit_code"],
            "iterations_seen": ",".join(str(record["iteration"]) for record in records),
            "prior_enabled_all": all(record["prior_enabled"] for record in records),
            "iteration_1_total_loss": records[0]["loss_total"],
            "iteration_1_vggt_depth_loss": records[0]["vggt_depth_loss"],
            "iteration_1_vggt_normal_loss": records[0]["vggt_normal_loss"],
            "iteration_2_total_loss": records[1]["loss_total"],
            "iteration_2_vggt_depth_loss": records[1]["vggt_depth_loss"],
            "iteration_2_vggt_normal_loss": records[1]["vggt_normal_loss"],
            "saved_artifacts_all": all(result["artifacts"].values()),
            "peak_gpu_memory_mib": max(gpu_values) if gpu_values else "",
            "max_host_rss_kib": time_value(log_text, "Maximum resident set size (kbytes)"),
            "elapsed_wall_clock": time_value(log_text, "Elapsed (wall clock) time (h:mm:ss or m:ss)"),
        }
        rows.append(row)
        exit_codes["scenes"][f"{result['dataset']}/{result['scene']}"] = result["exit_code"]
    with (args.run_root / "smoke_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.run_root / "smoke_exit_codes.json").write_text(
        json.dumps(exit_codes, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "pass", "scene_count": len(rows)}))


if __name__ == "__main__":
    main()
