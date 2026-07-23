#!/usr/bin/env python3
"""Read-only reconciliation of completed Ref-DGS experiment artifacts.

This script never launches training, inference, rendering, or mesh generation.
It reads frozen result files and writes only to a new comparison output root.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean


NO_ROOT = Path("/data1/liuly/reproduction/ref_dgs_paper_description_no_prior/20260722_011114")
VGGT_ROOT = Path("/data1/liuly/reproduction/ref_dgs_diagnostic_vggt_full/20260720_203206_resume")
CANONICAL_ROOT = Path("/data1/liuly/reproduction/ref_dgs_main_tables/20260720_010227")

CONFIGS = {
    "paper_reported": "PAPER-REPORTED REFERENCE",
    "no_external_geometry_prior": "PAPER-DESCRIPTION NO-GEOMETRY-PRIOR ABLATION",
    "regenerated_vggt_prior": "NON-CANONICAL VGGT-REGENERATED PRIORS",
}

SCENES = {
    "ShinySynthetic": {
        "ball": 15000,
        "car": 25000,
        "coffee": 15000,
        "helmet": 20000,
        "teapot": 20000,
        "toaster": 20000,
    },
    "GlossySynthetic": {
        "angel": 25000,
        "bell": 25000,
        "cat": 25000,
        "horse": 25000,
        "luyu": 25000,
        "potion": 25000,
        "tbell": 25000,
        "teapot": 25000,
    },
}

PAPER = {
    "ShinySynthetic": {
        "psnr": 35.21,
        "ssim": 0.975,
        "lpips": 0.053,
        "normal_mae_deg": {
            "ball": 0.61,
            "car": 1.72,
            "coffee": 1.86,
            "helmet": 1.35,
            "teapot": 0.58,
            "toaster": 2.43,
        },
        "reported_normal_mae_deg": 1.43,
        "reported_cd_x100": None,
        "fps": 76.34,
        "train_time_min": 12.6,
    },
    "GlossySynthetic": {
        "psnr": 30.63,
        "ssim": 0.958,
        "lpips": 0.052,
        "normal_mae_deg": {
            "angel": 2.05,
            "bell": 0.71,
            "cat": 1.37,
            "horse": 3.44,
            "luyu": 2.30,
            "potion": 2.48,
            "tbell": 1.84,
            "teapot": 0.86,
        },
        "cd_x100": {
            "angel": 0.38,
            "bell": 0.65,
            "cat": 0.93,
            "horse": 0.42,
            "luyu": 0.64,
            "potion": 0.66,
            "tbell": 0.52,
            "teapot": 0.79,
        },
        "reported_normal_mae_deg": 1.88,
        "reported_cd_x100": 0.62,
        "fps": None,
        "train_time_min": None,
    },
}

METRIC_RE = re.compile(
    r"psnr:(?P<psnr>[-+0-9.eE]+),\s*ssim:(?P<ssim>[-+0-9.eE]+),\s*"
    r"lpips:(?P<lpips>[-+0-9.eE]+),\s*fps:(?P<fps>[-+0-9.eE]+)\s*"
    r"mae:(?P<normal_mae_deg>[-+0-9.eE]+)"
)

UNIFIED_FIELDS = [
    "dataset",
    "scene",
    "configuration",
    "result_status",
    "psnr",
    "ssim",
    "lpips",
    "normal_mae_deg",
    "cd_x100",
    "fps",
    "train_time_min",
    "train_exit_code",
    "render_exit_code",
    "metrics_exit_code",
    "mesh_exit_code",
    "cd_exit_code",
    "gpu",
    "seed",
    "source_file",
    "auxiliary_source_files",
    "iterations",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_finite(value: object) -> bool:
    return value is None or (isinstance(value, (int, float)) and math.isfinite(float(value)))


def close(a: float, b: float, tolerance: float = 1e-12) -> bool:
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tolerance)


def parse_metric(path: Path) -> dict[str, float]:
    match = METRIC_RE.search(path.read_text())
    if not match:
        raise ValueError(f"Unrecognized metric format: {path}")
    values = {key: float(value) for key, value in match.groupdict().items()}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError(f"Non-finite metric: {path}")
    return values


def parse_cd_x100(path: Path) -> tuple[float, float]:
    fields = path.read_text().strip().split()
    if len(fields) != 2:
        raise ValueError(f"Unrecognized mesh log: {path}")
    raw = float(fields[1])
    scaled = raw * 100.0
    if not math.isfinite(raw) or not math.isfinite(scaled):
        raise ValueError(f"Non-finite CD: {path}")
    return raw, scaled


def parse_elapsed_minutes(path: Path) -> float:
    text = path.read_text(errors="replace")
    matches = re.findall(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*([^\s]+)", text)
    if not matches:
        raise ValueError(f"Elapsed time missing: {path}")
    parts = [float(part) for part in matches[-1].split(":")]
    if len(parts) == 2:
        seconds = parts[0] * 60.0 + parts[1]
    elif len(parts) == 3:
        seconds = parts[0] * 3600.0 + parts[1] * 60.0 + parts[2]
    else:
        raise ValueError(f"Bad elapsed time: {matches[-1]} in {path}")
    return seconds / 60.0


def read_int(path: Path) -> int:
    return int(path.read_text().strip())


def scene_paths(root: Path, dataset: str, scene: str, iterations: int) -> dict[str, Path]:
    attempt = root / "full" / dataset / scene / "attempt_001"
    model = attempt / "model"
    logs = root / "logs" / "full" / dataset / scene / "attempt_001"
    return {
        "attempt": attempt,
        "model": model,
        "metric": model / "test" / f"ours_{iterations}" / "metric.txt",
        "mesh_log": model / "train" / f"ours_{iterations}" / "mesh.log",
        "fuse": model / "train" / f"ours_{iterations}" / "fuse.ply",
        "fuse_post": model / "train" / f"ours_{iterations}" / "fuse_post.ply",
        "checkpoint": model / "point_cloud" / f"iteration_{iterations}",
        "train_log": logs / "train.log",
        "train_exit": logs / "train_exit_code.txt",
        "render_exit": logs / "render_exit_code.txt",
        "frozen_identity": attempt / "frozen_identity.txt",
        "complete": attempt / "scene_complete.ok",
    }


def required_input_files() -> dict[Path, str]:
    files: dict[Path, str] = {
        NO_ROOT / "results_no_prior.csv": "no-prior existing summary CSV",
        NO_ROOT / "results_no_prior.json": "no-prior existing summary JSON",
        NO_ROOT / "exit_codes.json": "no-prior aggregate exit-code summary",
        NO_ROOT / "verification_results.json": "no-prior completion verification",
        NO_ROOT / "environment.txt": "no-prior environment",
        NO_ROOT / "runtime/Ref-DGS/utils/general_utils.py": "no-prior seed implementation",
        VGGT_ROOT / "results_diagnostic.csv": "VGGT stale root summary CSV (audited, not used for metrics)",
        VGGT_ROOT / "results_diagnostic.json": "VGGT stale root summary JSON (audited, not used for metrics)",
        VGGT_ROOT / "exit_codes.json": "VGGT stale root exit summary (audited, not used for scene exits)",
        VGGT_ROOT / "environment.txt": "VGGT environment and checkpoint identity",
        VGGT_ROOT / "vggt_generation_config.json": "VGGT generation manifest",
        VGGT_ROOT / "prior_validation_all.json": "VGGT full prior validation",
        VGGT_ROOT / "prior_validation_summary.csv": "VGGT prior validation scene summary",
        VGGT_ROOT / "diagnostic_full_report.md": "VGGT stale root report (audited limitation)",
        VGGT_ROOT / "failure_evidence.md": "VGGT failure and retry evidence",
        VGGT_ROOT / "runtime/Ref-DGS/utils/general_utils.py": "VGGT seed implementation",
        CANONICAL_ROOT / "results.csv": "canonical blocked summary CSV",
        CANONICAL_ROOT / "results.json": "canonical blocker and paper reference JSON",
        CANONICAL_ROOT / "environment.txt": "canonical environment",
        CANONICAL_ROOT / "reproduction_report.md": "canonical blocker report",
    }
    for dataset, scene_map in SCENES.items():
        for scene, iterations in scene_map.items():
            manifest = VGGT_ROOT / "scene_manifests" / f"{dataset}_{scene}.json"
            files[manifest] = "VGGT scene manifest"
            for root, label in ((NO_ROOT, "no-prior"), (VGGT_ROOT, "VGGT")):
                paths = scene_paths(root, dataset, scene, iterations)
                for name in ("metric", "train_log", "train_exit", "render_exit", "frozen_identity", "complete"):
                    files[paths[name]] = f"{label} formal scene {name}"
                if dataset == "GlossySynthetic":
                    files[paths["mesh_log"]] = f"{label} formal scene raw CD"
    return files


def profile_file(path: Path, purpose: str) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    columns = "text"
    data_rows = 0
    if suffix == ".csv":
        with path.open(newline="") as handle:
            reader = csv.reader(handle)
            rows = [row for row in reader if any(cell != "" for cell in row)]
        columns = ",".join(rows[0]) if rows else ""
        data_rows = max(0, len(rows) - 1)
    elif suffix == ".json":
        payload = json.loads(path.read_text())
        if isinstance(payload, list):
            data_rows = len(payload)
            columns = ",".join(payload[0].keys()) if payload and isinstance(payload[0], dict) else "value"
        elif isinstance(payload, dict):
            if isinstance(payload.get("rows"), list):
                values = payload["rows"]
                data_rows = len(values)
                columns = ",".join(values[0].keys()) if values else ",".join(payload.keys())
            elif isinstance(payload.get("metrics"), list):
                values = payload["metrics"]
                data_rows = len(values)
                columns = ",".join(values[0].keys()) if values else ",".join(payload.keys())
            elif isinstance(payload.get("checks"), list):
                values = payload["checks"]
                data_rows = len(values)
                columns = ",".join(values[0].keys()) if values else ",".join(payload.keys())
            else:
                data_rows = len(payload)
                columns = ",".join(payload.keys())
        else:
            data_rows = 1
            columns = "value"
    else:
        text = path.read_text(errors="replace")
        data_rows = len(text.splitlines())
    return {
        "absolute_path": str(path.resolve()),
        "purpose": purpose,
        "columns": columns,
        "data_rows": data_rows,
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: "—" if row.get(field) is None else row.get(field) for field in fields})


def collect_experiment_scene(
    root: Path,
    dataset: str,
    scene: str,
    iterations: int,
    configuration: str,
    environment_path: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    paths = scene_paths(root, dataset, scene, iterations)
    required = [
        paths["metric"], paths["train_log"], paths["train_exit"], paths["render_exit"],
        paths["frozen_identity"], paths["complete"], paths["fuse"], paths["fuse_post"],
    ]
    required.extend(paths["checkpoint"] / name for name in (
        "gaussians_point_cloud.ply", "ref_gaussians_point_cloud.ply", "light_mlp.pt", "dir_encoding.pt"
    ))
    if dataset == "GlossySynthetic":
        required.append(paths["mesh_log"])
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError("; ".join(missing))
    metric = parse_metric(paths["metric"])
    train_rc = read_int(paths["train_exit"])
    render_rc = read_int(paths["render_exit"])
    identity = paths["frozen_identity"].read_text()
    identity_iteration_match = f"iterations={iterations}" in identity
    complete_label = paths["complete"].read_text().strip()
    cd_raw = cd_scaled = None
    if dataset == "GlossySynthetic":
        cd_raw, cd_scaled = parse_cd_x100(paths["mesh_log"])
    auxiliary = [
        paths["train_log"], paths["train_exit"], paths["render_exit"],
        paths["frozen_identity"], paths["complete"], environment_path,
    ]
    if dataset == "GlossySynthetic":
        auxiliary.append(paths["mesh_log"])
    row = {
        "dataset": dataset,
        "scene": scene,
        "configuration": configuration,
        "result_status": CONFIGS[configuration],
        **metric,
        "cd_x100": cd_scaled,
        "fps": metric["fps"],
        "train_time_min": parse_elapsed_minutes(paths["train_log"]),
        "train_exit_code": train_rc,
        "render_exit_code": render_rc,
        # render.py is the single process that renders, computes image/normal metrics,
        # fuses the mesh, and runs Glossy CD. The stage values therefore inherit its
        # exit code only when the corresponding required artifact is present.
        "metrics_exit_code": render_rc,
        "mesh_exit_code": render_rc,
        "cd_exit_code": render_rc if dataset == "GlossySynthetic" else None,
        "gpu": "NVIDIA RTX A5000 24 GB (physical GPU 1)",
        "seed": 0,
        "source_file": str(paths["metric"]),
        "auxiliary_source_files": "|".join(str(path) for path in auxiliary),
        "iterations": iterations,
    }
    validation = {
        "dataset": dataset,
        "scene": scene,
        "configuration": configuration,
        "expected_iterations": iterations,
        "identity_iteration_match": identity_iteration_match,
        "formal_attempt_path": str(paths["attempt"]),
        "source_is_not_smoke": "/smoke/" not in str(paths["metric"]),
        "completion_marker": complete_label,
        "checkpoint_artifacts_complete": True,
        "test_metric_finite": all(math.isfinite(value) for value in metric.values()),
        "train_exit_code": train_rc,
        "render_exit_code": render_rc,
        "mesh_outputs_present": paths["fuse"].is_file() and paths["fuse_post"].is_file(),
        "cd_raw": cd_raw,
        "cd_x100": cd_scaled,
        "cd_unit_rule_valid": dataset != "GlossySynthetic" or close(cd_scaled, cd_raw * 100.0),
        "valid": identity_iteration_match and train_rc == 0 and render_rc == 0,
    }
    return row, validation


def paper_rows(canonical_source: Path) -> list[dict[str, object]]:
    rows = []
    for dataset, scene_map in SCENES.items():
        for scene, iterations in scene_map.items():
            rows.append({
                "dataset": dataset,
                "scene": scene,
                "configuration": "paper_reported",
                "result_status": CONFIGS["paper_reported"],
                "psnr": None,
                "ssim": None,
                "lpips": None,
                "normal_mae_deg": PAPER[dataset]["normal_mae_deg"][scene],
                "cd_x100": PAPER[dataset].get("cd_x100", {}).get(scene),
                "fps": None,
                "train_time_min": None,
                "train_exit_code": None,
                "render_exit_code": None,
                "metrics_exit_code": None,
                "mesh_exit_code": None,
                "cd_exit_code": None,
                "gpu": "NVIDIA RTX 4090 24 GB (paper hardware)",
                "seed": None,
                "source_file": str(canonical_source),
                "auxiliary_source_files": "",
                "iterations": None,
            })
    return rows


def recompute_dataset_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    for dataset, scene_map in SCENES.items():
        expected = len(scene_map)
        for configuration in CONFIGS:
            subset = [row for row in rows if row["dataset"] == dataset and row["configuration"] == configuration]
            scenes = {str(row["scene"]) for row in subset}
            missing = sorted(set(scene_map) - scenes)
            if configuration == "paper_reported":
                psnr = PAPER[dataset]["psnr"]
                ssim = PAPER[dataset]["ssim"]
                lpips = PAPER[dataset]["lpips"]
                normal_value = PAPER[dataset]["reported_normal_mae_deg"]
                cd_value = PAPER[dataset]["reported_cd_x100"]
                fps = PAPER[dataset]["fps"]
                train_time = PAPER[dataset]["train_time_min"]
                basis = "paper dataset values; geometry scene mean independently recomputed"
            else:
                psnr = mean(float(row["psnr"]) for row in subset)
                ssim = mean(float(row["ssim"]) for row in subset)
                lpips = mean(float(row["lpips"]) for row in subset)
                normal_value = mean(float(row["normal_mae_deg"]) for row in subset)
                cd_value = mean(float(row["cd_x100"]) for row in subset) if dataset == "GlossySynthetic" else None
                fps = mean(float(row["fps"]) for row in subset)
                train_time = mean(float(row["train_time_min"]) for row in subset)
                basis = "unweighted arithmetic mean recomputed from formal scene rows"
            scene_recomputed_normal = mean(float(row["normal_mae_deg"]) for row in subset)
            scene_recomputed_cd = (
                mean(float(row["cd_x100"]) for row in subset)
                if dataset == "GlossySynthetic" else None
            )
            paper = PAPER[dataset]
            output.append({
                "dataset": dataset,
                "configuration": configuration,
                "result_status": CONFIGS[configuration],
                "valid_scene_count": len(scenes),
                "expected_scene_count": expected,
                "missing_scene_count": len(missing),
                "missing_scenes": "|".join(missing) if missing else None,
                "psnr_↑": psnr,
                "delta_paper_psnr": None if psnr is None else psnr - paper["psnr"],
                "ssim_↑": ssim,
                "delta_paper_ssim": None if ssim is None else ssim - paper["ssim"],
                "lpips_↓": lpips,
                "delta_paper_lpips": None if lpips is None else lpips - paper["lpips"],
                "normal_mae_deg_↓": normal_value,
                "delta_paper_normal_mae_deg": normal_value - paper["reported_normal_mae_deg"],
                "scene_recomputed_normal_mae_deg": scene_recomputed_normal,
                "paper_reported_normal_mae_deg": paper["reported_normal_mae_deg"],
                "cd_x100_↓": cd_value,
                "delta_paper_cd_x100": None if cd_value is None else cd_value - paper["reported_cd_x100"],
                "scene_recomputed_cd_x100": scene_recomputed_cd,
                "paper_reported_cd_x100": paper["reported_cd_x100"],
                "fps_↑": fps,
                "delta_paper_fps": None if fps is None or paper["fps"] is None else fps - paper["fps"],
                "train_time_min_↓": train_time,
                "delta_paper_train_time_min": (
                    None if train_time is None or paper["train_time_min"] is None
                    else train_time - paper["train_time_min"]
                ),
                "gpu": (
                    "NVIDIA RTX 4090 24 GB (paper hardware)"
                    if configuration == "paper_reported"
                    else "NVIDIA RTX A5000 24 GB (physical GPU 1)"
                ),
                "mean_basis": basis,
            })
    return output


def geometry_comparison(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    indexed = {(row["dataset"], row["scene"], row["configuration"]): row for row in rows}
    output = []
    for dataset, scene_map in SCENES.items():
        for scene in scene_map:
            paper = indexed[(dataset, scene, "paper_reported")]
            no_prior = indexed[(dataset, scene, "no_external_geometry_prior")]
            vggt = indexed[(dataset, scene, "regenerated_vggt_prior")]
            output.append({
                "dataset": dataset,
                "scene": scene,
                "paper_normal_mae_deg": paper["normal_mae_deg"],
                "no_prior_normal_mae_deg": no_prior["normal_mae_deg"],
                "no_prior_delta_paper_normal_mae_deg": no_prior["normal_mae_deg"] - paper["normal_mae_deg"],
                "regenerated_vggt_normal_mae_deg": vggt["normal_mae_deg"],
                "regenerated_vggt_delta_paper_normal_mae_deg": vggt["normal_mae_deg"] - paper["normal_mae_deg"],
                "paper_cd_x100": paper["cd_x100"],
                "no_prior_cd_x100": no_prior["cd_x100"],
                "no_prior_delta_paper_cd_x100": (
                    None if paper["cd_x100"] is None else no_prior["cd_x100"] - paper["cd_x100"]
                ),
                "regenerated_vggt_cd_x100": vggt["cd_x100"],
                "regenerated_vggt_delta_paper_cd_x100": (
                    None if paper["cd_x100"] is None else vggt["cd_x100"] - paper["cd_x100"]
                ),
            })
    return output


def efficiency_summary(dataset_summary: list[dict[str, object]]) -> list[dict[str, object]]:
    env = {
        "paper_reported": str(CANONICAL_ROOT / "environment.txt"),
        "no_external_geometry_prior": str(NO_ROOT / "environment.txt"),
        "regenerated_vggt_prior": str(VGGT_ROOT / "environment.txt"),
    }
    software = {
        "paper_reported": "paper reports RTX 4090 24 GB; software stack not fully reported",
        "no_external_geometry_prior": "Python 3.11.15; PyTorch 2.5.1+cu118; driver 535.247.01; CUDA runtime 11.8",
        "regenerated_vggt_prior": "Python 3.11.15; PyTorch 2.5.1+cu118; driver 535.247.01; CUDA runtime 11.8",
    }
    output = []
    for row in dataset_summary:
        output.append({
            "dataset": row["dataset"],
            "configuration": row["configuration"],
            "result_status": row["result_status"],
            "fps_↑": row["fps_↑"],
            "delta_paper_fps": row["delta_paper_fps"],
            "train_time_min_↓": row["train_time_min_↓"],
            "delta_paper_train_time_min": row["delta_paper_train_time_min"],
            "gpu": row["gpu"],
            "software_environment": software[row["configuration"]],
            "environment_file": env[row["configuration"]],
            "hardware_comparability_note": (
                "Reference only; paper and experiments use different GPU models. Do not attribute the speed gap to the algorithm."
                if row["configuration"] != "paper_reported" else
                "Paper hardware reference; no scene-level timing or software stack was published."
            ),
        })
    return output


def fmt(value: object, digits: int) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def render_markdown(
    summary: list[dict[str, object]],
    geometry: list[dict[str, object]],
    validation: dict[str, object],
) -> str:
    lines = [
        "# Ref-DGS existing-results comparison",
        "",
        "> **READ-ONLY AGGREGATION — NO NEW TRAINING, VGGT INFERENCE, RENDER, OR MESH GENERATION**",
        "",
        "## Technical summary",
        "",
        "The strict canonical paper reproduction remains incomplete because all 1,828 downloaded official depth `.pth` files are unreadable even though their SHA-256 values match the upstream LFS objects. This report therefore keeps three evidence classes separate: paper-reported references, a newly added no-external-prior ablation, and a non-canonical public-VGGT diagnostic.",
        "",
        "The current regenerated VGGT diagnostic is close to the paper on ShinySynthetic but degrades substantially on GlossySynthetic. The no-prior ablation improves over regenerated VGGT on GlossySynthetic but does not reproduce the paper geometry means. These observations do not establish that geometry priors are generally effective or ineffective; they show sensitivity to prior quality and data domain.",
        "",
        "## Dataset-level comparison",
        "",
        "All experimental means below are newly recomputed as unweighted arithmetic means over the 6 Shiny or 8 Glossy formal scene rows. Paper NVS, FPS, and Shiny training time are dataset-level reported references; paper geometry scene means are also recomputed separately to audit publication rounding.",
        "",
        "| Dataset | Configuration | Scenes | PSNR ↑ | Δpaper | SSIM ↑ | Δpaper | LPIPS ↓ | Δpaper | Normal MAE ↓ | Δpaper | CD×10² ↓ | Δpaper | FPS ↑ | Train min ↓ |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "paper_reported": "Paper reported",
        "no_external_geometry_prior": "No external geometry prior",
        "regenerated_vggt_prior": "Regenerated VGGT prior",
    }
    for row in summary:
        lines.append(
            f"| {row['dataset']} | {labels[row['configuration']]} | {row['valid_scene_count']}/{row['expected_scene_count']} | "
            f"{fmt(row['psnr_↑'],3)} | {fmt(row['delta_paper_psnr'],3)} | {fmt(row['ssim_↑'],4)} | {fmt(row['delta_paper_ssim'],4)} | "
            f"{fmt(row['lpips_↓'],4)} | {fmt(row['delta_paper_lpips'],4)} | {fmt(row['normal_mae_deg_↓'],3)} | "
            f"{fmt(row['delta_paper_normal_mae_deg'],3)} | {fmt(row['cd_x100_↓'],3)} | {fmt(row['delta_paper_cd_x100'],3)} | "
            f"{fmt(row['fps_↑'],2)} | {fmt(row['train_time_min_↓'],3)} |"
        )
    lines.extend([
        "",
        "`Δpaper = experiment value − paper-reported value`. A positive delta is favorable only for ↑ metrics; it is unfavorable for ↓ metrics. Missing paper metrics remain `—` rather than zero.",
        "",
        "## Scene-level geometry",
        "",
        "| Dataset | Scene | Paper MAE | No-prior MAE | Δpaper | VGGT MAE | Δpaper | Paper CD×10² | No-prior CD×10² | Δpaper | VGGT CD×10² | Δpaper |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in geometry:
        lines.append(
            f"| {row['dataset']} | {row['scene']} | {fmt(row['paper_normal_mae_deg'],3)} | "
            f"{fmt(row['no_prior_normal_mae_deg'],3)} | {fmt(row['no_prior_delta_paper_normal_mae_deg'],3)} | "
            f"{fmt(row['regenerated_vggt_normal_mae_deg'],3)} | {fmt(row['regenerated_vggt_delta_paper_normal_mae_deg'],3)} | "
            f"{fmt(row['paper_cd_x100'],3)} | {fmt(row['no_prior_cd_x100'],3)} | {fmt(row['no_prior_delta_paper_cd_x100'],3)} | "
            f"{fmt(row['regenerated_vggt_cd_x100'],3)} | {fmt(row['regenerated_vggt_delta_paper_cd_x100'],3)} |"
        )
    lines.extend([
        "",
        "## Scope, data, and metric definitions",
        "",
        "- Population: six ShinySynthetic and eight GlossySynthetic scenes from the formal `full/.../attempt_001` outputs only.",
        "- Excluded: every `smoke/` result, NeRF Synthetic, RefReal, GlossyReal, baselines, and ablations other than the explicitly named no-prior experiment.",
        "- PSNR/SSIM/LPIPS/FPS/normal MAE come from each formal test `metric.txt` without replacing full-precision values with rounded log summaries.",
        "- Glossy CD×10² is the raw `mesh.log` value multiplied by exactly 100. Shiny CD is not applicable.",
        "- FPS and training time are hardware-dependent. The experiments used RTX A5000 24 GB; the paper used RTX 4090 24 GB.",
        "",
        "## Methodology and source reconciliation",
        "",
        "Each experiment scene was reconstructed from its formal test metric, `/usr/bin/time -v` training log, per-scene train/render exit files, frozen identity, completion marker, final checkpoint artifacts, and Glossy mesh log. The released `render.py` is monolithic: render, image/normal metrics, mesh fusion, and Glossy CD share one process exit code. Stage exit columns therefore inherit that code only when the corresponding artifact exists.",
        "",
        "The no-prior CSV/JSON agrees with the newly parsed raw scene files. The VGGT root `results_diagnostic.csv/json`, `exit_codes.json`, and `diagnostic_full_report.md` are stale pre-resume resource-gate controls and contain no final metrics; they are retained as an audited conflict, not used as metric sources. Final VGGT numbers are reconstructed from `full/` and independently agree with the `prior_*` fields embedded in the later no-prior result JSON.",
        "",
        "## Limitations, uncertainty, and robustness checks",
        "",
        "- Official released depth priors are corrupted, so strict canonical reproduction remains blocked.",
        "- The no-prior run is a new paper-description ablation, not the original Table 1–4 configuration.",
        "- Regenerated VGGT uses public VGGT, deterministic groups of 20, and an unpublished-author-grouping mismatch; it cannot stand in for the authors' priors.",
        "- Paper NVS values are dataset-level only, so scene-level paper PSNR/SSIM/LPIPS cannot be populated.",
        "- The paper does not report Glossy FPS/training time or the full software stack; those cells remain missing.",
        f"- Validation assessment: `{validation['overall_assessment']}`; duplicate rows={validation['duplicate_key_count']}, non-finite values={validation['nonfinite_value_count']}.",
        "",
        "## Recommended next steps",
        "",
        "Keep these configurations as separate evidence classes. Do not replace the canonical blocker, and do not use the comparison to make a general causal claim about geometry priors. The evidence needed to resolve provenance is a valid original-prior release or an author statement documenting the exact paper-training configuration and prior-generation protocol.",
        "",
        "## Further questions",
        "",
        "Which prior source, view grouping, and confidence definition produced the publication tables, and did the released prior loss enter the submitted experiments despite being omitted from the paper description?",
        "",
    ])
    return "\n".join(lines)


def artifact_payload(
    summary: list[dict[str, object]],
    geometry: list[dict[str, object]],
    efficiency: list[dict[str, object]],
) -> dict[str, object]:
    generated = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    source = {
        "id": "unified_results",
        "label": "Read-only unified Ref-DGS comparison",
        "path": "unified_results.csv",
        "query": {
            "engine": "DuckDB",
            "language": "sql",
            "sql": "SELECT * FROM read_csv_auto('unified_results.csv', nullstr='—');",
            "description": "Reads the full-precision unified scene table reconstructed from formal result artifacts.",
            "tables_used": ["unified_results.csv"],
            "filters": ["formal full/.../attempt_001 only", "exclude all smoke paths"],
            "metric_definitions": {
                "dataset_mean": "Unweighted arithmetic mean over formal scene rows.",
                "delta_paper": "Experiment value minus paper-reported value.",
                "cd_x100": "Raw mesh.log Chamfer distance multiplied by 100.",
            },
        },
    }
    label_map = {
        "paper_reported": "Paper reported",
        "no_external_geometry_prior": "No external geometry prior",
        "regenerated_vggt_prior": "Regenerated VGGT prior",
    }
    chart_rows = []
    for row in geometry:
        for field, config in (
            ("paper_normal_mae_deg", "Paper reported"),
            ("no_prior_normal_mae_deg", "No external geometry prior"),
            ("regenerated_vggt_normal_mae_deg", "Regenerated VGGT prior"),
        ):
            chart_rows.append({"dataset": row["dataset"], "scene": row["scene"], "configuration": config, "normal_mae_deg": row[field]})
    cd_chart_rows = []
    for row in geometry:
        if row["dataset"] != "GlossySynthetic":
            continue
        for field, config in (
            ("paper_cd_x100", "Paper reported"),
            ("no_prior_cd_x100", "No external geometry prior"),
            ("regenerated_vggt_cd_x100", "Regenerated VGGT prior"),
        ):
            cd_chart_rows.append({"scene": row["scene"], "configuration": config, "cd_x100": row[field]})

    summary_table = []
    for row in summary:
        summary_table.append({
            "dataset": row["dataset"], "configuration": label_map[row["configuration"]],
            "scenes": f"{row['valid_scene_count']}/{row['expected_scene_count']}",
            "psnr": row["psnr_↑"], "ssim": row["ssim_↑"], "lpips": row["lpips_↓"],
            "normal_mae": row["normal_mae_deg_↓"], "cd_x100": row["cd_x100_↓"],
            "fps": row["fps_↑"], "train_min": row["train_time_min_↓"],
        })
    geometry_table = []
    for row in geometry:
        geometry_table.append({
            "dataset": row["dataset"], "scene": row["scene"],
            "paper_mae": row["paper_normal_mae_deg"], "no_prior_mae": row["no_prior_normal_mae_deg"],
            "no_prior_mae_delta": row["no_prior_delta_paper_normal_mae_deg"],
            "vggt_mae": row["regenerated_vggt_normal_mae_deg"],
            "vggt_mae_delta": row["regenerated_vggt_delta_paper_normal_mae_deg"],
            "paper_cd": row["paper_cd_x100"], "no_prior_cd": row["no_prior_cd_x100"],
            "no_prior_cd_delta": row["no_prior_delta_paper_cd_x100"],
            "vggt_cd": row["regenerated_vggt_cd_x100"],
            "vggt_cd_delta": row["regenerated_vggt_delta_paper_cd_x100"],
        })
    efficiency_table = []
    for row in efficiency:
        efficiency_table.append({
            "dataset": row["dataset"], "configuration": label_map[row["configuration"]],
            "fps": row["fps_↑"], "train_min": row["train_time_min_↓"], "gpu": row["gpu"],
            "software": row["software_environment"],
        })

    def chart(cid: str, title: str, subtitle: str, dataset: str, yfield: str, ylabel: str) -> dict[str, object]:
        return {
            "id": cid,
            "title": title,
            "subtitle": subtitle,
            "type": "bar",
            "dataset": dataset,
            "sourceId": source["id"],
            "encodings": {
                "x": {"field": "scene", "type": "nominal", "label": "Scene"},
                "y": {"field": yfield, "type": "quantitative", "label": ylabel, "format": "number"},
                "color": {"field": "configuration", "type": "nominal", "label": "Configuration"},
            },
            "yAxisTitle": ylabel,
            "valueFormat": "number",
            "layout": "full",
            "palette": {"kind": "categorical", "roots": ["blue", "gold", "olive"]},
        }

    blocks = [
        {"id": "title", "type": "markdown", "body": "# Ref-DGS existing-results comparison"},
        {"id": "summary", "type": "markdown", "sourceId": source["id"], "body": "## The evidence separates three configurations rather than declaring a canonical reproduction\n\nOfficial depth priors are corrupted, so strict paper reproduction remains blocked. The public-VGGT diagnostic is close to the paper on ShinySynthetic but degrades on GlossySynthetic; the no-prior ablation improves Glossy results relative to this regenerated prior but does not match the paper geometry means. This supports prior-quality and domain sensitivity, not a general causal claim about whether geometry priors help."},
        {"id": "dataset_text", "type": "markdown", "body": "## Dataset means are recomputed from formal scene rows\n\nThe exact table keeps ↑ and ↓ metrics separate. Paper NVS/FPS/time entries are reported dataset references; all experimental means are unweighted scene recomputations. Missing paper metrics remain blank."},
        {"id": "dataset_table", "type": "table", "tableId": "dataset_summary", "layout": "full"},
        {"id": "shiny_text", "type": "markdown", "sourceId": source["id"], "body": "## Shiny geometry is close for regenerated VGGT but unstable without priors\n\nLower normal MAE is better. The no-prior coffee and toaster scenes account for much of the aggregate gap, so the mean should not be interpreted as uniform degradation."},
        {"id": "shiny_chart", "type": "chart", "chartId": "shiny_mae", "layout": "full"},
        {"id": "glossy_mae_text", "type": "markdown", "sourceId": source["id"], "body": "## Glossy normal accuracy is strongly sensitive to the regenerated priors\n\nThe public-VGGT grouping/configuration degrades several Glossy scenes; the no-prior ablation is closer, but still above the paper mean."},
        {"id": "glossy_mae_chart", "type": "chart", "chartId": "glossy_mae", "layout": "full"},
        {"id": "glossy_cd_text", "type": "markdown", "sourceId": source["id"], "body": "## Glossy mesh distance shows the same domain sensitivity\n\nCD values use the released mesh log multiplied by 100. Lower is better; Shiny CD is intentionally omitted."},
        {"id": "glossy_cd_chart", "type": "chart", "chartId": "glossy_cd", "layout": "full"},
        {"id": "scene_text", "type": "markdown", "body": "## Exact scene geometry preserves full comparison detail\n\nDeltas are experiment minus paper. Positive values are unfavorable for both normal MAE and CD."},
        {"id": "scene_table", "type": "table", "tableId": "scene_geometry", "layout": "full"},
        {"id": "efficiency_text", "type": "markdown", "body": "## Efficiency values are hardware references, not algorithm attribution\n\nBoth experimental configurations use RTX A5000 24 GB, whereas the paper reports RTX 4090 24 GB. The difference is not used as an implementation correctness judgment."},
        {"id": "efficiency_table", "type": "table", "tableId": "efficiency", "layout": "full"},
        {"id": "scope", "type": "markdown", "body": "## Scope and metric definitions\n\nOnly the six ShinySynthetic and eight GlossySynthetic formal `full/.../attempt_001` outputs are included. PSNR/SSIM/LPIPS/FPS/MAE come from test metric files; training time comes from `/usr/bin/time -v`; Glossy CD×10² is raw mesh CD multiplied by 100. No smoke row is present."},
        {"id": "method", "type": "markdown", "body": "## Raw artifacts control when root summaries conflict\n\nThe no-prior summaries reconcile exactly with raw scenes. VGGT root summary files remain stale at an earlier resource gate, so metrics are reconstructed from formal scene files and cross-checked against the later no-prior JSON's embedded `prior_*` values. Per-stage metric/mesh/CD exit values inherit the monolithic render/evaluation process code only when the required artifact exists."},
        {"id": "limitations", "type": "markdown", "body": "## Provenance limits prevent a strict paper claim\n\nThe no-prior run is a new ablation, not a paper main-table configuration. Regenerated VGGT is non-canonical and uses an author-unknown grouping. Paper NVS is dataset-only, and Glossy speed/time were not reported. These comparisons cannot establish a general prior effect."},
        {"id": "next", "type": "markdown", "body": "## Keep the evidence classes isolated\n\nRetain the canonical blocker. Resolve publication provenance only with valid original priors or author documentation of the exact loss and prior-generation protocol."},
        {"id": "question", "type": "markdown", "body": "## Further question\n\nWhich prior source, grouping, and confidence definition produced the publication tables?"},
    ]
    tables = [
        {
            "id": "dataset_summary", "title": "Dataset-level comparison", "subtitle": "Six Shiny and eight Glossy formal scenes; arrows indicate preferred direction.",
            "dataset": "dataset_summary", "sourceId": source["id"], "defaultSort": {"field": "dataset", "direction": "asc"}, "layout": "full",
            "columns": [
                {"field": "dataset", "label": "Dataset", "type": "text"}, {"field": "configuration", "label": "Configuration", "type": "text"},
                {"field": "scenes", "label": "Scenes", "type": "text"}, {"field": "psnr", "label": "PSNR ↑", "format": "number"},
                {"field": "ssim", "label": "SSIM ↑", "format": "number"}, {"field": "lpips", "label": "LPIPS ↓", "format": "number"},
                {"field": "normal_mae", "label": "Normal MAE ↓", "format": "number"}, {"field": "cd_x100", "label": "CD×10² ↓", "format": "number"},
                {"field": "fps", "label": "FPS ↑", "format": "number"}, {"field": "train_min", "label": "Train min ↓", "format": "number"},
            ],
        },
        {
            "id": "scene_geometry", "title": "Scene-level geometry", "subtitle": "Exact formal-scene results; Δpaper is experiment minus paper.",
            "dataset": "scene_geometry", "sourceId": source["id"], "defaultSort": {"field": "dataset", "direction": "asc"}, "density": "dense", "layout": "full",
            "columns": [
                {"field": "dataset", "label": "Dataset", "type": "text"}, {"field": "scene", "label": "Scene", "type": "text"},
                {"field": "paper_mae", "label": "Paper MAE", "format": "number"}, {"field": "no_prior_mae", "label": "No-prior MAE", "format": "number"},
                {"field": "no_prior_mae_delta", "label": "No-prior Δ", "format": "number"}, {"field": "vggt_mae", "label": "VGGT MAE", "format": "number"},
                {"field": "vggt_mae_delta", "label": "VGGT Δ", "format": "number"}, {"field": "paper_cd", "label": "Paper CD", "format": "number"},
                {"field": "no_prior_cd", "label": "No-prior CD", "format": "number"}, {"field": "vggt_cd", "label": "VGGT CD", "format": "number"},
            ],
        },
        {
            "id": "efficiency", "title": "Efficiency and environment", "subtitle": "Hardware differs from the paper; values are descriptive only.",
            "dataset": "efficiency", "sourceId": source["id"], "defaultSort": {"field": "dataset", "direction": "asc"}, "layout": "full",
            "columns": [
                {"field": "dataset", "label": "Dataset", "type": "text"}, {"field": "configuration", "label": "Configuration", "type": "text"},
                {"field": "fps", "label": "FPS ↑", "format": "number"}, {"field": "train_min", "label": "Train min ↓", "format": "number"},
                {"field": "gpu", "label": "GPU", "type": "text"}, {"field": "software", "label": "Environment", "type": "text"},
            ],
        },
    ]
    charts = [
        chart("shiny_mae", "ShinySynthetic normal MAE by scene", "Three evidence classes; degrees, lower is better.", "shiny_mae", "normal_mae_deg", "Normal MAE (degrees)"),
        chart("glossy_mae", "GlossySynthetic normal MAE by scene", "Three evidence classes; degrees, lower is better.", "glossy_mae", "normal_mae_deg", "Normal MAE (degrees)"),
        chart("glossy_cd", "GlossySynthetic Chamfer distance by scene", "Raw mesh distance multiplied by 100; lower is better.", "glossy_cd", "cd_x100", "CD×10²"),
    ]
    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "Ref-DGS existing-results comparison",
            "description": "Read-only comparison of paper, no-prior, and regenerated-VGGT evidence.",
            "generatedAt": generated,
            "cards": [],
            "charts": charts,
            "tables": tables,
            "sources": [{"id": source["id"], "label": source["label"], "path": source["path"]}],
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated,
            "status": "ready",
            "datasets": {
                "dataset_summary": summary_table,
                "scene_geometry": geometry_table,
                "efficiency": efficiency_table,
                "shiny_mae": [row for row in chart_rows if row["dataset"] == "ShinySynthetic"],
                "glossy_mae": [row for row in chart_rows if row["dataset"] == "GlossySynthetic"],
                "glossy_cd": cd_chart_rows,
            },
        },
        "sources": [source],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False) if not output_root.exists() else None

    inventory_files = required_input_files()
    inventory = [profile_file(path, purpose) for path, purpose in sorted(inventory_files.items(), key=lambda item: str(item[0]))]
    inventory_fields = ["absolute_path", "purpose", "columns", "data_rows", "size_bytes", "sha256"]
    write_csv(output_root / "input_inventory.csv", inventory, inventory_fields)
    print("INPUT INVENTORY: absolute path | columns | data rows | SHA-256")
    for item in inventory:
        print(f"{item['absolute_path']} | {item['columns']} | {item['data_rows']} | {item['sha256']}")

    canonical = json.loads((CANONICAL_ROOT / "results.json").read_text())
    canonical_paper = canonical["paper"]
    canonical_checks = {
        "canonical_status": canonical["canonical_status"],
        "official_depth_files_checked": canonical["blocker"]["official_hash_evidence"]["all_downloaded_depth_files_checked"],
        "official_depth_torch_load_ok": canonical["blocker"]["official_hash_evidence"]["all_downloaded_depth_torch_load_ok"],
        "official_depth_hash_match": canonical["blocker"]["official_hash_evidence"]["all_downloaded_depth_hash_match"],
        "paper_constants_match_requested": (
            canonical_paper["table_1"]["normal_mae_deg"] == list(PAPER["ShinySynthetic"]["normal_mae_deg"].values())
            and canonical_paper["table_2"]["normal_mae_deg"] == list(PAPER["GlossySynthetic"]["normal_mae_deg"].values())
            and canonical_paper["table_2"]["cd_x1e2"] == list(PAPER["GlossySynthetic"]["cd_x100"].values())
        ),
    }

    no_seed_text = (NO_ROOT / "runtime/Ref-DGS/utils/general_utils.py").read_text()
    vggt_seed_text = (VGGT_ROOT / "runtime/Ref-DGS/utils/general_utils.py").read_text()
    for seed_text in (no_seed_text, vggt_seed_text):
        for marker in ("random.seed(0)", "np.random.seed(0)", "torch.manual_seed(0)"):
            if marker not in seed_text:
                raise AssertionError(f"Missing seed marker: {marker}")

    rows = paper_rows(CANONICAL_ROOT / "results.json")
    scene_validation = []
    for dataset, scene_map in SCENES.items():
        for scene, iterations in scene_map.items():
            no_row, no_check = collect_experiment_scene(
                NO_ROOT, dataset, scene, iterations, "no_external_geometry_prior", NO_ROOT / "environment.txt"
            )
            vggt_row, vggt_check = collect_experiment_scene(
                VGGT_ROOT, dataset, scene, iterations, "regenerated_vggt_prior", VGGT_ROOT / "environment.txt"
            )
            rows.extend((no_row, vggt_row))
            scene_validation.extend((no_check, vggt_check))

    rows.sort(key=lambda row: (row["dataset"], row["scene"], list(CONFIGS).index(row["configuration"])))
    write_csv(output_root / "unified_results.csv", rows, UNIFIED_FIELDS)

    summary = recompute_dataset_summary(rows)
    summary_fields = list(summary[0].keys())
    write_csv(output_root / "comparison_dataset_summary.csv", summary, summary_fields)
    geometry = geometry_comparison(rows)
    geometry_fields = list(geometry[0].keys())
    write_csv(output_root / "comparison_scene_geometry.csv", geometry, geometry_fields)
    efficiency = efficiency_summary(summary)
    efficiency_fields = list(efficiency[0].keys())
    write_csv(output_root / "comparison_efficiency.csv", efficiency, efficiency_fields)

    # Reconcile no-prior raw rows against both existing result formats.
    no_csv = {(row["dataset"], row["scene"]): row for row in csv.DictReader((NO_ROOT / "results_no_prior.csv").open())}
    no_json_payload = json.loads((NO_ROOT / "results_no_prior.json").read_text())
    no_json = {(row["dataset"], row["scene"]): row for row in no_json_payload["rows"]}
    no_exact = True
    vggt_embedded_exact = True
    for row in rows:
        if row["configuration"] == "no_external_geometry_prior":
            key = (row["dataset"], row["scene"])
            for unified_field, source_field in (
                ("psnr", "psnr"), ("ssim", "ssim"), ("lpips", "lpips"),
                ("normal_mae_deg", "normal_mae"), ("fps", "fps"),
                ("train_time_min", "train_minutes"),
            ):
                no_exact = no_exact and close(row[unified_field], float(no_csv[key][source_field]))
                no_exact = no_exact and close(row[unified_field], float(no_json[key][source_field]))
            if row["dataset"] == "GlossySynthetic":
                no_exact = no_exact and close(row["cd_x100"], float(no_csv[key]["cd_x100"]))
                no_exact = no_exact and close(row["cd_x100"], float(no_json[key]["cd_x100"]))
        elif row["configuration"] == "regenerated_vggt_prior":
            key = (row["dataset"], row["scene"])
            embedded = no_json[key]
            for unified_field, embedded_field in (
                ("psnr", "prior_psnr"), ("ssim", "prior_ssim"), ("lpips", "prior_lpips"),
                ("normal_mae_deg", "prior_normal_mae"), ("cd_x100", "prior_cd_x100"),
            ):
                if row[unified_field] is not None:
                    vggt_embedded_exact = vggt_embedded_exact and close(row[unified_field], float(embedded[embedded_field]))

    diagnostic_csv = list(csv.DictReader((VGGT_ROOT / "results_diagnostic.csv").open()))
    diagnostic_json = json.loads((VGGT_ROOT / "results_diagnostic.json").read_text())
    vggt_root_summaries_stale = (
        len(diagnostic_csv) == 14
        and all(not row["psnr"] and row["training_status"] == "incomplete_resource_gate" for row in diagnostic_csv)
        and diagnostic_json.get("metrics") == []
        and diagnostic_json.get("training_launched") is False
    )

    duplicate_counts = Counter((row["dataset"], row["scene"], row["configuration"]) for row in rows)
    duplicate_keys = [list(key) for key, count in duplicate_counts.items() if count > 1]
    numeric_fields = ["psnr", "ssim", "lpips", "normal_mae_deg", "cd_x100", "fps", "train_time_min"]
    nonfinite = []
    for index, row in enumerate(rows):
        for field in numeric_fields:
            if not is_finite(row[field]):
                nonfinite.append({"row": index, "field": field, "value": row[field]})
    missing_by_config_dataset = {}
    for dataset, scene_map in SCENES.items():
        for configuration in CONFIGS:
            found = {row["scene"] for row in rows if row["dataset"] == dataset and row["configuration"] == configuration}
            missing_by_config_dataset[f"{dataset}/{configuration}"] = sorted(set(scene_map) - found)

    paper_recomputed = {
        "ShinySynthetic": {
            "normal_mae_deg": mean(PAPER["ShinySynthetic"]["normal_mae_deg"].values()),
            "reported_normal_mae_deg": PAPER["ShinySynthetic"]["reported_normal_mae_deg"],
        },
        "GlossySynthetic": {
            "normal_mae_deg": mean(PAPER["GlossySynthetic"]["normal_mae_deg"].values()),
            "reported_normal_mae_deg": PAPER["GlossySynthetic"]["reported_normal_mae_deg"],
            "cd_x100": mean(PAPER["GlossySynthetic"]["cd_x100"].values()),
            "reported_cd_x100": PAPER["GlossySynthetic"]["reported_cd_x100"],
        },
    }
    summary_consistency = {
        "no_prior_raw_vs_existing_csv_json_exact": no_exact,
        "vggt_raw_vs_later_embedded_prior_fields_exact": vggt_embedded_exact,
        "vggt_root_results_csv_json_stale_pre_resume": vggt_root_summaries_stale,
        "paper_scene_recomputed_vs_reported": paper_recomputed,
    }
    validation = {
        "overall_assessment": "SHARE WITH CAVEATS",
        "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "read_only_scope": True,
        "input_roots": [str(NO_ROOT), str(VGGT_ROOT), str(CANONICAL_ROOT)],
        "canonical_blocker": canonical_checks,
        "unified_row_count": len(rows),
        "configuration_values": sorted({row["configuration"] for row in rows}),
        "result_status_values": sorted({row["result_status"] for row in rows}),
        "expected_scene_counts": {dataset: len(scenes) for dataset, scenes in SCENES.items()},
        "missing_scenes": missing_by_config_dataset,
        "duplicate_key_count": len(duplicate_keys),
        "duplicate_keys": duplicate_keys,
        "nonfinite_value_count": len(nonfinite),
        "nonfinite_values": nonfinite,
        "smoke_source_count": sum("/smoke/" in str(row["source_file"]) for row in rows),
        "formal_scene_checks": scene_validation,
        "all_formal_scene_checks_valid": all(check["valid"] for check in scene_validation),
        "all_cd_unit_checks_valid": all(check["cd_unit_rule_valid"] for check in scene_validation),
        "summary_consistency": summary_consistency,
        "exit_code_semantics": {
            "train_exit_code": "direct per-scene train_exit_code.txt",
            "render_exit_code": "direct per-scene render_exit_code.txt",
            "metrics_exit_code": "same monolithic render/eval process exit code, accepted only with finite metric.txt",
            "mesh_exit_code": "same monolithic render/eval process exit code, accepted only with fuse.ply and fuse_post.ply",
            "cd_exit_code": "Glossy only; same monolithic render/eval process exit code, accepted only with finite mesh.log",
        },
        "known_source_conflicts": [
            {
                "severity": "HIGH",
                "source": str(VGGT_ROOT / "results_diagnostic.csv"),
                "issue": "stale pre-resume resource-gate summary with empty metrics despite completed full scene artifacts",
                "resolution": "do not use as metric source; reconstruct from full formal scene files",
            },
            {
                "severity": "HIGH",
                "source": str(VGGT_ROOT / "results_diagnostic.json"),
                "issue": "stale pre-resume state says training_launched=false and metrics=[]",
                "resolution": "do not use as metric source; validate formal markers, exits, checkpoints, and raw outputs",
            },
        ],
        "paper_mean_rounding_audit": paper_recomputed,
        "input_inventory_file": str(output_root / "input_inventory.csv"),
    }
    critical = (
        canonical_checks["paper_constants_match_requested"]
        and canonical_checks["official_depth_files_checked"] == 1828
        and canonical_checks["official_depth_torch_load_ok"] == 0
        and len(rows) == 42
        and not duplicate_keys
        and not nonfinite
        and all(not missing for missing in missing_by_config_dataset.values())
        and validation["smoke_source_count"] == 0
        and validation["all_formal_scene_checks_valid"]
        and validation["all_cd_unit_checks_valid"]
        and no_exact
        and vggt_embedded_exact
        and vggt_root_summaries_stale
    )
    validation["all_required_checks_passed"] = critical
    (output_root / "validation_report.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")

    report_md = render_markdown(summary, geometry, validation)
    (output_root / "comparison_report.md").write_text(report_md)
    (output_root / "artifact.json").write_text(json.dumps(artifact_payload(summary, geometry, efficiency), indent=2, sort_keys=True) + "\n")

    chart_map = """# Chart map

| Section | Question | Family/type | Fields | Supported claim | Palette |
|---|---|---|---|---|---|
| Shiny geometry | How do scene MAEs differ across evidence classes? | Comparison/grouped bar | scene, configuration, normal_mae_deg | Public VGGT is close while no-prior has scene-specific failures | blue/gold/olive, legend plus labels |
| Glossy normal | How do normal errors differ on Glossy? | Comparison/grouped bar | scene, configuration, normal_mae_deg | Regenerated VGGT is strongly degraded on this domain | blue/gold/olive, legend plus labels |
| Glossy mesh | Does CD show the same sensitivity? | Comparison/grouped bar | scene, configuration, cd_x100 | CD degradation aligns with the normal-error pattern | blue/gold/olive, legend plus labels |

All bars start at zero. Exact values remain available in the adjacent tables. No chart is generated for efficiency because GPU hardware is not comparable.
"""
    (output_root / "chart_map.md").write_text(chart_map)

    # Rehash every consumed input after all local calculations to prove inputs stayed unchanged.
    after = {str(path.resolve()): sha256(path) for path in inventory_files}
    before = {item["absolute_path"]: item["sha256"] for item in inventory}
    validation["input_hashes_stable_during_build"] = before == after
    if before != after:
        validation["changed_inputs"] = sorted(path for path in before if before[path] != after.get(path))
    (output_root / "validation_report.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")
    if not critical or before != after:
        raise SystemExit(1)
    print(json.dumps({
        "output_root": str(output_root),
        "unified_rows": len(rows),
        "all_required_checks_passed": critical,
        "input_hashes_stable_during_build": before == after,
        "known_source_conflict": "VGGT root results_diagnostic CSV/JSON are stale; raw full artifacts control",
    }, indent=2))


if __name__ == "__main__":
    main()
