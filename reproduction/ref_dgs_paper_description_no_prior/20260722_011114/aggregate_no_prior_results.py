#!/usr/bin/env python3
"""Aggregate the paper-description no-geometry-prior Ref-DGS experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean


RUN_ROOT_DEFAULT = Path("/data1/liuly/reproduction/ref_dgs_paper_description_no_prior/20260722_011114")
PRIOR_ROOT_DEFAULT = Path("/data1/liuly/reproduction/ref_dgs_diagnostic_vggt_full/20260720_203206_resume")

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
        "nvs": {"psnr": 35.21, "ssim": 0.975, "lpips": 0.053},
        "normal_mae": {
            "ball": 0.61, "car": 1.72, "coffee": 1.86, "helmet": 1.35,
            "teapot": 0.58, "toaster": 2.43,
        },
        "mean_normal_mae": 1.43,
        "mean_train_minutes": 12.6,
        "fps": 76.34,
    },
    "GlossySynthetic": {
        "nvs": {"psnr": 30.63, "ssim": 0.958, "lpips": 0.052},
        "normal_mae": {
            "angel": 2.05, "bell": 0.71, "cat": 1.37, "horse": 3.44,
            "luyu": 2.30, "potion": 2.48, "tbell": 1.84, "teapot": 0.86,
        },
        "cd_x100": {
            "angel": 0.38, "bell": 0.65, "cat": 0.93, "horse": 0.42,
            "luyu": 0.64, "potion": 0.66, "tbell": 0.52, "teapot": 0.79,
        },
        "mean_normal_mae": 1.88,
        "mean_cd_x100": 0.62,
    },
}

METRIC_RE = re.compile(
    r"psnr:(?P<psnr>[-+0-9.eE]+),\s*ssim:(?P<ssim>[-+0-9.eE]+),\s*"
    r"lpips:(?P<lpips>[-+0-9.eE]+),\s*fps:(?P<fps>[-+0-9.eE]+)\s*"
    r"mae:(?P<normal_mae>[-+0-9.eE]+)"
)


def parse_metric(path: Path) -> dict[str, float]:
    match = METRIC_RE.search(path.read_text())
    if not match:
        raise ValueError(f"unrecognized metric format: {path}")
    values = {key: float(value) for key, value in match.groupdict().items()}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError(f"non-finite metric: {path}")
    return values


def parse_elapsed_minutes(path: Path) -> float:
    text = path.read_text(errors="replace")
    matches = re.findall(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*([^\s]+)", text)
    if not matches:
        raise ValueError(f"elapsed time missing: {path}")
    parts = [float(part) for part in matches[-1].split(":")]
    if len(parts) == 2:
        seconds = parts[0] * 60 + parts[1]
    elif len(parts) == 3:
        seconds = parts[0] * 3600 + parts[1] * 60 + parts[2]
    else:
        raise ValueError(f"bad elapsed time: {matches[-1]}")
    return seconds / 60


def parse_peak_gpu_mib(path: Path) -> float:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        values = [float(row["memory_used_mib"].strip()) for row in reader]
    if not values:
        raise ValueError(f"GPU trace empty: {path}")
    return max(values)


def parse_cd_x100(path: Path) -> float:
    fields = path.read_text().strip().split()
    if len(fields) != 2:
        raise ValueError(f"unrecognized mesh log: {path}")
    value = float(fields[1]) * 100.0
    if not math.isfinite(value):
        raise ValueError(f"non-finite CD: {path}")
    return value


def scene_paths(root: Path, dataset: str, scene: str, iterations: int) -> dict[str, Path]:
    attempt = root / "full" / dataset / scene / "attempt_001"
    model = attempt / "model"
    logs = root / "logs" / "full" / dataset / scene / "attempt_001"
    return {
        "attempt": attempt,
        "model": model,
        "metric": model / "test" / f"ours_{iterations}" / "metric.txt",
        "mesh": model / "train" / f"ours_{iterations}" / "mesh.log",
        "train_log": logs / "train.log",
        "render_log": logs / "render.log",
        "gpu": logs / "train_gpu_memory.csv",
        "train_rc": logs / "train_exit_code.txt",
        "render_rc": logs / "render_exit_code.txt",
        "complete": attempt / "scene_complete.ok",
    }


def load_scene(root: Path, dataset: str, scene: str, iterations: int) -> dict[str, object]:
    paths = scene_paths(root, dataset, scene, iterations)
    required = [paths[key] for key in ("metric", "train_log", "render_log", "gpu", "train_rc", "render_rc", "complete")]
    missing = [str(path) for path in required if not path.is_file()]
    if dataset == "GlossySynthetic" and not paths["mesh"].is_file():
        missing.append(str(paths["mesh"]))
    if missing:
        raise FileNotFoundError("; ".join(missing))
    train_rc = int(paths["train_rc"].read_text().strip())
    render_rc = int(paths["render_rc"].read_text().strip())
    if train_rc or render_rc:
        raise RuntimeError(f"nonzero exit code for {dataset}/{scene}: train={train_rc}, render={render_rc}")
    metrics = parse_metric(paths["metric"])
    row: dict[str, object] = {
        "dataset": dataset,
        "scene": scene,
        "iterations": iterations,
        "train_minutes": parse_elapsed_minutes(paths["train_log"]),
        "render_eval_minutes": parse_elapsed_minutes(paths["render_log"]),
        "peak_gpu_memory_mib": parse_peak_gpu_mib(paths["gpu"]),
        **metrics,
        "cd_x100": parse_cd_x100(paths["mesh"]) if dataset == "GlossySynthetic" else None,
        "model_path": str(paths["model"]),
        "status": "complete",
    }
    return row


def with_comparisons(row: dict[str, object], prior_row: dict[str, object]) -> dict[str, object]:
    dataset, scene = str(row["dataset"]), str(row["scene"])
    paper_mae = PAPER[dataset]["normal_mae"][scene]
    paper_cd = PAPER[dataset].get("cd_x100", {}).get(scene)
    row.update({
        "paper_normal_mae": paper_mae,
        "abs_diff_paper_normal_mae": abs(float(row["normal_mae"]) - paper_mae),
        "paper_cd_x100": paper_cd,
        "abs_diff_paper_cd_x100": None if paper_cd is None else abs(float(row["cd_x100"]) - paper_cd),
        "prior_psnr": prior_row["psnr"],
        "prior_ssim": prior_row["ssim"],
        "prior_lpips": prior_row["lpips"],
        "prior_normal_mae": prior_row["normal_mae"],
        "prior_cd_x100": prior_row["cd_x100"],
        "delta_vs_prior_psnr": float(row["psnr"]) - float(prior_row["psnr"]),
        "delta_vs_prior_ssim": float(row["ssim"]) - float(prior_row["ssim"]),
        "delta_vs_prior_lpips": float(row["lpips"]) - float(prior_row["lpips"]),
        "delta_vs_prior_normal_mae": float(row["normal_mae"]) - float(prior_row["normal_mae"]),
        "delta_vs_prior_cd_x100": None if row["cd_x100"] is None else float(row["cd_x100"]) - float(prior_row["cd_x100"]),
    })
    geometry_close = row["abs_diff_paper_normal_mae"] <= 0.20
    if paper_cd is not None:
        geometry_close = geometry_close and row["abs_diff_paper_cd_x100"] <= 0.05
    row["paper_geometry_comparison"] = "close to paper descriptive value" if geometry_close else "deviates from paper descriptive value"
    return row


def mean_row(dataset: str, rows: list[dict[str, object]], prior_rows: list[dict[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {"dataset": dataset, "scene": "mean", "iterations": None, "status": "complete"}
    for field in ("train_minutes", "render_eval_minutes", "peak_gpu_memory_mib", "psnr", "ssim", "lpips", "normal_mae", "fps"):
        result[field] = mean(float(row[field]) for row in rows)
        result[f"prior_{field}"] = mean(float(row[field]) for row in prior_rows)
    if dataset == "GlossySynthetic":
        result["cd_x100"] = mean(float(row["cd_x100"]) for row in rows)
        result["prior_cd_x100"] = mean(float(row["cd_x100"]) for row in prior_rows)
    else:
        result["cd_x100"] = result["prior_cd_x100"] = None
    paper_nvs = PAPER[dataset]["nvs"]
    result.update({
        "paper_psnr": paper_nvs["psnr"],
        "paper_ssim": paper_nvs["ssim"],
        "paper_lpips": paper_nvs["lpips"],
        "paper_normal_mae": PAPER[dataset]["mean_normal_mae"],
        "paper_cd_x100": PAPER[dataset].get("mean_cd_x100"),
        "paper_fps": PAPER[dataset].get("fps"),
    })
    for field in ("psnr", "ssim", "lpips", "normal_mae", "cd_x100", "fps"):
        paper_value = result.get(f"paper_{field}")
        result[f"abs_diff_paper_{field}"] = None if paper_value is None else abs(float(result[field]) - float(paper_value))
        result[f"delta_vs_prior_{field}"] = None if result[field] is None else float(result[field]) - float(result[f"prior_{field}"])
    close = (
        result["abs_diff_paper_psnr"] <= 0.30
        and result["abs_diff_paper_ssim"] <= 0.005
        and result["abs_diff_paper_lpips"] <= 0.010
        and result["abs_diff_paper_normal_mae"] <= 0.20
    )
    if dataset == "GlossySynthetic":
        close = close and result["abs_diff_paper_cd_x100"] <= 0.05
    result["paper_overall_comparison"] = "close to paper descriptive values" if close else "deviates from paper descriptive values"
    return result


def fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def render_markdown(rows: list[dict[str, object]], means: dict[str, dict[str, object]], root: Path) -> str:
    lines = [
        "# Ref-DGS Paper-Description No-Geometry-Prior Experiment",
        "",
        "> **PAPER-DESCRIPTION NO-GEOMETRY-PRIOR EXPERIMENT — NOT UNMODIFIED OFFICIAL CODE**",
        "",
        "## Technical summary",
        "",
        "This run disables all external depth, depth-confidence, and normal priors while preserving the official Ref-DGS model, scene splits, iteration counts, densification, 2DGS internal normal-consistency regularizer, rendering, TSDF, and metric definitions. The paper does not disclose the external VGGT supervision, whereas the released README and loader require it; therefore these measurements test the paper-description interpretation but are not an unmodified official-code reproduction.",
        "",
        "## Dataset-level comparison",
        "",
        "| Dataset | Variant | PSNR | SSIM | LPIPS | normal MAE | CD × 10² | FPS |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in SCENES:
        m = means[dataset]
        lines.append(f"| {dataset} | no external priors | {fmt(m['psnr'],2)} | {fmt(m['ssim'])} | {fmt(m['lpips'])} | {fmt(m['normal_mae'],2)} | {fmt(m['cd_x100'],2)} | {fmt(m['fps'],2)} |")
        lines.append(f"| {dataset} | regenerated VGGT diagnostic | {fmt(m['prior_psnr'],2)} | {fmt(m['prior_ssim'])} | {fmt(m['prior_lpips'])} | {fmt(m['prior_normal_mae'],2)} | {fmt(m['prior_cd_x100'],2)} | {fmt(m['prior_fps'],2)} |")
        lines.append(f"| {dataset} | paper Ours | {fmt(m['paper_psnr'],2)} | {fmt(m['paper_ssim'])} | {fmt(m['paper_lpips'])} | {fmt(m['paper_normal_mae'],2)} | {fmt(m['paper_cd_x100'],2)} | {fmt(m['paper_fps'],2)} |")
    lines.extend(["", "## Per-scene measurements", ""])
    for dataset in SCENES:
        lines.extend([
            f"### {dataset}",
            "",
            "| Scene | Train min | PSNR | SSIM | LPIPS | normal MAE | paper MAE | |Δ| | CD × 10² | paper CD | FPS | Geometry comparison |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ])
        for row in [item for item in rows if item["dataset"] == dataset]:
            lines.append(
                f"| {row['scene']} | {fmt(row['train_minutes'],2)} | {fmt(row['psnr'],2)} | {fmt(row['ssim'])} | {fmt(row['lpips'])} | "
                f"{fmt(row['normal_mae'],2)} | {fmt(row['paper_normal_mae'],2)} | {fmt(row['abs_diff_paper_normal_mae'],2)} | "
                f"{fmt(row['cd_x100'],2)} | {fmt(row['paper_cd_x100'],2)} | {fmt(row['fps'],2)} | {row['paper_geometry_comparison']} |"
            )
        lines.append("")
    lines.extend([
        "## Scope and metric definitions",
        "",
        "- Scope: six ShinySynthetic and eight GlossySynthetic scenes only; no NeRF Synthetic or real datasets.",
        "- External geometry priors: loader access disabled; `vggt_weight=0`, `vggt_until_iter=0`; all logged `vggt_depth_loss` and `vggt_normal_loss` values are null.",
        "- The retained `normal_consistency_loss` is the internal 2DGS rendered-normal versus depth-derived surface-normal regularizer, not an external normal prior.",
        "- Scene metrics are arithmetic means over the official test split. Dataset means are unweighted arithmetic means over scenes, matching the paper table convention.",
        "- GlossySynthetic CD is the official mesh log value multiplied by 100. ShinySynthetic CD is intentionally not reported.",
        "- FPS is measured on RTX A5000 24 GB using the released renderer; the paper value uses RTX 4090 24 GB and is not a correctness gate.",
        "",
        "## Methodology and evidence boundary",
        "",
        "The base code is commit `490dc585a2d329928363e94f5f91951a61ddee0c`. The isolated runtime patch only bypasses prior-file loading under an explicit environment variable and adds finite-loss audit logging. All training and render commands are recorded in `commands.sh`; per-scene stdout/stderr, exit codes, GPU traces, frozen identities, renders, normals, and meshes remain under the run root.",
        "",
        "## Limitations and robustness checks",
        "",
        "- The paper text omits external priors, but the released code includes them. This experiment resolves that discrepancy by testing the no-prior interpretation; it cannot establish which unpublished configuration generated the paper numbers.",
        "- Random sampling is not explicitly seeded by the released scripts, so exact reruns may vary.",
        "- The regenerated-VGGT comparison uses public VGGT at a known revision and deterministic grouping, but the authors' original grouping is unknown.",
        "- Completion requires 14/14 zero exit codes, finite metrics, exact render/normal counts, and present TSDF outputs; protected canonical hashes are checked before and after.",
        "",
        "## Recommended next step",
        "",
        "Treat the no-prior and regenerated-VGGT runs as two controlled implementation diagnostics. Do not replace the canonical blocked conclusion unless the authors release valid original priors or document the exact paper-training loss configuration.",
        "",
        "## Further question",
        "",
        "The remaining material uncertainty is whether the submitted paper results used the released VGGT supervision despite its absence from the paper, or whether the public code diverged after the reported experiments.",
        "",
        f"Run root: `{root}`",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT_DEFAULT)
    parser.add_argument("--prior-root", type=Path, default=PRIOR_ROOT_DEFAULT)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    prior_by_key: dict[tuple[str, str], dict[str, object]] = {}
    for dataset, scenes in SCENES.items():
        for scene, iterations in scenes.items():
            prior_by_key[(dataset, scene)] = load_scene(args.prior_root, dataset, scene, iterations)
            row = load_scene(args.run_root, dataset, scene, iterations)
            rows.append(with_comparisons(row, prior_by_key[(dataset, scene)]))

    means = {
        dataset: mean_row(
            dataset,
            [row for row in rows if row["dataset"] == dataset],
            [row for (name, _), row in prior_by_key.items() if name == dataset],
        )
        for dataset in SCENES
    }

    payload = {
        "classification": "PAPER-DESCRIPTION NO-GEOMETRY-PRIOR EXPERIMENT — NOT UNMODIFIED OFFICIAL CODE",
        "run_root": str(args.run_root),
        "prior_comparison_root": str(args.prior_root),
        "paper_source": "tmp/pdfs/ref_dgs_v3.pdf",
        "rows": rows,
        "dataset_means": means,
    }
    (args.run_root / "results_no_prior.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    fieldnames = list(rows[0].keys())
    with (args.run_root / "results_no_prior.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (args.run_root / "no_prior_report.md").write_text(render_markdown(rows, means, args.run_root))
    print(json.dumps({"scene_count": len(rows), "dataset_means": means}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
