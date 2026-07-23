#!/usr/bin/env python3
"""Build the canonical Data Analytics report artifact from aggregated results."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def round_or_none(value, digits=4):
    return None if value is None else round(float(value), digits)


def variant_rows(payload, dataset):
    mean_row = payload["dataset_means"][dataset]
    return [
        {
            "dataset": dataset,
            "variant": "No external priors",
            "psnr": round_or_none(mean_row["psnr"]),
            "ssim": round_or_none(mean_row["ssim"]),
            "lpips": round_or_none(mean_row["lpips"]),
            "normal_mae": round_or_none(mean_row["normal_mae"]),
            "cd_x100": round_or_none(mean_row["cd_x100"]),
            "fps": round_or_none(mean_row["fps"]),
        },
        {
            "dataset": dataset,
            "variant": "Regenerated VGGT diagnostic",
            "psnr": round_or_none(mean_row["prior_psnr"]),
            "ssim": round_or_none(mean_row["prior_ssim"]),
            "lpips": round_or_none(mean_row["prior_lpips"]),
            "normal_mae": round_or_none(mean_row["prior_normal_mae"]),
            "cd_x100": round_or_none(mean_row["prior_cd_x100"]),
            "fps": round_or_none(mean_row["prior_fps"]),
        },
        {
            "dataset": dataset,
            "variant": "Paper Ours",
            "psnr": round_or_none(mean_row["paper_psnr"]),
            "ssim": round_or_none(mean_row["paper_ssim"]),
            "lpips": round_or_none(mean_row["paper_lpips"]),
            "normal_mae": round_or_none(mean_row["paper_normal_mae"]),
            "cd_x100": round_or_none(mean_row["paper_cd_x100"]),
            "fps": round_or_none(mean_row["paper_fps"]),
        },
    ]


def geometry_rows(payload, dataset, field):
    rows = []
    for row in payload["rows"]:
        if row["dataset"] != dataset:
            continue
        paper_field = "paper_normal_mae" if field == "normal_mae" else "paper_cd_x100"
        prior_field = "prior_normal_mae" if field == "normal_mae" else "prior_cd_x100"
        rows.extend([
            {"scene": row["scene"], "variant": "No external priors", "value": round_or_none(row[field])},
            {"scene": row["scene"], "variant": "Regenerated VGGT diagnostic", "value": round_or_none(row[prior_field])},
            {"scene": row["scene"], "variant": "Paper Ours", "value": round_or_none(row[paper_field])},
        ])
    return rows


def scene_table_rows(payload, dataset):
    output = []
    for row in payload["rows"]:
        if row["dataset"] != dataset:
            continue
        output.append({
            "scene": row["scene"],
            "train_min": round_or_none(row["train_minutes"], 2),
            "psnr": round_or_none(row["psnr"], 3),
            "ssim": round_or_none(row["ssim"], 4),
            "lpips": round_or_none(row["lpips"], 4),
            "normal_mae": round_or_none(row["normal_mae"], 3),
            "paper_normal_mae": round_or_none(row["paper_normal_mae"], 3),
            "prior_normal_mae": round_or_none(row["prior_normal_mae"], 3),
            "cd_x100": round_or_none(row["cd_x100"], 3),
            "paper_cd_x100": round_or_none(row["paper_cd_x100"], 3),
            "prior_cd_x100": round_or_none(row["prior_cd_x100"], 3),
            "fps": round_or_none(row["fps"], 2),
            "status": row["paper_geometry_comparison"],
        })
    return output


def build(payload):
    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    shiny = payload["dataset_means"]["ShinySynthetic"]
    glossy = payload["dataset_means"]["GlossySynthetic"]
    source = {
        "id": "combined_experiment_results",
        "label": "Aggregated Ref-DGS experiment results",
        "path": "results_no_prior.json",
        "query": {
            "engine": "DuckDB-compatible SQL over the frozen JSON artifact",
            "language": "python",
            "sql": (
                "WITH payload AS (\n"
                "  SELECT * FROM read_json_auto('results_no_prior.json')\n"
                "), scene_rows AS (\n"
                "  SELECT UNNEST(rows) AS scene_result FROM payload\n"
                ")\n"
                "SELECT scene_result.* FROM scene_rows;\n"
                "-- Dataset means shown in the report are the frozen dataset_means "
                "object from the same JSON artifact."
            ),
            "tables_used": ["results_no_prior.json:rows", "results_no_prior.json:dataset_means"],
            "description": "Parsed official metric.txt, mesh.log, timing logs, GPU traces, paper table constants, and the regenerated-VGGT comparison run.",
            "filters": [
                "ShinySynthetic: ball, car, coffee, helmet, teapot, toaster",
                "GlossySynthetic: angel, bell, cat, horse, luyu, potion, tbell, teapot",
                "External depth, confidence, and normal priors disabled",
            ],
            "metric_definitions": {
                "dataset_mean": "Unweighted arithmetic mean over scene-level official test metrics.",
                "cd_x100": "Official mesh.log Chamfer distance multiplied by 100.",
                "normal_mae": "Mean angular error in degrees from the released evaluation pipeline.",
            },
        },
    }
    blocks = [
        {"id": "title", "type": "markdown", "body": "# Ref-DGS without external geometry priors"},
        {
            "id": "technical_summary",
            "type": "markdown",
            "sourceId": source["id"],
            "body": (
                "## The no-prior interpretation is measured separately from the official release\n\n"
                "This report tests Ref-DGS with external depth, confidence, and normal priors fully disabled. "
                f"ShinySynthetic mean normal MAE is **{shiny['normal_mae']:.2f}°** versus **{shiny['paper_normal_mae']:.2f}°** in the paper; "
                f"GlossySynthetic mean normal MAE is **{glossy['normal_mae']:.2f}°** and CD × 10² is **{glossy['cd_x100']:.2f}**. "
                "The paper does not disclose the released VGGT supervision, so this is a paper-description experiment, not an unmodified official-code reproduction."
            ),
        },
        {"id": "headline_metrics", "type": "metric-strip", "cardIds": ["shiny_mae", "glossy_mae", "glossy_cd", "shiny_psnr", "glossy_psnr"]},
        {
            "id": "dataset_comparison_text",
            "type": "markdown",
            "body": "## Dataset means separate image quality, geometry, and hardware-dependent speed\n\nThe exact lookup table keeps unlike units separate and shows all three evidence classes. FPS is included for completeness but is not used as a correctness gate because this run uses an RTX A5000 rather than the paper's RTX 4090.",
        },
        {"id": "dataset_comparison", "type": "table", "tableId": "dataset_means", "layout": "full"},
        {
            "id": "shiny_geometry_text",
            "type": "markdown",
            "sourceId": source["id"],
            "body": "## ShinySynthetic geometry varies by scene\n\nThe grouped bars compare normal MAE at identical scene grain. Lower is better; the exact values and training times are listed immediately afterward.",
        },
        {"id": "shiny_geometry", "type": "chart", "chartId": "shiny_normal_mae", "layout": "full"},
        {"id": "shiny_detail", "type": "table", "tableId": "shiny_scenes", "layout": "full"},
        {
            "id": "glossy_normal_text",
            "type": "markdown",
            "sourceId": source["id"],
            "body": "## GlossySynthetic normal errors reveal the strongest sensitivity\n\nNormal MAE is shown separately from Chamfer distance so scale and ranking remain interpretable. Lower values are better.",
        },
        {"id": "glossy_normal", "type": "chart", "chartId": "glossy_normal_mae", "layout": "full"},
        {
            "id": "glossy_cd_text",
            "type": "markdown",
            "sourceId": source["id"],
            "body": "## GlossySynthetic mesh distance uses the released ×100 unit\n\nEach grouped bar is the official mesh evaluation value multiplied by 100. No CD is reported for ShinySynthetic because its reference mesh contains an invisible inner layer.",
        },
        {"id": "glossy_cd", "type": "chart", "chartId": "glossy_cd_x100", "layout": "full"},
        {"id": "glossy_detail", "type": "table", "tableId": "glossy_scenes", "layout": "full"},
        {
            "id": "scope",
            "type": "markdown",
            "body": "## Scope and definitions\n\nThe population is the six ShinySynthetic and eight GlossySynthetic scenes in Tables 1–4. Dataset means are unweighted scene means. The retained 2DGS normal-consistency term compares two model-rendered geometry buffers and is not an external normal prior. NeRF Synthetic and real datasets are excluded.",
        },
        {
            "id": "method",
            "type": "markdown",
            "body": "## The loader gate removes the undisclosed dependency without changing Ref-DGS\n\nThe frozen base is commit `490dc585a2d329928363e94f5f91951a61ddee0c`. An isolated patch skips all prior-file access only when `REFDGS_DISABLE_GEOMETRY_PRIOR=1`; training also sets `vggt_weight=0` and `vggt_until_iter=0`. Model structure, loss terms disclosed through 2DGS, scene splits, iterations, densification, rendering, TSDF voxel size 0.002, and metric definitions are unchanged.",
        },
        {
            "id": "limitations",
            "type": "markdown",
            "body": "## The main uncertainty is provenance of the paper numbers\n\nThe paper omits external priors while the released code requires them. This experiment cannot prove which configuration generated the publication tables. The regenerated-VGGT run also cannot recover the authors' unknown original view grouping. Random camera sampling is not explicitly seeded, so exact reruns may vary.",
        },
        {
            "id": "next_steps",
            "type": "markdown",
            "body": "## Keep both runs as controlled diagnostics\n\nDo not replace the canonical blocked conclusion. The decisive next evidence would be valid original priors or an author statement documenting the exact training loss and prior-generation protocol used for the paper.",
        },
        {
            "id": "further_questions",
            "type": "markdown",
            "body": "## Further question\n\nDid the submitted results use the released VGGT supervision despite its absence from the paper, or did the public implementation diverge after the experiments?"
        },
    ]
    cards = [
        {"id": "shiny_mae", "description": "No-prior ShinySynthetic mean normal angular error.", "dataset": "headline", "sourceId": source["id"], "metrics": [{"label": "Shiny normal MAE", "field": "shiny_mae", "format": "number", "unit": "°"}]},
        {"id": "glossy_mae", "description": "No-prior GlossySynthetic mean normal angular error.", "dataset": "headline", "sourceId": source["id"], "metrics": [{"label": "Glossy normal MAE", "field": "glossy_mae", "format": "number", "unit": "°"}]},
        {"id": "glossy_cd", "description": "No-prior GlossySynthetic mean Chamfer distance times 100.", "dataset": "headline", "sourceId": source["id"], "metrics": [{"label": "Glossy CD × 10²", "field": "glossy_cd", "format": "number"}]},
        {"id": "shiny_psnr", "description": "No-prior ShinySynthetic mean test PSNR.", "dataset": "headline", "sourceId": source["id"], "metrics": [{"label": "Shiny PSNR", "field": "shiny_psnr", "format": "number", "unit": "dB"}]},
        {"id": "glossy_psnr", "description": "No-prior GlossySynthetic mean test PSNR.", "dataset": "headline", "sourceId": source["id"], "metrics": [{"label": "Glossy PSNR", "field": "glossy_psnr", "format": "number", "unit": "dB"}]},
    ]
    chart = lambda cid, title, subtitle, dataset, ylabel: {
        "id": cid, "title": title, "subtitle": subtitle, "type": "bar", "dataset": dataset,
        "sourceId": source["id"],
        "encodings": {
            "x": {"field": "scene", "type": "nominal", "label": "Scene"},
            "y": {"field": "value", "type": "quantitative", "label": ylabel, "format": "number"},
            "color": {"field": "variant", "type": "nominal", "label": "Variant"},
        },
        "yAxisTitle": ylabel, "valueFormat": "number", "layout": "full",
        "palette": {"kind": "categorical", "roots": ["blue", "gold", "olive"]},
    }
    charts = [
        chart("shiny_normal_mae", "ShinySynthetic normal MAE by scene", "Three fixed evidence classes; degrees, lower is better.", "shiny_geometry", "Normal MAE (degrees)"),
        chart("glossy_normal_mae", "GlossySynthetic normal MAE by scene", "Three fixed evidence classes; degrees, lower is better.", "glossy_normal", "Normal MAE (degrees)"),
        chart("glossy_cd_x100", "GlossySynthetic Chamfer distance by scene", "Official TSDF mesh evaluation multiplied by 100; lower is better.", "glossy_cd", "CD × 10²"),
    ]
    tables = [
        {
            "id": "dataset_means", "title": "Dataset-level comparison", "subtitle": "Unweighted scene means; em dash denotes a metric not reported by the paper.",
            "dataset": "dataset_means", "sourceId": source["id"], "defaultSort": {"field": "dataset", "direction": "asc"}, "layout": "full",
            "columns": [
                {"field": "dataset", "label": "Dataset", "type": "text"}, {"field": "variant", "label": "Variant", "type": "text"},
                {"field": "psnr", "label": "PSNR", "format": "number"}, {"field": "ssim", "label": "SSIM", "format": "number"},
                {"field": "lpips", "label": "LPIPS", "format": "number"}, {"field": "normal_mae", "label": "Normal MAE", "format": "number"},
                {"field": "cd_x100", "label": "CD × 10²", "format": "number"}, {"field": "fps", "label": "FPS", "format": "number"},
            ],
        },
        {
            "id": "shiny_scenes", "title": "ShinySynthetic scene results", "subtitle": "Official test split and full paper iteration count for each scene.",
            "dataset": "shiny_scenes", "sourceId": source["id"], "defaultSort": {"field": "scene", "direction": "asc"}, "density": "dense", "layout": "full",
            "columns": [
                {"field": "scene", "label": "Scene", "type": "text"}, {"field": "train_min", "label": "Train min", "format": "number"},
                {"field": "psnr", "label": "PSNR", "format": "number"}, {"field": "ssim", "label": "SSIM", "format": "number"},
                {"field": "lpips", "label": "LPIPS", "format": "number"}, {"field": "normal_mae", "label": "No-prior MAE", "format": "number"},
                {"field": "prior_normal_mae", "label": "VGGT MAE", "format": "number"}, {"field": "paper_normal_mae", "label": "Paper MAE", "format": "number"},
                {"field": "fps", "label": "FPS", "format": "number"}, {"field": "status", "label": "Comparison", "type": "text"},
            ],
        },
        {
            "id": "glossy_scenes", "title": "GlossySynthetic scene results", "subtitle": "Official test split, full iteration count, and voxel size 0.002.",
            "dataset": "glossy_scenes", "sourceId": source["id"], "defaultSort": {"field": "scene", "direction": "asc"}, "density": "dense", "layout": "full",
            "columns": [
                {"field": "scene", "label": "Scene", "type": "text"}, {"field": "train_min", "label": "Train min", "format": "number"},
                {"field": "psnr", "label": "PSNR", "format": "number"}, {"field": "ssim", "label": "SSIM", "format": "number"},
                {"field": "lpips", "label": "LPIPS", "format": "number"}, {"field": "normal_mae", "label": "No-prior MAE", "format": "number"},
                {"field": "prior_normal_mae", "label": "VGGT MAE", "format": "number"}, {"field": "paper_normal_mae", "label": "Paper MAE", "format": "number"},
                {"field": "cd_x100", "label": "No-prior CD", "format": "number"}, {"field": "prior_cd_x100", "label": "VGGT CD", "format": "number"},
                {"field": "paper_cd_x100", "label": "Paper CD", "format": "number"}, {"field": "fps", "label": "FPS", "format": "number"},
            ],
        },
    ]
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1, "surface": "report", "title": "Ref-DGS without external geometry priors",
            "description": "Technical comparison of no-prior, regenerated-VGGT diagnostic, and paper values.",
            "generatedAt": generated_at, "cards": cards, "charts": charts, "tables": tables,
            "sources": [{"id": source["id"], "label": source["label"], "path": source["path"]}], "blocks": blocks,
        },
        "snapshot": {
            "version": 1, "generatedAt": generated_at, "status": "ready",
            "datasets": {
                "headline": [{
                    "shiny_mae": round_or_none(shiny["normal_mae"]), "glossy_mae": round_or_none(glossy["normal_mae"]),
                    "glossy_cd": round_or_none(glossy["cd_x100"]), "shiny_psnr": round_or_none(shiny["psnr"]),
                    "glossy_psnr": round_or_none(glossy["psnr"]),
                }],
                "dataset_means": variant_rows(payload, "ShinySynthetic") + variant_rows(payload, "GlossySynthetic"),
                "shiny_geometry": geometry_rows(payload, "ShinySynthetic", "normal_mae"),
                "glossy_normal": geometry_rows(payload, "GlossySynthetic", "normal_mae"),
                "glossy_cd": geometry_rows(payload, "GlossySynthetic", "cd_x100"),
                "shiny_scenes": scene_table_rows(payload, "ShinySynthetic"),
                "glossy_scenes": scene_table_rows(payload, "GlossySynthetic"),
            },
        },
        "sources": [source],
    }
    return artifact


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.results.read_text())
    args.output.write_text(json.dumps(build(payload), indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
