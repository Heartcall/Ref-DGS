#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT=/data1/liuly/reproduction/ref_dgs_diagnostic_vggt_full/20260720_203206_resume
REFDGS_PYTHON=/home/liuly/anaconda3/envs/refdgs/bin/python
VGGT_SITE=/data1/liuly/reproduction/ref_dgs_diagnostic_vggt/20260720_151729/venv_vggt/lib/python3.11/site-packages
TASK_STOP_BYTES=48318382080
FILESYSTEM_RESERVE_BYTES=5368709120

check_storage() {
  local options available used
  options=$(findmnt -n -T "$RUN_ROOT" -o OPTIONS)
  case ",$options," in
    *,rw,*) ;;
    *) printf 'storage_gate_failed mount_options=%s\n' "$options" >&2; return 70 ;;
  esac
  available=$(df -B1 --output=avail "$RUN_ROOT" | tail -1)
  used=$(du -sb "$RUN_ROOT" | cut -f1)
  if (( available < FILESYSTEM_RESERVE_BYTES )); then
    printf 'storage_gate_failed available_bytes=%s\n' "$available" >&2
    return 71
  fi
  if (( used >= TASK_STOP_BYTES )); then
    printf 'storage_gate_failed run_bytes=%s\n' "$used" >&2
    return 72
  fi
  printf 'storage_gate_pass run_bytes=%s available_bytes=%s mount=%s\n' "$used" "$available" "$options"
}

validate_scene() {
  local dataset=$1 scene=$2 source=$3
  local visual_dir="$RUN_ROOT/visualizations/$dataset/$scene"
  local validation_dir="$RUN_ROOT/logs/validation/$dataset"
  mkdir -p "$visual_dir" "$validation_dir"
  "$REFDGS_PYTHON" "$RUN_ROOT/tools/validate_scene_vggt_priors.py" \
    --scene-root "$source" \
    --manifest "$RUN_ROOT/scene_manifests/${dataset}_${scene}.json" \
    --prior-root "$RUN_ROOT/priors" \
    --official-prior-root "$RUN_ROOT/priors" \
    --visualization-root "$visual_dir" \
    --output "$RUN_ROOT/logs/validation/${dataset}_${scene}.json" \
    2>&1 | tee "$validation_dir/${scene}.log"
}

generate_scene() {
  local dataset=$1 scene=$2 source=$3 prior_dir=$4 expected=$5
  local generation_dir="$RUN_ROOT/logs/generation/$dataset"
  local validation_json="$RUN_ROOT/logs/validation/${dataset}_${scene}.json"
  local existing
  mkdir -p "$generation_dir"
  if [[ -f "$validation_json" ]]; then
    "$REFDGS_PYTHON" -c "import json; d=json.load(open('$validation_json')); assert d['status']=='pass'; assert d['overall']['file_count']==$expected"
    printf 'scene_skip_validated dataset=%s scene=%s expected=%s\n' "$dataset" "$scene" "$expected"
    return 0
  fi
  existing=$(find "$prior_dir" -type f -name '*.pth' | wc -l)
  if (( existing != 0 )); then
    printf 'scene_refuse_partial dataset=%s scene=%s existing=%s\n' "$dataset" "$scene" "$existing" >&2
    return 73
  fi
  check_storage
  /usr/bin/time -v env \
    CUDA_VISIBLE_DEVICES=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="$VGGT_SITE" \
    "$REFDGS_PYTHON" "$RUN_ROOT/tools/generate_scene_vggt_priors.py" \
      --vggt-root "$RUN_ROOT/vendor/vggt" \
      --checkpoint "$RUN_ROOT/checkpoints/VGGT-1B/model.pt" \
      --scene-root "$source" \
      --manifest "$RUN_ROOT/scene_manifests/${dataset}_${scene}.json" \
      --output-root "$RUN_ROOT/priors" \
      --batch-log "$generation_dir/${scene}_batches.jsonl" \
      --split all --preprocess-mode crop --device cuda:0 \
      2>&1 | tee "$generation_dir/${scene}.log"
  validate_scene "$dataset" "$scene" "$source"
  "$REFDGS_PYTHON" -c "import json; d=json.load(open('$validation_json')); assert d['status']=='pass'; assert d['overall']['file_count']==$expected; assert len(d['file_sha256_manifest'])==$expected"
  sync
  check_storage
  printf 'scene_complete dataset=%s scene=%s expected=%s\n' "$dataset" "$scene" "$expected"
}

check_storage

# Revalidate the read-only reused ball prior in the resume tree.
if [[ ! -f "$RUN_ROOT/logs/validation/ShinySynthetic_ball.json" ]]; then
  validate_scene ShinySynthetic ball "/data/liuly/dataset/3DGS/Shiny Blender Synthetic/ball"
fi

for scene in car coffee helmet teapot toaster; do
  generate_scene ShinySynthetic "$scene" \
    "/data/liuly/dataset/3DGS/Shiny Blender Synthetic/$scene" \
    "$RUN_ROOT/priors/Ref-NeRF/refnerf/$scene" 300
done

for scene in angel bell cat horse luyu potion tbell teapot; do
  generate_scene GlossySynthetic "$scene" \
    "/data/liuly/dataset/3DGS/GlossySyntheticConverted/${scene}_blender" \
    "$RUN_ROOT/priors/Glossy/GlossySynthetic/${scene}_blender/depth" 128
done

printf 'phase_a_generation_and_validation_complete\n'
