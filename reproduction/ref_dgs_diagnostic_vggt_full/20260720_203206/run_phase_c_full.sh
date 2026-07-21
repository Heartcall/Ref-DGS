#!/usr/bin/env bash
set -euo pipefail

LABEL='NON-CANONICAL VGGT-REGENERATED PRIORS — NOT A STRICT PAPER REPRODUCTION'
RUN_ROOT=/data1/liuly/reproduction/ref_dgs_diagnostic_vggt_full/20260720_203206_resume
RUNTIME="$RUN_ROOT/runtime/Ref-DGS"
TRAIN_DIAGNOSTIC="$RUN_ROOT/tools/train_diagnostic.py"
GPU_PHYSICAL=1
TASK_LIMIT_BYTES=$((45 * 1024 * 1024 * 1024))
RESERVE_BYTES=$((5 * 1024 * 1024 * 1024))
TRAIN_HEADROOM_BYTES=$((512 * 1024 * 1024))
RENDER_HEADROOM_BYTES=$((2500 * 1024 * 1024))
mkdir -p "$RUN_ROOT/logs"
exec > >(tee -a "$RUN_ROOT/logs/phase_c_runner.log") 2>&1

storage_gate() {
  local required_bytes=${1:-0}
  local mount_options available run_bytes
  mount_options=$(findmnt -n -T /data1/liuly -o OPTIONS)
  available=$(df -B1 --output=avail /data1/liuly | tail -1 | tr -d ' ')
  run_bytes=$(du -sb "$RUN_ROOT" | cut -f1)
  [[ ",$mount_options," == *,rw,* ]] || { echo "storage_gate_failed mount=$mount_options"; return 1; }
  (( available - required_bytes >= RESERVE_BYTES )) || {
    echo "storage_gate_failed available_bytes=$available required_bytes=$required_bytes"; return 1;
  }
  (( run_bytes + required_bytes < TASK_LIMIT_BYTES )) || {
    echo "storage_gate_failed run_bytes=$run_bytes required_bytes=$required_bytes"; return 1;
  }
  echo "storage_gate_pass run_bytes=$run_bytes required_bytes=$required_bytes available_bytes=$available mount=$mount_options"
}

monitor_gpu_until_exit() {
  local pid=$1 output=$2
  echo 'timestamp_utc,memory_used_mib,utilization_gpu_percent' > "$output"
  while kill -0 "$pid" 2>/dev/null; do
    nvidia-smi --id="$GPU_PHYSICAL" --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader,nounits >> "$output" || true
    sleep 0.5
  done
}

run_scene() {
  local dataset=$1 scene=$2 source=$3 iterations=$4 render_dataset=$5 train_count=$6 test_count=$7
  shift 7
  local attempt="$RUN_ROOT/full/$dataset/$scene/attempt_001"
  local model="$attempt/model"
  local log_dir="$RUN_ROOT/logs/full/$dataset/$scene/attempt_001"
  local manifest="$RUN_ROOT/scene_manifests/${dataset}_${scene}.json"
  local final_pc="$model/point_cloud/iteration_$iterations"
  local test_dir="$model/test/ours_$iterations"
  local train_dir="$model/train/ours_$iterations"
  local train_log="$log_dir/train.log"
  local render_log="$log_dir/render.log"
  local train_rc render_rc train_pid render_pid
  local extra_args=("$@")

  if [[ -f "$attempt/scene_complete.ok" ]]; then
    echo "full_scene_already_complete dataset=$dataset scene=$scene"
    return 0
  fi
  [[ ! -e "$attempt" ]] || {
    echo "refusing to overwrite incomplete attempt dataset=$dataset scene=$scene path=$attempt"
    return 1
  }
  storage_gate "$TRAIN_HEADROOM_BYTES"
  mkdir -p "$model" "$log_dir"
  {
    echo "label=$LABEL"
    echo "dataset=$dataset"
    echo "scene=$scene"
    echo "source=$source"
    echo "iterations=$iterations"
    echo "refdgs_commit=$(git -C "$RUNTIME" rev-parse HEAD)"
    echo "refdgs_status=$(git -C "$RUNTIME" status --porcelain | tr '\n' ';')"
    echo "scene_manifest_sha256=$(sha256sum "$manifest" | cut -d' ' -f1)"
    echo "vggt_generation_config_sha256=$(sha256sum "$RUN_ROOT/vggt_generation_config.json" | cut -d' ' -f1)"
    echo "train_diagnostic_sha256=$(sha256sum "$TRAIN_DIAGNOSTIC" | cut -d' ' -f1)"
    printf 'extra_args='; printf '%q ' "${extra_args[@]}"; echo
  } > "$attempt/frozen_identity.txt"
  {
    echo "#!/usr/bin/env bash"
    echo "# $LABEL"
    printf 'conda run -n refdgs env CUDA_VISIBLE_DEVICES=%q REFDGS_DIAGNOSTIC_LOSS_LOG=1 PYTHONPATH=%q python %q -s %q -m %q --eval --iterations %q --run_dim 64' \
      "$GPU_PHYSICAL" "$RUNTIME" "$TRAIN_DIAGNOSTIC" "$source" "$model" "$iterations"
    printf ' %q' "${extra_args[@]}"; echo
  } > "$log_dir/train_command.sh"
  tail -n 1 "$log_dir/train_command.sh" >> "$RUN_ROOT/commands.sh"

  set +e
  (
    cd "$RUNTIME"
    /usr/bin/time -v conda run -n refdgs env \
      CUDA_VISIBLE_DEVICES="$GPU_PHYSICAL" \
      REFDGS_DIAGNOSTIC_LOSS_LOG=1 \
      PYTHONPATH="$RUNTIME" \
      python "$TRAIN_DIAGNOSTIC" -s "$source" -m "$model" --eval \
      --iterations "$iterations" --run_dim 64 "${extra_args[@]}"
  ) > "$train_log" 2>&1 &
  train_pid=$!
  monitor_gpu_until_exit "$train_pid" "$log_dir/train_gpu_memory.csv"
  wait "$train_pid"
  train_rc=$?
  set -e
  printf '%s\n' "$train_rc" > "$log_dir/train_exit_code.txt"
  (( train_rc == 0 )) || { echo "train_failed dataset=$dataset scene=$scene exit_code=$train_rc"; return 1; }
  rg -q 'Training complete' "$train_log"
  ! rg -q 'FloatingPointError|CUDA out of memory|nan|NaN|inf|Inf' "$train_log"
  for artifact in gaussians_point_cloud.ply ref_gaussians_point_cloud.ply light_mlp.pt dir_encoding.pt; do
    [[ -s "$final_pc/$artifact" ]] || { echo "missing_final_artifact $final_pc/$artifact"; return 1; }
  done
  local early_records
  early_records=$(rg -c '^\[DIAGNOSTIC_PRIOR\].*"iteration": ([1-9]|[1-9][0-9]|100),' "$train_log" || true)
  (( early_records == 100 )) || { echo "early_loss_record_count_failed count=$early_records"; return 1; }
  echo "train_complete dataset=$dataset scene=$scene iterations=$iterations early_records=$early_records"

  storage_gate "$RENDER_HEADROOM_BYTES"
  {
    echo "#!/usr/bin/env bash"
    echo "# $LABEL"
    printf 'conda run -n refdgs env CUDA_VISIBLE_DEVICES=%q PYTHONPATH=%q python %q -m %q --iteration %q --dataset %q --voxel_size 0.002\n' \
      "$GPU_PHYSICAL" "$RUNTIME" "$RUNTIME/render.py" "$model" "$iterations" "$render_dataset"
  } > "$log_dir/render_command.sh"
  tail -n 1 "$log_dir/render_command.sh" >> "$RUN_ROOT/commands.sh"

  set +e
  (
    cd "$RUNTIME"
    /usr/bin/time -v conda run -n refdgs env CUDA_VISIBLE_DEVICES="$GPU_PHYSICAL" PYTHONPATH="$RUNTIME" \
      python "$RUNTIME/render.py" -m "$model" --iteration "$iterations" \
      --dataset "$render_dataset" --voxel_size 0.002
  ) > "$render_log" 2>&1 &
  render_pid=$!
  monitor_gpu_until_exit "$render_pid" "$log_dir/render_gpu_memory.csv"
  wait "$render_pid"
  render_rc=$?
  set -e
  printf '%s\n' "$render_rc" > "$log_dir/render_exit_code.txt"
  (( render_rc == 0 )) || { echo "render_failed dataset=$dataset scene=$scene exit_code=$render_rc"; return 1; }
  [[ -s "$test_dir/metric.txt" && -s "$train_dir/metric.txt" ]]
  [[ -s "$train_dir/fuse.ply" && -s "$train_dir/fuse_post.ply" ]]
  local actual_train actual_test actual_train_normals actual_test_normals
  actual_train=$(find "$train_dir/renders" -maxdepth 1 -type f -name '*.png' | wc -l)
  actual_test=$(find "$test_dir/renders" -maxdepth 1 -type f -name '*.png' | wc -l)
  actual_train_normals=$(find "$train_dir/vis/normal" -maxdepth 1 -type f -name '*.png' | wc -l)
  actual_test_normals=$(find "$test_dir/vis/normal" -maxdepth 1 -type f -name '*.png' | wc -l)
  (( actual_train == train_count && actual_test == test_count ))
  (( actual_train_normals == train_count && actual_test_normals == test_count ))
  if [[ "$render_dataset" == glossy ]]; then
    [[ -s "$train_dir/mesh.log" ]]
  else
    [[ ! -e "$train_dir/mesh.log" ]]
  fi
  printf '%s\n' "$LABEL" > "$attempt/scene_complete.ok"
  storage_gate 0
  echo "full_scene_complete dataset=$dataset scene=$scene train_count=$actual_train test_count=$actual_test"
}

echo "# Phase C executed commands — $LABEL" >> "$RUN_ROOT/commands.sh"
run_scene ShinySynthetic ball "$RUN_ROOT/staging/refnerf/ball" 15000 shiny 100 200
run_scene ShinySynthetic car "$RUN_ROOT/staging/refnerf/car" 25000 shiny 100 200
run_scene ShinySynthetic coffee "$RUN_ROOT/staging/refnerf/coffee" 15000 shiny 100 200 --albedo_lr 0.002
run_scene ShinySynthetic helmet "$RUN_ROOT/staging/refnerf/helmet" 20000 shiny 100 200
run_scene ShinySynthetic teapot "$RUN_ROOT/staging/refnerf/teapot" 20000 shiny 100 200
run_scene ShinySynthetic toaster "$RUN_ROOT/staging/refnerf/toaster" 20000 shiny 100 200
run_scene GlossySynthetic angel "$RUN_ROOT/staging/GlossySynthetic/angel_blender" 25000 glossy 112 16 --albedo_bias 2 --albedo_lr 0.0005
run_scene GlossySynthetic bell "$RUN_ROOT/staging/GlossySynthetic/bell_blender" 25000 glossy 112 16 --albedo_bias 2 --albedo_lr 0.0005
run_scene GlossySynthetic cat "$RUN_ROOT/staging/GlossySynthetic/cat_blender" 25000 glossy 112 16 --albedo_bias 2 --albedo_lr 0.0005
run_scene GlossySynthetic horse "$RUN_ROOT/staging/GlossySynthetic/horse_blender" 25000 glossy 112 16 --albedo_bias 2 --albedo_lr 0.0005
run_scene GlossySynthetic luyu "$RUN_ROOT/staging/GlossySynthetic/luyu_blender" 25000 glossy 112 16 --albedo_bias 2 --albedo_lr 0.0005
run_scene GlossySynthetic potion "$RUN_ROOT/staging/GlossySynthetic/potion_blender" 25000 glossy 112 16 --albedo_bias 2 --albedo_lr 0.0005
run_scene GlossySynthetic tbell "$RUN_ROOT/staging/GlossySynthetic/tbell_blender" 25000 glossy 112 16 --albedo_bias 2 --albedo_lr 0.0005
run_scene GlossySynthetic teapot "$RUN_ROOT/staging/GlossySynthetic/teapot_blender" 25000 glossy 112 16 --albedo_bias 2 --albedo_lr 0.0005
echo phase_c_full_complete
