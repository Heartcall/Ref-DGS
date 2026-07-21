#!/usr/bin/env bash
set -euo pipefail

LABEL='NON-CANONICAL VGGT-REGENERATED PRIORS — NOT A STRICT PAPER REPRODUCTION'
RUN_ROOT=/data1/liuly/reproduction/ref_dgs_diagnostic_vggt_full/20260720_203206_resume
RUNTIME="$RUN_ROOT/runtime/Ref-DGS"
TRAIN_DIAGNOSTIC="$RUN_ROOT/tools/train_diagnostic.py"
VALIDATE_SMOKE="$RUN_ROOT/tools/validate_smoke.py"
GPU_PHYSICAL=1
TASK_LIMIT_BYTES=$((45 * 1024 * 1024 * 1024))
RESERVE_BYTES=$((5 * 1024 * 1024 * 1024))
mkdir -p "$RUN_ROOT/logs"
exec > >(tee -a "$RUN_ROOT/logs/phase_b_runner.log") 2>&1

storage_gate() {
  local mount_options available run_bytes
  mount_options=$(findmnt -n -T /data1/liuly -o OPTIONS)
  available=$(df -B1 --output=avail /data1/liuly | tail -1 | tr -d ' ')
  run_bytes=$(du -sb "$RUN_ROOT" | cut -f1)
  [[ ",$mount_options," == *,rw,* ]] || { echo "storage_gate_failed mount=$mount_options"; return 1; }
  (( available >= RESERVE_BYTES )) || { echo "storage_gate_failed available_bytes=$available"; return 1; }
  (( run_bytes < TASK_LIMIT_BYTES )) || { echo "storage_gate_failed run_bytes=$run_bytes"; return 1; }
  echo "storage_gate_pass run_bytes=$run_bytes available_bytes=$available mount=$mount_options"
}

run_scene() {
  local dataset=$1 scene=$2 source=$3
  shift 3
  local model="$RUN_ROOT/smoke/$dataset/$scene/i2"
  local log="$RUN_ROOT/logs/smoke/$dataset/$scene/train_i2.log"
  local gpu_log="$RUN_ROOT/logs/smoke/$dataset/$scene/gpu_memory.csv"
  local result="$RUN_ROOT/logs/smoke/$dataset/$scene/result.json"
  local exit_file="$RUN_ROOT/logs/smoke/$dataset/$scene/exit_code.txt"
  local command_file="$RUN_ROOT/logs/smoke/$dataset/$scene/command.sh"
  local command

  storage_gate
  [[ -f "$RUN_ROOT/logs/validation/${dataset}_${scene}.json" ]]
  if [[ -f "$result" ]] && grep -q '"status": "pass"' "$result"; then
    echo "smoke_scene_already_validated dataset=$dataset scene=$scene"
    return 0
  fi
  [[ ! -e "$model" ]] || { echo "refusing to overwrite existing smoke model: $model"; return 1; }
  mkdir -p "$(dirname "$log")" "$model"
  command="conda run -n refdgs env CUDA_VISIBLE_DEVICES=$GPU_PHYSICAL REFDGS_DIAGNOSTIC_LOSS_LOG=1 PYTHONPATH=$RUNTIME python $TRAIN_DIAGNOSTIC -s $source -m $model --eval --iterations 2 --run_dim 64 --save_iterations 2 $*"
  {
    echo "#!/usr/bin/env bash"
    echo "# $LABEL"
    printf '%s\n' "$command"
  } > "$command_file"
  printf '%s\n' "$command" >> "$RUN_ROOT/commands.sh"
  echo 'timestamp_utc,memory_used_mib,utilization_gpu_percent' > "$gpu_log"

  set +e
  (
    cd "$RUNTIME"
    /usr/bin/time -v conda run -n refdgs env \
      CUDA_VISIBLE_DEVICES="$GPU_PHYSICAL" \
      REFDGS_DIAGNOSTIC_LOSS_LOG=1 \
      PYTHONPATH="$RUNTIME" \
      python "$TRAIN_DIAGNOSTIC" \
      -s "$source" -m "$model" --eval --iterations 2 --run_dim 64 --save_iterations 2 "$@"
  ) > "$log" 2>&1 &
  local train_pid=$!
  while kill -0 "$train_pid" 2>/dev/null; do
    nvidia-smi --id="$GPU_PHYSICAL" --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader,nounits >> "$gpu_log" || true
    sleep 0.25
  done
  wait "$train_pid"
  local rc=$?
  set -e
  printf '%s\n' "$rc" > "$exit_file"

  conda run -n refdgs python "$VALIDATE_SMOKE" \
    --dataset "$dataset" --scene "$scene" --log "$log" --model-path "$model" \
    --exit-code "$rc" --output "$result"
  storage_gate
  echo "smoke_scene_complete dataset=$dataset scene=$scene exit_code=$rc"
}

echo "# Phase B executed commands — $LABEL" >> "$RUN_ROOT/commands.sh"
run_scene ShinySynthetic ball "$RUN_ROOT/staging/refnerf/ball"
run_scene ShinySynthetic car "$RUN_ROOT/staging/refnerf/car"
run_scene ShinySynthetic coffee "$RUN_ROOT/staging/refnerf/coffee" --albedo_lr 0.002
run_scene ShinySynthetic helmet "$RUN_ROOT/staging/refnerf/helmet"
run_scene ShinySynthetic teapot "$RUN_ROOT/staging/refnerf/teapot"
run_scene ShinySynthetic toaster "$RUN_ROOT/staging/refnerf/toaster"
run_scene GlossySynthetic angel "$RUN_ROOT/staging/GlossySynthetic/angel_blender" --albedo_bias 2 --albedo_lr 0.0005
run_scene GlossySynthetic bell "$RUN_ROOT/staging/GlossySynthetic/bell_blender" --albedo_bias 2 --albedo_lr 0.0005
run_scene GlossySynthetic cat "$RUN_ROOT/staging/GlossySynthetic/cat_blender" --albedo_bias 2 --albedo_lr 0.0005
run_scene GlossySynthetic horse "$RUN_ROOT/staging/GlossySynthetic/horse_blender" --albedo_bias 2 --albedo_lr 0.0005
run_scene GlossySynthetic luyu "$RUN_ROOT/staging/GlossySynthetic/luyu_blender" --albedo_bias 2 --albedo_lr 0.0005
run_scene GlossySynthetic potion "$RUN_ROOT/staging/GlossySynthetic/potion_blender" --albedo_bias 2 --albedo_lr 0.0005
run_scene GlossySynthetic tbell "$RUN_ROOT/staging/GlossySynthetic/tbell_blender" --albedo_bias 2 --albedo_lr 0.0005
run_scene GlossySynthetic teapot "$RUN_ROOT/staging/GlossySynthetic/teapot_blender" --albedo_bias 2 --albedo_lr 0.0005
echo phase_b_smoke_complete
