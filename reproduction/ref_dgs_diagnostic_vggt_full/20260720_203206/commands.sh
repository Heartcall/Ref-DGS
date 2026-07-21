#!/usr/bin/env bash
# NON-CANONICAL VGGT-REGENERATED PRIORS — NOT A STRICT PAPER REPRODUCTION
# This file separates executed preparation commands from blocked GPU commands.

RUN_ROOT=/data1/liuly/reproduction/ref_dgs_diagnostic_vggt_full/20260720_203206
LOCAL_ROOT=/home/liuly/Surface_Reconstruction/Glossy/Ref-DGS/reproduction/ref_dgs_diagnostic_vggt_full/20260720_203206
OLD_BALL=/data1/liuly/reproduction/ref_dgs_diagnostic_vggt/20260720_151729
DATA_ROOT=/data/liuly/dataset/3DGS

# EXECUTED: create isolated detached worktree.
git worktree add --detach "$RUN_ROOT/worktree" 490dc585a2d329928363e94f5f91951a61ddee0c

# EXECUTED: test generalized tools plus the 12 completed-ball regression tests.
conda run -n refdgs python -m unittest -v \
  "$LOCAL_ROOT/tests/test_full_diagnostic_tools.py" \
  /home/liuly/Surface_Reconstruction/Glossy/Ref-DGS/reproduction/ref_dgs_diagnostic_vggt/20260720_151729/tests/test_diagnostic_tools.py

# EXECUTED: generate deterministic manifests for all 14 scenes.
for scene in ball car coffee helmet teapot toaster; do
  conda run -n refdgs python "$LOCAL_ROOT/tools/build_scene_manifests.py" \
    --dataset ShinySynthetic --scene "$scene" \
    --scene-root "$DATA_ROOT/Shiny Blender Synthetic/$scene" --batch-size 20 \
    --output "$LOCAL_ROOT/scene_manifests/ShinySynthetic_${scene}.json"
done
for scene in angel bell cat horse luyu potion tbell teapot; do
  conda run -n refdgs python "$LOCAL_ROOT/tools/build_scene_manifests.py" \
    --dataset GlossySynthetic --scene "$scene" \
    --scene-root "$DATA_ROOT/GlossySyntheticConverted/${scene}_blender" --batch-size 20 \
    --output "$LOCAL_ROOT/scene_manifests/GlossySynthetic_${scene}.json"
done
conda run -n refdgs python "$LOCAL_ROOT/tools/build_scene_manifests.py" \
  --dataset ShinySynthetic --scene ball \
  --scene-root "$DATA_ROOT/Shiny Blender Synthetic/ball" --batch-size 20 \
  --recovered-ball-manifest "$OLD_BALL/ball_batch_manifest.json" \
  --output "$LOCAL_ROOT/scene_manifests/ShinySynthetic_ball.json"

# NOT EXECUTED: Phase A GPU inference is blocked by the explicit 5.4 GB storage gate.
# Every remaining scene would use the following exact command shape, serially on physical GPU 1:
# PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=1 "$OLD_BALL/venv_vggt/bin/python" \
#   "$RUN_ROOT/tools/generate_scene_vggt_priors.py" \
#   --vggt-root "$OLD_BALL/vendor/vggt" \
#   --checkpoint "$OLD_BALL/checkpoints/VGGT-1B/model.pt" \
#   --scene-root <scene-root> --manifest <scene-manifest> \
#   --output-root "$RUN_ROOT/priors" --batch-log <scene-jsonl> \
#   --split all --preprocess-mode crop --device cuda:0

# EXECUTED: reuse through a read-only symlink and fully revalidate the completed ball priors.
conda run -n refdgs python "$RUN_ROOT/tools/validate_scene_vggt_priors.py" \
  --scene-root "$DATA_ROOT/Shiny Blender Synthetic/ball" \
  --manifest "$RUN_ROOT/scene_manifests/ShinySynthetic_ball.json" \
  --prior-root "$RUN_ROOT/priors" \
  --official-prior-root /data1/liuly/reproduction/ref_dgs_main_tables/20260720_010227/priors \
  --visualization-root "$RUN_ROOT/visualizations/ShinySynthetic/ball" \
  --output "$RUN_ROOT/logs/prior_validation_ShinySynthetic_ball.json"

# EXECUTED, EXIT 1: final metadata synchronization after /data1 remounted read-only.
conda run -n refdgs cp -a "$LOCAL_ROOT/." "$RUN_ROOT/"
