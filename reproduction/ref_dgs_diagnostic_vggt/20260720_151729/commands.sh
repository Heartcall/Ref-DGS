#!/usr/bin/env bash
# NON-CANONICAL: regenerated with public VGGT; not an official paper reproduction
# Audit trail; paths and commands are literal. No Table 1--4 canonical command was run.

RUN_ROOT=/data1/liuly/reproduction/ref_dgs_diagnostic_vggt/20260720_151729
CHECKPOINT_PARTIAL="$RUN_ROOT/checkpoints/VGGT-1B/.cache/huggingface/download/7LKmnjt53bdRhNP0DZo5ghr1Kmo=.d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0.32de60ee.incomplete"

# Audit/generation commands used the repository root as their working directory.
# Ref-DGS train/render commands used "$RUN_ROOT/runtime/Ref-DGS".

rg -n "torch.load|depth_conf|depth_map|prior|lambda_prior|depth_loss|normal_loss|vggt_until_iter" train.py scene arguments utils

git clone --depth 1 https://github.com/facebookresearch/vggt.git "$RUN_ROOT/vendor/vggt"
git -C "$RUN_ROOT/vendor/vggt" rev-parse HEAD
conda run -n refdgs python -m venv --system-site-packages "$RUN_ROOT/venv_vggt"
"$RUN_ROOT/venv_vggt/bin/pip" install numpy==1.26.1 einops safetensors

# The HF CLI connection stalled twice but preserved a partial. The second line
# completed the same official object; exact size and SHA were verified.
/home/liuly/anaconda3/envs/refdgs/bin/hf download facebook/VGGT-1B model.pt --revision 860abec7937da0a4c03c41d3c269c366e82abdf9 --local-dir "$RUN_ROOT/checkpoints/VGGT-1B" --max-workers 1 --format agent
curl --location --fail --retry 20 --retry-delay 5 --continue-at - --output "$CHECKPOINT_PARTIAL" https://huggingface.co/facebook/VGGT-1B/resolve/860abec7937da0a4c03c41d3c269c366e82abdf9/model.pt
sha256sum "$CHECKPOINT_PARTIAL"

conda run -n refdgs python -m unittest discover -s reproduction/ref_dgs_diagnostic_vggt/20260720_151729/tests -v

# One recovered 20-view batch probe; exit 0 after output-layout compatibility fix.
conda run -n refdgs env CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$RUN_ROOT/venv_vggt/lib/python3.11/site-packages" python reproduction/ref_dgs_diagnostic_vggt/20260720_151729/generate_ball_vggt_priors.py --vggt-root "$RUN_ROOT/vendor/vggt" --checkpoint "$RUN_ROOT/checkpoints/VGGT-1B/model.pt" --scene-root "/data/liuly/dataset/3DGS/Shiny Blender Synthetic/ball" --manifest reproduction/ref_dgs_diagnostic_vggt/20260720_151729/ball_batch_manifest.json --output-root "$RUN_ROOT/smoke/vggt_probe_prior" --batch-log "$RUN_ROOT/logs/vggt_probe_batches.jsonl" --split train --preprocess-mode crop --device cuda:0 --probe-first-batch-only

# All ball priors only: 5 train batches + 10 test batches, exit 0.
conda run -n refdgs env CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$RUN_ROOT/venv_vggt/lib/python3.11/site-packages" python reproduction/ref_dgs_diagnostic_vggt/20260720_151729/generate_ball_vggt_priors.py --vggt-root "$RUN_ROOT/vendor/vggt" --checkpoint "$RUN_ROOT/checkpoints/VGGT-1B/model.pt" --scene-root "/data/liuly/dataset/3DGS/Shiny Blender Synthetic/ball" --manifest reproduction/ref_dgs_diagnostic_vggt/20260720_151729/ball_batch_manifest.json --output-root "$RUN_ROOT/priors/Ref-NeRF/refnerf/ball" --batch-log "$RUN_ROOT/logs/vggt_ball_batches.jsonl" --split all --preprocess-mode crop --device cuda:0

conda run -n refdgs python reproduction/ref_dgs_diagnostic_vggt/20260720_151729/validate_ball_vggt_priors.py --scene-root "/data/liuly/dataset/3DGS/Shiny Blender Synthetic/ball" --manifest reproduction/ref_dgs_diagnostic_vggt/20260720_151729/ball_batch_manifest.json --prior-root "$RUN_ROOT/priors/Ref-NeRF/refnerf/ball" --visualization-root "$RUN_ROOT/visualizations" --output "$RUN_ROOT/prior_validation.json"

# Exact canonical smoke flags with diagnostic source/model paths; exit 0.
conda run -n refdgs env CUDA_VISIBLE_DEVICES=0 REFDGS_DIAGNOSTIC_LOSS_LOG=1 PYTHONPATH="$RUN_ROOT/runtime/Ref-DGS" python reproduction/ref_dgs_diagnostic_vggt/20260720_151729/train_diagnostic.py -s "$RUN_ROOT/staging/refnerf/ball" -m "$RUN_ROOT/smoke/ball_i2" --eval --iterations 2 --run_dim 64 --save_iterations 2

# Independent short chain; exit 0.
conda run -n refdgs env CUDA_VISIBLE_DEVICES=0 REFDGS_DIAGNOSTIC_LOSS_LOG=1 PYTHONPATH="$RUN_ROOT/runtime/Ref-DGS" python reproduction/ref_dgs_diagnostic_vggt/20260720_151729/train_diagnostic.py -s "$RUN_ROOT/staging/refnerf/ball" -m "$RUN_ROOT/smoke/ball_i20_chain" --eval --iterations 20 --run_dim 64 --save_iterations 20
conda run -n refdgs env CUDA_VISIBLE_DEVICES=0 python render.py -m "$RUN_ROOT/smoke/ball_i20_chain" --iteration 20 --dataset shiny --voxel_size 0.002
