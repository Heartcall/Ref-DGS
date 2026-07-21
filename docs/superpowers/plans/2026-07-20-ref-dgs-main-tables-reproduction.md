# Ref-DGS Main Tables Reproduction Execution Plan

> **For agentic workers:** Execute this plan inline and update each checkbox from fresh command evidence. Do not dispatch subagents unless the user explicitly authorizes delegation. Do not treat smoke or diagnostic outputs as canonical results.

**Goal:** Reproduce the Ref-DGS `Ours` entries from Tables 1-4 on ShinySynthetic and GlossySynthetic, while preserving the source datasets as read-only and retaining a complete audit trail.

**Architecture:** Use official commit `490dc585a2d329928363e94f5f91951a61ddee0c` and its scene-specific flags. Construct writable per-scene staging roots under the timestamped reproduction directory because the Blender loader writes `points3d.ply`; all images, transforms, normals, alpha masks, GT depth, and `eval_pts.ply` remain read-only links to `/data/liuly/dataset/3DGS`. Run one complete `ball` smoke, then serialize canonical training, rendering, geometry evaluation, and metric collection on one GPU.

**Tech Stack:** Conda `refdgs`, Python 3.11, PyTorch 2.5.1+cu118, CUDA extensions shipped by Ref-DGS, Open3D 0.19.0, NVIDIA RTX A5000/RTX 3090 Ti GPUs.

---

## Fixed paths and scope

- Repository: `/home/liuly/Surface_Reconstruction/Glossy/Ref-DGS`
- Evidence root: `/data1/liuly/reproduction/ref_dgs_main_tables/20260720_010227`
- Shiny source: `/data/liuly/dataset/3DGS/Shiny Blender Synthetic`
- Glossy converted source: `/data/liuly/dataset/3DGS/GlossySyntheticConverted`
- Glossy raw GT depth source: `/data/liuly/dataset/3DGS/GlossySynthetic`
- Priors target: `/data1/liuly/reproduction/ref_dgs_main_tables/20260720_010227/priors`
- Explicitly excluded: NeRF Synthetic, RefReal, GlossyReal, all baselines, and Table 5.

## Task 1: Repository, paper, environment, and data audit

**Files:**
- Create: `reproduction/ref_dgs_main_tables/20260720_010227/environment.txt`
- Create: `reproduction/ref_dgs_main_tables/20260720_010227/reproduction_report.md`
- Create: `reproduction/ref_dgs_main_tables/20260720_010227/commands.sh`

- [x] Record `git rev-parse HEAD`, `git status --short`, branch, remote, log, and submodule status.
- [x] Read `README.md`, `requirements.txt`, `scripts/run_shiny.py`, `scripts/run_glossy_syn.py`, `train.py`, `render.py`, `arguments/__init__.py`, scene/model/renderer/mesh/metric code, and the v3 paper.
- [x] Record GPU, driver, CUDA, Python, PyTorch, compiler, extension imports, storage, and source-data permissions.
- [x] Verify all 6 Shiny and 8 Glossy scenes, official splits, image resolution, GT normals/depth/evaluation point clouds, and checksums of protocol files.
- [x] Verify that NeRF Synthetic is present but never selected by any planned command.
- [ ] Re-run the audit capture immediately before the first smoke and append the timestamp.

## Task 2: Prior availability hard gate

**Files:**
- Create after authorization: `/data1/liuly/reproduction/ref_dgs_main_tables/20260720_010227/priors/`
- Create after download: `/data1/liuly/reproduction/ref_dgs_main_tables/20260720_010227/audit/prior_manifest.tsv`

- [x] Search repository, `/data`, `/data1`, and relevant local project roots for compatible prior files.
- [x] Query official Hugging Face metadata without downloading blobs.
- [x] Confirm the required synthetic subsets total 5,648 files and 25,373,292,855 bytes (23.63 GiB).
- [ ] Obtain user authorization for this large download, as required by the task instructions.
- [ ] Download only `Ref-NeRF/refnerf/**` and `Glossy/GlossySynthetic/**` at repository revision `f7a411e493f5aa621a1bd3dcf5344058cf8bfa0e`.
- [ ] Validate every required `.pth` with `torch.load(..., map_location="cpu", weights_only=False)` and every normal image with Pillow.
- [ ] Require exactly 300 depth + 300 normal files per Shiny scene and 128 depth + 128 normal files per Glossy scene before smoke.

**Gate:** No training, including smoke training, may run while any required prior is missing or unreadable.

## Task 3: Create read-only-safe staged datasets

**Files:**
- Create: `/data1/liuly/reproduction/ref_dgs_main_tables/20260720_010227/staging/refnerf/<scene>/`
- Create: `/data1/liuly/reproduction/ref_dgs_main_tables/20260720_010227/staging/GlossySynthetic/<scene>_blender/`

- [ ] For each Shiny scene, link `train/`, `test/`, `transforms_train.json`, `transforms_test.json`, and optional GT mesh into a writable scene directory whose path contains `refnerf`.
- [ ] For each Glossy scene, link `rgb/`, `transforms_train.json`, `transforms_test.json`, `eval_pts.ply`, and `depth/` (from the raw scene) into a writable scene directory whose path contains `GlossySynthetic`.
- [ ] Copy the deterministic existing `points3d.ply` into each staging scene so the official loader may overwrite only the staged copy.
- [ ] Record `readlink -f` mappings and verify that no planned output path resolves under `/data/liuly/dataset/3DGS`.
- [ ] Hash all source protocol files again after staging and compare with the pre-staging manifest.

## Task 4: Complete `ball` smoke chain

**Files:**
- Create: `/data1/liuly/reproduction/ref_dgs_main_tables/20260720_010227/smoke/ball_i2/`
- Create: `/data1/liuly/reproduction/ref_dgs_main_tables/20260720_010227/logs/smoke_ball/`

- [ ] Run a 2-iteration official-model smoke with `--eval --iterations 2 --run_dim 64 --save_iterations 2` and `/usr/bin/time -v`; label it non-paper.
- [ ] Confirm training loads all 100 train cameras with authentic VGGT priors and writes both Gaussian point clouds.
- [ ] Run `render.py -m <smoke-model> --dataset shiny --iteration 2`.
- [ ] Confirm 100 train and 200 test render sets, finite PSNR/SSIM/LPIPS/FPS/MAE, and a non-empty TSDF mesh.
- [ ] Save exact stdout/stderr, exit codes, command line, environment snapshot, and representative render/normal/mesh paths.
- [ ] If a failure occurs, reproduce it unchanged, identify the root cause, change only one factor in a new diagnostic directory, and rerun the original failing command.

## Task 5: Freeze and run canonical ShinySynthetic training

**Files:**
- Create: `/data1/liuly/reproduction/ref_dgs_main_tables/20260720_010227/canonical/ShinySynthetic/<scene>/`
- Create: `/data1/liuly/reproduction/ref_dgs_main_tables/20260720_010227/logs/canonical_shiny_<scene>/`

Run serially with `/home/liuly/anaconda3/envs/refdgs/bin/python` and one selected free GPU:

- [ ] `ball`: `train.py --eval --iterations 15000 --run_dim 64`
- [ ] `car`: `train.py --eval --iterations 25000 --run_dim 64`
- [ ] `coffee`: `train.py --eval --iterations 15000 --run_dim 64 --albedo_lr 0.002`
- [ ] `helmet`: `train.py --eval --iterations 20000 --run_dim 64`
- [ ] `teapot`: `train.py --eval --iterations 20000 --run_dim 64`
- [ ] `toaster`: `train.py --eval --iterations 20000 --run_dim 64`

For every scene:

- [ ] Save wall-clock training time, exit code, `cfg_args`, code snapshot, final point clouds, and `light_mlp.pt`.
- [ ] Require the expected final iteration directory and finite final log loss before starting the next scene.
- [ ] Never infer or backfill a metric from a failed or incomplete scene.

## Task 6: Freeze and run canonical GlossySynthetic training

**Files:**
- Create: `/data1/liuly/reproduction/ref_dgs_main_tables/20260720_010227/canonical/GlossySynthetic/<scene>_blender/`
- Create: `/data1/liuly/reproduction/ref_dgs_main_tables/20260720_010227/logs/canonical_glossy_<scene>/`

- [ ] For each of `angel bell cat horse luyu potion tbell teapot`, run the official default 25,000 iterations with `--eval --run_dim 64 --albedo_bias 2 --albedo_lr 0.0005`.
- [ ] Apply the same per-scene checkpoint/log/finite-value gate as ShinySynthetic.

## Task 7: Canonical rendering, geometry, and metrics

For each completed scene:

- [ ] Run `render.py -m <model> --dataset shiny` for Shiny or `--dataset glossy` for Glossy at the final saved iteration.
- [ ] Validate exact test-view counts (200 Shiny; 16 Glossy) and finite per-scene `metric.txt` values.
- [ ] For Shiny, record normal MAE only; do not report CD.
- [ ] For Glossy, derive GT normals through the official depth-to-normal path and record MAE.
- [ ] For Glossy, extract TSDF mesh with official `--voxel_size 0.002`, keep the postprocessed mesh, and parse raw CD from `mesh.log`.
- [ ] Determine the paper table scaling from paper/code evidence; retain both raw CD and `CD x 10^2` without silently rescaling.
- [ ] Compute dataset means as unweighted arithmetic means of completed scene metrics only; if any scene is missing, mark the dataset mean incomplete.

## Task 8: Rendering speed

- [ ] First report the repository's exact official timing result from `GaussianExtractor.reconstruction`: 800x800 inputs, one pass over each view, no explicit warmup, `time.time()` around `render` + `render_ref`, and no explicit `torch.cuda.synchronize()`.
- [ ] Record GPU model, driver, PyTorch/CUDA, scene/view count, and the exact checkpoint used.
- [ ] Because the repository has no explicit warmup or synchronized benchmark, label this as the released-code FPS protocol and document its timing limitations rather than inventing a different official protocol.
- [ ] Do not classify algorithmic reproduction success/failure from cross-hardware FPS alone.

## Task 9: Aggregate and verify final evidence

**Files:**
- Update: `/data1/liuly/reproduction/ref_dgs_main_tables/20260720_010227/results.csv`
- Update: `/data1/liuly/reproduction/ref_dgs_main_tables/20260720_010227/results.json`
- Update: `/data1/liuly/reproduction/ref_dgs_main_tables/20260720_010227/reproduction_report.md`
- Update: `/data1/liuly/reproduction/ref_dgs_main_tables/20260720_010227/commands.sh`

- [ ] Compare canonical dataset means to the paper bands without tuning on test metrics.
- [ ] List all failures, retries, diagnostics, and compatibility fixes with separate paths and diffs.
- [ ] Verify every cited render, normal, mesh, checkpoint, metric file, and log exists and is non-empty.
- [ ] Parse `results.csv` and `results.json`, confirm they agree, and reject NaN/Inf.
- [ ] Re-run git status and source-data protocol hashes to prove the source dataset was not modified.
- [ ] Only then mark Tables 1-4 complete; otherwise report exact incomplete/blocked cells.

## Current gate state (2026-07-20 01:02 +0800)

`BLOCKED_MISSING_PRIORS`: the official synthetic VGGT prior files are not present locally. The required scoped download is 25,373,292,855 bytes (23.63 GiB), so the task's explicit large-download rule requires user authorization before smoke or canonical training can begin.
