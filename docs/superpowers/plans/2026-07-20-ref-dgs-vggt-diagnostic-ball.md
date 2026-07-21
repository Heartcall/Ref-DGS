# Ref-DGS VGGT Regenerated-Prior Ball Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a strictly non-canonical `ball` diagnostic that regenerates authentic VGGT depth/confidence priors, validates them, and exercises Ref-DGS train-render-evaluate without altering the blocked Table 1-4 canonical record.

**Architecture:** Use a new `/data1` evidence root containing an official VGGT checkout/checkpoint, diagnostic priors, a clean Ref-DGS runtime snapshot, staging links, logs, visualizations, and reports. Recover the author's 20-view batch membership and within-batch ordering from the intact pickle metadata of the damaged official priors, run public VGGT on exactly those groups, and save cloned per-view tensors. Keep all instrumentation inside the runtime snapshot and label every output `NON-CANONICAL`.

**Tech Stack:** Python 3.11, PyTorch 2.5.1+cu118, CUDA, official `facebookresearch/vggt`, official `facebook/VGGT-1B`, Pillow, NumPy, Matplotlib, Ref-DGS CUDA extensions.

---

### Task 1: Freeze boundaries and protocol evidence

**Files:**
- Create: `reproduction/ref_dgs_diagnostic_vggt/20260720_151729/execution_plan.md`
- Create: `reproduction/ref_dgs_diagnostic_vggt/20260720_151729/environment.txt`
- Create: `reproduction/ref_dgs_diagnostic_vggt/20260720_151729/vggt_generation_config.json`
- Create: `reproduction/ref_dgs_diagnostic_vggt/20260720_151729/commands.sh`

- [ ] Record canonical deliverable hashes and permissions without writing under `/data1/liuly/reproduction/ref_dgs_main_tables/20260720_010227`.
- [ ] Record Ref-DGS requirements: dict keys, tensor shapes, resize modes, confidence normalization/masking, depth scale-shift alignment, iteration range, and weight.
- [ ] Parse official `ball` pickle metadata and assert five train plus ten test batches, each with slots 0-19 and 518x518 storage slices.
- [ ] Create the isolated diagnostic root `/data1/liuly/reproduction/ref_dgs_diagnostic_vggt/20260720_151729` and its requested subdirectories.

### Task 2: Audit and acquire official VGGT

**Files:**
- Create: `/data1/liuly/reproduction/ref_dgs_diagnostic_vggt/20260720_151729/vendor/vggt/`
- Create: `/data1/liuly/reproduction/ref_dgs_diagnostic_vggt/20260720_151729/checkpoints/VGGT-1B/model.pt`
- Create: `/data1/liuly/reproduction/ref_dgs_diagnostic_vggt/20260720_151729/logs/vggt_audit.log`

- [ ] Shallow-clone `https://github.com/facebookresearch/vggt.git`, record `HEAD`, and inspect `README.md`, `requirements.txt`, `vggt/utils/load_fn.py`, `vggt/models/vggt.py`, aggregator, and depth head completely.
- [ ] Check every required inference dependency in `refdgs`; install only missing packages into that environment and record changes.
- [ ] Download only official `facebook/VGGT-1B/model.pt` at pinned HF revision `860abec7937da0a4c03c41d3c269c366e82abdf9`.
- [ ] Assert checkpoint size `5026874952` and SHA-256 `d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0` before inference.

### Task 3: Implement metadata recovery, generation, and validation

**Files:**
- Create: `reproduction/ref_dgs_diagnostic_vggt/20260720_151729/recover_ball_batches.py`
- Create: `reproduction/ref_dgs_diagnostic_vggt/20260720_151729/generate_ball_vggt_priors.py`
- Create: `reproduction/ref_dgs_diagnostic_vggt/20260720_151729/validate_ball_vggt_priors.py`
- Create: `reproduction/ref_dgs_diagnostic_vggt/20260720_151729/tests/test_diagnostic_tools.py`

- [ ] Write tests for exact five/ten batch recovery, unique slots, schema rejection, non-finite rejection, negative-confidence rejection, and tensor-storage independence.
- [ ] Run tests before implementation and preserve the expected failing output.
- [ ] Implement batch recovery from ZIP local headers, pickle storage offsets, and fixed-length raw storage hashes.
- [ ] Implement public VGGT inference with one 20-view scene batch at a time, official preprocessing, autocast selected from GPU capability, and direct depth/confidence outputs.
- [ ] Squeeze only documented singleton dimensions; save `float32` CPU `contiguous().clone()` tensors under `priors/Ref-NeRF/refnerf/ball/{train,test}/depth/r_N.pth`.
- [ ] Validate all 300 files, aggregate quantiles/statistics, detect constants/NaN/Inf/negative confidence, and write five deterministic RGB/depth/confidence panels.
- [ ] Run tests again and require zero failures.

### Task 4: Generate and validate ball priors

**Files:**
- Create: `/data1/liuly/reproduction/ref_dgs_diagnostic_vggt/20260720_151729/priors/Ref-NeRF/refnerf/ball/`
- Create: `/data1/liuly/reproduction/ref_dgs_diagnostic_vggt/20260720_151729/prior_validation.json`
- Create: `/data1/liuly/reproduction/ref_dgs_diagnostic_vggt/20260720_151729/visualizations/`

- [ ] Run a one-batch memory/shape probe on one recovered 20-view train group and record peak allocated/reserved GPU memory.
- [ ] If the 20-view probe succeeds, generate all five train and ten test groups without changing grouping; validate each group immediately.
- [ ] If the probe OOMs, stop and report the exact 20-view protocol as infeasible on 24 GB rather than inventing an independent-view fallback.
- [ ] Require 100 train plus 200 test files, exact image-name coverage, finite same-shaped depth/confidence, nonnegative confidence, and non-degenerate distributions.

### Task 5: Build isolated Ref-DGS runtime and run two-iteration smoke

**Files:**
- Create: `/data1/liuly/reproduction/ref_dgs_diagnostic_vggt/20260720_151729/runtime/Ref-DGS/`
- Create: `reproduction/ref_dgs_diagnostic_vggt/20260720_151729/runtime_train_diagnostic.patch`
- Create: `/data1/liuly/reproduction/ref_dgs_diagnostic_vggt/20260720_151729/logs/smoke_i2_train.log`

- [ ] Copy a clean tracked Ref-DGS snapshot to the diagnostic runtime and link `priors` only to regenerated depth plus official read-only normal PNGs.
- [ ] Apply logging-only instrumentation that prints finite total, PBR, alpha, normal, VGGT-normal, and VGGT-depth losses when `REFDGS_DIAGNOSTIC_LOSS_LOG=1`; do not change optimization.
- [ ] Run the same prior-failing command with GPU selected after a fresh occupancy check: `train.py --eval --iterations 2 --run_dim 64 --save_iterations 2`.
- [ ] Require exit 0, camera count 100/200, two finite logged iterations, enabled nonzero VGGT terms, and both Gaussian PLY files at iteration 2.

### Task 6: Run short renderable diagnostic chain

**Files:**
- Create: `/data1/liuly/reproduction/ref_dgs_diagnostic_vggt/20260720_151729/smoke/ball_short/`
- Create: `/data1/liuly/reproduction/ref_dgs_diagnostic_vggt/20260720_151729/logs/short_train.log`
- Create: `/data1/liuly/reproduction/ref_dgs_diagnostic_vggt/20260720_151729/logs/short_render.log`

- [ ] Run 20 iterations with official ball flags plus `--save_iterations 20`; this is diagnostic, not extrapolated quality evidence.
- [ ] Run `render.py --dataset shiny --iteration 20` and preserve render metrics and TSDF output.
- [ ] Require 200 test renders, finite PSNR/SSIM/LPIPS/MAE, and non-empty mesh; label all artifacts `NON-CANONICAL` in the report and metadata.

### Task 7: Final verification and handoff

**Files:**
- Create: `/data1/liuly/reproduction/ref_dgs_diagnostic_vggt/20260720_151729/diagnostic_report.md`
- Create: `/data1/liuly/reproduction/ref_dgs_diagnostic_vggt/20260720_151729/code_diff.patch`

- [ ] Re-run JSON, Python, tests, prior schema, file-count, finite-metric, checkpoint, render-count, and mesh checks from scratch.
- [ ] Recompute canonical hashes and assert they equal Task 1 exactly.
- [ ] Record code/runtime diff, VGGT/weight identities, generation timing, peak memory, smoke exits, unresolved author-protocol differences, and estimated 14-scene cost.
- [ ] Recommend expansion only if both smokes and all prior quality gates pass; otherwise report the new exact blocker.

