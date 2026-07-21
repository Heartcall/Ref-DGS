# Ref-DGS Full VGGT-Regenerated Prior Diagnostic Execution Plan

> **For Codex:** REQUIRED SUB-SKILL: Use `executing-plans` to execute this plan task-by-task with evidence checkpoints.

**Goal:** Complete an isolated, non-canonical VGGT-regenerated-prior diagnostic across the 6 ShinySynthetic and 8 GlossySynthetic paper scenes without modifying the canonical or completed ball diagnostic deliverables.

**Architecture:** Reuse the fixed public VGGT repository, checkpoint revision, checkpoint hash, preprocessing, inference dtype, and tested tensor-layout normalization from the completed ball diagnostic. Generalize only the diagnostic manifest/generation/validation orchestration. Preserve original RGB and normal priors through read-only staging links; write regenerated depth/depth_conf, logs, outputs, and reports only under a new timestamped full-diagnostic root.

**Tech Stack:** Python 3.12/3.11, PyTorch CUDA, VGGT-1B, Ref-DGS, pytest, JSON/CSV/Markdown, shell subprocess orchestration.

---

### Task 1: Freeze scope, provenance, and isolation

**Files:**
- Create: `/data1/liuly/reproduction/ref_dgs_diagnostic_vggt_full/<timestamp>/execution_plan.md`
- Create: `/data1/liuly/reproduction/ref_dgs_diagnostic_vggt_full/<timestamp>/environment.txt`
- Create: `/data1/liuly/reproduction/ref_dgs_diagnostic_vggt_full/<timestamp>/commands.sh`
- Create: `/data1/liuly/reproduction/ref_dgs_diagnostic_vggt_full/<timestamp>/vggt_generation_config.json`
- Create: `/data1/liuly/reproduction/ref_dgs_diagnostic_vggt_full/<timestamp>/protected_delivery_sha256_before.txt`

1. Record Ref-DGS/VGGT commits, checkpoint revision/hash, runtime versions, GPU 1 identity, disk budget, source dataset roots, and protected-directory hashes.
2. Create a detached Ref-DGS worktree and independent data/output hierarchy under the new run root.
3. Copy the fixed VGGT source and isolated inference environment into the new root; link the immutable checkpoint read-only.
4. Record the non-canonical label and deterministic grouping policy before generation.

### Task 2: Generalize manifest and validator with tests first

**Files:**
- Create: `tools/build_scene_manifests.py`
- Create: `tools/generate_scene_vggt_priors.py`
- Create: `tools/validate_scene_vggt_priors.py`
- Create: `tests/test_full_diagnostic_tools.py`
- Create: `scene_manifests/<dataset>_<scene>.json`

1. Write failing tests for Shiny split mapping, Glossy disjoint split mapping, deterministic consecutive grouping, final partial batch, tensor layout normalization, schema validation, constant-map rejection, and uniform visualization selection.
2. Run tests and save the expected red evidence.
3. Implement the minimum generalization while preserving the 12-tested ball normalization behavior.
4. Run all diagnostic tests and save green evidence.
5. Generate and audit all 14 scene manifests before any GPU inference.

### Task 3: Phase A generate and validate all priors

**Files:**
- Create: `priors/<dataset>/<scene>/...`
- Create: `logs/generation/<dataset>/<scene>/...`
- Create: `visualizations/<dataset>/<scene>/...`
- Create: `prior_validation_all.json`
- Create: `prior_validation_summary.csv`
- Create: `prior_generation_failures.md`

1. Reuse the already validated ball priors by copying them into the new root and verifying exact source hashes/config metadata.
2. For each remaining scene, run deterministic transform-order consecutive 20-view VGGT inference on physical GPU 1.
3. Immediately validate camera/prior bijection, load/schema/shape/dtype/finite/range/non-constant checks, per-file SHA-256, aggregate statistics, adjacent-view diagnostics, and five uniform visualizations.
4. Fail closed on any invalid scene; diagnose one variable at a time and keep all failure evidence.
5. Aggregate Phase A JSON/CSV/failure records and compare actual storage/time against the approved budget.

### Task 4: Phase B run 14 two-iteration smoke tests

**Files:**
- Create: `staging/<dataset>/<scene>/...`
- Create: `logs/smoke/<dataset>/<scene>/...`
- Create: `smoke_results.csv`
- Create: `smoke_exit_codes.json`

1. Build read-only staging trees combining source RGB/data, official normal priors, and regenerated depth/depth_conf.
2. Derive exact two-iteration commands from the canonical command/config snapshots, changing only staging/prior/output paths.
3. Run scenes serially on physical GPU 1, validating iterations 1/2, finite total/depth/normal losses, prior activation, saved model components, peak memory, and exit code.
4. Stop before full training if any systematic failure remains.

### Task 5: Phase C run serial full diagnostics and official evaluations

**Files:**
- Create: `outputs/<dataset>/<scene>/...`
- Create: `logs/train/<dataset>/<scene>/...`
- Create: `renders/<dataset>/<scene>/...`
- Create: `meshes/<dataset>/<scene>/...`
- Create: `exit_codes.json`

1. Freeze config/code/prior manifest IDs and start full ShinySynthetic runs, then GlossySynthetic, one scene at a time.
2. Monitor the first 100 iterations, all save/eval points, finite losses, Gaussian counts, peak memory, duration, and resumability metadata.
3. Immediately run official test rendering, NVS metrics, normal output, TSDF extraction, geometry evaluation, and official FPS measurement after each successful training.
4. Report Shiny normal MAE only; report Glossy normal MAE and CD x 10^2 using voxel size 0.002.

### Task 6: Phase D aggregate and independently verify

**Files:**
- Create: `diagnostic_full_report.md`
- Create: `results_diagnostic.csv`
- Create: `results_diagnostic.json`
- Create: `failure_evidence.md`
- Create: `code_changes.diff`
- Create: `protected_delivery_sha256_after.txt`

1. Produce per-scene diagnostic results with generation config ID, timings, NVS, geometry, FPS, paper values, absolute differences, and allowed diagnostic status vocabulary.
2. Mark every report/table prominently as `NON-CANONICAL VGGT-REGENERATED PRIORS — NOT A STRICT PAPER REPRODUCTION`.
3. Re-run test suites and artifact-integrity/count/schema/finite checks against the latest files.
4. Recompute the five canonical delivery hashes and protected ball diagnostic hashes, compare before/after, and record exact isolation evidence.
5. Report incomplete/blocked items only from observed evidence; never populate canonical reproduction-value columns.
