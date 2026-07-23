# Ref-DGS Paper-Description No-Geometry-Prior Experiment Plan

> **For Codex:** Execute this plan inline with verification checkpoints. Preserve the official-code canonical and VGGT-regenerated diagnostic runs.

**Goal:** Run the 14 synthetic Ref-DGS scenes without loading or optimizing against depth/normal priors, while keeping all other official scene settings unchanged and comparing the result with the prior-assisted diagnostic run.

**Classification:** `PAPER-DESCRIPTION NO-GEOMETRY-PRIOR EXPERIMENT — OFFICIAL LOADER ADAPTED ONLY TO REMOVE UNDISCLOSED PRIOR DEPENDENCY`. This is not the unmodified official-code canonical result.

**Isolation:** Freeze Ref-DGS commit `490dc585a2d329928363e94f5f91951a61ddee0c` in a detached runtime under `/data1/liuly/reproduction/ref_dgs_paper_description_no_prior/<timestamp>/`. Do not modify `/data1/liuly/reproduction/ref_dgs_main_tables/20260720_010227` or `/data1/liuly/reproduction/ref_dgs_diagnostic_vggt_full/20260720_203206_resume`.

---

### Task 1: Freeze evidence and resources

- Record the paper-text search, README/code mismatch, git identity, environment, mount/write probe, GPU inventory, dataset mappings, and protected hashes.
- Verify `/data1/liuly` is writable, has at least 55 GiB free, and GPU 1 is idle.
- Create an independent run root and copy only the required runtime/source metadata.

### Task 2: Remove prior dependency without changing the model

- Add an isolated loader gate controlled by `REFDGS_DISABLE_GEOMETRY_PRIOR=1`.
- When enabled, set `vggt_depth`, `vggt_depth_conf`, and `vggt_normal` to `None` without opening prior files.
- Always pass `--vggt_weight 0 --vggt_until_iter 0`; the latter prevents the training branch from dereferencing the disabled tensors.
- Save `code_changes.diff`; verify the main worktree core files remain unchanged.

### Task 3: Ball smoke gate

- Run ball for 2 iterations from a runtime with no `priors` directory.
- Confirm iterations 1 and 2 complete, losses are finite, outputs save, and logs contain the explicit no-prior marker.
- Run a short train-render-metrics chain before the full batch.

### Task 4: Full ShinySynthetic batch

- Run ball 15k, car 25k, coffee 15k with `albedo_lr=0.002`, and helmet/teapot/toaster 20k.
- After each scene, run official render/evaluation at voxel size 0.002, validate outputs, metrics, finite values, duration, and exit code.

### Task 5: Full GlossySynthetic batch

- Run all eight scenes for 25k with `albedo_bias=2` and `albedo_lr=0.0005`.
- After each scene, run official render/evaluation at voxel size 0.002 and validate NVS, normals, TSDF mesh, CD, duration, and exit code.

### Task 6: Aggregate and compare

- Produce scene-level CSV/JSON and a report comparing no-prior, regenerated-VGGT diagnostic, and paper values.
- Preserve distinct labels; do not write no-prior values into the canonical reproduction columns.
- Recompute protected hashes and verify the canonical and prior-diagnostic directories were not modified.

### Task 7: Safe output links

- Create non-overwriting symlinks under repository `output/` for the completed regenerated-VGGT run and this no-prior run.
- Resolve and verify both links and record them in the final report.
