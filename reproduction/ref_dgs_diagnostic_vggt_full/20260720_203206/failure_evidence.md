# NON-CANONICAL VGGT-REGENERATED PRIORS — RESOURCE GATE

No GPU inference has been launched in this full-diagnostic directory.

The user-defined stop condition requires pausing when estimated new storage materially exceeds 5.4 GB. The estimate was recalculated from measured completed-ball artifacts before Phase A generation:

- Completed ball priors: 300 files, 644,410,600 payload bytes; mean 2,148,035.33 bytes/view.
- Remaining views to generate: 2,524.
- Projected remaining prior payload: 5,421,641,181 bytes (5.422 decimal GB), before manifests, validation JSON, SHA manifests, logs, and 70 visualizations.
- New detached Ref-DGS worktree already occupies approximately 89–101 MB.
- Minimum Phase A run-root estimate: 5,510,955,927 bytes (5.511 decimal GB) before validation/log/visualization overhead. ball is planned as a read-only reuse rather than a copy.
- The completed 20-iteration ball chain's train/test/point-cloud outputs occupy 2,197,757,840 bytes. Scaling only that observed render/output footprint to 14 scenes gives a conservative 30.8 GB proxy; full 30,000-iteration checkpoints, meshes, logs, and additional evaluations would add more.

Therefore the complete requested Phase A–D run is expected to exceed 5.4 GB by a large margin. Under the explicit stopping rule, generation is paused pending an increased storage authorization or a user-selected external output root/budget. No model, loss, split, iteration, prior dtype, or evaluation setting was changed to reduce storage.

After the ball revalidation completed, `/data1` changed from read-write to a read-only mount (`/dev/sdc on /data1 type ext4 (ro,nosuid,nodev,relatime)`). A subsequent metadata synchronization failed with `Read-only file system`. No remount or sudo action was attempted. The full local metadata mirror remains at `/home/liuly/Surface_Reconstruction/Glossy/Ref-DGS/reproduction/ref_dgs_diagnostic_vggt_full/20260720_203206`; the external run root contains the earlier synchronized tools/manifests plus the completed ball validation log and visualizations, but not every later summary update.

Tooling failures retained during preparation:

1. The first generalized tree validator incorrectly treated Glossy test priors as extra train files because both splits share one depth directory. Fixed by moving the exact bijection check to whole-scene scope; 28 tests pass afterward.
2. The first manifest builder rejected Shiny's legal train/test reuse of names such as `r_0`, although Shiny prior paths are split-scoped. Fixed by applying the overlap rejection only to Glossy, where train/test share one depth directory; 28 tests pass afterward.
