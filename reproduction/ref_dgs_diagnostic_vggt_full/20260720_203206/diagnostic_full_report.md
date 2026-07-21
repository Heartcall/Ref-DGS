# NON-CANONICAL VGGT-REGENERATED PRIORS — NOT A STRICT PAPER REPRODUCTION

## Status

`incomplete_resource_gate`. This run does not reproduce Table 1–4 and contains no newly trained diagnostic metrics.

All 14 camera manifests and the generalized generation/validation tools are ready. The tools pass 16 new tests plus the 12 completed-ball regression tests. The existing ball priors were reused read-only and passed a fresh full validation: 300 files, correct schema and camera mapping, finite non-constant values, 300 readable official normals, 300 SHA-256 records, and five visual checks without an obvious flip, crop, index, or foreground-confidence error.

## Blocker

The remaining 2,524 priors alone project to 5.422 decimal GB from measured ball file sizes. With the already created detached worktree, the minimum Phase A footprint is 5.511 GB before logs and visualizations. A completed ball render chain occupies 2.20 GB; the same output class across 14 scenes is conservatively about 30.8 GB before full-training checkpoints and meshes. This exceeds the explicit 5.4 GB stop threshold, so no new VGGT inference, smoke, full training, rendering, geometry evaluation, or FPS test was launched.

In addition, `/data1` later remounted read-only. The completed ball validation and its visualizations were already durable there, but the final local summary updates could not be synchronized. Resolving this requires host/storage administration; no sudo or remount was attempted.

## Fixed configuration

- Ref-DGS commit: `490dc585a2d329928363e94f5f91951a61ddee0c`
- VGGT commit: `a288dd0f14786c93483e45524328726ab7b1b4ce`
- VGGT-1B revision: `860abec7937da0a4c03c41d3c269c366e82abdf9`
- Checkpoint SHA-256: `d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0`
- Planned runtime: physical GPU 1, RTX A5000 24 GB; paper hardware is RTX 4090 24 GB.
- Grouping: transform-file order, consecutive groups of 20, final partial group allowed; no author-equivalence claim. ball retains its previously recovered storage-metadata grouping.
- Serialization: float32 CPU `[518,518]` `depth_map` and `depth_conf`, each independently contiguous and cloned.

## Isolation

The canonical and completed-ball directories are referenced only read-only. The new root contains a detached worktree and read-only symlinks to the fixed VGGT code/environment/checkpoint and completed ball priors. Before/after hashes are recorded in `protected_delivery_sha256_before.txt` and `protected_delivery_sha256_after.txt`.

## Required decision

Resume only after authorizing a larger new-storage budget or naming another output root. A practical conservative allowance for Phase A–D is at least 40 GB; 50 GB leaves safer room for checkpoints, meshes, logs, retries, and filesystem overhead. This is an engineering storage estimate, not an estimate of algorithm quality.

The host must also restore `/data1` to read-write or provide a different writable output root.
