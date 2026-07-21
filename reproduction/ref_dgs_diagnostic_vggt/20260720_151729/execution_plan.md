# Ref-DGS ball VGGT regenerated-prior diagnostic plan

> NON-CANONICAL: regenerated with public VGGT; not an official paper reproduction.

1. Preserve and hash-check the canonical delivery at
   `/data1/liuly/reproduction/ref_dgs_main_tables/20260720_010227`; never write
   into it.
2. Audit Ref-DGS prior loading, resizing, confidence normalization, masking,
   loss scheduling, and depth scale/shift alignment from source.
3. Pin the public VGGT repository and checkpoint, recover the author's exact
   20-view batch membership/order from intact metadata in the corrupt official
   prior archives, and generate only `ball` priors.
4. Validate all 100 train and 200 test files for schema, shape, dtype, finite
   values, confidence sign/distribution, image/transform alignment, per-view
   variation, adjacent-view jumps, and five deterministic visualizations.
5. Run the original two-iteration ball smoke flags with only the staging paths
   changed; log both prior-loss components at every iteration.
6. Run an independent 20-iteration diagnostic chain through checkpoint,
   official render/NVS metrics, normal output, and voxel-size 0.002 TSDF mesh.
7. Re-hash canonical deliverables and report diagnostic evidence separately;
   do not write diagnostic values into any Table 1--4 reproduction field.

