# Ref-DGS ball regenerated-VGGT-prior diagnostic report

> **NON-CANONICAL: regenerated with public VGGT; not an official paper reproduction.**
>
> This report does not replace the canonical conclusion that all 1,828 released
> depth priors are upstream-corrupt. None of the values below is a reproduced
> Table 1--4 value.

## Outcome

The `ball` diagnostic successfully traversed the complete Ref-DGS chain:

`public VGGT multi-view inference -> 300 validated depth/confidence priors -> all cameras loaded -> prior-weighted training -> saved two-Gaussian checkpoint -> official train/test rendering -> PSNR/SSIM/LPIPS/normal MAE -> voxel-size 0.002 TSDF mesh`.

All commands in that chain exited 0 after two generator output-layout
compatibility fixes. The result establishes that current public VGGT outputs can
be serialized in Ref-DGS's required schema and can stably drive a 20-iteration
diagnostic run. It does **not** establish numeric equivalence to the unpublished
VGGT generation setup used by the authors, nor stable full 15,000-iteration
convergence.

The canonical directory
`/data1/liuly/reproduction/ref_dgs_main_tables/20260720_010227` was never used as
an output. Its five delivery hashes were rechecked at the end and are unchanged.

## Isolation and identities

- Ref-DGS commit: `490dc585a2d329928363e94f5f91951a61ddee0c`.
- Ref-DGS runtime: detached disposable worktree at the same commit under this
  diagnostic root.
- VGGT repository: `https://github.com/facebookresearch/vggt`.
- VGGT commit: `a288dd0f14786c93483e45524328726ab7b1b4ce`.
- Checkpoint: official `facebook/VGGT-1B/model.pt`, revision
  `860abec7937da0a4c03c41d3c269c366e82abdf9`.
- Checkpoint size: `5,026,874,952` bytes.
- Checkpoint SHA-256:
  `d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0`
  (exactly the official LFS object hash).
- GPU: physical GPU 0, NVIDIA RTX A5000 24 GB; driver 535.247.01.
- Original RGB/alpha/camera files stayed under the read-only dataset. The
  diagnostic staging tree contains only symlinks to them.
- Official intact normal PNGs are mounted through read-only symlinks. Only
  `depth_map` and `depth_conf` were regenerated.

Full software versions are in `environment.txt`; exact generation choices and
known differences are in `vggt_generation_config.json`.

## Ref-DGS prior contract found in code

For ShinySynthetic, `scene/cameras.py:104-128` builds a path relative to the
current repository root:

`priors/Ref-NeRF/refnerf/<scene>/<train|test>/depth/<image>.pth`.

The loaded object must be a dictionary containing tensor keys `depth_map` and
`depth_conf`. The practical input contract is a two-dimensional map `[H,W]` for
each key; this diagnostic saves both as CPU contiguous `torch.float32` tensors.

- Depth is expanded to `[1,1,H,W]`, bilinearly resized with
  `align_corners=False` to the RGB size, then squeezed to `[1,H,W]`.
- Confidence is nearest-neighbour resized, transformed with `log(conf+1)`,
  min-max normalized per image, zeroed outside the alpha mask, and raised to
  `tau=1`. There is no additional threshold.
- The official normal PNG is decoded from `[0,255]` to `[-1,1]` and rotated by
  the camera world-view rotation. This diagnostic did not regenerate it.
- Ref-DGS does not crop, invert, normalize, scale, or shift the loaded depth in
  the camera loader.
- `utils/loss_utils.py:101-162` uses confidence-weighted least squares to align
  rendered depth by a detached scale and shift to VGGT depth on foreground
  pixels, then applies confidence-weighted L2. Thus the prior can be arbitrary
  scale, while no numeric scale is manually imposed during serialization.
- `train.py:129-138` applies the same confidence to both the VGGT normal term
  and VGGT depth term. Defaults are weight `0.05` and active iterations
  `iteration < 15000` (`arguments/__init__.py:119-120`).
- Transform-frame names map directly to prior filenames. `Scene` shuffles train
  and test camera lists with seed 42 after loading (`scene/__init__.py:70-73`),
  so prior lookup is name-based rather than inference-order-based.

VGGT's current model describes depth as `[B,S,H,W,1]` and confidence as
`[B,S,H,W]`; this is also what was measured. Depth uses an `exp` activation and
confidence uses `expp1`, so both outputs are positive. The VGGT geometry helpers
treat the depth as per-camera Z depth for OpenCV unprojection; predicted camera
extrinsics, when requested, are camera-from-world with x-right, y-down,
z-forward. No camera output was substituted into Ref-DGS. The public code does
not give these predicted values a guaranteed dataset metric unit, and Ref-DGS's
own scale/shift fit is retained unchanged.

## Multi-view generation protocol

The intact pickle metadata inside the released corrupt files contains tensor
storage offsets and storage prefixes even though the actual tensor archive is
truncated. Those metadata recover exactly five 20-view train groups and ten
20-view test groups for `ball`, including slot order. The manifest is
`ball_batch_manifest.json`.

Each recovered group was passed jointly through VGGT's global aggregator at
`518x518`. The current DPT head then processes its output in frame chunks of 8;
this is the public implementation's memory-saving head behavior, not independent
per-view VGGT inference. No cross-group depth stitching, artificial confidence,
or Metric3D output was used.

Input preprocessing used the official `load_and_preprocess_images(...,
mode="crop")`: Pillow bicubic resize of the square 800x800 input to 518x518,
RGB `[0,1]`, white background. Ampere inference used bfloat16 autocast. The raw
outputs were float32. Every saved view used
`detach().cpu().contiguous().clone()` before `torch.save`, producing independent
tensor storage. Files are 2,147,968--2,148,044 bytes; their smaller size than
the released archives is expected and is not a failure criterion.

Measured 20-view probe: 2.847 s model inference, peak 9,068,212,224 allocated
bytes and 13,503,561,728 reserved bytes. The 15 inference batches summed to
39.874 s; after model initialization, preprocessing/inference/save spanned
48.56 s.

## Prior validation

All 300 files passed a fresh `torch.load(..., map_location="cpu")`, dictionary
schema, tensor type, identical shape, finite depth/confidence, and nonnegative
confidence assertions. Counts exactly match 100 train and 200 test transform
frames and RGB/alpha pairs. Every source image is 800x800; every prior is
518x518 float32. No depth or confidence map is constant.

| Split | Files | Depth min / mean / std / max | Depth p01 / p50 / p99 | Conf min / mean / std / max | Conf p01 / p50 / p99 | Conf zero |
|---|---:|---|---|---|---|---:|
| train | 100 | 0.49239 / 0.83482 / 0.10249 / 1.89630 | 0.67332 / 0.82204 / 1.05242 | 1.0 / 23.4478 / 24.8634 / 97.3416 | 1.0 / 23.9626 / 85.7883 | 0% |
| test | 200 | 0.51194 / 0.87321 / 0.09846 / 1.57790 | 0.71869 / 0.85706 / 1.09017 | 1.0 / 43.3124 / 42.8579 / 122.4124 | 1.0 / 45.6294 / 106.2220 | 0% |
| all | 300 | 0.49239 / 0.86041 / 0.10145 / 1.89630 | 0.67931 / 0.84648 / 1.08384 | 1.0 / 36.6909 / 38.9652 / 122.4124 | 1.0 / 27.7744 / 104.6357 | 0% |

Quantiles use a deterministic uniform sample of up to 4,096 pixels per view;
min/mean/std/max and zero ratios use every pixel. The complete per-view data and
all p00/p01/p05/p25/p50/p75/p95/p99/p100 values are in
`prior_validation.json`.

Median-depth relative changes between adjacent numbered camera views are small
and finite: train mean/p50/max `0.03067/0.02748/0.09589`, test
`0.02542/0.02244/0.08721`. Five visualizations (`train r_0/r_25/r_50`, `test
r_0/r_100`) show correct silhouette, orientation, crop, and indexing. Depth is
smooth on the sphere; noisy white-background depth is assigned VGGT's minimum
confidence and is subsequently removed by Ref-DGS's alpha mask.

## Two-level smoke results

### Original 2-iteration flags

The original canonical smoke flags were retained:

`--eval --iterations 2 --run_dim 64 --save_iterations 2`

Only the diagnostic source/model roots and prior tree changed. Exit code was 0;
wall time was 35.47 s (including all 300 camera loads). Both Gaussian groups,
the light MLP, and direction encoding were saved at iteration 2.

| Iteration | Camera | Total loss | VGGT normal loss | VGGT depth loss | Prior active |
|---:|---|---:|---:|---:|---|
| 1 | r_79 | 0.7684005 | 0.0511649 | 0.00008591 | yes |
| 2 | r_96 | 0.7029419 | 0.0455318 | 0.00008977 | yes |

All logged component values were finite. Loaded confidence ranges were `[0,1]`
after the official loader transform.

### 20-iteration complete chain

The independent 20-iteration run exited 0 and saved iteration 20. All 20
iterations had the prior active and all logged components finite:

- total loss range: `0.471744--0.768400`;
- VGGT normal loss range: `0.038850--0.054079`;
- VGGT depth loss range: `0.00007683--0.00009686`;
- training wall time: 26.08 s, of which most was initial data loading.

Official `render.py --dataset shiny --voxel_size 0.002` exited 0 after 16:27.83
and produced exact counts for renders, GT, depth, and normal: train 100 each,
test 200 each. The test metrics are finite but reflect only 20 iterations and are
reported solely as a chain-integrity diagnostic:

| Diagnostic split | PSNR | SSIM | LPIPS-VGG | normal MAE | renderer FPS |
|---|---:|---:|---:|---:|---:|
| test, iteration 20 | 11.8744 | 0.72254 | 0.56711 | 27.9833° | 51.36 |

These values must not be compared to or entered into paper Table 1--4 results.
The rendered image is visibly an untrained grey Gaussian cloud, as expected at
20 iterations.

TSDF used the code-derived depth truncation 8.06226, voxel size 0.002, and SDF
truncation 0.01. Raw mesh: 3,128,372 finite vertices and 4,125,865 triangles.
Because the model is barely trained, it contains 192,697 connected components;
the official largest-component post-process leaves 1,573 finite vertices and
2,417 triangles. This is valid geometry output, not meaningful surface quality.

## Failures and fixes

1. The Hugging Face CLI connection twice stopped growing while downloading the
   5.03 GB checkpoint. It was stopped without discarding the partial; the same
   official revision URL was then retrieved with curl. The final exact size and
   official LFS SHA-256 were verified before model load.
2. First VGGT probe inference succeeded but the serializer asserted the stale
   DPT-head doc layout `[B,S,1,H,W]`; measured depth was
   `[B,S,H,W,1]`. A channels-last regression test and layout normalization were
   added.
3. Second probe inference succeeded but exposed confidence as `[B,S,H,W]`
   rather than the initially assumed rank-5 layout. A 4-D confidence regression
   test and independent normalization were added.
4. The identical third probe passed, followed by the final 300-file generation.

No Ref-DGS model, loss, optimizer, split, iteration schedule, renderer, or
metric definition was changed. Twelve diagnostic unit tests pass. The only
training variant is `train_diagnostic.py`, a copy of the official script with
environment-gated JSON logging and a finite-value assertion immediately before
the unchanged backward call. Its complete diff is `train_diagnostic.diff`.
Generation/validation tools are new isolated files:
`recover_ball_batches.py`, `generate_ball_vggt_priors.py`, and
`validate_ball_vggt_priors.py`.

The disposable runtime worktree accumulated only Python bytecode cache changes;
the main Ref-DGS tracked source has no modifications. No canonical file was
edited.

## Canonical integrity

End-of-run hashes exactly match the protected baseline:

- `reproduction_report.md`: `49b59306f598c6baac72fee82da3db666e498cf722e479f90ac4783cbd77b8f3`
- `results.csv`: `3c557d6df86892c30ae5b9913dca899aa1ce2e2ff06e8d1875cd94f64af80847`
- `results.json`: `a5f37d83bc9ab544e7747b629881804083cf76e358450c39e8cb616731d5f18c`
- `commands.sh`: `59212a4e91565757f3906d51c461aa8f0a4a2019c1993bee40207665fc8b575c`
- `environment.txt`: `b1939a41467c7532c66843006c80e3d8ad483d4f07fe980154de96356a0094e8`

The canonical result therefore remains: **official depth priors are corrupt and
the strict paper reproduction is blocked**.

## Expansion decision and estimate

The ball gates requested for considering expansion are satisfied: all priors
pass, 2-iteration and 20-iteration losses are finite, a checkpoint exists,
render/NVS metrics/normal/mesh outputs exist, and visualized priors have no
obvious alignment defect.

I recommend expansion only as a clearly named `diagnostic_vggt` experiment,
after user confirmation, not as canonical reproduction. Remaining raw target
views total 2,524 (five ShinySynthetic scenes x 300 plus eight GlossySynthetic
scenes x 128). At ball's measured throughput, pure VGGT inference is about 5.6
GPU-min; preprocessing/save and per-scene model loads make a realistic prior
generation estimate about 12--20 GPU-min plus roughly 5.4 GB additional prior
storage. This estimate excludes training.

Full Ref-DGS training is much larger: the paper's 14-scene lower-bound arithmetic
is about 176 GPU-min on an RTX 4090 if every scene took the reported 12.6 min;
the RTX A5000 is different and likely slower, and the 20-iteration diagnostic
cannot safely extrapolate densification-era runtime. A cautious reservation is
4--8 GPU-hours before rendering/TSDF.

The largest remaining methodological risk is batching provenance. Exact 20-view
group/order was recoverable for `ball`, but released Shiny prior coverage is
incomplete for several scenes, so the author's missing group assignments cannot
always be reconstructed. Any new deterministic grouping would be another
explicit non-canonical choice. GlossySynthetic's current converted data layout
and per-scene 128-view grouping also require a separate audited adapter before
generation.

## Evidence index

- Commands: `commands.sh`
- Environment: `environment.txt`
- Generation protocol: `vggt_generation_config.json`
- Batch manifest: `ball_batch_manifest.json`
- Full prior validation: `prior_validation.json`
- Prior visualizations: `visualizations/`
- Generated priors: `priors/Ref-NeRF/refnerf/ball/`
- Probe/batch timings: `logs/vggt_probe_batches.jsonl`,
  `logs/vggt_ball_batches.jsonl`
- 2-iteration log: `logs/smoke_ball_i2_train.log`
- 20-iteration log: `logs/smoke_ball_i20_train.log`
- Official render/metric/mesh log: `logs/smoke_ball_i20_render_mesh_eval.log`
- Checkpoints and outputs: `smoke/ball_i2/`, `smoke/ball_i20_chain/`
- Representative render:
  `smoke/ball_i20_chain/test/ours_20/renders/r_0.png`
- Representative normal:
  `smoke/ball_i20_chain/test/ours_20/vis/normal/r_0.png`
- Meshes: `smoke/ball_i20_chain/train/ours_20/fuse.ply` and
  `fuse_post.ply`
- Diagnostic code diff: `train_diagnostic.diff`

