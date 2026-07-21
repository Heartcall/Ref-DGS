# Ref-DGS Tables 1-4 reproduction report

Status captured: 2026-07-20 14:35 +0800  
Canonical status: **BLOCKED_CORRUPT_OFFICIAL_DEPTH_PRIORS**

## Outcome first

Repository, paper, environment, GPU, storage, code configuration, all 14 target scenes, evaluation inputs, official per-scene flags, and the released FPS implementation were audited. After user authorization, the two required synthetic-prior subtrees were downloaded from the exact official Hugging Face revision. A real 2-iteration `ball` smoke command was then launched with the frozen official flags. It exited with status 1 while constructing the first training camera because the official depth-prior `.pth` files are truncated PyTorch ZIP archives. The downloaded files match the official object hashes, so this is an upstream-data defect rather than a local transfer error. No canonical training was launched and no reproduced metric, average, training time, FPS, render, normal, or mesh is claimed.

## Repository and workspace

- Official remote: `https://github.com/njfan/Ref-DGS.git`
- Branch/commit: `main` at `490dc585a2d329928363e94f5f91951a61ddee0c`, equal to `origin/main` at audit time.
- `git submodule status --recursive` returned empty; the extension directories are tracked as ordinary repository content rather than registered submodules.
- Existing untracked build/vendor paths were present before this reproduction and were left untouched: `nvdiffrast/`, extension `.egg-info/` directories, and `submodules/simple-knn/build/`.
- The official batch scripts were not executed directly because they hard-code another machine's paths and run `rm -rf <model_path>/*`. Equivalent per-scene flags are frozen in the execution plan, with every result written to a new timestamped directory.

## Paper and implementation settings

The arXiv v3 PDF (`2603.07664v3`, 2 June 2026) was rendered locally with Ghostscript and visually checked on pages containing Tables 1-4 and Section 4. The audit confirms:

- The two Gaussian sets are independently instantiated, trained, stepped, densified, pruned, and opacity-reset in the same training loop.
- `SphMipEncoding` uses `n_levels=9`, `plane_size=512`, a base parameter tensor with spatial size `512 x 1024`, and `sph_dim=4`.
- The local reflection Gaussian feature dimension is `gsfeat_dim=4`, matching the global Sph-Mip feature dimension.
- Official scripts set `--run_dim 64`; `light_mlp` has three hidden Linear/ReLU blocks of width 64 and a final RGB layer.
- GlossySynthetic normal MAE is computed by reading GT depth and passing it through the official `depth_to_normal` path.
- ShinySynthetic reports only normal MAE because its GT meshes contain visible outer and invisible inner layers.
- `render.py` defaults to TSDF `--voxel_size 0.002`; the PDF Table 2 caption explicitly reports `CD x 10^2`.
- NVS metrics are per-view PSNR, SSIM, and LPIPS with `net_type='vgg'`, then arithmetic means over views.

## Dataset audit and safe mapping

### ShinySynthetic

Source: `/data/liuly/dataset/3DGS/Shiny Blender Synthetic`

Each of `ball`, `car`, `coffee`, `helmet`, `teapot`, and `toaster` has 100 train and 200 test frames at 800 x 800, with 100/200 GT normal maps. `ball` also has explicit alpha maps; the other images carry RGBA channels. The stock loader recognizes the Blender transforms.

The loader unconditionally writes `points3d.ply`, so formal runs must not point directly at these source directories. A writable staging scene will link the immutable images/transforms and hold a local `points3d.ply`. Its path must include `refnerf` because `scene/cameras.py` uses that substring and parent directory names to resolve priors.

### GlossySynthetic

Raw source: `/data/liuly/dataset/3DGS/GlossySynthetic`  
Converted source: `/data/liuly/dataset/3DGS/GlossySyntheticConverted`

Each converted target scene has 112 train and 16 test frames at 800 x 800 plus `eval_pts.ply`. Each raw scene has 128 camera pickles and 128 GT depth PNGs. The converted directories do not contain the `depth/` directory that `render.py --dataset glossy` expects. Formal staging will therefore link `depth/` to the corresponding raw scene root without changing either source tree.

NeRF Synthetic exists under the data root but is excluded from every command, mapping, and result file.

## Environment and hardware

- Conda: `refdgs`; interpreter `/home/liuly/anaconda3/envs/refdgs/bin/python`
- Python 3.11.15; PyTorch 2.5.1+cu118; torchvision 0.20.1+cu118; cuDNN 9.1
- Open3D 0.19.0; NumPy 2.3.5; OpenCV 4.13.0; SciPy 1.17.1
- All required CUDA extension imports passed.
- Driver 535.247.01; four RTX A5000 24 GB and four RTX 3090 Ti 24 GB are visible with approved host execution. The paper used one RTX 4090 24 GB.
- `/data1` has about 2.3 TiB free and is the required destination; the repository filesystem has only 9.5 GiB free and `/data` is full.

## Prior gate and exact blocker

No compatible local files matching the official Shiny or GlossySynthetic prior layouts were found before download. Official Hugging Face metadata at revision `f7a411e493f5aa621a1bd3dcf5344058cf8bfa0e` reports:

| Required subtree | Files | Bytes |
|---|---:|---:|
| `Ref-NeRF/refnerf/**` | 3,600 | 15,923,811,588 |
| `Glossy/GlossySynthetic/**` | 2,048 | 9,449,481,267 |
| Total | 5,648 | 25,373,292,855 |

The authorized download was started with the repository's documented `hf download` method and exact revision. The high-concurrency attempts were stopped after reproducible stalls and resumed with two workers; every attempt is preserved in separate logs. Once every target scene had direct invalid-object evidence, the remaining bulk transfer was stopped to avoid downloading about 9 GiB of already-proven unusable payload. The preserved partial mirror contains 3,482 official files totaling 16,563,484,949 bytes, plus untouched Hugging Face cache metadata.

The files that have completed download reveal an upstream defect:

- Across all files actually mirrored, 1,654/1,654 normal PNGs open successfully and 0/1,828 depth `.pth` files load successfully. All 1,828 failing depth files match their official Hugging Face SHA-256 metadata.
- GlossySynthetic is complete: each of 8 scenes has 128/128 readable normal PNGs and 0/128 readable depth priors.
- Shiny `ball` and `car` are complete: each has 300/300 readable normals and 0/300 readable depth priors.
- `coffee` has 201 downloaded depth priors, all official-hash matches and all unreadable; its targeted `train/r_0.pth` is included.
- `helmet`, `teapot`, and `toaster` each have a targeted official `train/r_0.pth`; all four targeted files including `coffee` are exactly 8,388,608 bytes, match official metadata, and raise the same error. Their SHA-256 values are recorded in `prior_validation_final.log`.
- For example, `ball/test/depth/r_0.pth` has SHA-256 `ffb6a3d71d86d3ee9cebdc80aaa93a9f4f472e4fe741ab25b424652bfb0b1cac`, identical to the official metadata.
- A representative Glossy object, `angel_blender/depth/0.pth`, has local SHA-256 `356c651bd80a4d9542f8324a76a03a66f14374c515b22260c339f2586f4dd33e`, identical to the official `x-linked-etag` and metadata.

Binary inspection shows why a container-only repair is impossible: `angel_blender/depth/0.pth` declares two FloatStorage objects of 5,366,480 floats each (about 42.9 MB total raw payload) but the official object is only 8,912,896 bytes and stops inside the first storage. `scene/cameras.py` requires both `depth_map` and `depth_conf`, so missing numerical payload cannot be reconstructed by adding a ZIP central directory. Dummy confidence, zero priors, skipped cameras, or disabling the VGGT loss would change the method and are forbidden for canonical reproduction.

## Released-code FPS protocol

The only official repository timing implementation is inside `GaussianExtractor.reconstruction`. It processes the same 800 x 800 scene views used for metrics, records `time.time()` immediately around `render` and `render_ref`, performs one pass, has no explicit warmup, and has no `torch.cuda.synchronize()` before or after timing. FPS is `1 / mean(render_times)`. Git history shows this implementation is unchanged since the initial release commit.

This exact released-code value will be recorded if the run is unblocked. Because asynchronous CUDA timing can affect this method and the available GPUs differ from the paper's RTX 4090, cross-hardware FPS will be reported descriptively rather than used as an algorithmic pass/fail criterion.

## Results versus paper

All reproduced fields are intentionally empty; the paper values and blocked statuses are in `results.csv` and `results.json`. No mean is computed from missing cells.

| Table | Dataset/metric | Reproduced | Paper | Absolute difference | Status |
|---|---|---:|---:|---:|---|
| 1 | Shiny normal MAE mean | - | 1.43 deg | - | blocked_corrupt_official_depth_priors |
| 1 | Shiny training time mean | - | 12.6 min | - | blocked_corrupt_official_depth_priors |
| 2 | Glossy CD x 10^2 mean | - | 0.62 | - | blocked_corrupt_official_depth_priors |
| 2 | Glossy normal MAE mean | - | 1.88 deg | - | blocked_corrupt_official_depth_priors |
| 3 | Shiny PSNR / SSIM / LPIPS | - | 35.21 / 0.975 / 0.053 | - | blocked_corrupt_official_depth_priors |
| 3 | Glossy PSNR / SSIM / LPIPS | - | 30.63 / 0.958 / 0.052 | - | blocked_corrupt_official_depth_priors |
| 4 | Shiny released-code FPS | - | 76.34 | - | blocked_corrupt_official_depth_priors |

## Failures, retries, and fixes

- The authentic-prior `ball` smoke was actually launched on RTX A5000 GPU 1 with `--eval --iterations 2 --run_dim 64 --save_iterations 2`. It exited 1 after 12.28 seconds in `scene/cameras.py:111`, at the first `torch.load(depth_path)`. The command, full traceback, resource usage, and exit code are preserved.
- The first download at eight workers reached about 8.9 GiB and then stopped making progress for more than seven minutes. It was interrupted with exit 130 and resumed with two workers without deleting completed objects. A later 4-worker trial added only 16 files in about 30 minutes and was also interrupted with exit 130. The final four-file targeted request spent about three hours under service-side throttling before completing.
- The first local PDF tooling attempt found Poppler absent. No package was installed; the already available Ghostscript rendered the official PDF successfully for visual inspection.
- No core Ref-DGS code was modified and no compatibility fix was applied. Reconstructing absent tensor payload is not a compatibility fix.
- No checkpoint was reused, and no prior smoke result was promoted to canonical evidence.

## Artifact locations at this gate

- Execution plan: `docs/superpowers/plans/2026-07-20-ref-dgs-main-tables-reproduction.md`
- Environment: `environment.txt`
- Command record: `commands.sh`
- Machine-readable status: `results.csv`, `results.json`
- Full report: `reproduction_report.md`
- Paper audit temp files: `tmp/pdfs/ref_dgs_v3.pdf` and rendered page PNGs
- Smoke log: `logs/smoke_ball_i2_train.log`; exit code: `logs/smoke_ball_i2_train.exitcode`.
- Final prior validation: `prior_validation_final.log` (1,828 hash-matched depth failures and 1,654 readable normals).
- Staging audit: `audit/staging_validation.txt` (14 writable staging scenes, no broken symlinks).
- Smoke directory: `smoke/ball_i2`; it contains initialization metadata only and no checkpoint/render/normal/mesh.
- Download logs: `logs/prior_download.log`, `logs/prior_download_w8.log`, `logs/prior_download_retry_w2.log`, `logs/prior_download_retry_w4.log`, `logs/prior_download_retry2_w2.log`, and `logs/prior_download_targeted_shiny_r0.log`.
- Canonical checkpoints/renders/normals/meshes: none, because the authentic-prior smoke failed before iteration 1.

## Exact continuation condition

Full 14-scene execution requires corrected official ShinySynthetic and GlossySynthetic depth priors containing both complete `depth_map` and `depth_conf` tensors, or an author-specified deterministic regeneration procedure with the exact VGGT version, checkpoint, preprocessing, batching, and serialization settings. The replacement must pass full `torch.load` and shape/finite-value validation before the same `ball` smoke is retried.
