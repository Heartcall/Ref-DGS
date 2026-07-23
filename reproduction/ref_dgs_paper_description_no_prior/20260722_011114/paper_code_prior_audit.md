# Paper/code audit: external geometry priors

Classification: **PAPER-DESCRIPTION NO-GEOMETRY-PRIOR EXPERIMENT — NOT UNMODIFIED OFFICIAL CODE**.

## Paper evidence

Source: `tmp/pdfs/ref_dgs_v3.pdf` (11 pages, local repository evidence).

- Full-text counts: `VGGT=0`, `depth prior=0`, `normal prior=0`.
- Section 3 describes the dual Gaussian representation, global/local specular features, and the physically-aware shader, but does not define an external depth or normal supervision term.
- Section 4.1 says the two Gaussian sets are jointly optimized using the 2DGS optimization/densification strategy and lists Sph-Mip/shader dimensions and RTX 4090 hardware; it does not disclose external geometry priors.
- The paper's “label-free” discussion concerns reflection-fraction labels, so it is not affirmative proof that no depth/normal priors were used. The correct conclusion is narrower: the external priors used by the release are undisclosed in the paper.

## Released-code evidence

Base commit: `490dc585a2d329928363e94f5f91951a61ddee0c`.

- `README.md` explicitly instructs users to download depth and normal priors and states that synthetic priors are inferred by VGGT.
- `arguments/__init__.py` defaults to `vggt_weight=0.05` and `vggt_until_iter=15000`.
- `train.py` adds confidence-weighted external normal and depth losses while `iteration < vggt_until_iter`.
- `scene/cameras.py` unconditionally loads each synthetic camera's `depth_map`, `depth_conf`, and normal PNG. Setting only `vggt_weight=0` would therefore still require and load prior files.

## Frozen no-prior interpretation

- `REFDGS_DISABLE_GEOMETRY_PRIOR=1` prevents the isolated camera loader from opening any depth, confidence, or normal prior.
- `--vggt_weight 0 --vggt_until_iter 0` disables the complete external-prior loss branch and prevents dereferencing disabled tensors.
- The internal 2DGS rendered-normal/surface-normal consistency regularizer remains enabled. It is derived from the model's own rendering buffers and is not an external normal prior.
- The runtime intentionally contains no `priors/` path. Successful camera construction is an executable negative-access check.
- All other model, split, iteration, densification, rendering, TSDF, and metric definitions remain unchanged.

## Claim boundary

This experiment tests the implementation implied by the paper's disclosed method, but it cannot establish whether the paper numbers were generated without priors. It must not replace either the canonical “official priors corrupted” conclusion or the separate regenerated-VGGT diagnostic result.
