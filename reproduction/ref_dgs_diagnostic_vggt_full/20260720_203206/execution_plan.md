# NON-CANONICAL VGGT-REGENERATED PRIORS — NOT A STRICT PAPER REPRODUCTION

Run ID: `20260720_203206`

1. Freeze commits, checkpoint, source mappings, deterministic batching, protected hashes, resource estimate, and an isolated detached worktree.
2. Test and generalize manifest/generator/validator tools without modifying Ref-DGS core code.
3. Phase A: reuse and revalidate the completed ball priors; generate and immediately validate the other 13 scenes on physical GPU 1.
4. Phase B: only after all Phase A scenes pass, run serial two-iteration smoke tests for all 14 scenes.
5. Phase C: only after all smoke tests pass and the resource gate is approved, run official full configurations serially, followed by render/NVS/normal/TSDF/geometry/FPS evaluation.
6. Phase D: aggregate diagnostic-only comparisons and re-check the protected canonical and ball diagnostic hashes.

Current checkpoint: manifest tooling and all 14 camera manifests are complete. GPU generation is paused at the user-defined storage stop condition documented in `failure_evidence.md`.

