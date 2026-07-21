# Diagnostic failure evidence

All entries are non-canonical and occurred before the final successful outputs.

## Checkpoint transfer stalls

The official Hugging Face CLI transfer stopped changing size at 908,831,903
bytes, was interrupted with exit 130, then stopped again at 1,041,651,849 bytes
and was aborted with exit 1. The same official revision URL was completed with
curl. Final size was 5,026,874,952 bytes and final SHA-256 exactly matched
`d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0`.

## VGGT probe attempt 1: stale DPT layout assumption

Inference completed, then serialization stopped before writing final priors:

```text
ValueError: expected 20 views in depth [1,S,1,H,W], found 20: (1, 20, 518, 518, 1)
```

Root cause: the DPT-head docstring says channels-first, while current VGGT's
top-level docstring and measured output are channels-last. A regression test was
added and the serializer now normalizes either official layout without changing
values.

## VGGT probe attempt 2: confidence rank assumption

Inference completed again, then serialization stopped before final writes:

```text
ValueError: depth/confidence output shape mismatch: (1, 20, 518, 518, 1) vs (1, 20, 518, 518)
```

Root cause: current public VGGT returns depth `[B,S,H,W,1]` and confidence
`[B,S,H,W]`, exactly as the top-level model docstring states. A 4-D confidence
regression test was added and confidence is only unsqueezed to the serializer's
internal channels-first representation.

## Verification after fixes

The final identical probe exited 0, saved 20 valid files, and recorded a peak of
9,068,212,224 allocated GPU bytes. The final suite has 12 passing tests, and the
subsequent 300-file generation, validation, 2-iteration training, 20-iteration
training, rendering, metrics, and TSDF steps all exited 0.

