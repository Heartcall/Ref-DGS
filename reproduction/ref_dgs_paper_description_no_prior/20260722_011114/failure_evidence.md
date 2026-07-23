# Failure and retry evidence

> **PAPER-DESCRIPTION NO-GEOMETRY-PRIOR EXPERIMENT — NOT UNMODIFIED OFFICIAL CODE**

## Experiment execution

- No formal scene training, rendering, NVS evaluation, normal evaluation, mesh extraction, or GlossySynthetic CD evaluation failed.
- All 14 formal scene attempts are `attempt_001`; none was overwritten or silently excluded.
- The ball 2-iteration and 100-iteration chain tests both exited successfully before the formal batch.

## Setup and reporting retries

1. The first isolated `git apply --check` rejected the hand-written runtime patch as `corrupt patch at line 20`. The root cause was an incorrect added-line count in two hunk headers. Only those hunk counts were corrected; the resulting diff was then applied successfully and is frozen in `code_changes.diff`.
2. The first symlink creation attempt failed because the repository had no `output/` directory. The directory was created without touching an existing output, then both requested links were created and resolved to their independent `/data1/liuly` roots.
3. During the ball render smoke, the command wrapper returned control before its buffered child output was visible. Host PID/GPU/output checks showed the render was still progressing, so no duplicate render was launched; the original process later exited 0.
4. The first portable-report package attempt failed schema validation because the canonical source lacked `source.query.sql`. The report generator was amended with the exact frozen-JSON extraction query and source tables. The second package attempt passed validation and packaging. Browser-based QA could not run because no Chromium executable is installed; structural verification passed and no browser was downloaded.

## Evidence boundary

These retries changed neither model behavior nor any metric. The only runtime change is the explicit loader gate and loss-audit instrumentation recorded in `code_changes.diff`. The released code's external-prior dependency and the canonical corrupted-prior conclusion remain unchanged.
