# Source Observation Audit and Masked-Neighbor Repair

Status: registered before new observation summaries; source development is
already exposed. This is not independent confirmation or a trajectory result.

## Question and Scope
The cap experiment failed. Does its source input omit useful current-visible
neighbors, and is its eight-step motion estimate sensitive to observation
quality? Audit the same 318,969 complete-history source targets from 163
recordings in 12 admitted source-training localities. Preserve target ordering,
sampling, roles, raw box-center coordinates and future labels. No reserved
selection, calibration or confirmation data are read.

## Fixed Diagnostics
Verify source-role, cache, recording, agent, frame and packed-history hashes.
Reparse only authorized raw recording members and compare exact history boxes
at q-84,...,q with cached inputs. Measure raw-frame presence inside this prefix,
box-size changes, half-pixel concentration, stationary steps, direction
reversals, line-fit residuals and last-difference versus OLS disagreement.
Frame gaps are observation gaps, not proven broken identities. Residuals and
box changes are noise proxies, not human-validated detection errors.

Compare fixed FD, OLS4 and OLS6 extrapolations from the first six observed
steps to the last two observed steps. These are within-prefix checks, not
12-step forecast evaluation. All current targets remain included. No threshold
or smoothing estimator will be selected by future outcomes in this audit.

## Versioned Repair
The old packer uses complete-history targets as its neighbor candidate pool.
The new input-only packer uses every currently visible agent, chooses at most
eight by current distance with agent-ID tie breaks, and preserves per-step
valid masks. Missing coordinates and times are zero; there is no imputation,
future visibility or future-dependent sampling. Ego history and baseline tokens
stay byte-identical. Existing packer/checkpoints remain unchanged. Audit how
often missing-history neighbors enter the new nearest-eight pool and how much
neighbor geometry changes. The new schema is not compatible with treating
old checkpoints as already trained on it: matched retraining is required.

## Decision Boundary
Input omission and its repair can be demonstrated without a learned head.
Prediction lift, rare-event learnability and statistical independence cannot.
If omission is material, the next experiment must compare unchanged legacy
and masked-neighbor input under the same producer/controller exclusions,
seeds, losses, budgets and training-only preprocessing. No test-based policy
search or deployment promotion is authorized by a coverage increase.
The earlier event-label decomposition remains cached evidence; a fresh
event-stratified quality analysis is not part of this input-only run.

## Resource and Claims
Native arm64 Python, four compute threads, zero workers; sequential raw-member
parsing and immutable per-record receipts support interruption recovery.
Maintain at least 10 GiB free space. Local pilot estimates cost; use CREATE only
if justified. Stage5C and SMC remain disabled. Coordinates remain dataset-local
raw-frame, not metric, seconds-level, human gold, true 3D or foundation evidence.
