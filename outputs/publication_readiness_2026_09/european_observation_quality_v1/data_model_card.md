# Input Repair Card

Scope: source-development observation audit plus an untrained neural-input
variant, not a deployable checkpoint. `audit.json` contains fresh aggregate
diagnostics from cached-verified source inputs and freshly reparsed authorized
raw members. Whole files are parsed for row identity; features use only the
query and its history. Future labels are neither scored nor used to select rows.

Population: 318,969 complete-history targets, 163 recordings, 12 localities.
Incomplete-history agents are context candidates only; target eligibility and
future-label availability rules do not change. The fixed task remains eight
observations and twelve requested steps at raw-frame stride12 for this source.
Nothing here calibrates seconds, homography, meter-per-pixel or physical safety.

New input: up to eight nearest current-visible agents, each with eight position
and relative-time slots plus validity mask. Missing slots are zero; no history
or future is imputed. Current position and agent ID determine ordering. The
original ego and causal baseline tokens remain unchanged.

New neural variant: observed-token Transformer attention with a conditioner
whose radius includes only valid observations, including partial histories.
The same ego-motion bounded output and baseline choice API are preserved. It
has the same trainable parameter shapes as the old model. Initialization,
forward, backward and masking tests are engineering checks only. Real training,
validation selection, safety calibration and predictive comparison: not_run.

Limitations: detector tracks may have identity or localization errors;
direction/box changes are ambiguous; short-neighbor history can itself be noisy;
neighbor-count capacity remains eight; image evidence is not used in this
variant. Many overlapping queries and uneven locality sizes preclude treating
the row count as independent sample size. More visible tokens may worsen
prediction. Independent roles, frozen policies and prior negative evidence
remain unchanged. Stage5C and SMC remain disabled.
