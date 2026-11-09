# Accuracy repair at fixed predicted risk

I test whether learning OLS4-relative utility can recover the accuracy lost by
the lower-coverage component-risk joint policy. This is a development study on
the same nine exposed source groups, 24 recordings, three role rotations and
three seeds, with eight observations and twelve forecast steps. Units remain
local pixels. DroneCrowd's approved 15 calibration / 30 independent confirmation
groups are not loaded. No original evaluation threshold changes.

Each query inherits the incumbent's exact number of edited agents. The joint
solver cannot increase any of three predicted risk sums, decrease either
predicted selected reference sum, or decrease frozen CV-relative shape gain.
The LP gives a search direction; rounded actions must pass all inequalities and
beat the incumbent objective or the solver retains the incumbent. These are
predicted budgets, not certificates for observed risk.

The frozen motion rays, risk head, and fitting sources are shared. One head
regresses the two conservative gains directly. The factorized head predicts
the observed CV-relative contribution and adds the geometric missing-step
debit weighted by past-only estimates of missingness at each lead. Both train
with the same gains, observed-benefit and missingness supervision on the same
sampled actions. A no-fit control ranks the exact OLS4 geometric term under
fully missing outcomes. We use 1,400 updates per head and fold/seed cell.

I report paired missing-step-conservative gain against CV and OLS4, differences
between arms, intervention counts, unchanged recording-level all/easy risk and
signed easy degradation, and source-group bootstrap intervals. A calibration
cell qualifies only if every calibration recording supports the original rule,
there is at least one edit, and both conservative baseline gains are positive;
among qualifying arms I maximize the smaller gain. No qualified arm means CV
fallback. After freezing the causal decisions I also run an outcome-aware
search with the identical solver and budgets as a capacity diagnostic, not an
oracle optimum, candidate model, or deployment claim.
