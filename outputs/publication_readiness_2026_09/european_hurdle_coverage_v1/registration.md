# Matched-Coverage Risk Ranking: Source Diagnostic

## Material Passport

25 September 2026. Registered before this diagnostic's new decisions/readout.
The parent hurdle experiment is completed and has already informed this question;
its sources are opened development data, not independent final-test data.
Parent analysis SHA-256:
`29bb7dc9a39bda42699094b7176e313934766e2202e1d3b74b19e6845fb1b331`.

Question: does explicit occurrence/severity supervision improve intervention
ordering at the same count, or mainly change how often intervention occurs?
The preceding study changed coverage as well as accuracy. This diagnostic uses
frozen forecast and score checkpoints; it trains nothing and relaxes no
deployment risk constraint.

## Fixed Comparison

Keep both candidates (neural and damping0.97), both risk events (all and easy),
all three folds and seeds 17/29/43: 36 groups, four views each. The original
utility head is identical within each group. Original product-MSE and hurdle
decisions must reproduce their parent hashes exactly.

Within each excluded locality, form the common causal pool: observed movement,
positive original predicted net utility, and positive predicted reference mass
from both risk heads. Nonfinite or negative scores are errors. If an original
decision falls outside this common pool, fail rather than silently trim it.
Rank by ascending predicted event-harm/reference-mass ratio, breaking ties by
the frozen integer row ID. Labels, future masks and future-defined easy/hard
membership never enter the ranking or count.

Two anchors are retained without selecting one afterwards:

1. At the original product-MSE count in each locality, compare product ranking
   with hurdle ranking (`hurdle_at_product`).
2. At the original hurdle count in each locality, compare hurdle ranking with
   product ranking (`product_at_hurdle`).

Each ranking must reconstruct its own thresholded anchor at its own count.
Counts include all causally eligible rows before outcome-support filtering.
Evaluated counts may differ because unavailable future labels stay unavailable.
This is an offline full-locality ranking diagnostic, not a causal streaming
deployment procedure. Counterfactual top-k choices may violate the original
predicted 2% risk rule; retain and count those violations, never call them safe
deployments. Do not invent an optimized k, threshold, new candidate or new split.

## Additive Decomposition

Let P and H be errors under original product and hurdle rules, Hp the hurdle
ranking at product counts, and Ph product ranking at hurdle counts. Within each
locality/subset, normalize error-sum differences by the SAME CV error sum:

```text
total = (P - H) / CV
path 1: total = (P - Hp) / CV + (Hp - H) / CV
                  ranking             coverage
path 2: total = (P - Ph) / CV + (Ph - H) / CV
                  coverage            ranking
```

Multiply by 100 for percentage points of CV-normalized improvement. This is
an exact accounting identity, not a unique causal allocation: report both paths.
Do not add percentages with changing reference denominators. Preserve undefined
zero-reference or unsupported localities instead of dropping them from means.

Report all/easy/hard ADE, actual FDE label support, worst-locality positive-easy
degradation, zero-CV added error, intervention counts, selected unknown labels,
predicted-risk violations, selected harm and tail errors. Keep 3,000 paired
locality-bootstrap draws per contrast, all three seeds and all 144 views.
Report 72 matched-ranking contrasts and both coverage decompositions. Views
share twelve development localities; uncertainty is conditional, not adjusted
for multiplicity. Do not select the favorable subset, fold or anchor.

## Execution and Verification

Local arm64 environment, CPU4/inter-op1/workers0. First run one decision-only
group as a runtime/memory pilot, then freeze all 36 decision archives before
outcome evaluation. File locks, atomic checkpoints per group, manifest hashes
and PID/heartbeat records support restart without redoing completed work.
No optimizer or new trajectory inference is required. Available disk is about
27GiB; stop safely below10GiB, never delete unrelated data.

Verify decisions with a separate scalar sorting implementation; recompute raw
coordinate errors and per-locality sums independently, then reproduce the
bootstrap arithmetic and both additive paths. Retain the parent 72 original
view matches and scoped test scope. Private decision arrays are Git-ignored;
public outputs contain only aggregate metrics, hashes, code and reports.

The preceding CREATE queue observation at09:47:01UTC is cached evidence, not a
new live query or job-completion claim. This small local score analysis needs
no remote training. The specific remote M3W asset directory remains unverified.
Other projects and their jobs are untouched.

No independent confirmation or deployment promotion. Released detector-track
image pixels,8observed/12predicted rawstride12, not legacyt50, seconds, metric,
human gold, physical safety, true3D or foundation. Stage5C andSMC remain off.
