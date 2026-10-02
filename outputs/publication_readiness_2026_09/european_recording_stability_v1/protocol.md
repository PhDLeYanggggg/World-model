# Source-Recording Decision Stability

## Material Passport

Code experiment, run/validate; frozen source forecasts and source calibration
labels only. Independent roles closed. This follow-up is motivated by already
exposed development failures, not prospectively independent confirmation.

## Question and Prior Evidence

Does a decision's sensitivity to deleting one calibration recording identify
harmful retained interventions, or merely remove useful coverage? The previous
selected-set recalibration did not improve support (23/72 joint OOF views).
Its four nonempty failures include three unknown-completion cases and one
fully observed harmful set. No case ID defines this experiment's rule.

Existing support-factorization controls rejected useful forecasts and did not
establish equal-coverage utility lift. Crossed-head seeds also did not repair
risk. This control changes neither features nor head seed: it varies only the
source recordings used for empirical calibration, holding the forecast fixed.
It is not a new architecture or a repeated percentile support filter.

A pre-computation metadata audit found 48/72 heads with two source calibration
recordings,12 with three,6 with17 and6 with19. There are58 distinct recordings
in the frozen source roster. These counts already informed this diagnostic;
they are not an unseen finding. Nested deletions may have no supporting record.

## Frozen Computation

Use all72 existing heads, including their three cost-head seeds. Primary is
the original raw-pool joint calibrator; secondary is the rejected iterative
selected-set joint calibrator for mechanism comparison only. Retain q=.9,
higher-order-statistic quantiles,8-round cap for the secondary, and original
2% selected positive-harm/reference budget. No threshold search or new neural/
forest fit; only the specified empirical calibrators are recomputed.

For each held source recording r, reproduce its existing calibrator excluding r.
For every other recording s, refit excluding BOTH r and s. Apply every calibrator
to the same past-only inputs in r. Thus no calibration coefficient for r uses
r's outcomes, even for an alternative decision. Cache unordered deletion pairs
to avoid duplicate fitting. Record support counts, action flips and hashes.

The fixed consensus diagnostic retains a parent action only if every nested
calibrator supports and selects it. If no deletion is estimable, selection is
empty and cannot pass safety. No future-availability filter is permitted.
Parent, retained and removed outcome masses must add exactly within tolerance.

Compare conservative net-utility lower mass against the parent and against the
exact expectation of uniform count-matched thinning within each held recording.
The latter is a linear utility expectation, NOT a risk-ratio expectation,
realized random policy, frame-matched control or safety certificate. No random
seed is selected. Report removed known harm, lost benefit and unknown-envelope
mass separately. All repeated occurrences remain dependent.

Primary descriptive contrasts are consensus-minus-parent and consensus-minus-
matched expected utility, divided by full known reference mass (not a changed
risk denominator). Use3000 bootstrap draws of locality means, averaging heads
within locality first. Intervals are nominal, development-only and unadjusted;
empty or undefined risk remains visible. No superiority is inferred just from
fewer violations or cleaner risk after rejecting most samples.

## Decision and Resource Rules

If consensus loses utility without same-count evidence, reject it as a repair.
If it retains unsupported or harmful sets, stability is insufficient. Even a
positive source diagnostic does not authorize deployment or independent-role
opening. No target transfer for the already rejected selected-set calibrator.

Read the hash-bound CREATE packet bundle into local memory, decode each member
once, and use native arm64 CPU,4threads,workers0. Existing fitting functions
import Torch but no neural training or GPU/resource probing is performed.
Pilot the source with the most recordings before full execution to measure cost.
Save small group checkpoints and heartbeat, support resume and exact replay.
The10GiB local numerical-cache reserve stays unchanged; save no input arrays,
forecasts, latent caches or model checkpoints. Aggregate output cap4MiB.

Protocol: obs8/pred12,stride12raw frames,image-local detector-silver on exposed
European development localities. No metric,seconds,human-gold,physical-safety,
true3D,foundation or submission-ready claim. Stage5C and SMC off.
