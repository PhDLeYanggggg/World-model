# Explicit Zero-Reference Atom on Frozen Risk Partitions

Registered locally before the new readouts, choices and evaluation. Four already
design-exposed SDD sites, seeds17/29/43, obs8/pred12 stride12 annotation pixels.
All175756 past-eligible windows remain. No new data role, outcome filtering,
predictor, trajectory, easy cutoff, 2% positive-easy limit or exact-zero reference
tolerance is introduced. DroneCrowd confirmation stays closed; IMPTC quarantined.

## Diagnosis and Hypothesis

Zero-CV cases are already included in the six-moment easy labels. Omission is
not the cause. Expected easy net harm can average over a mixture containing
zero-reference cases, other easy cases and benefit. It need not protect the
zero-reference component. Previous input changes recover positive-easy averages
but leave absolute harm of 1.4--6.2 pixels on a few zero-CV windows. A prior
source diagnosis also rejected exact-past-CV vetoes as a general repair.

Test whether explicitly retaining the zero-reference atom in the learned
conditional distribution can protect that component while preserving useful
intervention. This is a source-only risk-accounting/support experiment, not a new
neural dynamics architecture or independent calibrated-safety method.

## Fixed Readout Fit

Reuse all36 cutoff-relative forest partitions and source-only preprocessing.
For each tree leaf, estimate P(complete CV ADE = 0 | leaf) using exactly the
original known-source draws times the unchanged D>0 fitting weight. Unknown
labels have zero training weight. Their labels may not enter the counts. Fit36
new leaf readouts, not36 new forests. Check leaf weighted support against the
frozen tree, record unique support, effective zero-CV counts and moving/stopped
composition. The source-excluded producer chain remains fixed.

The event is recovered from complete training targets: easy probability target1
and easy denominator target0. A validation check confirms that effective event
rows have harm fraction1 and benefit0. At inference, only past-conditioned leaf
membership and training counts are used, never an observed future error. Average
the128 tree frequencies. No independent calibration or class reweighting.
Save every16 tree readouts with exact source/checkpoint identities and resume.
First fit16 readouts for coupa/seed17/Transformer, then resume all36.

## Fixed Policies and Same-Count Controls

The zero-reference risk tolerance is already0. Add a veto when the estimated
zero-reference probability is nonzero, or any readout lacks source support.
This uses exact0, not a fitted small-probability threshold. A zero empirical
frequency is not proof of population absence; unseen events remain a risk.

For each frozen point/population/selected rule, compare:
1. Existing cutoff-relative rule, cached_verified.
2. Atom-guarded rule, recomputed with the veto before allocation. Risk scores,
   objective and denominator conventions remain unchanged.
3. Same-query, same-count original-eligibility control, whose count equals the
   guarded rule. Point controls rank predicted gain among the original point-
   feasible pool; population/selected controls use the original query budget.

The guarded solution is a feasible incumbent for its matched control. If a
numeric solve fails count, original-unit risk or objective checks, retain that
incumbent rather than silently lose count or force an unsafe addition. Record
such cases as feasible-incumbent, not proved optimal. No solver tolerance or risk
allowance is widened. Every original and new policy is reported; no winner is
selected after readout. Include old strict as a background reference.

Freeze all decisions before aggregate readout. Primary comparisons are guarded
minus original and guarded minus matched for each action/rule, reporting all,
hard and positive-easy paired scene contrasts. Report all30 action/policy rows,
ADE/FDE, three seeds, sites, zero-CV harms, target fitting support, switch rates,
unknown/incomplete selections and full-grid future bounds. Three thousand paired
four-site bootstrap resamples remain nominal development uncertainty, not
window-level independence or multiplicity-adjusted confirmation.

Protection with no useful intervention, worse utility than the matched control,
or any remaining zero-CV harm is retained as a failure/limitation. No thresholds
are tuned from exposed outcomes. No external predictive readout, deployment,
metric/seconds, true3D/foundation claim, Stage5C or SMC execution. All original
failures remain intact; separate arithmetic, replay and scoped tests are required.
