# European Candidate Opportunity and Rejection Diagnosis V1

## Material Passport

Source-development diagnostic after the completed protected-motion experiment;
not independent confirmation or a new predictor/policy selection. Freeze this
diagnostic before reading row-level decompositions. Existing outcomes motivate
the question; source scenes do not become untouched data by renaming them.

## Fixed Questions

1. Does the neural candidate have less attainable gain than damping, or does the
   learned controller fail to retain its available gains?
2. Which sequential causal rule loses gross attainable gain: source-support
   abstention, no observed motion, nonpositive predicted utility, zero predicted
   event mass, or predicted-risk veto? How much harmful switching remains?
3. Do four-locality and eight-locality forecasters differ on exactly the same
   outer held source rows? Retain both eligible single-fold producers, ordered
   by fold id, rather than choosing the better one using outcomes.

All 48 frozen pointwise views, both candidates, all/easy events, ridge/neural
risk heads, guarded/unguarded support and seeds 17/29/43 are retained. Reuse the
318,969 source target population and the original per-fold easy/hard cutoffs.
Keep future-unknown rows in the denominator of intervention rates. Source-only
labels may be used for diagnosis, never for causal reason assignment or scoring
by a deployed policy. No new thresholds, architecture, fitting or deployment.

For each candidate, the hindsight oracle chooses the lower observed ADE of CV
and that candidate. It is an error-opportunity bound, not an inference model.
Gross positive opportunity equals captured benefit plus beneficial error
reduction rejected by the five ordered gates. Actual net gain equals captured
benefit minus switched positive harm. Oracle regret equals missed benefit plus
switched harm. Normalize each sum by the same locality/subset CV error sum,
then average localities equally. An empty or zero-reference locality remains
undefined, not omitted. These percentage-point contributions sum to the frozen
policy gain; they are not raw pixel sums pooled across localities.

Report all/easy/hard subsets, candidate gain opportunities, per-gate rejection,
utility sign precision/recall, and predicted-risk bands fixed at 0.02, 0.1,
0.5 and 1 plus zero-mass. Risk bands are diagnostic, not a threshold search.
Retain per-locality rows and 3,000 paired locality resamples, seed 39271.
Intervals remain conditional on shared source data and fitted producers.

For producer comparison, use the frozen complement model trained on eight
localities and both frozen single-fold models trained on four, with every
compared model excluding the entire current held fold. Measure ADE and oracle
opportunity on that same held population. This compares fitting population
and fitted predictors, not a randomized isolated sample-size intervention.
It does not prove that producer shift causes the controller failure. Do not
substitute a four-locality predictor into deployment based on this readout.

Verification must bind prior analysis/checkpoint/decision receipts, reproduce
all 48 frozen full-population ADE gains, conserve every reason/count/gain/harm,
and reproduce the full diagnostic without refitting. No reserved selection,
calibration, confirmation or DroneCrowd is opened. Native arm64 local execution
is appropriate: cached arrays and aggregate arithmetic only, no new CREATE job.
Record process, heartbeat, resumable per-seed receipts and exact replay.

Image-pixel obs8/pred12 raw stride12; not raw t50, meters, seconds, verified
sensor-online causality, human gold, physical safety, true 3D or foundation.
Stage5C and SMC remain off. A negative result must change the next experiment,
not merely generate another status report.
