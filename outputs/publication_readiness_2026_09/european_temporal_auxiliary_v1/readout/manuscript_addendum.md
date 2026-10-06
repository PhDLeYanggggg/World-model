# Matched Temporal Supervision Does Not Establish Safer Intervention

This dated result addendum supersedes the **pending-training status only** in
the version-three draft. Its older empirical tables and negative results remain
unchanged. This is exposed development evidence, not a submission-ready claim.

## Experiment

We trained 216 bounded neural cost heads across 24 source contexts and three
head seeds, comparing no auxiliary supervision, row-mean auxiliary supervision,
and stepwise temporal auxiliary supervision. Each fit used 2000 fixed updates.
Model architecture, initialization, query sampling, primary loss, source-recording
partitions and inference eligibility were matched. Only the auxiliary target
changed. All models were frozen before evaluating the seven registered arms,
which also include original forest, additive, positive-harm and squared-cost
controls. The trajectory forecasters themselves were not retrained.

The readout covers 72 source/seed views across 12 already exposed localities.
Within-locality averaging precedes 3000 paired locality bootstrap draws. The
intervals are nominal and are not independent or multiplicity-adjusted
confirmation. Original controls reproduce their prior predictions, actions and
metrics exactly. An independent scalar implementation cross-checks the sums,
decisions, completion bounds and intervals; this is algorithmic verification,
not independent scientific replication.

## Results

Temporal supervision reduced normalized signed-score MSE compared with row-mean
supervision: -0.024422, nominal 95% CI [-0.054556, -0.001405]. Nevertheless, its
paired lower utility was worse: -0.353836 percentage points of full known
reference cost, CI [-0.545292, -0.186078]. The difference remained negative at
matched per-query intervention counts: -0.231955, CI [-0.372209, -0.112760].
Relative to no auxiliary supervision, full paired utility was also worse:
-1.026746, CI [-1.557310, -0.574154]. These are decision-utility contrasts,
not trajectory ADE/FDE improvements.

The temporal policy improved full paired utility against the original forest
by +1.390725, CI [0.566962, 2.193647], but failed stronger neural controls and
absolute safety. All 72 temporal views exceeded the registered 2% completion-
upper selected easy-risk budget; 60 already exceeded it using known outcomes.
The median known selected easy-harm/reference ratio was 7.283471%. No temporal
full-policy view satisfied complete finite-outcome utility and risk support.
Twenty-nine original-selected views lacked known support, leaving the registered
common-cohort MSE contrast undefined rather than permitting selective omission.

## Interpretation and Limitations

This matched experiment rejects advancement of the present temporal-auxiliary
head. It supports a narrower negative observation: improving a global temporal
prediction objective does not necessarily improve selected-policy utility or
easy-case preservation. The precise causal mechanism remains unresolved; a
TRAIN-only harm/reference decomposition is the next diagnostic, not an already
successful repair. Independent calibration, confirmation and transfer are not
opened by these findings.

The data remain image-local detector-silver trajectories with obs8/pred12 and
raw stride12. No verified metric scale, seconds-level interpretation, physical
safety, human-gold, true-3D or foundation-model claim is made. Stage5C and SMC
remain disabled. This result is suitable as a traceable development ablation
or negative finding, not as evidence that M3W's principal research goal is met.

Full contrasts, safety/support accounting and reproducible provenance are in
[the complete readout](report.md), [failure analysis](failure_analysis.md),
[summary](summary.json) and [verification receipt](complete.json).
