# Full Current-Query Readout: Negative Primary Result

Status update2026-10-30T15:00Z:the separate full replay below subsequently
finished in623.356s at peakRSS11,088,609,280B. All72 groups,144 controls and
the canonical summary agree exactly. [Complete readout](readout.md) and
verification.json record that later result;the first-readout timing is retained.

## Evidence status

fresh_run:36 native Torch risk-critic fits,72,000 primary updates and4,000
separate exact-refit updates. Training PID69758 exits0 in1877.402s at peak
RSS10,798,153,728B. Both first complete2,000-step refits agree scientifically.
cached_verified:36 fixed-query controls,unchanged source data,normalizers,
draws,original context and all seven forecast paths. No new trajectory model.

Full inference/allocation/scoring PID73583 exits0 in628.182s at peak
RSS10,999,578,624B. All72 groups and144 fixed-query controls complete.
Separate full replay PID75450 is running. Its exact agreement is not yet claimed.
No partial policy result selected a checkpoint,threshold or model.

## Prespecified result

Selected-set joint known ADE gain over the floor falls8.215347% to6.783336%.
The paired current-minus-fixed difference is-1.432011 percentage points,
nominal conditional95% locality CI[-3.335655,+0.608111]. The prespecified full
wholly-unknown-completion-lower difference is-7.087372pp
CI[-13.238339,-2.904036]. All three critic-seed means of that difference
are negative. These are fitting summaries,not independent significance.

Only11/444 required recording-risk views are supported. There are268 known
all-risk violations,313 known easy-risk violations and21 undefined easy
views. Selection decreases410,575 to137,815 repeated views;3,872 selected
views have unknown outcomes. Fewer interventions do not establish safety.
The original all/easy2% positive-harm/reference and easy-preservation criteria
are unchanged. Empty risk is not credited as safe zero. Four of eight fitting
checks pass;the advancement gate fails and deployment is unchanged.

## Secondary diagnostics do not override the primary

Current-target set versus row has known+0.768069pp CI[+0.267183,+1.340767],
but completion-lower-3.525013pp CI[-7.281836,-0.772340]. Its explicitly
reoptimized same-count context control has known+0.179028pp
CI[-0.159545,+0.584688] and lower-1.589234pp CI[-2.936624,-0.477403].
Full-policy counts differ in the first contrast. Count controls are utility
comparisons only,not replacements for complete original recording risk.

The common fixed-original-set monitor worsens under current-query training.
The moving masked query term is a different selected population and excludes
unknown/empty contributions from supervision. Its reduction would not prove
calibrated selected risk. [Actual traces and coverage](training_report.md).

## Reproduction and scope

The metric-only ZIP preserves all72 registered aggregate JSON files exactly;
it contains no per-row arrays,raw data or model weights. The summary retains
all subsets,locality/seed contrasts and endpoint-not-estimable entries.
The complete replay and report will be recorded separately when finished.

Exposed European SOURCE fitting/resubstitution,not whole-pipeline OOF or
independent confirmation. Twelve localities are the bootstrap units after
context/head-seed averaging;3,000 resamples do not remove prior exposure.
Three critic seeds share forecast producer seed43. Obs8/pred12/rawstride12,
local-image detector-silver. Not historical t+50,metric,seconds,human gold,
true3D,foundation,world-dynamics improvement or submission-ready evidence.
Independent roles,Stage5C execution and SMC remain closed.
