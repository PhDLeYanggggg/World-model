# Frozen Query Allocation Model and Data Card

- Purpose: test allocation under frozen predictive utility and signed-risk scores.
- Status: development-only. Mean accuracy primary passes; observed risk gate fails.
- New training: none. Cached forecasts, protected floor, utility ridge and 108
  descriptor risk heads are checksum-verified through the registered seal chain.
- Data: only the same twelve opened European source-training localities.
- Roles: inherited four/four/two/two source separation; no independent
  selection/calibration/confirmation access and no test-based refitting.
- Protocol: eight past positions, twelve future targets, raw-frame stride 12.
  Image-local detector-derived silver. No metric, seconds or human-gold labels.
- Inputs: causal past geometry/context, frozen predicted rollouts and scores,
  eligibility, current frame/recording/locality identities and stable row IDs.
- Input exclusion: future coordinates, future masks and future targets are
  removed before inference. Central velocity and test-endpoint goals are absent.
- Output: binary choice between the frozen protected floor and frozen neural
  predictor for each retained query row. Not a new trajectory generator.
- Constraints: same count as frozen independent admission; all/easy aggregate
  predicted signed excess <=0 at nominal 2% budget. Numerical failures abstain
  to the independent anchor rather than changing the intervention count.
- Risk limitation: the anchor itself is not certified. Its use is a controlled
  experimental fallback, not a guarantee of safe deployment.
- Solver: SciPy/HiGHS binary allocation, 256-node limit, zero requested relative
  gap, integrality/count/support/constraint/utility/primal-dual checks.
- Missing labels: included in causal action counts; unknown errors stay unknown.
- Statistics: three forecast seeds, 108 groups, 216 dependent locality views per
  arm; twelve locality means are the bootstrap units, not 747,900 query views.
- Arm limitations: uniform-query admission is not rate matched; utility-top-k
  deliberately ignores risk and is diagnostic only.
- Confirmed limitations: no all-visible completeness guarantee, no pairwise
  interaction penalty, no physical validity certificate or causal mechanism proof.
- Deployment: unchanged. No independent success, true 3D, foundation, Stage5C
  execution or SMC claim. Actual paper submission is not authorized by this result.
