# Fixed-Predictor Deferral Control for v7

This supplementary experiment is fixed before fitting any deferral head. The
completed v7 outcomes are already known; this is an added development control,
not a retrospective preregistration or an independent test. The parent protocol,
forecaster checkpoints, metric, masks, seeds and selection decisions remain unchanged.

## Question

Does the proposed gain/harm selector offer empirical value over a cost-sensitive
two-action deferral objective with the same frozen forecasters and OOF inputs?
Architecture composition and a lower training loss are not method contributions.

## Fixed Grid

- Both v7 families: baseline-skip and observed-motion-bounded Transformer.
- Seeds 17, 29, 43; three held-physical-group predictors and one full-fit predictor
  per seed are reused with verified hashes, not retrained.
- One linear and one width-64 GELU deferrer, each at cost bounds 1 and 10 in the
  unchanged past-normalized coordinate system. All four results are retained.
- 1,000 Adam updates, batch 128, learning rate 0.0003, gradient clip 1;
  checkpoint every 100 updates, heartbeat every 20. CPU4/interop1/workers0.
- Inputs and their OOF fingerprint must exactly match the existing ridge and
  neural gain/harm heads. Labels are the two absolute ADE costs. Costs are
  divided by the registered bound and clipped at one; clipping fractions are
  reported. Evaluation errors are not clipped or redefined.
- The single-expert logistic surrogate is adapted from Mao, Mohri and Zhong,
  [Regression with Multi-Expert Deferral](https://proceedings.mlr.press/v235/mao24d.html),
  ICML 2024. Opposite-action costs weight the base-two log loss. This is a
  task-specific implementation, not a reproduction of their published benchmarks.
- Argmax routing, ties to CV. No validation thresholds or winner selection.
  Logit margins are not calibrated safety probabilities.

## Matching and Evidence

Extract the existing causal rollout features before reading future labels.
Use future trajectories only for OOF supervision and development scoring.
Compare all original ridge/neural-cost and conservative/moderate controls, not
only the previously selected arm. Fresh floor and candidate ADE/FDE must exactly
match every hash-verified original query export before controls are reused.
Deferral has no intervention cap or gain/harm budget. Actual switch rate is
reported; neither equal realized coverage nor equal risk is claimed.

Primary task remains eight observations and twelve native annotation steps;
raw-frame t+50 remains supplementary in the completed parent study. This added
control evaluates the primary task, not a new t+50-conditioned predictor.
Development contains two recordings of one physical site. Do not create a
window-level bootstrap CI or claim three seeds establish cross-site validity.
No metric/seconds, independent confirmation, deployment, Stage5C or SMC claim.
The four comparator settings cannot enter parent model selection retroactively.

## Outcomes That Matter

Report predictive errors, easy/hard slices, actual switching, positive harm,
per-recording results and all three seeds, including unfavorable outcomes. A
deferrer that preserves easy cases only by selecting CV is not positive neural
transfer. A winner on mean error that violates easy preservation is not deployable.
Given the small v7 candidate/CV oracle ceiling, this control can falsify a claimed
selection advantage but cannot create missing forecasting headroom.
