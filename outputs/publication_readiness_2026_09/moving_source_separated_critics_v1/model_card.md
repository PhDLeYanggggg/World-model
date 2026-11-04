# Model Card: Source-Separated Neural Cost Critics

**Status:** research-only negative result. No deployment upgrade or CCF-A
submission-candidate claim. These are cost heads over frozen EqMotion
forecasts, not newly trained trajectory dynamics, JEPA or generative models.

## Model And Training

- 108 actual Torch heads: 18 original outer contexts, three existing inner-C
  parts, two cost seeds. Original producer seeds are 17, 29 and 43; the second
  cost seed adds 1009 without changing the forecaster seed.
- 28,229 parameters per head; width 64. The 428-field input uses 380 causal
  local-history/geometry/rollout fields and 48 zeroed fields. Query-context
  pooling uses contemporaneous observed agents, not future trajectories.
- Outputs: predicted benefit B, harm H, reference R, easy reference ER and
  easy harm EH for the fixed amplitude bank. Targets use registered
  available-label ADE, not raw FDE classification labels.
- Original bound-aware point regression and net-gain regression loss;
  2,000 saved Adam updates per head, 216,000 primary updates total. Pilot
  1,200 updates and the first full-refit 2,000 updates are additional.
- Own-part physical fitting features/labels determine normalization, target
  RMS and bias. Validation, outer H and producer-fitting labels do not enter
  a critic's fit. All 108 checkpoint scope/optimizer audits pass.

Native arm64 CPU, four Torch threads, interop one, workers zero. No resource
probing or NumPy training fallback. Checkpoints every 100 updates, heartbeat,
process lock, fixed 12-hour phase boundary and same-binding resume. The first
process disappeared after 71 full models and a step-100 partial checkpoint;
the cause and exit code are unknown. Same-binding resume completed the bank.
Saved primary fitting time is 8,822.70 s; the terminal resumed attempt is
2,760.31 s and peaked at 9.30 GB RSS. They are not additive definitions.

## Decision Rule And Evaluation

A proposer selects its original positive predicted-gain, 2%-admissible
amplitude. A same-data seed replica or source-disjoint critic may veto the
action, never reselect its amplitude. The threshold is not tuned. All models
freeze before new nested-C validation. The original full-policy risk,
easy-net, support and empty-utility rules remain unchanged.

Primary is unavailable: 18 required disjoint-data source views have empty
matched controls. Full-strategy secondary known contrast is -0.56277 pp
[-1.09473, -0.19341]; direct lower is -1.27394 pp [-2.08928, -0.58580].
Only 44/696 recording views meet the original risk/support contract. Exact
replay is an engineering pass, not an admission or generalization pass.

## Access And Reproduction

Checkpoints remain local under
`data/stage_cvpr2027_experiments/moving_source_separated_critics_v1/fits/`.
Each public fit receipt binds its exact local checkpoint, fitting scope and
preprocessing. No raw data, weights or per-agent prediction cache is in Git.

The runnable read-only replay is
`scripts/replay_m3w_source_separated_critics.py`. It requires the existing
legal data/checkpoints and checks all 216 arrays, 108 policy blocks and 1,620
cost cells without modifying existing public study files. It is not a public
no-data demo or a stand-alone trajectory-prediction deployment API.

## Limitations

All 23 supported source groups are exposed development, not verified IID
physical scenes or independent final tests. Smaller fitting parts and
different normalization ranges complicate causal attribution. Source
separation does not imply probabilistic independence or calibrated risk.
No scene-image, JEPA, Transformer-dynamics or physical-validity contribution
is established here. No true 3D, foundation, metric, seconds-level or
human-gold claim. Stage5C execution and SMC remain disabled.
