# Full-Range Harm Cost Head

- Status: research-only, registered advancement gate failed; not deployable.
- Training:18 fresh cost-head continuations, three seeds,2,000 updates each.
- Inputs: inherited380-dimensional causal features and causal envelope, frozen
  source-only preprocessing. No future endpoint, central velocity or test goal.
- Outputs: benefit, all harm, floor reference, easy reference, easy harm moments.
  Benefit/reference remain frozen; only the bounded harm branch is trained.
- Parameters:25,407 total,12,258 trainable. No new trajectory producer, JEPA,
  generative rollout, pixel reconstruction or world dynamics training.
- Data: original European source TRAIN only, whole-recording inner fit/held
  partition,12 localities. Held records have previous development exposure.
- Evaluation:144 views, six frozen controls exactly reproduced, source-locality
  bootstrap3,000. Row counts repeat across contexts/seeds, not independent n.
- Main negative result: same-count held utility versus isolated odds+0.243pp
  with interval crossing zero; signed MSE significantly worse under nominal
  intervals.44/72 known easy-risk violations and0/72 complete policy support.
- Missing outcomes: retained at inference and bounded at evaluation; undefined
  risk never counted as safe. No threshold tuning or selected favorable seed.
- Runtime: native arm64 CPU4/workers0; checkpoint and resume verified; all runs
  terminal. No new CREATE job; read-only hash-verified TRAIN streaming only.
- Restrictions: no deployment upgrade, independent calibration guarantee,
  physical safety, metric/seconds, human-gold, true3D or foundation claim.
  Stage5C and SMC remain disabled. Stage37 is not recertified by this work.

The code is available for reproducibility, not as a recommended policy.
Checkpoints remain local and excluded from Git.
