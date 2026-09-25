# Registered Ranking-Auxiliary Experiment

## Material Passport

- Role: opened-source development experiment, not independent confirmation.
- Previous turn: progress; 216 matched-count views verified and published.
- Hypothesis: explicitly supervising within-locality harm/reference ordering
  improves the risk controller beyond occurrence/severity regression alone.
- Frozen parent: completed hurdle heads and coverage diagnosis, commit7276538b.
- New work:36 Torch heads,72,000 updates,three seeds; no new trajectory fitting.

## Single Changed Factor

Keep each three-output network, initialization, fitting rows, causal355 features,
training-only scalers, optimizer, original balanced-locality minibatch sequence,
2,000 steps, utility head and 2% predicted-risk rule identical to the verified
hurdle control. Add exactly one ranking auxiliary with coefficient1, no sweep.
Only the frozen control is reused; new heads train from matched initialization,
not from an extra-trained checkpoint. A zero-coefficient unit test must exactly
reproduce the control fitting path. Both neural and damping candidates are fitted.

For known fitting labels (B,H), define observed risk share q=H/(B+H) when B+H>0.
Within each minibatch/locality, pair each row with its cyclic next row; there are
no additional random draws and no cross-locality pairs. Skip undefined q pairs
and ties for the ranking term only. They remain in all original losses. Weight
each logistic pair loss by abs(q_i-q_j), normalize by total pair weight, and rank
predicted log(Hhat+epsilon)-log(Bhat+epsilon), epsilon=1e-6 in training cost units.
The added loss is a margin-weighted RankNet-style logistic auxiliary. Existing
moment MSE, occurrence BCE and positive-only severity losses remain unchanged.
This is a sample-ordering hypothesis, not an unbiased estimator of a ratio of
conditional expectations and not a new probabilistic calibration guarantee.

The pairwise logistic idea is established: Burges et al., ICML2005,
[Learning to Rank using Gradient Descent](https://www.microsoft.com/en-us/research/wp-content/uploads/2005/08/icml_ranking.pdf),
Section3, equations1-3. Its use here is a controlled adaptation, not a novelty
claim. The new weighting/target choice is an experimental design, not a result
reported by that paper. No test labels or held-out outcomes enter training pairs.

## Frozen Evaluation Matrix

Three folds,three seeds17/29/43,two candidates,two event targets give36groups.
Retain six views each: original hurdle control, ranking-augmented head, both
common-pool anchors, ranked order at control counts, control order at ranked
counts. Count decisions use only causal scores and are saved before new outcome
evaluation. Full/common support differences remain explicit. Every one of216
views is reported;36 full controls must exactly reproduce parent metrics.

The shared matching helper's internal names `product` and `hurdle` mean
`control_hurdle` and `ranking_augmented_hurdle` in this experiment. Public tables
must use control/ranked names; no product-MSE refit is claimed here.

Report all/easy/hard ADE,FDE,tail/worst-locality errors,zero-CV harm,switch rate,
predicted-budget violations,both matched-count ranking components,and direct
neural-versus-equally-protected-damping contrasts. Both anchor decompositions
use the same CV denominator. All216 views replay; separate scalar sorting and
coordinate-error arithmetic verify them. Bootstrap3,000 paired localities.

Do not select an event,seed,fold,count or loss coefficient after these outcomes.
Stable ordering and neural advantage require consistent signs across folds and
seeds with easy/zero-reference preservation, not one favorable interval.
Forced-count arms may violate predicted risk and are never deployment candidates.
No independent confirmation, submission readiness or deployment upgrade is
authorized by this source experiment alone, even if point estimates improve.

## Execution and Boundaries

Native arm64 CPU4/inter-op1/workers0,atomic checkpoint every200 updates,heartbeat,
resume. Run100-step fitting-only pilot, estimate local cost, then complete all
36 heads and replay before outcome readout. Disk below10GiB is a hard stop;
slow healthy execution is not a reason to reduce the registered matrix.
Prior fit times make local execution appropriate; no remote resource changes.

Each fit uses four fitting/eight complete-chain-excluded localities. All twelve
are already opened European Squares development data. Released detector-track
image pixels,obs8/pred12 rawstride12,not historical t50,seconds,metric,human gold,
physical safety,true3D or foundation. Conditional bootstrap and shared seeds do
not create independent samples; comparisons are not multiplicity corrected.
Reserved calibration/confirmation remain closed. Historical Stage37 is not
recertified. Stage5C and SMC stay disabled. No threshold/calibration refitting.
