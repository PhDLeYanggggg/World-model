# Direct Neural Intervention: Source Versus Recording Constraints

## Material Passport

Prospective fixed method comparison on original TRAIN only. The preceding
easy-constrained fit repaired donor loss but failed full policy budgets (5/72)
and matched utility versus original. Earlier task-risk neural fits already
tried a product of three soft moment gates with a quadratic risk loss. Their
failure cannot be presented as an untried method. No change to the approved
2% budget, data units, held roles, or historical verdicts.

## Hypothesis and Novelty Boundary

Test direct action learning without a moment-MSE objective. Two controlled
contrasts isolate source-averaged versus recording-specific constraints, and
soft versus hard forward decisions. A small MLP adds a learned score to the
frozen original decision (+1/-1); zero output initialization exactly reproduces
that decision. Inputs are380 frozen causal features plus5 frozen risk/utility
moments, normalized by the original inner-fit source preprocessing. There is
no record-ID input, future position, label-knownness or new goal construction.

Arms: soft_source, soft_record, hard_record (PRIMARY). Same32 hidden units,
initialization,64 eligible rows sampled per recording with replacement, training
draws and2,000updates;6 contexts x3 seeds x3 arms=54 fits/108,000 updates.
Each source has equal objective mass; recordings are weighted by their eligible
row count within source. Thus empirical constraints target row-mass ratios,
not equal-query-average ratios. Recordings with no eligible training rows have
no gradient but remain in full evaluation. Training includes unknown rows.

Known targets define gain B-H and two risk excesses H-.02R,EH-.02ER.
Wholly unknown targets contribute gain -envelope and both excesses +envelope,
the existing worst-case completion loss, not an imputed observed outcome.
No unknownness is supplied to inference. The objective minimizes negative gain
plus nonnegative dual-weighted excess, with source or recording duals. Positive
fixed TRAIN scales stabilize gradients without changing excess signs. Empty
reference scales use positive causal mean-envelope scales, not safe zero risks.

soft_source/soft_record use sigmoid(logit/T) in the objective. hard_record uses
the exact indicator(logit>0) forward, with sigmoid-gradient straight-through
backward. T decreases linearly1->.25; lr.001, dual lr1/cap100, gradient clip5,
AdamW decay.0001 are fixed before data scoring. Deployment decisions always use
logit>0 and causal eligibility. Scores are not calibrated probabilities.

This uses known building blocks: the straight-through gradient family of
[Bengio et al.](https://arxiv.org/abs/1308.3432) and empirical constrained learning
related to [Cotter et al.](https://proceedings.mlr.press/v98/cotter19a.html).
It is not their exact proxy-Lagrangian algorithm or its stochastic-classifier
guarantee. Their separate work on
[constraint generalization](https://proceedings.mlr.press/v97/cotter19b.html)
also does not license a generalization claim from training constraints here.
These components alone are not novel; useful agent-scene intervention evidence
is the research question. Current inference remains per-agent, not a joint
scene optimization or a newly trained trajectory world model.

## Frozen Readout

Freeze every final model before scoring the existing inner-held recordings.
Reuse the input-only whole-recording partition from shared_train_risk_v1;
inner-held was exposed by prior experiments and is exploratory, not a new test.
No best-step selection, threshold tuning or independent-role opening.
Report both fit and inner-held,72 source/context/head-seed views each.

Compare floor, original frozen gate, and all3 direct gates. Primary hard_record
versus original and soft_record; soft_record versus soft_source isolates dual
grouping. Full and per-record/frame equal-count utility use existing completion
bounds. Original rank is B-H; direct rank is the learned logit. Keep all empty
controls, with risk not estimable and not a matched safety gate. Full deployment
safety still requires the original all/easy2% budget and easy preservation on
every required scene. Lower risk MSE is not an advancement requirement. No better
secondary arm will replace primary after results are read.

Report actual intervention counts, unknown counts, known/completion risk,
per-record failures, paired utility intervals, train-held gaps and source/seed
slices. Completion intervals are not sampling confidence intervals. Upstream
producer exposure prevents whole-pipeline OOF/independent significance claims.
No deployment promotion from this fitting/previously-exposed inner-held test.

## Execution

Native arm64 CPU4/interop1/workers0; no resource probing or x86 Conda. Pilot runs
100updates per arm, verifies interrupted50+50 equals direct100 on real TRAIN,
then estimates full runtime. Complete all54 fits if local resources suffice;
do not downsize scientific scope because it is slow. Atomic checkpoints every
100steps, heartbeat every50, fixed12h boundary,10GiB reserve and256MiB model cap.
Final readout is replayed exactly. Targeted tests suffice for unchanged old code.
No new CREATE job is submitted; existing authorized TRAIN packet reads are
hash-verified. Raw data, features and model checkpoints remain outside Git.

obs8/pred12 rawstride12, pixel/local detector-silver. No metric/seconds, human
gold, physical-safety, true3D, foundation or submission-readiness claim.
Stage5C/SMC stay off. Prior failed experiments remain unchanged.
