# Contemporaneous Peer Context: Fixed TRAIN-Only Experiment

## Question and Prior Evidence
Can a matched risk head distinguish harmful neural interventions using the
other available agent rows from the same recording and current frame? The
current rational head processes each row separately. Its 380 inputs already
contain up to eight neighbor histories, so this is an incremental query-wide
context test, not the first interaction-aware model.

The completed conditional-calibration comparison reduced actions but lost
utility at matched counts. Eleven positive-easy-excess retained cells had an
offset covering the raw cell mean but not the retained subgroup. Group-average
calibration is not sufficient evidence for calibrated subselection. Earlier
fixed/adaptive decision-region weighting, selected-group penalties, hurdle
ranking and temporal auxiliary supervision also failed their strong gates.
Those negative results are not being relabeled as a new hypothesis here.

## Single Intervention
Use the unchanged shared-utility anchor. Keep benefit, reference cost and
easy-reference cost exactly frozen. Clone its 32-wide encoder for the risk
branch; concatenate two encoded vectors into the two-output rational harm
decoder. Both new arms have identical parameters, initialization and compute:
- self: own causal vector in both encoder calls;
- peer: own vector plus the leave-self-out mean of all available causal rows
  with the exact same recording and current frame. Singletons use self.

Average already TRAIN-standardized/clipped features, before the shared risk
encoder. No outcome, known-future mask, selection mask, recording-wide average,
future frame or held statistics enter pooling. Rows without future labels stay
in the inference roster. This is not transductive fitting over a held dataset:
context is restricted to one contemporaneous query. Source identifiers only
retrieve previously frozen source preprocessing; no new unseen-domain mapping.

Pooling is an established permutation-invariant device, not a novel method by
itself. See Zaheer et al., [Deep Sets](https://proceedings.neurips.cc/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html),
NeurIPS2017. Our limited mean-before-encoder construction does not invoke its
universal representation result or imply physical interaction understanding.

## Fixed Execution
Six existing producer/controller contexts, four excluded source localities per
context, seeds17/29/43, two new arms: 36 fits, 2,000 updates each. These are risk
head seeds, not three newly trained forecasters. Same query draws, query-balanced
five-moment/signed-score loss, task-regret term, scaled dual risk penalty,
learning rate0.001, AdamW, gradient clip5 and2% risk budget as rational_harm_v1.
Whole-recording inner fit/held roles remain inside the original TRAIN split.
All twelve localities were already exposed in development. No outer development,
independent selection, reserved calibration or confirmation is opened.

Real native arm64 CPU4/inter-op1/workers0 pilot:100updates per arm plus an exact
50+50resume/direct replay. Then full2,000updates, checkpoint/heartbeat100updates,
12h stop-and-resume boundary,10GiB free-disk reserve,256MiB checkpoint cap.
No outcome-based early stopping, threshold sweep or source/seed selection.
Commit code/protocol before real fitting; freeze all36finals before held readout.

## Readout and Advancement Screen
Retain all eight rational_harm comparators (local, anchor, old_risk,
utility_continue, joint_risk, isolated_risk, full_range, rational), plus matched
self and peer heads. Reproduce old predictions and policy readouts exactly.
Report full and per-query matched-count paired completion-lower utility,
conditional MSE, known/unknown support, easy positive-harm risk, action counts,
recording-level results and selected moment/tail diagnostics. Unknown outcomes
remain unknown and their exchanged envelope mass penalizes the lower bound.

Average repeated head/context views within locality, then3,000 locality
bootstrap draws with the existing frozen seed. Intervals are nominal exploratory
TRAIN evidence, not adaptive-search-adjusted or independent confirmation.
Advancement requires positive lower full AND matched utility CIs against every
retained comparator, and finite completion support in all72peer views. A positive
information contrast alone is not a deployment gate. No automatic promotion.

If unsuccessful, record whether context was actually available, whether the
fitting objective changed, selected harm underestimation and the split between
lost benefit and added harm. Do not tune pooling/thresholds on these readouts.

## Claims
Obs8/pred12 at raw stride12; image-local detector-silver research. Not the
historical raw-frame t50 endpoint. No metric, seconds, human-gold, physical
safety, true3D, foundation, submission-ready or new neural-dynamics claim.
Stage5C and SMC remain disabled. Historical Stage37 is not recertified.
