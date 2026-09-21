# Why the Protected-Risk Head Did Not Solve Safe Intervention

## Verified Failures

1. **Wrong inference from global ranking.** Protected-event AUROC is 0.866-0.985,
   but the admitted low-score region still contains 25 zero-CV harmful instances.
   Some have predicted probabilities below 1e-14. AUROC over a predominantly
   negative population is not evidence of precision where the policy operates.
   No independently calibrated probabilities or risk bounds were claimed.
2. **Target scope is narrower than easy preservation.** Predicting harm to exact
   zero-CV outcomes does not protect small positive-CV errors. Positive-easy
   diagnostic degradation remains 7.1601%, versus 7.1111% without the new guard.
   The protected head also fails its own exact-zero objective in every seed.
3. **A causal proxy is not enough.** A past-stop veto reduces zero-CV harmed
   instances from 79 to 17. The learned rule leaves 25, including eight with a
   zero last observed step and seventeen outside that group. Future perfect-CV
   predictability cannot be treated as a past-stop input flag. No row-specific
   exception is added after observing outcomes.
4. **Expected severity remains weak.** Protected-cost MSE beats the training
   constant in only 4/12 outer views despite falling batch loss. Rare-event
   recognition and useful magnitude estimation are different learning problems.
5. **Broad safety collapses capacity.** General-harm score <=0.01 admits only
   eight eligible query/seed rows in total. Matching its capacity shrinks the
   experiment from 4,437 requested switches to eight. Protected, net-only and
   stop-veto matched choices are identical: no incremental mechanism evidence.
6. **Incomplete outcomes remain unresolved.** Protected online selection includes
   30,511 repeated instances without complete future risk labels; 2,176 also
   lack any ADE support. Training never labels these rows safe. Evaluation
   cannot fill their missing risk with zero or exclude them from coverage.
7. **Independent evidence is still absent.** Clean pair-excluded producers remove
   direct fitting leakage, but these four sites have been used in method design.
   The 3,000 scene-bootstrap intervals are conditional development uncertainty,
   not independent confirmation or evidence from hundreds of thousands of scenes.

## What Was Not the Cause in This Run

All fixed endpoints, paired training draws, label hashes, causal feature hashes
and saved risk predictions replay. Real arm64 CPU training is complete, with no
unknown-label training draws or crashed/incomplete model used as an endpoint.
Independent scalar errors and selected identities agree. The experiment did
not need larger hardware; its failure is predictive/safety-related, not a
training process being stuck or a cache-row mismatch in the checked chain.

The independent verifier initially used exact/float64 tolerances on float32
ECE mean versus sum reductions and on scalar versus vector distance maxima.
Those overly strict verifier comparisons were repaired with dtype-based ECE
rounding and 1e-12 error tolerance, preserving frozen scores, counts and code.
Observed maximum differences are 7.12094e-9 ECE and 7.10543e-15 absolute error.
Regression tests cover float32 rounding. No model result was rewritten.

## Next Falsifiable Step

Audit training-only conditional false negatives, magnitude tails and event
support among predicted-gain-positive rows, not only all-row MSE. Test whether a
conditional severity/support objective improves the joint accuracy-harm frontier
over a simple past-motion veto, using the same frozen forecast and a registered
budget. Any new thresholds require source-only selection with clean producer
lineage and a separate calibration role; no sweep on these displayed outcomes.
Do not promise that strict empirical protection is achievable at useful coverage.
If support cannot distinguish future departures/continuations from past context,
that information limit must constrain the method and its deployment claim.

No new deployment, Stage5C, SMC, metric/seconds or foundation claim.
