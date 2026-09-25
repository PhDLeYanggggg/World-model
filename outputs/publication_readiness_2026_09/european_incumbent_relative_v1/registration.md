# Incumbent-Relative Incremental Intervention: Registered Development Test

## Material Passport
Experiment run; existing opened EuropeanSquares source-locality development data.
New controllers: not_run at registration. Existing assets: cached_verified.
No independent confirmation, promotion, calibration fit, Stage5C or SMC.

## Hypothesis
The preceding fixed-producer study lost more useful incumbent switches than it
gained by avoiding harmful ones in its three negative all-ADE CI cases. I will
test learning the value of overriding the existing decision instead of learning
the whole floor-versus-neural decision again. This is a target/policy-reference
comparison, not a new world-dynamics architecture or a proven new method.

## Fixed Design
- Same twelve already-opened sources, same six ordered A/B/C role rotations,
  seeds17/29/43, and all/easy risk events:36 dependent groups.
- A fits the cached four-source trajectory producer, protected floor and original
  stopping controller. B supervises new heads; C supplies development readout.
  No A/B/C row overlap per rotation. Reserved roles remain closed.
- Both new arms receive exactly the same381 causal inputs: previous380 features
  plus the replay-verified original stopping-controller choice. No future labels
  or errors are inputs. B and C use the same A-fitted forecast/controller stack.
- Floor-reference control learns neural benefit/harm versus the protected floor.
  Incumbent-reference learns alternate benefit/harm versus the original choice;
  when the original chooses neural, the alternate is the floor, and vice versa.
- Width64 utility and risk heads,2,000 updates, batch256, lr0.0003, clip5;
 144 heads/288,000 updates. Native arm64 CPU4/inter-op1/workers0. A100-step
  utility/risk pilot resumes inside the fixed budget. Heartbeat/checkpoint200.
- Both arms use identical known masks, source-weighted draws, feature means/stds,
  CV cost scales, capacity and seeds. Target means, initial target biases and
  target-dependent fixed ranking denominators may differ intentionally.
- Utility predicts expected positive benefit and harm; risk predicts reference
  cost and positive harm on the fixed event. Easy/hard thresholds stay A-derived.
  B only supplies preprocessing and loss targets. Unknown futures are never zero.
- A symmetric causal rollout-distance envelope bounds both orientations' harm.
  Latest observed step must be moving; predicted benefit must exceed harm and
  predicted harm/reference <=2%. No threshold search or risk calibration claim.
- Floor arm falls back to floor. Incumbent arm falls back to the original
  decision. Add-only and remove-only reuse the same incumbent scores and are
  predeclared diagnostic decompositions, not a post-readout winner selection.
-72 fixed-alpha0.01 ridge controls fit incumbent-relative targets with the same
  inputs. Cached original and preceding matched controller plus raw neural are
  comparisons:8 policies x36 groups=288 views, all retained.

## Readout And Claims
All fits and all decisions freeze before new comparative readout. Primary:
incumbent-reference versus newly trained floor-reference and original stopping
controller, same candidate/floor forecasts. Also report ridge, directional rules,
preceding matched policy and raw neural without selecting the best afterwards.
All/easy/hard/complete-window ADE, actual endpoint FDE, worst-locality easy loss,
tail errors, zero-CV harm, changed-action accounting, selected positive-harm
reliability and unknown-label switches. Equal-source means and3,000 paired source
bootstrap resamples, seed39271; four source localities per view, not window IID.
Overlapping rotations/seeds/events are not independent replications. Negative
intervals and missing/unsupported subsets remain visible. No confirmation claim.

Evidence of consistent incumbent improvement requires at least one positive
all-ADE interval and no negative all-ADE intervals across registered views, plus
all36 defined worst-locality easy losses <=2% and no new zero-CV harm. This is a
development evidence condition, not automatic deployment authorization. Risk
calibration, joint-scene intervention and independent sources remain later work.

Image-pixel obs8/pred12 at raw annotation stride12; not historical raw-t50, meters,
seconds, human gold, physical safety, true3D, foundation or submission readiness.
CREATE was checked read-only; no remote job/upload. About19GiB local disk remains;
stop only on hard resource failure, preserve checkpoints, never silently shrink
the registered budget. No raw data, cache, checkpoint or private queue stdout
will enter Git. Existing unrelated staged changes stay untouched.
