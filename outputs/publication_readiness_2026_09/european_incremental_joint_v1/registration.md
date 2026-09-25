# Frozen Incremental Joint Control: Development Registration

I am testing whether coordinate-proximity joint selection improves the residual
hard errors of incumbent-protected additions. This is a controller experiment,
not new neural trajectory training. Previous outcomes motivated this question;
the same opened sources remain development data, never independent confirmation.

## Fixed Assets And Population
Use all 36 preceding fixed-producer/controller/readout role/seed/event groups.
All forecasts, 381-feature cost heads, stopping incumbent and add-only eligibility
are unchanged and hash-verified. The original 96 hash-selected current queries
per locality, using the September24 salt, define this comparison. Include every
indexed eight-history-eligible target at these queries, regardless of future
label availability. This is not all visible people: agents without a full indexed
history are absent from pairwise control. No forecast is invented for them.

This restricted population is reported explicitly, not passed off as the full
318,969-row parent evaluation. Four source localities per readout view and 12
unique localities overall; overlapping windows and 36 views are dependent.
Reserved model-selection, risk-calibration and confirmation roles remain closed.

## Frozen Decisions
Protect every incumbent neural choice. Only the frozen add-only candidate pool
may change from floor to neural. Use its frozen positive-gain and learned2% risk
gate; no new threshold fitting. Pair geometry starts from the incumbent trajectory
combination, including interactions with fixed incumbent choices.

At each current recording/frame, let k=floor(pool_size/2). The independent
reference takes the k highest predicted utility gains, with stable row-ID ties.
Its predicted positive-harm sum is the shared cap. Compare:

1. Original incumbent, with no additions.
2. All eligible additions, replaying the earlier frozen add-only rule.
3. Half-count independent gain ranking.
4. Half-count outcome-independent hash priority, constrained by the same cap.
5. Half-count unary proximity objective, removing only pair-product terms.
6. Half-count joint proximity objective with the same risk/count constraints.
7. Uniform add-all-or-none across the candidate pool under that same cap/count;
   ordinarily this must abstain. It is not called a matched-rate comparator.

Use existing geometry settings without a sweep: edges within3 median current
bounding-box widths, proximity threshold0.5 widths, pair weight0.1. Utility and
risk costs use the prior controller-training cost scale, not readout statistics.
Pair costs are increased mean squared proximity hinge over the incumbent rollout;
they are not measured collisions, physical safety, or future ground truth.

Two-second numerically certified MILP solves. When any matched solver fails,
all matched variants retain the feasible independent choice; mark that query
unmatched, retain it in intention-to-compare results and exclude it from the
matched-only diagnostic. No timeout is called optimal. k<2 or no supported
pair-products is algebraically unable to demonstrate a joint interaction effect.

## Readout And Gates
Freeze all decisions before newly scoring them. Evaluate all/easy/hard/complete
ADE, true-endpoint FDE, tails, zero-CV errors, intervention counts, unknown labels,
realized harm and predicted proxy. Easy/hard use producer-only CV cutoffs.
Use paired source-locality bootstrap with3,000 resamples and seed39271, including
all four registered source localities; missing/zero denominators stay undefined.
Independent arithmetic verifies coordinate errors and metric reductions.

Primary method evidence requires joint vs independent AND joint vs unary to have
at least one positive all-ADE interval, no negative all/hard intervals, and easy
degradation<=2% in all36 defined views. Also require nontrivial pair support,
actual changed identities and certified matched solves. Report the hash control,
all negative branches and full-add comparison, not a post-readout winner.
This exploratory consistency criterion is not multiplicity-corrected evidence.
No automatic promotion or risk guarantee follows even if it passes.

No training, threshold selection, independent risk calibration, test tuning,
new scene/goal prototypes, Stage5C or SMC. Image-pixel native obs8/pred12, no
metric/seconds, human-gold, true3D, foundation or physical-safety claims.
