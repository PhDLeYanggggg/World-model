# Frozen Utility-Aware Query Allocation

## Material Passport

New causal decision rules on cached_verified frozen estimators, not new
training. Twelve already-opened development localities, three forecasting
seeds, 108 four/four/two/two source-role groups. Registration and action
freeze precede this policy's outcome readout. No independent sources opened.

## Hypothesis

The preceding decomposition shows that risk-only admission sacrifices more
benefit than harm. Use the existing utility magnitude, not just its sign,
to allocate the same number of interventions within the current query.
Test this policy factor before changing a training objective again.

## Frozen Arms

1. Protected floor: no neural interventions.
2. Independent: unchanged causal-descriptor policy, all original guards.
3. Control matched count: earlier risk-ranking diagnostic, unchanged.
4. Utility top-k: maximum frozen expected utility at the independent count,
   ignoring risk; explicitly an unsafe diagnostic control.
5. Whole-query uniform: select all retained query rows only if all are
   eligible and both aggregate predicted excesses are nonpositive; otherwise
   select none. This arm is not rate matched.
6. Joint utility: maximize frozen expected utility at exactly the independent
   count, with both aggregate predicted all/easy excesses nonpositive.

Eligibility retains the same moving, input-support and positive-utility
guards. The two score constraints remain sum(Hhat - .02 Rhat) <= 0, using
the same frozen signed-excess estimators and fitting-only scale. They now
pool predicted slack within a current query, not across recordings, frames,
sources or future arrivals. This changes the allocation unit, not the 2%
tolerance. No future label or label mask reaches the allocation API.

The retained cohort is not guaranteed to contain every visible agent. This
is joint budget allocation without a pairwise interaction penalty; it cannot
establish nonadditive interaction, collision or full-scene safety benefits.

## Solver and Failure Handling

Use stable utility top-k when already feasible. Otherwise solve the binary
linear problem with SciPy/HiGHS, a fixed 256-node limit and zero requested
relative optimality gap. Scale constraint rows numerically; check integrality,
count, support, original score constraints, primal/dual agreement and utility
nondecrease before accepting. Incomplete or numerically invalid solutions
retain the independent anchor, never an unmatched intervention count.
No held solver tuning. Tie/no-improvement solutions retain the anchor.

## Readout and Gates

Primary: paired ADE gain of joint utility over independent, with identical
current-query counts; lower exploratory 95% CI must exceed zero. Also compare
the matched risk control and unsafe utility top-k. Report all/easy/hard,
complete/partial labels, true-endpoint FDE, tail error, source and seed
breakdowns, unknown-label actions, solver fallbacks and changed query counts.

Retain the real outcome requirements: every held view must have a defined
selected positive-harm ratio <=2%, easy degradation <=2% versus CV, no
zero-CV harm, and nonempty locality coverage. Missing ratios do not pass.
Predicted feasibility does not imply observed or calibrated safety. A positive
mean primary without the risk gates is diagnostic, not deployable.

Use the existing 3,000 paired locality-bootstrap draws, seed101531, after
dependent-view averaging. No IID-window or independent-confirmation claim.
Keep every arm and source regardless of sign; no outcome-based policy selection.

## Execution

Native arm64 CPU4/interop1, workers0, no resource probing. Atomic compressed
action masks, per-group completion and heartbeat/resume. Maintain10GiB free
disk, no duplicate score banks. Local runtime assessed from actual prior
replays and a first-group causal allocation pilot. Existing CREATE restrictions
unchanged. Image-local detector silver, obs8/pred12 rawstride12. No metric,
seconds, human-gold, physical-safety, true3D, foundation, Stage5C or SMC claim.
