# Explicit Zero-Reference Atom: Completed Negative Result

**The added readout does not repair zero-reference protection. No deployment change.**

I fitted 36 new leaf-frequency readouts on the frozen cutoff-relative forests
and ran the full registered control matrix. The experiment uses four already
development-exposed SDD sites, three seeds and all 175756 past-eligible windows.
Its protocol is obs8/pred12, stride12 annotation steps, annotation pixels. These
are not historical raw t50 results, independent confirmation or metric safety.

## What Changed

The prior easy labels already included zero-CV cases. I did not find or repair a
missing-label bug. Instead, I tested a separate conditional component for the
event that constant velocity has exactly zero complete-label ADE. Its probability
was fitted using the same source-only weighted rows on the same 128-tree
partitions. At inference, a positive estimated probability vetoes intervention;
the fixed risk scores, budgets, cutoffs and forecasts remain unchanged.

This is new risk-readout fitting, not 36 new forests or new Transformer/EqMotion
training. The [registration](registration.md) was written before fitting, choices
and readout. All 188388 query/action/seed choices were frozen before aggregation.

## Main Results

The following are equal-site ADE percentage gains over constant velocity, not
over a selected strongest baseline. Easy degradation is the worst positive-easy
site/seed value. Zero harms count repeated window/seed instances, not independent
events. The full [30-row table](results.md) retains every rule and action.

| Population rule | ADE gain % | Hard gain % | Worst easy degradation % | Zero-CV harms | Mean selected windows |
|---|---:|---:|---:|---:|---:|
| Transformer, original | 3.6046 | 3.0823 | 1.2732 | 5 | 44673.7 |
| Transformer, atom guard | 2.9702 | 2.4535 | 1.0152 | 5 | 35645.3 |
| Transformer, same-count control | 3.2336 | 2.7827 | 1.2687 | 5 | 35645.3 |
| EqMotion, original | 3.3370 | 2.0532 | 1.4580 | 6 | 28666.7 |
| EqMotion, atom guard | 2.7964 | 1.5117 | 1.2015 | 8 | 24308.0 |
| EqMotion, same-count control | 3.0904 | 1.8669 | 1.0023 | 5 | 24308.0 |

The atom guard lowers average utility and does not remove Transformer zero-CV
harms. EqMotion has more such harms after joint reallocation, despite fewer
total interventions. The policy vetoes actions before re-solving a joint budget;
its final selected set need not be a subset of the previous solution.

At the same intervention count in every recording/frame/seed, the guarded
population policy trails its control by -0.2634 percentage points for Transformer
(nominal 95% CI [-0.3817, -0.1820]) and -0.2939 for EqMotion
([-0.5176, -0.1261]). Thus reduced coverage alone does not explain the utility
loss: which windows are kept also matters. All nine guarded rules have lower
point-estimate ADE gain than their same-count controls. These are conditional
development comparisons on only four physical sites, not adjusted confirmation.

The more restrictive Transformer point/selected rules still harm one zero-CV
window/seed. EqMotion point/selected retain zero observed harms but have lower
ADE gain than their original counterparts. Simple damping remains free of
observed zero-CV harms here and also loses utility under the added veto. None of
these controls is promoted as a new winner.

## What the Failure Reveals

- Effective source support for **moving** zero-reference cases is only 2--7
  distinct windows per fitted readout. Overlapping windows are not independent
  tracks. This is the support relevant to a policy that does not intervene on
  last-step-stopped histories.
- Transformer has 1999--2646 effective zero-reference source windows and EqMotion
  6993--9811, but almost all are stopped. A large total count therefore does not
  establish coverage of the rare moving event.
- Across the four held-source sites there are only seven moving zero-CV windows:
  five in coupa and two in hyang, reused across the three seeds. Both hyang cases are
  admitted by every action/seed readout. Finite-leaf empirical absence can miss
  precisely the cases the guard is intended to protect.
- An estimated probability of zero is not a calibrated upper risk bound. The
  existing partitions optimize six conditional moments, not rare-event recall;
  this experiment does not isolate partition design from sparse event support.
- Easy population averages below 2% and feasible predicted budgets do not imply
  protection of the zero-reference component. The separate requirement fails.

[Fit support and losses](fit_support.md) retain all 36 fits, including unknown-label
draws of zero. The full-complete-label event Brier in that table includes stopped
rows outside intervention support; it must not be presented as rare moving-event
recall or independent calibration. The separate descriptive diagnostic reports
the moving/effective subset and actual harmful-case probabilities.

The [descriptive harm audit](postreadout_diagnostics.json) confirms that every
guarded harmed case receives empirical zero probability. Population Transformer
harms two unique windows, repeated five times across seeds, with ADE
1.4141--2.7050 pixels. EqMotion harms three unique windows in eight seed instances,
with ADE 2.6943--5.8874 pixels. Reallocation avoids one original EqMotion harmed
window/seed instance but introduces three other instances, for a net increase
of two. This is not floating-point noise or a failure to apply the veto.

Only 7 of 108462 complete moving/effective evaluation windows have the zero-CV
event. Event Brier can therefore be around 0.00007 while missing most or all
positive cases. A constant all-zero probability already has Brier 0.00006454
on that slice. A low overall squared probability loss is not evidence that the
rare-event guard works. These are post-readout explanations, not a new selection
criterion or threshold adjustment.

## What This Does Not Establish

The result rejects this **fixed-partition empirical-atom veto** as a repair. It
does not prove that every conditional-distribution model must fail, that adding
more trees would fix the problem, or that sparse support is the sole cause.
Leaf counting is an established construction, not a novel architecture claim;
see the [method positioning](method_positioning.md).

Unknown and incomplete futures remain in the index and in explicit bounds.
Source exclusion is checked, but these sites remain design-exposed. DroneCrowd
confirmation is closed and IMPTC remains quarantined, with no external forecast
errors opened. No safety certificate, seconds, metric, true3D, foundation-model
or submission-ready claim follows. Stage5C and SMC remain disabled.

## Next Decision

Do not loosen the zero-reference tolerance, choose a probability epsilon from
these outcomes or promote the original policy based on this failed comparison.
The next source diagnostic should separate insufficient relevant event support
from representation aliasing: compare causal neighborhoods of moving zero-CV,
beneficial and harmful windows, with frozen site exclusions and no new threshold
selection. Any subsequent support-aware model must use matched controls and
retain abstention when support is inadequate. Independent-site admission and
calibration are still prerequisites for a paper safety claim; more source-only
variants do not substitute for them.

See [statistical limits](statistical_interpretation.md), [operation/recovery](operation_zh.md)
and [execution evidence](execution_notes.md). The research goal remains active.
