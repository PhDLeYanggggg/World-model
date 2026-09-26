# Failure Analysis

## Supported Findings

1. The primary magnitude objective failed in held-locality evaluation: none
   of six full-pair MSE intervals is positive. The worst assignment is -25.741%
   with CI [-51.920%, -0.066%]. This cannot be relabeled as a successful repair.
2. Event discrimination is not completely unlearnable from these inputs.
   Conditional AUROC improves with five positive role-level intervals. The
   intervention distinguishes some supported harm cases more effectively.
3. Better average coverage does not imply smaller squared error. Coverage
   aggregates signed over/underestimation; MSE also penalizes misplaced or
   excessive harm predictions. The paired tail metric is not uniformly positive.
4. Fitting and transfer differ. Full training MSE improves in 56/72 dependent
   views, held MSE in 33/72, with 29 fitting-only improvements. The change is
   not just an optimizer that never learned. It is also not a universally
   helpful representation waiting only for a larger training budget.
5. Other outputs incur a tradeoff: two full D_all MSE intervals worsen.
   Shared parameters allow the auxiliary to change reference-cost estimation.
6. Motion-only conditional AUROC has no positive interval. Its 30/72 weak
   event-support views remain visible. A full-pair ranking improvement is not
   evidence of a universally transferable harm score.

## Mechanisms to Test, Not Facts Already Proved

Fractional cross-entropy changes the finite-capacity optimization emphasis
from raw cost magnitude toward cost relative to disagreement. It can improve
ordering and typical coverage without repairing rare, high-cost prediction
errors. The loss is mean-target consistent at the unrestricted pointwise
optimum; that fact gives no finite-model or shifted-locality guarantee.

The error decomposition in `replay_receipt.json` separates zero/positive easy-harm
and causal disagreement above/below the largest training-fitted bin edge.
This is descriptive post-readout diagnosis, not a new selection rule. Of 39
full views with worse held MSE, the zero-easy-harm partition contributes more
of the excess squared error in 37. Only 5/39 have the above-training-edge
partition as the larger contributor. Motion-only has 38 worse views, with
34 dominated by zero-easy-harm error and 11 by the upper envelope partition.

This makes excess predicted easy-harm on zero-target rows the main observed
MSE penalty, not universally a missed costly tail or large-envelope outlier.
Crucially, H_easy=0 includes rows outside the train-defined easy event, as
well as easy rows with no positive harm. These are not all harmless rows.
The present decomposition does not separate easy-event membership errors
from overestimated harm within easy cases. That distinction must precede a
targeted repair; it cannot be inferred from aggregate coverage or AUROC.
Label-based partitions are never inference inputs.

Earlier occurrence/severity hurdle loss, pairwise ordering, rare-harm sampling,
selected-group objectives, extra continuation and reference protection have
already been tested. This experiment does not erase those negative results.
Repeating them under a new name, tuning C thresholds or globally rescaling
held costs would not be a justified next repair.

## Next Falsifiable Repair

First separate outside-easy zero targets from inside-easy zero-harm targets
under the same frozen cuts, using them as diagnostic labels only. Before
another policy run, test whether the new representation contains useful
ranking information that a separate magnitude readout can exploit. Compare a
small nonnegative magnitude readout on frozen fractional features against the
same readout on frozen mean-head features, with an unchanged reference-cost
path and matched fit budget. Keep the fourth locality outside every fitting
and preprocessing step. Register the exact readout, objective and budget before
evaluation, and first check whether this repeats an earlier calibration design.

The earlier bridge/nested calibration used population rescaling and selected
risk grids. A new frozen-feature magnitude readout would be a different fit;
it still needs its own controlled registration and cannot inherit their results.

This is a proposed next experiment, not a model already trained or evidence
that calibration will solve the problem. No coefficient/threshold is selected
on these held results. The primary gate, 2% tolerance and data roles are fixed.
Reserved calibration and confirmation data remain unopened.

## Implementation and Claim Limits

A targeted reporting test caught a duplicate dictionary key between a locality
identifier and held-gain value; the diagnostic identifier was renamed before
real diagnostic execution. This did not change training, predictions, labels
or the primary readout. Earlier float32 equivalence testing uses a numerical
tolerance only for equivalent summation expressions; exact model/resume and
matched-sampler checks remain exact. Negative research results are retained.

Four resampled localities per assignment give coarse uncertainty, and source
assignments overlap. No multiple-comparison correction or independent test
claim is made. Detector-label uncertainty remains. No new ADE, physical-safety,
metric/seconds, true3D, foundation or submission-readiness claim. Stage5C/SMC off.
