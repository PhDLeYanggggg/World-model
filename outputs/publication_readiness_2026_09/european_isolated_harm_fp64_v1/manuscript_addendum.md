# Experimental Addendum: Protecting Benefit Does Not Ensure Safe Selection

We tested whether risk learning could be separated from an accurate
utility-trained benefit/reference representation. From18 frozen shared-utility
anchors, we trained three matched continuations: utility-only, joint-risk, and
an isolated harm branch with an immutable base. Each received2,000 further
updates with identical continued query draws. All54 finals were frozen before
the complete144-view fit/held-recording readout. A numerical backward defect in
an earlier float32 precursor was repaired; all arms were then restarted, with
no precursor result reused.

The isolated branch preserved benefit and reference predictions exactly but
did not improve intervention quality. Relative to equal-budget joint risk,
its held paired completion-lower utility decreased1.628 percentage points
(nominal95%CI[-2.464,-0.853]); at matched intervention counts the decrease was
0.276points[-0.491,-0.077]. Known easy-risk violations occurred in47/72 views
versus45/72 for joint risk, and only3/72 isolated views had complete policy
support. The fixed2% risk criterion was not relaxed.

On an identical anchor-selected cohort, the isolated branch increased the
median predicted-to-observed easy-harm ratio from0.0049 to1.531. Nevertheless,
the ratio on its own selected cohort was0.0133. These are descriptive aggregate
ratios with different defined-view counts, not calibration guarantees. Together
with the negative matched-count contrast, they show that bulk harm estimation
and selected-tail reliability remain distinct in this experiment.

Continued utility training also improved fitting MSE without held improvement.
This result does not support solving the problem through longer optimization
alone. Nor does it identify gradient conflict as a unique mechanism: isolation
changes trainable capacity and freezes potentially imperfect reference outputs.

All evidence is exploratory on previously exposed original TRAIN recordings:
74 fit and31 held source-recording pairs across12 localities. Confidence
intervals use3,000 locality bootstrap draws after averaging repeated views.
They are nominal, not independent confirmation or simultaneous guarantees.
The utility is baseline-reference-normalized completion-lower mass, not raw
trajectory ADE/FDE. No new forecasting producer or deployment was introduced.

The original SelectiveNet comparison and remaining methodological distinctions
are documented in [the literature note](literature_position.md). A separate
risk branch alone is not claimed as novelty. Positive independent evidence,
published-method comparisons and safe scene-level contribution are still
required before the project can support a submission-ready central claim.
