# OpenDD: how often does a uniformly chosen fixed-spend plan pass?

Use the exposed OpenDD development roster, frozen query prefixes,
recording-seed edit count, chronological credit rule and peak-movement
cap of the full-grid hindsight audit. Do not refit the forecast or score
new external labels. Count prefix vectors, not equivalent edit orders.
For each view, sample uniformly from all affordable prefix vectors of
the same count, then reject vectors above the original movement cap.
The full-grid and original risk tests remain at 2%; success also needs
conservative gain above the original solver's 0.000001 pp threshold
against both CV and OLS4. A no-edit view is a failed pass, not a risk pass.

Enumerate spaces of at most 4,096 plans before the movement-cap filter.
Otherwise sample 8,192 ranks with replacement, using a fixed per-view
seed; report accepted draws and do not substitute a nonuniform
alternative when rejection leaves no draws. For a no-draw view, bound
the unknown success probability by [0,1] unless the exact audit already
rules out a pass. Sum expectations across views and report contrasts
by equally weighted locations, averaging three seeds and resampling
seven locations 3,000 times. This is a diagnostic distribution over
whole-recording plans, not an online policy or an independent test.

Check coefficient scores against the direct scorer for the frozen plan,
every near-boundary sampled plan and representative passing and failing
plans. Reproduce the frozen allocation and hindsight counts unchanged.
