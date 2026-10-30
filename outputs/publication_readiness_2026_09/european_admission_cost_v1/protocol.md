# Online selected-action false-admission learning

## Motivation and fixed hypothesis

The completed, exactly replayed selected-action attribution finds that 32532
of 33241 easy false admissions in the upper-score arm are numerator-only
witnesses (repeated SOURCE fitting views, not independent samples). Selected
easy harm mass is 19341.876351 versus predicted 255.500950. In the all channel,
the reference is inflated and aggregate harm is conservative, demonstrating
that cancelling global biases do not diagnose the selected risk problem.
These label-informed arithmetic witnesses are not inference rules or causal
identification. They motivate one new loss, not another threshold search.

## Single controlled change

Retain the five-moment/ranking objective and upper-score auxiliary. In each
training batch, recompute the model's current causal admissible winning action
using the unchanged positive-gain and two 2% inequalities. Detach the discrete
choice. Only for that supervised chosen action and channels with true signed
score positive, add the squared positive underprediction of `q=H-.02R`,
normalized by the unchanged physical fitting RMS quadrature. Average over
all batch rows and both channels, with unit weight. This is a safety surrogate,
not expected-risk calibration or a guarantee. It may suppress useful switches
or remove easy support; these remain failures, not successes.

Unlike previous fixed teacher regions or all-action upper-score weighting,
the action receiving this additional gradient changes with the current model
on every optimizer step. No label chooses the deployment action. Unknown
rows remain in the past-only sampler and receive zero observed-cost loss.
There is no new input, model, trajectory producer, bank, sampler, normalization,
decoder, risk tolerance or evaluation cohort.

## Complete experiment and stopping rule

Six original contexts x three controller seeds x three objectives (original,
upper_score, admitted_score), all using the frozen new-neural seven-action
bank, gives 54 full heads and 108000 primary optimizer updates. Each head
retains 2000 updates, batch128, width64, 30405 parameters and 462 causal fields.
The 36 original/upper full-state controls must match the previous experiment.
One new full 2000-step refit and a real 160-update pilot verify recovery.
Pilot is ten-plus-ten/direct20 for all three arms plus two20-step legacy
controls; no policy scoring in the pilot. Retain all arms, no early stopping
on outcomes or adaptive weight tuning. Register endpoints before readout.

Primary: admitted_score minus upper_score full-policy known and
completion-lower utility, with 3000 locality bootstrap draws after averaging
overlapping context/seed views. Also report original, same-head physical
restriction, selected-action attribution, hard/easy, tails, support and all
required recording risks. Same-count controls are utility-only; empty is
not estimable. The unchanged complete-policy 2% safety rule must pass in all
required views; no positive mean may hide violations. No held or independent
role is opened. All evidence is SOURCE fitting/resubstitution, not whole-
pipeline OOF or independent confirmation. End after the fixed matrix and
full separate replay; diagnose rather than silently add trials.

Execution: native arm64, CPU4/interop1, workers0, checkpoint100/heartbeat50,
12h bound, 10GiB free reserve, 256MiB checkpoint cap, explicit resume. The
previous 162-head experiment fit locally in 778s and 10.91GB; a new measured
pilot must precede this smaller matrix. No reason to duplicate HPC work.
Outputs contain only code, aggregate reports and light metrics. Weights,
inputs and caches stay private. No metric/seconds, human-gold, true3D,
foundation, submission-ready or deployment claim. No Stage5C or SMC.
