# Producer-Relative Cap-Event Learnability

## Material Passport
Code experiment, source-development only. Registration precedes new support
summaries and all fitting. Predictions must be committed before held-locality
labels are scored. This diagnostic is not independent confirmation, a risk
bound, a new trajectory result or a deployment gate.

## Question and Controls
Can causal history identify easy-harm realizations above the frozen all-harm
estimate? Prior any-harm classification and context/risk-bin MSE correction
failed. A large empirical output-cap floor does not establish conditional
bias: future variation can exceed an accurate conditional expectation.
This study tests predictability, not an automatic reason to raise a cap.

The matched arms are logistic regression and a one-hidden-layer SiLU network
(width32), both trained in native Torch. Both receive the existing frozen
native causal vector plus seven context summaries, seven missing indicators,
and H/envelope and H_E/H. No ID embedding, observed future, realized error or
event label enters prediction. Native inputs include causal rollouts, not
future targets. The forecaster and original cost estimators remain frozen.

## Fitting and Transport
Each of144 fixed views has three meta-fitting localities and one held locality.
For each fitting row the nested cost producer excludes that row's locality.
The binary label is common-meta-event easy harm greater than this producer's
predicted H. Common easy thresholds use only the three meta-fitting localities;
they are supervised labels, never inputs to the two-locality inner teacher.
At held readout, the event uses the frozen three-locality outer producer.
This two-to-three-locality producer transport mismatch remains explicit.
Support includes positive/negative rows, recordings and recording-agent pairs;
overlapping windows are not independent samples. Numerical fitting requires
both classes overall and known rows in every fitting locality, not a claimed
power threshold. Unsupported views must stop the registered all-view run.

Preprocessing is fitted only on known positive-disagreement fitting rows,
with equal locality weight. Missing values use fitting means; all-missing
columns use zero. Standardized inputs are clipped to[-20,20]. Constant-column
scale is one. Site-balanced minibatches optimize unweighted binary log loss,
not class-balanced loss; class frequency remains meaningful. Each arm trains
2000 fixed updates, AdamW lr0.0003, weight decay0.0001, batch256, clip5.
The intercept starts at the fitting prior. Seeds17/29/43, six frozen producer/
controller assignments and full/motion-only inputs are all retained. No early
stopping, hyperparameter search, threshold fitting or favorable-arm selection.

## Readout and Evidence Rule
Primary descriptive contrasts are held binary-log-loss/Brier gain versus
the fitting-only constant prior, MLP versus linear, and AP/top10 overshoot-mass
capture versus the causal disagreement envelope. Frozen H_E/envelope and H/
envelope rankings are additional fixed controls, not calibrated event
probabilities. Positive-disagreement rows are the primary population; zero
disagreement is deterministically no harm and cannot establish learnability.
All support, AUROC, AP, predicted prevalence and training losses are reported.

Average three seeds inside each held locality, then3000 paired bootstrap
resamples of four localities within each assignment. Report all six assignment
intervals separately. Assignments overlap; intervals are exploratory with
only four localities and no multiplicity adjustment. Missing class support
makes that contrast not_estimable, never zero or silently discarded.
A consistent diagnostic signal requires all six full-input MLP intervals
positive for log-loss gain over prior, AP and overshoot capture over envelope.
This deliberately does not claim calibrated cost magnitudes or policy utility.
No new deployment is possible from this experiment, even if that rule passes.

## Runtime and Boundaries
CPU4/interop1/workers0, native arm64, no resource probing. Pilot first100
updates of the first MLP; resume the same checkpoint to the fixed budget.
Checkpoint every200 updates; heartbeat includes PID/step. Preserve10GiB disk.
Local placement follows pilot cost; CREATE jobs are queried read-only first.
Source development only; independent selection, reserved calibration and
confirmation stay unopened. Obs8/pred12 native annotation steps and detector
pixels. No metric/seconds, physical-safety, human-gold, true3D or foundation
claims. Stage5C/SMC remain off.
