# Fixed-Budget Risk-Head Continuation

## Material Passport

Experiment: moving_risk_convergence_v1. Mode: local code experiment.
Data role: already-exposed SOURCE / own-B development, not independent confirmation.
Question: does insufficient optimization explain the previous neural risk failure?

## Single Changed Factor

Continue all eight registered 32-epoch risk heads from their exact model,
AdamW, random, shuffle and row-visit state. Add at least 4,096 updates per fold,
rounded up to complete epochs. Keep the 120 features, source exclusion, 9,890
parameters, learning rate, loss, weights, reference IDs, target scales and risk
threshold unchanged. There is no architecture, threshold, seed or loss search.

The final fixed-budget checkpoint is the only new evaluated checkpoint. Training
does not read held labels for loss, early stopping or checkpoint selection.
No validation-best claim is made. The old 32-epoch failure remains unchanged.
All eight fits must finish before the new source-held readout starts.

Fitting loss is logged every 128 additional updates and at completion. An
absolute relative change of at most 1% over the last quarter of the new budget
is an operational plateau diagnostic, not proof of mathematical convergence.
It does not change the fixed stopping budget or any safety gate. Improving
fitting while harming source-held decisions would weaken the hypothesis that
undertraining alone explains the previous failure.

## Comparisons And Fixed Screen

Compare cached-verified shared64, track32, 32-epoch neural plain/transport and
fresh continued plain/transport. Keep all 48,061 known proposals; 1,274 unknown
outcomes remain unknown. Report per-source fitting loss, source-held risk MSE,
joint decision cost, harmful admissions and selected support, not only averages.

Primary continued transport must reduce source-macro decision cost by at least
1% versus shared64, not increase its false-safe cost, and beat continued plain
in at least two of three contexts. This is the existing component information
screen, not the original full-policy safety estimator. Secondary improvement
against the short-budget network cannot replace the primary comparison.

The original 2% all/easy positive-harm, easy-net and support requirements remain.
Empty support/control is NA. No identity-fallback exemption. Original 24-group
primary remains NA. No new nested-C/H, Moving20, DUT or DroneCrowd readout.
Native obs8/pred12 image coordinates and silver labels remain, with no metric,
seconds-level, true3D or foundation claim. Stage5C/SMC stay disabled.

## Runtime And Verification

Use native arm64 .venv-pytorch, CPU 4 threads, inter-op 1, workers 0. Save every
128 updates and on interruption; no new resource probing or CREATE submission.
Private cap 512 MiB and disk reserve 5 GiB; a 12-hour phase limit preserves
recoverable state. Pilot on one real fitting fold for 256 additional updates,
checking interrupted 128+128 against continuous 256. It cannot select the model.

Register and commit this design before training. Verify all eight continuations
by full repeat from the same parents, then replay with fitting forbidden.
Use hashes for ancestry, schema, references and row alignment. Keep checkpoints
and data private; publish code, configurations, reports and light aggregates.

This is a risk-model optimization diagnostic, not a new trajectory model, new
independent scene evidence, three-seed main experiment or submission success.
