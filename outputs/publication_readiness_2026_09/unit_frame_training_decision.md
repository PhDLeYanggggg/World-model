# Fixed Observed-Unit Mechanism Comparison

The v2 input frame repairs a native-unit threshold inherited through cached turn
rollouts. Actual parameter-gradient probes on training roles show under-response
to starts, while naive radius decoding under the old loss greatly increases
gradient magnitude. These are optimization diagnostics, not forecast gains.

Before any new held-fit forecast is read, retain three fixed arms:

1. `unit_inputs_only`: reconstructed unit-frame features, canonical rotation,
   but correction magnitude in the original primary coordinate scale.
2. `unit_primary_log`: same features, radius-decoded correction, original log-ADE
   training loss.
3. `unit_internal_log`: same features and decoder, log-ADE divided by past-only
   context radius for both source and main training.

All use the existing geometry branch, three seeds (17/29/43), all three exposed
physical-site fit folds, 2,000 SDD auxiliary updates plus 4,000 main updates,
batch 64 and unchanged optimization settings. This is 27 new fits and nine
hash-verified previous same-schedule geometry controls. No held-score selection.
This isolates numerical conditioning before any new multimodal expansion; it
does not establish that geometry suffices or replace the multimodal research goal.

Full populations remain 11,966 main and 229,333 original SDD train-40 windows.
Main primary evaluation remains equal-physical-site, past-normalized ADE on
8-observed/12-predicted native annotation steps. Source spacing remains stride
12 raw frames, endpoint +144, not time-equated to main. No sealed role is opened.

Rows with no spatial anchor remain in evaluation and fall back exactly to CV.
Their undefined internal loss is omitted only for the internal-loss arm. Record
counts and an old-model guard-only counterfactual to expose this support change.
Partial future labels affect losses only. Standardization fits only the main
training fold. No future input, central velocity or endpoint-derived goal.

Run a 100-update real pilot without held forecasts, resume its state into the
full fixed comparison, replay all predictions, and verify completed resume makes
zero updates. Preserve every result, including easy harm. The site bootstrap
has only three repeatedly exposed clusters and is descriptive, not confirmation.
No deployment, 3D, foundation, metric, seconds, Stage5C or SMC claim is authorized.
