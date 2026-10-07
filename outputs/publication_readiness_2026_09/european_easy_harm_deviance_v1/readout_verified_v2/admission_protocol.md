# Complete-grid admission after exact-reference recovery

This readout uses the unchanged registered six-arm calculation and scalar verifier
from `readout/calculation_registration.json`, and calls the original readout
implementation without changing its model inference, selection, unknown-outcome
bounds, matched intervention counts, bootstrap or pass criteria.

Admission requires 144 complete paired final fits: 54 historical-exact quadratic
controls, 18 exact original-trainer replays and 72 candidates. All 110 previously
accepted receipts must remain byte-identical. Reference identities and checkpoint
hashes are those in `control_execution_amendment_v2.json`; both old and new
registrations stay immutable. All four contributing tasks and the final join must
complete successfully. Failed historical reproduction remains a reported limitation.

The original readout registration remains retained and unexecuted. This directory
has its own provenance registration before inference and retains the same 72
views, 12 localities, 3,000 paired-locality bootstrap draws, four primary contrasts,
full and matched risk checks and 2% positive-easy-harm/reference budget. Unknown
outcomes are not imputed as safe. Development is exposed, not confirmation.

No weights are cached locally. No independent calibration or confirmation labels
are opened. No checkpoint selection, threshold search, deployment change, Stage5C
or SMC execution is authorized by this readout. Raw image coordinates and frame
horizons remain unchanged; no metric, seconds or physical-safety claim follows.
