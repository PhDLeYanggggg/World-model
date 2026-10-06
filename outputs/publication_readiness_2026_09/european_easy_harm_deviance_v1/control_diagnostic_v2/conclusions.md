# Original and New Implementations Agree, Historical Shard Zero Does Not

## Completed Evidence

CREATE diagnostic37817976 completed0:0 in4m10s. It performed68,000 TRAIN-only
verification updates:17 original-trainer direct fits and17 new quadratic direct
fits, each with2,000 updates. These are diagnostic replays, not34 additional
accepted scientific fits. All17 per-identity result hashes verify against
[the completion receipt](complete.json). No development inference, independent
role access, threshold change or new acceptance exception occurred.

Across all17 identities, the unmodified original trainer and the new quadratic
implementation match exactly in model, optimizer, initialization, preprocessing,
input hashes, seed, step, sampler state, Torch state, draw hash and query counts.
Neither direct implementation matches its historical checkpoint on any of these
17 identities. The saved failed checkpoint for eu-locality-020/head29 also
matches both new direct replays exactly. The new loss implementation therefore
does not explain that particular historical-control discrepancy.

Historical/current differences are not all negligible. Their largest model or
optimizer tensor differences range from1.460314e-6 to0.252738. For the failed
eu-locality-020/head29 case, the original error message reports0.113232 in its
first weight tensor; the diagnostic's maximum across all model/optimizer tensors
is0.171524. These are different summaries, not conflicting measurements.
The first unequal recorded monitor appears between steps100 and600, generally
at roughly1e-8 to1e-6 before larger final-state divergence. Monitors are sparse;
they do not identify the first differing arithmetic operation.

## Environment Evidence and Limits

Scheduler accounting places all18 historical shard-zero identities on
`erc-hpc-vm-hj7o46`. That task took3h42m21s. Other historical shards ran on
different nodes and their54 quadratic controls reproduce exactly in the new
training. Current direct diagnostics ran on`erc-hpc-comp186`.

A read-only scheduler query reports the original shard-zero node **not found**.
The scheduler metadata are
[preserved separately](../control_node_accounting_20261006.json); its displayed
Start/End timestamps retain the scheduler's timezone and are not reinterpreted
as UTC. Node availability is summarized in
[the environment receipt](environment_receipt.json), without internal addresses.

The pattern is consistent with historical execution-environment sensitivity.
It does not identify a specific CPU instruction, reduction kernel, floating
mode or library difference as the proven root cause. The original node is
unavailable for a controlled same-node replay. Training-time sensitivity also
must not be interpreted as a small prediction difference or negligible policy
impact: those claims have not been evaluated.

## Preserved State and Next Action

There are110 accepted final fit receipts:54 historical-exact quadratic controls,
one previously registered exact original-trainer replay, and55 deviance fits.
The17 remaining pairs are unaccepted. Their direct diagnostic states remain
on CREATE; they are not silently inserted into the accepted training grid.
The failed tasks, old receipts and the first narrow amendment remain intact.

The one-identity acceptance amendment is now insufficient. Before resuming the
last shard, the next execution contract must explicitly address the18 historical
shard-zero identities, using their hash-frozen unmodified-original-trainer
replays and retaining all historical failures. It must preserve the54 exact
historical controls and110 completed fits, and must not relax numerical
tolerances or select a checkpoint using development outcomes. This broader
contract has **not yet been implemented or executed**. The scientific loss,
fixed update budget, data roles, utility contrasts and2% selected easy-harm
screen are unchanged.

The144-head scientific training freeze remains incomplete. The six-arm
development readout is still`not_run`; independent calibration/confirmation
remain closed. Ten focused diagnostic tests pass. The full legacy suite was
not rerun, and successful replays are not evidence of predictive or safety lift.

Image-local detector-silver obs8/pred12 at raw stride12. No metric, seconds,
human-gold, physical-safety, true-3D or foundation claim. Stage5C and SMC remain
off. The research goal is not achieved.
