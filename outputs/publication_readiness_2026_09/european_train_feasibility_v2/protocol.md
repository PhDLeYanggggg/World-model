# TRAIN Feasibility: Execution Revision 2

The scientific design is unchanged from
[the v1 protocol](../european_train_feasibility_v1/protocol.md): exact label-aware
maximum-utility witnesses for the fixed 2% all/easy and easy-degradation budgets,
fitting recordings only, complete source/context/head-seed roster, frozen controls.
No oracle decision is an inference feature or deployable policy.

This revision repairs a numeric/string recording-ID mismatch in the first
execution. The initial faulty output remains preserved and marked invalid.
The new version checks every frozen per-recording score against its previously
verified parent readout, as well as complete-view scores, action hashes and
per-record count sums. Small exhaustive optimality tests and numeric/string
ID regression tests must pass before the new real diagnostic.

Raw legacy files physically contain all original-TRAIN rows; only already
assigned fitting records are used for new preprocessing and diagnosis. No
inner-held model scoring, independent role, new fit, or threshold search occurs.
Report 24 packet views and 148 record/context views without counting head seeds
as independent data, plus the 72 fixed head-seed control views. Preserve missing
labels, unknown risk, prior failures, and all unsupported recordings.

No new optimizer steps, CREATE job, deployment, Stage5C or SMC. Local native
arm64 CPU4/interop1/workers0, resumable per-view output, process lock, heartbeat,
10GiB reserve and 12h invocation limit. Same executable commands as v1 now use
the v2 output directory; `check` must exactly reproduce all new outputs.
