# Execution Record

## Material Passport

Fresh frozen-model diagnosis; no fresh training. Hash-verified models and
source caches reused. No independent roles, deployment, Stage5C or SMC.

- Registration: `111157b4`, pushed before the first slice readout.
- Pilot: PID32796, one complete group,8.49process seconds,6,130,581,504bytes
  peak RSS. This group was reused by the full run, not counted twice.
- Full108groups: PID32831,210.88process seconds,11,409,784,832bytes peak RSS.
- Full replay: PID33166,219.94process seconds,12,081,725,440bytes peak RSS.
  Every aggregate, fitting cutoff, prediction and action matched exactly.
- Native arm64 environment, CPU4/interop1, workers0. No NumPy fallback
  presented as training. No CREATE job submitted or environment modified.
- Latest pre-existing approved CREATE queue receipt remains the27September
  read-only check from the parent experiment, not a newly polled live claim.
- New private aggregate store is about31MiB; no raw, feature, latent or model
  cache was duplicated. Public reports contain aggregates and hash references.
- Disk is close to the10GiB reserved-free threshold. Check actual free space
  before the next checkpoint-writing experiment; do not delete unrelated data.

The report writer initially refused an attempt to append tied-bin metadata to
an immutable diagnostic JSON. It stopped before changing that evidence. Tied
cutpoints were instead written to a separate `bin_ties.json`; the final report
writer then completed. This was a reporting/provenance guard, not a failed
model fit or altered numeric result.

`verification.json` records final targeted tests, independent reductions and
byte-reproducible reports. Full legacy suite and cold raw rebuild are not_run.
The figure was inspected for readable labels, intervals and missing-bin marks.
All local work processes are terminal; resume uses saved per-group records.

Git updates are explicit-path only. The large unrelated staged change set
is preserved. No raw data, private caches, checkpoints, third-party data or
environment directory is included in this diagnostic commit.
