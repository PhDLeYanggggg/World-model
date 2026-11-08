# Who Receives a Fixed-Radius Edit?

I tested whether a causal policy can recover easy-reference support by
changing who receives an edit rather than changing the times, counts or
displacement radii. The prior study changed path shape at fixed recipients;
here all arms use the same observed-prefix shape rule and can only change
the recipient within each active query. A substitute must have enough native
CV-to-OLS4 radius and no agent can take two slots in one query.

## Result

All arms apply exactly 1,457 edits at identical query times and slot radii.
An eligible recipient's candidate is a six-shape past-only path ranked by
short backtests inside the observed eight steps. The learned head predicts
two gain terms, three original-risk terms, and the probability of a positive
easy reference. It fits six other OpenDD development locations at four
fixed amplitudes. There are 21 fresh Torch fits (three seeds per held
location, 800 updates each) and no held-site parameter or threshold tuning.
Joint-local values coverage on every query; joint-prefix discounts it by the
product of earlier no-reference probabilities in that recording. The product
is a scoring proxy, not a calibrated probability bound or safety certificate.

| Recipient rule | Reassigned slots | Easy risk undefined | Original risk passes | Joint passes | Direct gain vs frozen (pp, 95% CI) |
|---|---:|---:|---:|---:|---:|
| Frozen roster, prefix shape | 0 | 334 | 112 | 77 | reference |
| Learned gain | 697 | 364 | 93 | 58 | +0.004 [-0.016, +0.021] |
| Learned support only | 721 | 351 | 101 | 66 | -0.169 [-0.329, -0.041] |
| Joint-local | 652 | 353 | 95 | 67 | -0.060 [-0.175, +0.009] |
| Joint-prefix | 666 | 354 | 96 | 65 | -0.060 [-0.175, +0.009] |

Counts are over 606 recording-seed views (202 distinct recordings at seven
locations); these are not 606 independent test samples. Joint passes require
the original 2% recording check plus positive conservative gain versus both
CV and OLS4. Gain is the paired missing-step-debited ADE difference, in
percentage points of the common CV error mass. The percentile intervals
resample the seven development locations 3,000 times after averaging seeds.
Joint-prefix versus joint-local is +0.0002 pp [-0.0002, +0.0008].

## What Can and Cannot Be Fixed at These Times

The outcome-aware assignment can certify 206 joint passes, versus 77 when
the recipients remain fixed. It uses the same causal shape per candidate
recipient and never changes an edit time or slot radius. This is capacity,
not a learned result. Another 92 views are risk-infeasible, and in 51 the
solver's upper bound excludes strict positive two-reference gain under the
original checks. None of the solves remained unresolved.

At least **257/606** views cannot gain an easy denominator from recipient
reassignment: 172 have no edit slots, 2 have no easy reference anywhere
despite nonzero slots, and 83 have easy references but no eligible recipient
for any frozen slot. This rules out a complete recovery by merely changing
the recipient model. It does not rule out recovering support with different
causal edit times or a different allocation of the same displacement budget.
The learned support rule also loses utility, and adding the chronological
coverage term does not recover it. I do not deploy any of these policies.

Next, isolate the effect of changing edit times under matched budgets and
unchanged risk checks, first determining where support is actually reachable.
The earlier [streaming menu ceiling](../streaming_menu_capacity_v1/README.md)
already allowed changes in time and amplitude under a common credit cap.
It did not isolate time at equal realized edit counts and radii. The next
comparison should keep those distinctions, and separate recordings with no
reachable budget from those where the old policy simply made no edit.
This is OpenDD development, not independent confirmation. The 15/30
DroneCrowd roles remain unchanged; no independent confirmation outcome was
opened. Observation 8 and prediction 12 use native annotation steps. No
physical-safety, metric, seconds-level, true-3D or foundation claim.
Stage5C and SMC remain off.

## Reproduction

```sh
.venv-pytorch/bin/python scripts/run_m3w_recipient_slot.py
.venv-pytorch/bin/python scripts/run_m3w_recipient_slot.py --replay
```

The runner requires the frozen OpenDD caches and prior count and recipient
models. Native Darwin arm64 CPU Torch 2.12.0, four compute threads, one
interop thread, zero DataLoader workers. Checkpoints store optimizer and RNG
state every 100 updates and the runner writes a PID heartbeat. The first run
finished in 14.71 seconds (16,800 updates); replay finished in 6.91 seconds
with zero updates, matching the immutable readout exactly. The local run
does not require a new CREATE job. Historical CREATE completion records are
not a fresh remote readout.

[Registration](registration.json) |
[Aggregates](analysis.json) |
[Checkpoint receipts](head_freeze.json) |
[Supplement v71](../evidence_manuscript_v71/README.md).
