# Can a Recording Risk Ledger Improve the Same Edits?

The [direct pair score](../opendd_slate_relative_gain_v1/README.md)
improved an earlier joint policy, but actual harm still blocked
deployment. I replaced its predicted net risk terms with six separately
supervised quantities: positive harm on observed rows, positive harm
on easy rows, signed easy error change, CV and easy reference masses,
and the missing-row debit. Harm and debit are bounded by the edit
radius. The reference masses depend only on observed motion, not on
which trajectory is proposed. The loss is donor-scaled squared error,
so the targets are conditional means rather than clipped risk residuals.

I froze the previous cost and pair-gain heads, the turn menu, the
per-query edit count and radii, and the original recording-level 2%
checks. The soft arm replaces the risk terms in the previous utility.
Local and prefix arms prioritize the worst predicted risk violation,
then total violation, then shared utility. The prefix arm accumulates
only estimated contributions from queries seen so far in that
recording. The last arm uses the same prefix rule but retains the
original singleton edits. These are fixed-budget experiments: if all
options look bad, the least-bad option still has to fill the slots.
An estimated ratio is not a certificate.

## Same-Budget Comparison

The 21 arm64 CPU fits cover seven held locations at three seeds.
Training donors exclude the scored location; donor neural forecasts
also exclude the donor from their own fit. All decisions use eight
observed positions and predict 12 native steps. Each policy fills
3,690 slots across 439 edited recording-seed views, within 606 total
views from 202 distinct recordings. No confirmation recording is read.

| Rule | Passes all risk checks | Easy reference undefined | Signed easy > 2% | ADE / available-endpoint FDE |
|---|---:|---:|---:|---:|
| Additive | 42 | 116 | 170 | 0.54830 / 1.25312 |
| Frozen pair-gain policy | 47 | 117 | 164 | 0.54826 / 1.25305 |
| Soft moment penalty | 32 | 132 | 143 | 0.54796 / 1.25303 |
| Local predicted risk first | 21 | 132 | 152 | 0.54887 / 1.25490 |
| Prefix predicted risk first | 24 | 132 | 151 | 0.54894 / 1.25498 |
| Prefix with fixed singleton edits | 41 | 120 | 164 | 0.54834 / 1.25299 |

The soft moment penalty improves observed ADE slightly and reduces
signed easy violations, but all-positive-harm violations rise from
323 to 333 and reference support falls. The prefix rule's paired
full-grid lower interaction gain against the frozen pair policy is
**-6.787 [-16.103, +2.145]**, in 10^-3 native units. With fixed
singletons it is **-0.186 [-0.698, +0.296]**. The latter finds eight
strict positive views against the pair policy, but does not improve
aggregate risk or establish an average gain. Intervals resample the
six estimable locations 3,000 times after averaging seeds within
recording. The [readout](readout.json) includes paired accuracy,
hard-subset gain, and recording-mean p95 ADE; the broad prefix arm's
p95 rises from 1.3932 to 1.3983. All errors use native coordinates.

## Separating Recipient Choice from Path Shape

I replayed the frozen outputs as an exploratory check, accepting a
proposed replacement only when it keeps the same edited recipients,
or restricting the changes to singleton or double-edit queries.
Every replay retains the same slots and per-query radii.

| Replay against frozen pair policy | Risk passes | Easy reference undefined | Newly undefined / recovered |
|---|---:|---:|---:|
| Frozen pair policy | 47 | 117 | reference |
| Soft, same recipients only | 41 | 117 | 0 / 0 |
| Prefix, same recipients only | 40 | 117 | 0 / 0 |
| Prefix changes only at singleton queries | 30 | 132 | 32 / 17 |
| Prefix changes only at double-edit queries | 40 | 119 | 6 / 4 |

Preserving recipients preserves easy-reference support exactly, but
it does not preserve observed harm: the prefix shape replay rescues
four views and loses eleven. Its conservative pair interval against
the frozen parent spans zero. Singleton substitutions alone reproduce
the net loss of 15 easy references. Those substitutions dominate the
full-grid loss as well. The two subset replays are not independent
trained policies, and their risks do not add: the denominator and
threshold are computed after combining decisions across each recording.

**Decision:** keep the frozen pair policy as the development comparator.
Do not use predicted masses as evidence of recording risk control.
The next experiment should permit abstention on harmful or unsupported
edits instead of forcing a different recipient into every slot. It
should preserve the matched-budget utility control separately from
the deployment rule. Unlike
[online conformal prediction](https://proceedings.mlr.press/v235/angelopoulos24a.html),
this prefix calculation has no realized feedback in its update and
makes no calibration guarantee.

## Replay

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_opendd_risk_moments.py freeze
.venv-pytorch/bin/python scripts/run_m3w_opendd_risk_moments.py score
.venv-pytorch/bin/python scripts/report_m3w_opendd_risk_recipient_guard.py
```

21 fresh fits, 600 updates per fold, four CPU threads and zero data
workers. Re-run the interrupted command to resume from the checkpoint
written every 50 updates or the completed score fold. The private
experiment directory retains the PID heartbeat. No CREATE submission,
threshold change, Stage5C, SMC or deployment change.

[Registration](registration.json) | [Freeze](freeze.json) |
[Primary readout](readout.json) | [Recipient replay](recipient_guard.json)
