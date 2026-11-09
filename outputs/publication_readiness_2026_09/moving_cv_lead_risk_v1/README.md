# Better Annotation Prediction Did Not Improve Edit Decisions

I separated two changes on the frozen motion bank. One risk head
triples the Huber penalty when it underestimates observed easy excess,
using the same initialization and draws as the original conditional
head. A history-only head predicts annotation availability at each
future step; another predicts observed benefit at each step for each
candidate. The flat-availability control uses the same two heads and
the per-row mean availability, holding the total expected number of
visible steps fixed. Availability here is a dataset-label property,
not an estimate of physical observability.

| Fixed edit-count strategy | Edits | Original risk passes /72 | Undefined easy /72 | Worst known signed easy change |
|---|---:|---:|---:|---:|
| Conditional risk, flat availability | 51,614 | 6 | 20 | +27.62% |
| Conditional risk, lead-specific availability | 51,614 | 7 | 20 | +25.90% |
| Asymmetric risk, flat availability | 51,614 | 12 | 19 | +64.23% |
| Asymmetric risk, lead-specific availability | 51,614 | 13 | 19 | +61.90% |

All four use exactly the same edit count in every query, but can
choose different recipients and paths. Lead-specific availability
improves query-weighted Brier score from 0.18185 to 0.16559
(reduction 0.01626 [0.00447, 0.03070]). It does **not** improve the
conservative decision score. Its paired lower utility versus flat
availability is -0.7673 pp [-1.8435, -0.1614] with conditional risk,
and -0.3416 pp [-0.7613, -0.0789] with asymmetric risk. For conditional
risk, the paired neighbor lower contrast is -0.4839 pixels
[-0.8805, -0.1844]. Intervals resample nine sources 3,000 times after
averaging the three seeds. These conservative contrasts are not
point estimates of the unseen trajectories.

At fixed lead-specific gain, asymmetric risk raises original passes
from 7 to 13 and reduces known easy positive-harm violations from
39 to 23. It increases the worst known signed easy change, and its
direct conservative utility contrast is -1.9501 pp
[-4.5112, -0.3825]. Of 30 otherwise passing observed views, 17 still
fail when wholly missing outcomes are charged. The uncapped
asymmetric/lead variant passes 14/72 at 54,748 edits, but its lower
utility against OLS4 remains -55.044 pp [-96.583, -23.007].
**No calibration qualifies (0/9).** The preceding 73,477-edit control
reproduces exactly. The matched and uncapped comparisons have
different exposure and are not interchangeable.

## Is the late disagreement avoidable?

The conditional-risk lead/flat comparison gains +0.0387 pp on known
steps. The paired missing-step debit is 0.8060 pp, of which 0.5821 pp
(72.2%) is on steps 9--12 and 0.5178 pp is on agents edited by both
policies. This is mostly a path-disagreement problem, not only
recipient changes.

As a secondary control I shifted the same per-agent total displacement
earlier with a fixed taper, leaving the last four steps at CV. The
control shrinks uniformly to the same total displacement. Neither
changes recipients, and neither is calibrated. Taper-minus-uniform
lower utility is -0.4216 pp [-0.8833, -0.1242] with conditional risk
and -0.1973 pp [-0.3751, -0.0630] with asymmetric risk. Uniform/taper
risk passes are 21/15 and 24/23. The taper also increases peak edit
displacement (2.42 to 2.86 pixels, and 1.44 to 1.73 pixels), which
matters for the wholly missing risk debit. Its lower utility versus CV
has a positive point estimate but crosses zero in both arms; neither
approaches OLS4. Moving displacement earlier at the same total is
not supported as a repair.

Better annotation probability and lower average harm do not identify
a good allocation under this rule. The next experiment should
supervise **selected sets** from outcome-aware feasible training
witnesses and evaluate with the same fixed bank, rather than changing
one more rowwise visibility or risk cutoff. Outcomes for such
witnesses are training targets only. The original per-recording rule
and disjoint confirmation roles stay fixed.

This differs from [TrajImpute](https://arxiv.org/abs/2411.00174),
which studies missing observations in the input trajectory. Here
observed history is intact and the missing values are future labels
used in the training and evaluation targets.

## Reproduction

`fresh_run`: 27 Torch fits, 27,000 updates, 101.61 s on native arm64
CPU, 24 held fitting recordings, nine source groups, three seeds.
Four Torch threads, zero loader subprocesses, checkpoint every
100 updates. `cached_verified`: parents, source-excluded donors and
exact zero-update replay. Forty-five focused tests pass; the full
legacy suite was not rerun.

```sh
.venv-pytorch/bin/python scripts/run_m3w_cv_lead_risk.py
.venv-pytorch/bin/python scripts/run_m3w_cv_lead_risk.py --replay
.venv-pytorch/bin/python scripts/audit_m3w_cv_lead_risk_evidence.py
.venv-pytorch/bin/python scripts/audit_m3w_cv_lead_disagreement.py
.venv-pytorch/bin/python scripts/audit_m3w_cv_late_edit_taper.py
```

The anonymous scalar archive
`031e24f26d74fa12be680fbc434ab0c7f2f9c389c3a6d85aa17f7eafe2fc77ed`
reproduces risk and utility after clean extraction with NumPy only.
It does not reproduce training, geometry or the taper.

`not_run`: new A or DroneCrowd calibration outcomes, independent
confirmation, deployment, Stage5C and SMC. All 30 independent groups
remain sealed. The original 2% rule is unchanged. Native 8/12 annotation
steps and image pixels are not verified seconds or physical units.
This is not a true 3D or foundation model.

[Protocol](protocol.md) | [Primary results](readout.json) |
[Risk and replay](failure_replay.json) |
[Missing-width attribution](missing_disagreement.json) |
[Taper](late_taper.json) | [Manuscript v132](../evidence_manuscript_v132/README.md)
