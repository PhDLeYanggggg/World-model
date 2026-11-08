# Past Sampling Reverses Utility but Does Not Repair Risk

**Fresh paired diagnostic.** We keep every forecast anchor, future
label mask, CV fallback, intervention recipient and half-blend weight
from the frozen hindcast. The only change is which past positions fit
the alternative velocity. Unselected trajectories remain exactly CV;
a selected row can become a no-op if its fitted velocity equals CV.
The four layouts are defined inside the existing eight observations,
so no rows are removed to obtain longer history. Slopes are per native
annotation index, not seconds. No new thresholds or parameters are fit.

| Velocity fit (offsets from query) | Fitting gain vs recent (pp) | Opened gain vs recent (pp) | Original fitting / opened passes |
|---|---:|---:|---:|
| Recent four: -3,-2,-1,0 | 0 | 0 | 0/24; 40/40 |
| Wide uniform four: -6,-4,-2,0 | -1.737 | +0.829 | 0/24; 39/40 |
| Wide endpoint four: -6,-5,-1,0 | -1.654 | +2.615 | 0/24; 38/40 |
| Wide dense seven: -6 through 0 | -1.659 | +2.465 | 0/24; 38/40 |

Gain is the direct paired conservative contrast on the fixed CV
denominator. The primary wide-uniform contrast has 95% grouped
intervals [-3.354,-0.322] on nine original fitting groups and
[+0.609,+1.067] on 15 opened DroneCrowd groups, using 3,000
bootstrap draws. The corresponding *known-only* fitting contrast is
-1.006 pp [-2.474,+0.364], so the conservative result should not be
read as a decisive observed-label-only loss. The two additional wide
layouts repeat the conservative reversal. Longer span and number of
samples are separate from layout: at fixed four points and six-index
span, wide uniform trails wide endpoints by 1.789 pp on the opened
cohort. This establishes sensitivity to the chosen past observations,
not that actual release cadence causes the gap. Even the wide endpoint
half blend (+29.100 pp vs CV) has less gain than full OLS4 (+51.288 pp)
on the opened roster and loses two original risk passes.

## A capacity limit before training another selector

We first give every original recipient the best of the four layouts
using future labels on the fitting cohort only. At a known row, choose
minimum available-label ADE; at a wholly missing row, choose minimum
maximum disagreement from CV. This hindsight witness passes 0/24.
Even the optimistic bound using all eligible reference mass rules
out every recording.

The second certificate relaxes the discrete layouts further. For each
future step, allow any point in the convex hull of the four half-blend
paths at that step. The distance from the target to the hull lower
bounds any single blend's step error. On wholly missing rows, distance
from the final CV position to that step's hull lower bounds maximum
path disagreement. Subtract known CV error and keep positive harm.
Using the largest possible selected reference masses makes both ratios
optimistic, even when a mixture cancels a change. No-op or unsupported
recordings do not pass automatically.

**24/24 excluded even by this relaxation:** 20 have defined optimistic
easy-harm lower bounds of at least 7.1706%, and four have no eligible
easy reference mass. The relaxation can use a different mixture at
each step and can inspect outcome labels; it is not a forecast or
deployment result. The conclusion is limited to the frozen recipients
and convex combinations of these four half-blend paths. It does not
rule out new recipients, withdrawals, new amplitudes or different
forecast geometry.

**Next decision:** Do not train another selector over this unchanged
candidate bank. Screen capacity for a genuinely different path family
and recipient allocation against full OLS4 on fitting sources first.
Keep the original 2% rule and all 30 independent confirmation groups
sealed. Stage5C and SMC remain off.

~~~sh
p=scripts/run_m3w_past_sampling_audit.py
for phase in register freeze score replay; do .venv-pytorch/bin/python "$p" "$phase"; done
for x in risk convex; do
  p="scripts/audit_m3w_sampling_${x}_capacity.py"
  .venv-pytorch/bin/python "$p"
  .venv-pytorch/bin/python "$p" --verify
done
~~~

Native arm64, per-recording receipts and heartbeat. Path hashes for
all 40 opened recordings freeze before scoring. The original recent
and OLS4 comparisons reproduce. Both capacity receipts and the
overall score replay exactly without reopening labels. Twenty-two
focused tests passed. No independent confirmation or deployment.
[Analysis](analysis.json) | [Discrete bound](fitting_capacity.json) |
[Convex relaxation](convex_capacity.json).
