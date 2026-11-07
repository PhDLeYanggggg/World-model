# A Wider Forecast Bank Recovers Options, Not Observed Safety

I replayed the same membership ranker on the previous spend-dominant
slate and on all eight candidates with the original causal ledger.
I then compared the original full-bank utility score with a fixed
easy-reference bonus and a matched easy-excess penalty. Both use
the original frozen causal cost estimates. No models are refitted,
and all five arms make the same 6,184 switches at the same queries.

| Policy | Completion Lower vs CV (pp) | All Risk Above 2% | Easy Risk Above 2% | Undefined Easy | Original Passes / 102 |
|---|---:|---:|---:|---:|---:|
| Original | +0.10154 | 68 | 65 | 34 | 3 |
| Restricted membership | +0.09783 | 68 | 64 | 35 | 3 |
| Full membership | +0.10822 | 70 | 64 | 36 | 2 |
| Easy-reference bonus | +0.10122 | 66 | 64 | 35 | 3 |
| Easy-excess caution | +0.10046 | 68 | 65 | 34 | 3 |

The direct full-versus-restricted paired lower is +0.00151 pp
[-0.01207, +0.01563]. The reference bonus changes the paired lower
against the original by -0.00453 pp [-0.01309, +0.00186], and against
the caution control by -0.00388 pp [-0.01283, +0.00296]. The pooled
observed selected easy reference rises 4.13% versus original, but
undefined easy views rise from 34 to 35. Pooled reference is a
descriptive image-error mass over repeated views, not certification
or a view count. Intervals use 3,000 paired draws over 14 sources
after averaging three actual forecast banks.

## Where The Remaining Block Occurs

With the previous predicted ledger balance held fixed at each query,
expanding to the full bank reduces the number ruled out by necessary
observed conditions from **75/102 to 47/102**. The other 55 are not
proved feasible. The diagnostic minimizes observed all and easy excess
separately and maximizes the easy reference. These optima may use
different choices, and they do not update future predicted slack.
The proof applies to the fixed-prefix allowance, not every possible
dynamic policy.

Of the 47 ruled-out views, 33 cannot acquire positive easy reference
at the original counts. A posthoc check finds that all 33 contain
legal single and paired options with positive easy reference and
nonpositive observed all/easy excess when considered locally at
zero credit. Five have an originally inactive query where the
original predicted allowance would also admit such a single switch.
For the other 28, that allowance rejects all easy-supported one-
and two-switch options across the recording. This implicates
predicted risk in the rejected region, rather than absent easy
labels. It does not prove full-recording feasibility, a safe learned
policy, or the behavior under a different earlier allocation.

**Decision.** Enlarging the bank helps opportunity, but none of the
policies improves the full original pass count. A pooled bonus is
not a remedy for scene-level missing reference. The next repair
should test whether causal risk estimation can distinguish observed
feasible options from truly harmful ones among rejected options,
without loosening the final 2% rule.

Exposed Moving A development data; eight observed and twelve future
annotation steps, image coordinates. No independent outcome access,
deployment, Stage5C, or SMC. The posthoc audits export no choices.

## Reproduce

```sh
.venv-pytorch/bin/python scripts/run_m3w_expanded_risk_support.py freeze
.venv-pytorch/bin/python scripts/run_m3w_expanded_risk_support.py score
.venv-pytorch/bin/python scripts/run_m3w_expanded_risk_support.py replay
.venv-pytorch/bin/python scripts/audit_m3w_easy_reference_timing.py
```

Design `c16da762` precedes the native arm64 run. Original and both
membership controls replay exactly; new choices freeze before
scoring. Full scalar replay is exact and 52 scoped tests pass.
[Protocol](protocol.md) | [Replay receipt](replay_execution.json) |
[Timing](reference_timing.json) | [Paired timing](paired_reference_timing.json) |
[Observed opportunities](observed_reference_opportunities.json)
