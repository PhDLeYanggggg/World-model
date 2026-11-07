# Changing When To Switch At The Same Recording Total

Fresh development analysis on the same 14 sources, 34 recordings and 102
views from three existing forecaster seeds. Paths, original 2% observed
risk/support rules, eight observed and twelve future annotation steps,
and the direct control are unchanged. Image coordinates only.

| Constraints besides the original observed rule | Prior fixed query: feasible / infeasible / unresolved | Same recording total: feasible / infeasible / unresolved |
| --- | ---: | ---: |
| None | 81 / 20 / 1 | 84 / 18 / 0 |
| Final predicted budgets | 62 / 36 / 4 | 72 / 30 / 0 |
| Every predicted prefix | 48 / 40 / 14 | 57 / 33 / 12 |

All 84 nonzero-total views are feasible under observed risk alone; the
other 18 have no interventions to move. Of the former 20 proved
infeasible at fixed query counts, two become feasible and 18 remain
empty. Under prefix constraints, seven previously proved infeasible
views become feasible, and the number of checked positive witnesses
rises from 48 to 57. For the terminal-budget condition, 70 of the 72
checked witnesses have positive paired completion-lower utility; two
are feasible without positive utility.

Keeping the recording total fixed moves 612 switches in the
observed-only witnesses and 233 in the prefix witnesses. Comparing
only pairs with both optima certified, the timing relaxation improves
47/81 observed-only views and 18/31 prefix views. These counts describe
hindsight, not an inferential estimate or an independent learned result.
The prefix solve has 12 unresolved views; numerical rejections or limits
are not infeasibility. No aggregate ADE improvement is imputed for
unsupported or unproved views.

## Causal Comparisons

The local and banked greedy arms choose exactly the direct action in
every row of all three seeds. They each make 6,952 switches, pass 2/102
views, and have zero paired utility difference. The rule is inert
because it spends each earned slot rather than carrying it forward.
The [one-slot reservation](reservation/readout.md) actually moves 70
switches but leaves 67 unused and loses 0.00505 pp of paired lower
utility (95% source interval [-0.01073, -0.00031]). It is not an
equal-volume control and does not advance deployment.

Timing matters for some views at the same total, but a greedy
predicted-gain rule does not turn hindsight opportunity into reliable
online improvement. The 18 zero-total views also require a coverage
change, not retiming. The next model test should predict where an
intervention can provide an easy reference without increasing observed
harm, then test allocation without using the final total at inference.

## Verification

Native arm64 local run, 579.1 seconds for the registered scoring and
solves. Checkpoint aggregation replay matches byte for byte; it does
not re-solve. All nine learned members of the timing bank (three seeds
each of PCT, EqMotion and self-history) are linked to the cached training
image screen. That lineage check is not physical site certification,
external policy freeze or evaluation. No external outcomes, Stage5C,
SMC or deployment change.

[Summary](summary.json) | [Frozen action agreement](frozen_agreement.json) |
[Model-bank lineage](bank_lineage.json) | [Related work](related_work.md)
