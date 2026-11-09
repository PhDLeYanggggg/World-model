# What Happens When the Model Can Decline an Edit?

The [fixed-budget harm study](../opendd_risk_moments_v1/README.md) had
to fill every slot even when none of the alternatives looked useful.
Here each query may retain neither, either or both of its original
slots. Dropped trajectories revert to CV. The model can change the
recipient and shape of a retained edit, but cannot invent or amplify
its radius. I froze all forecasters and heads from the prior studies:
this is a new decision comparison, not another set of training fits.

Release requires positive predicted increments for the two individual
gain targets and nonpositive predicted excess for all three original
2% risk checks. The local gate sees just the current query. The prefix
gate also sees its previous estimated commitments in the recording,
never later observations or realized forecast errors. The additive
and joint controls share the prefix release and choose from the same
risk-eligible candidates at the same retained radii. The relative-motion
head is used only when two slots survive; the controls are identical
on singletons. No risk threshold was fitted to the readout.

## Lower Coverage, No Established Gain

There are 606 recording-seed views (202 distinct recordings, seven
locations and three existing seeds). All use eight observed positions
and 12 forecast steps in native coordinates. CV gain is conservative
individual gain, with 3,000 paired location draws after averaging
seeds within recording. Risk passes and dual gains are counts, not
an extrapolation to new sites.

| Rule | Edits | Edited views | Risk passes | Risk pass and positive vs CV/OLS4 | Signed easy >2% | CV gain pp [95% CI] |
|---|---:|---:|---:|---:|---:|---:|
| CV only | 0 | 0 | 0 | 0 | 0 | reference |
| Frozen pair policy | 3,690 | 439 | 47 | 41 | 164 | +0.0218 [-0.5361, +0.6887] |
| Query-local release | 212 | 128 | 39 | 15 | 3 | +0.0040 [-0.0007, +0.0128] |
| Prefix release, additive | 282 | 128 | 35 | 15 | 3 | +0.0041 [-0.0004, +0.0128] |
| Prefix release, joint | 282 | 128 | 35 | 15 | 3 | +0.0041 [-0.0003, +0.0128] |

The joint rule keeps 7.6% of the previous edit count. It releases
260 singletons and only 11 pairs, and changes six choices relative
to matched additive. There are **no strict positive full-grid pair
wins** over matched additive among the 128 edited views. The
full-grid lower gain is +0.0018 [0.0000, +0.0046] in 10^-3 native
units (four estimable locations). This is neither a clear utility
gain nor a risk pass. Signed easy harm still reaches 5.45% in the
worst view. Of the 128 edited views, 45 have no selected easy
reference; another 478 have no edits. Neither group passes the
unchanged rule. Observed ADE/FDE are 0.54724/1.25322 versus
0.54726/1.25329 for CV; the endpoint score uses available labels.

## Can the Frozen Menu Do Better?

I used future labels **only in a hindsight optimization**. It can
drop any of the frozen parent policy's edits, but cannot substitute a
new recipient, radius or path. It maximizes the lesser conservative
gain against CV and OLS4, requires an edited easy reference and
enforces the original three risk limits. All feasible witnesses were
rescored with the original risk evaluator.

Of the parent's 439 edited views, 117 lack a selected easy reference
even before dropping edits; 128 are infeasible; 194 solve to verified
optima. **179 views at 87 distinct recordings** admit a strictly
positive solution. Those solutions use 1,303 of the original slots.
The causal prefix policy edited in 81 of those 179 views and
achieved a risk pass and dual positive gain in 12. These figures are
an overlap diagnostic, not a recall score: the causal policy can
choose alternatives outside the restricted oracle's menu. There were
no unresolved solver views.

**Decision:** lower coverage reduced measured harm but did not produce
a candidate for independent confirmation. The attainable subset result
points to the admission score as the next target, especially whether
an edit has positive paired utility while contributing usable easy
reference. A reject option alone is not a new contribution; see
[SelectiveNet](https://proceedings.mlr.press/v97/geifman19a.html).
The specific test here is rejection and joint utility at the same
released budget, with unchanged recording risks and incomplete labels.

## Replay

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_opendd_optional_edits.py freeze
.venv-pytorch/bin/python scripts/run_m3w_opendd_optional_edits.py score
```

The arm64 CPU run reuses 21 hash-bound fold checkpoints, with four
compute threads and zero data workers. Inference saves predictions
before scoring; each fold writes a PID heartbeat and an atomic output.
Rerun the interrupted command to resume. No new fit, CREATE submission,
reserved outcome access, threshold change, Stage5C, SMC or deployment.

[Registration](registration.json) | [Freeze](freeze.json) |
[Results](readout.json)
