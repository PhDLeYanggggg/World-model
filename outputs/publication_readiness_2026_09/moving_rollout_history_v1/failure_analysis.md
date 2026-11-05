# Interpretation And Remaining Failures

## What The Feature Change Achieved

Six actual neural fits and one full treatment refit are complete. Adding32
causal rollout-history relations reduces full-fitting risk MSE relative to the
masked control by8.737%,10.484%,1.980% across the three contexts. Joint decision
cost improves in two contexts (8.833%,1.472%) but worsens by0.0137% in critic0.
There, false-safe count falls18to17 while false-safe cost rises27.664%: fewer
mistakes are not necessarily cheaper mistakes. The relation basis helps some
risk estimation, but does not consistently align decisions with harm severity.

On already exposed nested-C, known-label macro ADE gain changes0.894772%to
0.981299%, a+0.086527percentage-point contrast. Hard gain changes1.169812%to
1.303188%. The missing-label completion interval is[+0.045112,+0.127942]pp.
It is not a sampling confidence interval, and does not establish stable
generalization. This is modest development evidence, not independent success.

## Why It Is Not Deployable

| Failure | Evidence | Consequence |
|---|---|---|
| Original easy constraint violated | Worst defined easy-net8.660257%, versus8.633364% control and2% requirement | No deployment |
| Positive-harm risk remains high | Known-all12/17 and known-easy13/17 recording views violate2% | Net mean gain cannot substitute for harm control |
| Missing selected-easy support | Three views have undefined selected-easy completion risk | NA remains NA; no identity exception |
| Source instability | Two sources improve, four worsen, one unchanged | No stable cross-source claim |
| Concentrated gain | Source078 contributes+0.922930pp; the other six average-0.052874pp | Do not promote the macro alone |
| Removing the decision head is unsafe | Risk-only gain3.268218%, worst easy25.726249% | Higher intervention is not the repair |
| Independent evidence absent | One producer seed; reused exposed development; original24-group primary NA | Not submission-ready |

All seven sources remain in the main result. The six-source figure is a posthoc
influence diagnostic, not an exclusion rule or alternative primary. The largest
easy-net violation is source062, recording27:8.660257%. Its known selected-easy
positive-harm ratio is80.196125%, a different denominator/estimand from easy-net.
Recordings26and30 also show substantial easy-net harm. Fixing missing support
alone would therefore not fix the numerical risk violations.

Full original risk/support is0/17 for both the matched control and the new
complete policy. These are17 recording views, not17 independent sources. No
threshold, support definition, source roster or reference was changed.

## Causal And Scientific Boundaries

The added features use past motion and frozen causal model forecasts, never
future ground truth. Their normalization is own-B only. Matching preserves
initial functions, common weights, query draws and optimization budget. The
masked model has the same parameter count but lower effective capacity.
This experiment does not isolate information from capacity perfectly, prove
the basis novel, or test new trajectory dynamics/JEPA training.

Parent assets are cached_verified. Six fits, exact controls/refit, development
predictions and error attribution are fresh_run; replay verifies those same
results rather than adding a new independent test. Reserved Moving20, DUT,
DroneCrowd and H remain not_run and closed. Source-level influence is posthoc.
Native obs8/pred12 image coordinates do not imply metric scale or seconds.
Stage5C/SMC remain disabled; no true3D, foundation or human-gold claim.

## Next Model Decision

Keep the current deployment unchanged. Do not increase thresholds or remove
the decision head to capture the risk-only gain. The next targeted diagnostic
should use own-fitting data to separate harmful and beneficial candidates with
similar rollout-history relations, and quantify neighbor-relative proposal
conflicts versus residual source shift. That can justify either an explicit
candidate-neighbor interaction encoder or a revised fitting target; it does not
yet justify another large sweep. The present branch remains diagnostic until
the unchanged full risk/support requirements and independent evaluation pass.

The next research turn must also keep the support gap separate from model
error: unavailable reference events require valid additional evidence, not a
larger model or relabeling empty controls as safe. No new evaluation rule is
proposed or silently enacted here.
