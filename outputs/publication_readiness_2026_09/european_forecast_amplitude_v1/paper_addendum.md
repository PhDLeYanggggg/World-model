# Output Amplitude Does Not Remove Selection-Conditioned Risk

Working evidence addendum, 29 October 2026. Complete fitting-only control;
independent inference verification is recorded separately. No independent
confirmation, new model training or deployment result is claimed.

After a frozen-expert control recovered some utility but worsened risk, we
test an actual output intervention rather than another cost-loss adjustment.
For every physical and neural candidate P and causal floor B, construct
P_alpha = B + alpha(P-B) at five predeclared amplitudes. We compare keeping
each original action selection set with reselecting under transported cost
surrogates. All six contexts, three head seeds and two estimator families
remain: 144 fitting groups and 720 amplitude views.

For fixed-mask Euclidean ADE, convexity ensures that fractional realized
benefit is no less than alpha times endpoint benefit, and fractional positive
harm no greater than alpha times endpoint harm. This elementary property
does not calibrate the endpoint cost estimator. We recompute actual new-path
errors and canonical completion envelopes; scaling observed endpoint errors
would be incorrect. Future labels and masks are used only for scoring.

| Neural cost-head mixed policy | Known fitting ADE gain over floor (%) | Required recording risk support | Known easy selected-risk violations | Unknown selected views |
|---|---:|---:|---:|---:|
| Original full output | 0.678064 | 77/444 | 230 | 853 |
| Fixed original set, alpha=1/16 | 0.047258 | 291/444 | 0 | 853 |
| Reselect, alpha=1/16 | 0.818671 | 40/444 | 36 | 4,968 |

Thus output shrinkage reduces observed harm on a fixed set but forfeits
utility and cannot restore undefined support. Reselection recovers utility
while entering a larger, insufficiently observed region. No fixed setting
passes complete recording risk. The full tables retain all banks and
coefficients, not a result-selected winner. Count-matched utility is separate
from the unchanged all/easy 2% safety requirement.

Unknown outcomes are not observed harm: unsupported completion bounds do not
reveal their true errors. Conversely, zero known violations do not establish
safety under missing outcomes. This suggests a concrete supervision question:
the existing cost heads sample no wholly unknown rows, but their decisions
are evaluated with missing-outcome completion mass. Completion-aware interval
supervision is a proposed next comparison, not a completed method.

The locality intervals are 3,000 fitting-resubstitution resamples after
averaging dependent context/head-seed views, not independent generalization
or three forecasting pipelines. Seventy-four unique fitting recordings are
repeated in 444 recording views per family. SOURCE coordinates are oblique
detector-track midpoint pixels; obs8/pred12 uses twelve-raw-frame steps.
These are not SDD gold, metric/seconds, true 3D or foundation results. Stage5C
and SMC remain disabled; protected roles and deployment are unchanged.

[Protocol](protocol.md), [convexity and scope](method_note.md),
[complete results](readout.md), [failure analysis](failure_analysis.md),
[Chinese reproduction guide](reproduction_zh.md).
