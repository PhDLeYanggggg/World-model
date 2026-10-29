# Why Smaller Outputs Still Do Not Establish A Safe Policy

## Scope

The complete first inference run finishes 144 fitting groups and 720 fixed
amplitude views. Exact replay is a separate required artifact, not assumed
from successful execution. The readout and verification files determine its
final status. No head is retrained, no held label selects an amplitude, and
no new dynamics contribution is claimed.
Separate process PID 21457 subsequently replays all groups and summary exactly
in 574.746712 seconds. The negative conclusion is unchanged.

## Output Size And Selection Are Different Mechanisms

Keeping the original neural-cost mixed-policy selection set and shrinking
its paths to 1/16 reduces known easy selected-risk violations from 230 to 0.
Recording risk support rises from 77 to 291/444. But known fitting ADE gain
falls from 0.678064% to 0.047258%; 49 easy-risk views remain undefined.
The intervention set is unchanged at 70,561 repeated query views, including
853 unknown outcomes. This is a tradeoff, not a successful repair.

When transported costs instead allow reselection at the same fixed 1/16
amplitude, known fitting gain is 0.818671%, but support is only 40/444.
Interventions rise to 269,038, including 4,968 unknown outcomes; 36 known
easy-risk violations remain. The smaller per-action harm does not control
the cost of a much broader selected region. Shrinking outputs and expanding
coverage cannot be treated as one isolated amplitude effect.

The forest gives a particularly clear distinction. At 1/16, its reselected
mixed policy has no known all/easy selected-risk violation, but supports
only 251/444 recording views. The unchanged-set version supports 336/444
while retaining 104 undefined easy-risk views. No known-label violation is
not the same as complete risk support. Unknown outcomes and unidentified
references cannot be silently treated as favorable labels.

## A Higher Fitting Gain Is Not A Winning Model

For illustration, the separate-neural-expert policy at fixed 1/4 amplitude
has known fitting gain 1.398761%, compared with 1.122806% at full amplitude.
Its recording support falls from 57 to 51/444; 116 known easy-risk violations
remain. This is not a selected deployment setting. Every fixed bank and
coefficient is retained in the complete table, including worse results.

The five coefficients do not train a neural dynamics model. Costs are
transported convex-loss surrogates from the original heads, not accurately
observed or newly calibrated expected errors. Convexity holds for realized
fixed-mask Euclidean losses, but cannot repair incorrect conditional cost
estimates. Two-expert composition still doubles estimator capacity.

## Data And Evaluation Limits

Unknown selected outcomes could be good or bad. A completion bound failing
does not reveal their actual errors. It means the declared safety conclusion
is unsupported by these data. The aggregate report separates known violations,
undefined risk and unsupported views with selected unknown outcomes; these
categories overlap and are not a unique causal attribution.

At 1/16 amplitude, the forest mixed policy has 188 unsupported recording views
with selected unknown outcomes and no known selected-risk violation. The
neural mixed policy has 356 such views, alongside 37 views with a known
selected-risk violation. These are overlapping descriptive categories, not
proof that unknown outcomes actually cause harm or the only model weakness.

All statistics are fitting resubstitution over 74 unique recordings with
repeated context/head-seed/query/amplitude views. The 3,000 locality resamples
are not independent confirmation or repeated-research-selection correction.
Future masks enter scoring only. Unestimable endpoint denominators remain
unestimable. Count matching concerns utility, not safety.

## Next Testable Hypothesis

The current cost learner explicitly samples zero unknown-outcome rows and
learns costs conditional on available labels, whereas the safety readout
charges conservative completion mass for selected unknown rows. That target/
evaluation mismatch now deserves a controlled test. First verify whether
prior completion-screen and quality-extension controls already test actual
completion-aware training; they must not be renamed as new experiments.

If not previously tested, compare matched known-only supervision with
interval-bound supervision on the same causal query draws. Derived lower/
upper endpoints must be named as interval surrogates, not imputed future
outcomes; preserve the original NaN labels for evaluation. Train-only future
availability may define supervision, never an inference field. Retain the
same complete 2% recording contract, physical-only control, fixed forecast
bank and all required recordings. This is a proposed next training experiment,
not work already run or a promise it will pass.

Matched current-source RGB inputs are also not established; see the separate
[availability audit](../source_rgb_availability_20261029.md). Lack of present
pixel assets is not proof that causal geometry is information-theoretically
insufficient. No deployment, independent-role opening, Stage5C or SMC.
The model remains a pixel/raw-frame research track, not true 3D or foundation.
