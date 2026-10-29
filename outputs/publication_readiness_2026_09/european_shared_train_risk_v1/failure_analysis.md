# Sharing Cost Heads Does Not Resolve Recording Transfer

## Decision and Scope

Both registered repairs fail. All108 fits and all72 inner-held views completed;
the result is not a partial run or a downgrade. No deployment, new outer
development evaluation or independent-role access follows. These are small
neural cost heads on frozen forecasting outputs, not new world-dynamics models.

The experiment tests shared learning because16/24 source/context groups have
only two TRAIN recordings. Holding out a whole recording leaves one fit recording
in those groups. Pooling four sources within a fixed forecasting/controller
context adds recording diversity but also couples source-specific cost maps.
All per-source query draws match. Shared-global/source use identical initial
weights; only their grouping of risk duals differs. Relative to local heads,
pooling also changes priors, standardization and the number of parameter sets.

## Observed Failures

| Finding | Evidence | Interpretation boundary |
|---|---|---|
| Shared cost prediction worsens | Global-minus-local signed-score MSE+7.845471, nominal95%CI[2.612407,15.439524]; source-specific+3.438174[0.831990,6.243236] | Does not isolate whether capacity, normalization, objective or source interference caused it |
| Utility falls | Global paired lower utility-2.094903pp[-3.479909,-0.824900]; source-specific-2.604299[-4.255594,-1.128204] | These are conservative pairwise utility bounds, not raw ADE/FDE differences |
| Fewer switches do not explain the entire loss | Both same-count contrasts are negative: global-0.519429pp; source-specific-0.381723pp, with negative nominal intervals | Equal-count comparison controls intervention count, not every selection mechanism |
| Separate source constraints do not rescue sharing | Source-minus-global full-1.145644pp[-1.672529,-0.637533], matched-0.158733[-0.340291,-0.006414] | Pair-specific lower bounds are not transitive point utilities |
| Safety remains unsupported | Known easy violations46/46/40; finite completion support0/0/1, for local/global/source | Fewer violations alongside more undefined cohorts is not a safety certificate |
| Selected easy harm is underestimated | Median predicted/observed selected easy harm0.022185/0.105573/0.046836 on61/62/56 defined views | Different selected cohorts prevent a controlled calibration-improvement claim |

All numbers retain the full72-view roster. The source-constrained model selects
27,826 row occurrences versus57,903 for local and34,810 for shared-global. Its
undefined easy-risk count increases from the local7 to12. Unknown labels remain
unknown: each arm evaluates419,766 repeated row occurrences, including9,228
wholly unknown outcomes. No unknown label is filtered from action selection.

## Heterogeneous Source Effects

Source067 improves its known easy violation count from6/6 local views to1/6
source-constrained shared views; source112 improves from5/6 to0/6. Conversely,
source074 changes from0/6 to5/6, and source008 from0/6 to1/6. Sources007,110,119
and124 remain6/6 violations. These descriptive changes are consistent with
cross-source interference or source-dependent underestimation, but do not prove
which mechanism caused them. Removing the unfavorable sources would be invalid.

## Follow-Up That Is Admitted

The registered frozen inner-fit census reads every inner-fit recording using
the same108 final weights, with zero new optimizer steps. It compares fit risk
and signed-score MSE against already saved held aggregates. This determines
whether the shared model already fails to fit its training costs or mainly
fails when recordings change. It cannot select a better checkpoint or threshold.
The first attempt stopped before any group inference because a TRAIN stream
returned SSH255. A subsequent read-only connection returned0; same-code resume
completed all72 views in77.52s. Scientific readout completion is independent of
that transport interruption. All census ratios, recordings and summary hashes
verify; no new optimizer or held inference ran.

The census finds local fit/held known easy violations1/46, shared-global10/46
and shared-source9/40. Local mean signed MSE increases from0.559739 on fit to
4.013844 held; shared-global is1.968616/11.859315 and shared-source1.965634/
7.452018. Thus recording shift remains, but pooling already worsens the fitting
problem. A smaller shared held/fit ratio is not a gain when both absolute errors
are worse. [Full fit/held diagnosis](fit_census/conclusions.md).

Further model repair must follow that distinction. A poor TRAIN fit motivates a
matched source-conditional decoder or objective-isolation test; a good fit with
held failure motivates observable support and recording-shift analysis. These
are hypotheses, not completed fixes or proof that either will improve results.
No outer/test threshold sweep is justified by this negative result.

## Limits

The3,000-draw paired-locality bootstrap is nominal TRAIN-internal evidence on12
localities. Repeated windows, three cost-head seeds and shared training do not
create independent datasets or three independent forecasting models. The source
corpus has prior exposure; the new inner partition cannot restore independence.
Historical Stage26/37 scores remain exploratory. Observation8/prediction12 at
raw-frame stride12 is distinct from historical t+50. Coordinates remain local,
with no metric/seconds claim; not true3D, not foundation. Stage5C and SMC remain
disabled, and the project is not yet submission-ready.

[Complete evidence](readout/conclusions.md), [frozen design](protocol.md),
[inner-fit diagnostic](fit_census/protocol.md).
