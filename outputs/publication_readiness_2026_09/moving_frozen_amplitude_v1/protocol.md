# Count-Preserving Forecast Amplitude Control

Use the original 34 exposed native development recordings,
14 source groups and three forecast seeds. Preserve all original
choices, CV risk reference, source-specific fitting cuts, unknown
support, 2% checks and annotation windows. Do not retrain or tune
thresholds, and do not open external outcomes.

For each frozen substituted forecast A with original floor C,
evaluate C + alpha (A - C). Use three fixed coefficients 1/8, 1/4,
1/2 and a past-only variant limiting maximum displacement from C
to the original causal scale (one plus 6.5 times median observed
step speed). Verify every nonzero switch stays nonzero, with no
changes in which agents switch. These are count-matched amplitude
controls, not candidate reranking or correction training.

The primary comparison is quarter versus the original full
amplitude policy. Report paired completion-lower utility both
against CV and against the original policy, source-group bootstrap
after averaging seeds, passes, unsupported easy references and
signed easy risk. Other amplitudes are diagnostic, not a search
for a held winner. The all-risk taxonomy separates already
observed violations from cases in which the unknown trajectory
disagreement alone moves the original bound over 2%.

For fixed alpha, convexity of Euclidean ADE on observed points
bounds positive harm by alpha times the original harm. The
unknown disagreement scales exactly by alpha, the denominator is
unchanged, and the signed easy net harm also has an alpha-scaled
upper bound. Verify the corresponding row-weighted inequalities
for the adaptive cap as well. This is a finite-roster arithmetic
bound, not a population guarantee or a way to pass empty support.

No independent confirmation, deployment, metric/seconds,
Stage5C or SMC claim follows.
