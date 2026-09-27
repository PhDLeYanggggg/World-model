# Temporal Context: No Advancement

## Material Passport
Fresh_run: 1,728 ridge probes, 432 inner fitting-locality screens, support and
cost diagnostics. Cached_verified: frozen forecasts, single-locality neural
references and two-locality risk heads. No new neural training. Registration
7481acc3 and prediction freeze cd61e6dd preceded scoring. Outer conditional-head
evaluation and independent evaluation are not_run by design, not completed.

## Main Result
The fixed ordered-history hypothesis does not pass its information screen.
For the full-input forecaster family, history plus neighbors versus score-only
has one positive, one negative and four overlapping primary intervals. Its
point range is -2.25% to +4.53%. Against the old seven summaries it has zero
positive, two negative and four overlapping intervals, ranging -2.04% to +0.17%.

Adding neighbors to ordered history gives one positive, three negative and
two overlapping primary intervals. Top10 harm-mass capture also has three
negative intervals for that comparison. Improved coverage in some contrasts
does not replace the primary and protection screens. All-harm MSE is unchanged
by construction, not evidence of a learned all-harm improvement.

Motion-only retains partial positive signals against score-only, but against
the old summaries it has one positive and five negative primary intervals.
The -113.24% worst point against score-only and -108.25% against old summaries
are retained with absolute-cost context. Two motion-only assignments lack
complete tail/coverage support and remain not_estimable. No subgroup is
selected as the new winner.

These are expected easy-harm cost MSE gains, not trajectory ADE/FDE, raw-frame
t50 improvement or easy-case trajectory degradation. The three seeds and
three outer contexts are averaged inside scoring locality before 3,000 paired
four-locality resamples. Six assignments overlap; intervals are exploratory.

## Why This Matters
There are 240,809 distinct supported positive-envelope agent-query keys across
the union of source views, 52,056 recording-scoped tracks, 35,389 scene queries,
162 recordings and 12 localities. These are deduplicated physical-key counts,
not independent observations. The two largest localities contribute 202,390
of those agent-query keys. Balanced fitting weights do not create new sites.

Event support is much smaller than the row count suggests. Among 216 full-input
scoring views, 35 have fewer than ten harm-mass effective event tracks. The
motion-only counts are 158/216, and 66/216 have fewer than ten distinct event
tracks. The effective count is a concentration diagnostic, not a power estimate.

The fixed all-harm cap also creates a substantial error floor. Its unavoidable
easy-harm MSE is a median 65.18% of raw MSE in full-input views and 70.75% in
motion-only views. It exceeds half of raw MSE in 154/216 and 141/216 views,
respectively. This establishes a limitation of this easy-only clipped probe,
not that removing a safety constraint would improve generalization.

Primary and protection screens fail. No outer conditional-head study is
promoted, no deployment changes and no independent confirmation is claimed.
The wider research goal remains active. Stage5C and SMC remain disabled.

See [all contrasts](results.md), [intervals](context_intervals.svg),
[support](event_support.svg), [absolute costs and ceiling](support_diagnostics.md),
[failure taxonomy](failure_analysis.md), and [next boundary](project_gap.md).
