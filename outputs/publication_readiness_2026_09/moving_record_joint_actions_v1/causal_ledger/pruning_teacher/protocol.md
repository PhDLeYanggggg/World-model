# Training From Checked Whole-Recording Removal Sets

Keep the direct stream, original 2% and reference-support rules, and
at most one deletion per query. Restore the causal floor on deletion;
never add, replace or move an action. Native 8/12 annotation-step
Moving A development only. No new forecast training.

For each of 42 outer source/seed cells, build teaching examples only
on its seven fitting sources. Regenerate each fitting source's direct
choices using the other six fitting sources, excluding the entire
calibration and held cohort. Obtain a maximum-retention deletion
solution under the original recording risk/support rules and
nonnegative paired lower gain versus that regenerated direct policy.
Verify the solver-optimal witness with the original scorer.
Unsupported, infeasible and unresolved examples are excluded, not
assigned all-negative membership. Do not export teacher choices.

Fit two ridge outputs on the same observed features and source/
record/query weights as the prior pruner: deletion membership and
paired deletion lower gain divided by causal motion scale. For each
query remove at most the top-scoring member. Compare imitation alone
to a guard that rejects negative predicted paired deletion gain.
If no positive training deletion exists, preserve direct. Tune only
on the same six donors using the fixed score grid (0, .01, .025, .05,
.10, .20, .35). A threshold must recover a full pass, lose none, add
no signed easy violation and preserve aggregate paired lower gain.
Otherwise make no deletion. Freeze before scoring the held source.

Separately use a fixed .10 score cutoff as a nondeployment diagnostic
for both learned arms. Match each to the prior moment head deleting
the same number of actions at the same queries. An empty diagnostic
has no ranking contrast. Report the operating policies separately.
Report threshold availability and verified training witnesses as
repeated outer-fold occurrences, not independent samples.

Policy improvement requires more complete held passes, no increase
in signed easy violations, and a strictly positive paired 95% lower
utility interval versus direct. Resample 14 sources 3,000 times after
averaging the three seeds within source. No change to the rule,
independent outcome access, deployment, Stage5C, SMC, or physical
safety claim.
