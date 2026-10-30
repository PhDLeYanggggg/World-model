# What goes wrong in the selected actions?

Result source: fresh frozen Torch inference, followed by a separate complete
exact replay. No new training, policy or threshold. PID87277 completed all216
groups in93.049436s, peak RSS10960896000B; PID88396 replayed every group exactly
in93.284951s, peak RSS10910662656B. All prediction, action, original full-policy
metrics and row hashes match the frozen budget-score experiment.

The support remains149510 unique queries,74 recordings and12 localities.
Each objective contains897060 overlapping context/seed row views,878082
known and18978 wholly unknown. These are repeated views, not independent
samples. All findings below concern this exposed fitting cohort only.

## The easy problem is predominantly underestimated selected harm

|Objective|Selected known views|Easy false admissions|Numerator-only witnesses|Selected easy observed harm|Selected easy predicted harm|
|---|---:|---:|---:|---:|---:|
|Original|263581|39562|38666|24672.550435|281.923647|
|Symmetric score|344902|47529|46840|38020.153870|321.658700|
|Upper score|262370|33241|32532|19341.876351|255.500950|

For upper score,97.867% of easy false admissions are numerator-only: replacing
predicted harm with its true value would reject the fixed selected action,
whereas replacing just the reference would not. The other disjoint witnesses
are denominator-only22, either single replacement failing334, and only both
replacements failing353. These substitutions are future-informed arithmetic
diagnoses, not available policies and not proof of causality.

Easy reference is underpredicted in aggregate:154143.364112 predicted versus
249543.148576 observed. Consequently its signed contribution is conservative
(-1907.995689), not the main aggregate optimism. Easy harm underprediction
contributes+19086.375401. Merely fixing easy reference cannot cure the numerator
failure. Known noneasy selected rows receive60360.012604 of predicted easy
reference mass, so conditional support is also misallocated.

## The all-channel problem is different

Upper-score selected all harm is48319.825508 predicted versus36162.433431
observed, conservative in aggregate. Reference is4148774.651601 predicted
versus2295100.360660 observed, yielding+37073.485819 signed optimism that
outweighs the-12157.392077 aggregate numerator contribution. There are45272
individual all-channel false admissions:5307 numerator-only,12108
denominator-only,26426 either single replacement failing,1431 joint-only.
Global mean bias can therefore mask the wrong selected decisions.

Actions4/5 account for18916 of33241 easy false admissions, and neural action6
accounts for9337. This is not a failure confined to the neural proposal.
The decomposition residual is below1e-10 in magnitude in pooled channels.

## Support and unknown outcomes remain part of the failure

Upper score selects135506 known easy views and126864 known noneasy views;
all214257 full known easy views have426216.404625 reference mass. No selected
known easy view has zero reference in this real run, but the implementation
and tests retain that possibility. Full easy support does not establish
selected support for every recording. The parent's42undefined selected-easy
recording views and312known easy violations are unchanged, not repaired.

Upper score also selects3658 wholly unknown views with62829.450789 causal
envelope mass. They receive no imputed outcome or arithmetic attribution.
The evaluation preserves unknown-policy sensitivity and recording-level
requirements. Pooled easy mean net gain cannot cancel positive harm.

## Decision

Run one separately registered online false-admission loss. It recomputes the
current winning action during training and penalizes its optimistic signed
score on supervised false admissions, while retaining the full original
objective and unchanged causal inference. This differs from previous fixed
teacher regions and all-action score weighting. A positive result still
requires complete-policy recording risk and independent evidence; neither
is supplied by this diagnosis. No deployment or independent role opens.

The216aggregate groups are packed in a1634665B verified archive; no raw data,
weights or per-query features are included. Unit, unknown-label, nonmutation
and exact-control tests pass; the report is not an A-conference result.
