# Route Repairable Recordings Before Deleting Actions

On the exposed Moving A 8/12 annotation-step task, preserve the
exact direct stream, original 2% risk/support/NA rules, and one
deletion per query. Use the previously fitted imitation action
score without changing the head. Fit a separate prefix score to
estimate whether direct fails yet has a non-loss deletion solution.

Use only the original seven fitting sources in each source/seed
cell. Generate each training source's direct stream from the
other six fitting sources. Label as positive only a failed record
with a checked non-loss optimum; keep existing passes and proved
infeasible or unsupported active recordings as negatives.
Do not turn unresolved solves into negative labels. Zero-action
recordings have no routing query. The score sees only current
and preceding proposed direct actions: counts and current,
running mean and running maximum summaries of predicted gain,
all/easy excess and predicted easy reference. Weight recordings
within source and sources equally. No final totals or labels enter
inference.

Compare routing alone to routing with a reserve of at least half
the original *predicted* easy reference mass seen so far. The
reserve can choose the next eligible action if the first violates
it; this is not a guarantee of observed easy support. Tune each
policy on six disjoint donors using routing cuts 0/.2/.4/.6 and
the previous imitation cutoff grid. Require additional complete
passes, no lost passes, no added signed easy violations, and
nonnegative paired lower utility. If no threshold qualifies,
retain direct. Report coverage differences; this is not a
matched-count comparison of action ranking. Freeze all choices
before held scoring.

Report AUROC/AP at the first active query for 84 active views,
with 28 failed views repairable under the fixed-action non-loss
hindsight test. Use 3,000 source-cluster resamples for the detection
intervals. Policy effects use paired source resampling after
averaging the three forecast seeds, full passes, signed easy and
paired transitions. A policy advance requires strictly positive
paired lower interval, more passes and no added signed easy
violations versus direct. No reserved outcomes, deployment,
identity certification, metric or seconds claim, Stage5C, or SMC.
