# Query-Regret Training at Fixed Intervention Counts

Development diagnostic on the previously read OpenDD cohort. Same seven
leave-location-out folds, three seeds (17, 29, 43), 8/12 stride-12 queries,
CV fallback and five alternative trajectories as the nested abstention run.
No threshold search, official-risk revision, or DroneCrowd access.

Train one pointwise control and two structured variants. All share the
same past-only inputs, scalar-output MLP, initialization and draws (site
then query uniform), 400 AdamW steps at .001, batch 64 queries. Use the
same query scale for every recipient: median of the greater of CV and OLS4
speed for each eligible history, floored at .05 in publisher coordinates.
Train a Huber regression on 12-step conservative CV-relative ADE gain;
every unknown step pays CV-to-candidate disagreement. CV ranking adds
expected normalized opportunity regret under a softmax over legal edits
(temperature .10, regret coefficient 1). Dual ranking keeps the same
Huber control but deducts conservative positive OLS4-relative harm from the
regret utility. Label values are clipped to [-4, 4] after normalization;
negative lower bounds and missing future observations never affect which
candidate is legal. No future information is a predictor input.

Primary comparison: CV ranking vs pointwise regression with exactly one
edit in every eligible query. Compare dual ranking, minimum change, and
history hindcast at the same queries and counts. Report matched causal
20% coverage (floor, at least one per site with support), choosing queries
by best predicted gain times the common scale. No count is chosen on
outcomes. OLS4 and unchanged CV are no-edit references. Outcomes are used
only for a per-query oracle, supervision, and scoring.

Two diagnostic ablations separate choosing the recipient from choosing
its motion forecast. For recipient-only, every agent's candidate is fixed
to its best legal history-hindcast option. For candidate-only, the agent is
fixed to the one chosen by the full history-hindcast rule. All models and
controls edit exactly once on the same queries in each ablation. Also score
the offline oracle in each restricted menu for capacity only.

Use observed ADE/FDE and conservative full-grid paired gain as points
relative to observed CV mass. Debit missing-step disagreement directly
between each pair of actual predictions. Report worst signed and positive
easy ADE change on the existing descriptive train-only CV-easy slice,
site results, switches, and equal-site 3,000-resample intervals over seed
averages. Save checkpoints every 100 steps, native arm64 CPU, four threads,
zero workers. No external confirmation, deployment change, Stage5C or SMC.
