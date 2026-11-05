# Candidate-Neighbor Mechanism Check

## Scope

Keep the original 2% risk, signed easy-net, reference and support requirements.
No identity-fallback certification, threshold search, new H or reserved scenes.
This is a model-side diagnostic, not a new evaluation contract. Old failures stand.
Native obs8/pred12 image coordinates; no metric, seconds or physical safety claim.
Stage5C and SMC remain off. Independent confirmation is not_run.

## Hypothesis

The current neural risk critics see observed neighbor distances and candidate
motion, but do not explicitly encode the candidate's proximity to each
neighbor's causal constant-velocity continuation. This missing relationship
may help distinguish useful interventions from harmful ones across sources.
This is a testable feature hypothesis, not a confirmed failure cause.

## Fixed Experiment

Use the three existing own-B fitting partitions only. The parent producer,
proposal and current risk models are hash-verified and remain frozen. Construct
24 features from eight observed states, masks and two causal forecast paths.
Use neighbors with the last two observed timestamps aligned to the ego packet;
never infer unavailable neighbors from future masks. Coordinates share the
existing causal scale and observed ego origin. Neighbor CV is a proxy, not a
measured future path or collision ground truth.

Compare a fixed ridge probe on kinematic64 + rollout/history32 against that
same probe plus neighbor24. Leave one own-B source group out at a time; all
preprocessing and target RMS are fitted on the remaining source groups. Use
only known proposed outcomes for this diagnostic; record unsupported counts.
No row-random split, hyperparameter search or repeated validation selection.
Penalty is 0.01 in the source/query-balanced mean squared objective, intercept
unpenalized. Decision accepts only when both predicted signed costs are <=0.
The original sum-over-channels decision-cost helper is reused unchanged.

Primary diagnostic summaries are source-macro normalized MSE and original
joint sign-cost, separately for each critic context. A follow-up neural
candidate-neighbor experiment is justified only if at least two of three
contexts reduce MSE by >=1% without increasing joint sign-cost. This is an
internal experimental screen, NOT a deployment gate or safety certificate.
All folds and negative results are retained. Source groups have been used in
earlier development and their frozen producer lineage is not newly held out:
these probes cannot establish independent generalization or manuscript success.

Also decompose the existing relational critic's own-B errors by train-only
candidate-minus-floor nearest-distance quartiles, without refitting thresholds.
This is descriptive association, not a causal attribution of failures.

## Operational Boundaries

Commit this protocol and code before the real probe. Run native arm64 PyTorch
CPU4, interop1, workers0. Save fitted probe coefficients locally, hashes and
aggregate metrics publicly; keep row arrays/checkpoints out of Git. Re-run
deterministically to verify outputs. No new deployment, paper-ready claim or
independent scene opening follows merely from completing this experiment.
