# Cost-Directed Partitions With Unchanged Risk Means

Registered after the negative directional-peer probe, before these fits.
The previous turn is progress:48 fits/refits reject expansion of that feature
branch. This study uses the unchanged summary120 features, not the unqualified
directional180 branch. It does not reopen any reserved data or change a gate.

## Hypothesis

The continuous-risk regressor can reduce MSE while retaining positive estimated
risk on every candidate. Changing a leaf mean into probability-times-severity
with the same fitting measure and partition is algebraically identical. Test
whether a different fitted partition improves decisions while preserving the
original risk targets and readout. This is an objective/partition control, not
a new neural dynamics model or novel tree algorithm.

The earlier signed-logistic and joint-event neural studies stay negative.
Those changed neural supervision/decoders. This study instead holds the
two-channel leaf-mean readout fixed and varies the split criterion.

## Fixed Comparison

- mse_partition: the prior frozen64-tree summary120 regressor, verified by hash,
  prediction and a fresh full refit; no new control result claimed.
- sign_partition:64 ExtraTrees classification partitions, supervised by the sign
  of the original joint diagnostic cost, with source/query/row-balanced weights.
- cost_partition: same classifier settings, but split weights multiply the
  original measure by absolute joint diagnostic cost.

Both new partition arms then recompute each leaf's two original signed-risk
means under the ORIGINAL source/query measure, not the classification weights.
Average these means across trees. Acceptance still requires both risk channels
nonpositive; classifier probabilities never determine acceptance. No threshold
search. Max depth8,min leaf32,max features1,seed17,64trees,n_jobs4. Capacity
limits match; learned structures and node counts need not match.

Zero-cost labels have zero split weight only in the cost arm but remain in the
original-measure leaf readout. Unknown outcomes are not imputed or labeled safe;
these are known-outcome fitting probes, not full-policy risk certification.
Preprocessing and target RMS use fitting sources only. Inputs are past-only;
future observations supply supervision/evaluation only. All sources retained.

## Scope And Execution

Eight own-B source-held folds in three previously exposed contexts, one
producer seed. Sixteen new fitted models and eight cached controls. Refit all
sixteen models and all eight controls for deterministic verification. One full
64-tree real-data pilot repeats before primary fitting. Models/cells are atomic
checkpoints; explicit resume only after actual interruption. Native arm64,
Torch CPU4/inter-op1/workers0; no resource probing. Twelve-hour runtime guard.
Use local execution unless the measured pilot requires CREATE. No duplicate jobs.

For every fitted source measure, reconstruct a frozen partition's prediction
as P(y>0)E[y|y>0]-P(y<0)E[-y|y<0], per channel. Report arithmetic differences,
sign changes, target sign/magnitude distribution and decision counts. This
posthoc decomposition is not a separately learned hurdle model or new result.

## Primary And Readout

Primary: source-macro joint diagnostic cost, cost_partition versus mse_partition.
Internal follow-up screen requires at least1% cost reduction, no increase in
false-safe cost, and total cost no worse than sign_partition, in at least2/3
contexts. MSE and all individual sources reported even if negative. Sign-only
cannot replace primary after inspection. The screen is not the original safety
gate and cannot promote a model. No new nested-C/H in this study even if it passes;
a positive result would justify a separately registered neural mechanism test.

Original2% all/easy positive harm, signed easy-net, support and reference rules
remain. No identity certification. Empty support remains NA. Old failures and
original24-group primary NA are retained. Moving20,DUT,DroneCrowd,H closed.
Native obs8/pred12 silver image coordinates, not metric, seconds, true3D,
foundation or human gold. Stage5C/SMC off. Not submission-ready.

## Implementation Source

Weighted classification partitions and squared-error regression are standard
controls, not a novelty claim. See the official
[ExtraTrees documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html)
and [tree objectives](https://scikit-learn.org/stable/modules/tree.html).
The local installed scikit-learn1.8.0 is recorded in registration; no library
upgrade or new theoretical safety guarantee is implied by current web docs.
