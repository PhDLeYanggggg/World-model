# Input-Support Diagnostic

Descriptive source slices, not a new selection criterion or a causal mediation analysis.
Groups depend only on whether the observed neighbor geometry changed. No outcome defines a group.
All models and primary endpoints remain fixed. Inference for both groups uses the trained full model.

| Input group/subset | Equal-locality ADE gain vs legacy (%) | Locality interval |
|---|---:|---|
| partial_context_changed_all | -0.04976494145651622 | [-0.4662105356400974, 0.3774388039103918] |
| partial_context_changed_positive_easy | 0.2976266468305741 | [-0.07123489148960709, 0.6369929196707134] |
| partial_context_changed_hard | -0.0376093770434994 | [-0.5216600290031633, 0.42027794187241435] |
| no_observed_input_change_all | -0.38172466457106435 | [-1.0301790462694107, 0.1876035441114261] |
| no_observed_input_change_positive_easy | -0.06562926629937509 | [-0.39709553627341615, 0.2664596391488269] |
| no_observed_input_change_hard | -0.7033830402215827 | [-1.4429196473710422, -0.07152597299477971] |

An unchanged input row can still change prediction because training changed shared model weights.
A contrast between these groups cannot separate normalization, nearest-agent membership, mask support,
detector noise or the true usefulness of interactions. Undefined fixed-roster percentages stay undefined.
Slice percentage gains have different denominators and locality weights; they are not additive components of the primary.
There is no seed selection, threshold search, risk-policy refitting or independent-role access.

## Observed Support

| Quantity | Value |
|---|---:|
| old_mean_current_neighbors | 6.762393837645664 |
| new_mean_current_neighbors | 7.304480999720976 |
| old_mean_valid_neighbor_slots | 54.099150701165314 |
| new_mean_valid_neighbor_slots | 50.51280218453831 |
| changed_queries | 282529 |
| changed_queries_fewer_valid_slots | 215971 |
| changed_queries_more_valid_slots | 65280 |
| changed_queries_equal_valid_slots | 1278 |

These counts describe observed inputs, not interaction relevance or annotation accuracy.
