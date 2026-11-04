# Method Position And Limits

The diagnosis belongs to predict-then-optimize, not to a new JEPA or Transformer
architecture claim. Prediction error and decision error are different targets.
[Elmachtoub and Grigas, Smart Predict then Optimize, Management Science68(1),9-26](https://pubsonline.informs.org/doi/10.1287/mnsc.2020.3922)
build a loss around the downstream optimization problem. Our fixed aggregate
excess penalty is not their SPO+surrogate and inherits none of its consistency
or optimization guarantees.

[Smith and Winkler, The Optimizer's Curse, Management Science52(3),311-322](https://pubsonline.informs.org/doi/10.1287/mnsc.1050.0451)
explain how selection on noisy value estimates can make chosen alternatives
look too optimistic even without globally biased estimates. That is a relevant
alternative explanation, not a verified causal diagnosis of this dataset.

The new test changes only a training objective for an existing causal cost
network. It asks whether sampled selected-record excess supervision improves
the complete joint controller versus its actual matched neural control and
the stronger existing conservative tree policy. The original2%contract,
unsupported views, independent data requirements and unknown-label bounds
are unchanged. A loss-only improvement is not paper-level novelty or safety.

## Earlier Negative Aggregate-Loss Experiment

This is not the project's first aggregation test. The earlier
[European query-excess study](../european_query_excess_refit_v1/conclusions.md)
trained216heads/432000updates and found pure query-aggregate supervision worse
than its pointwise joint control by0.15468%in its paired ADE contrast, with a negative
95%development interval. Its same-count ADE contrast crossed zero, and its
selected-risk primary remained unavailable. That failure is retained.

The present within-study contrast adds an aggregate term while preserving the
original point cost objective, focuses it on frozen fitting teacher selections,
uses coherent five-moment bound-aware cost outputs and samples recording
aggregates on the Moving SOURCE population. These differences motivate a new
controlled test; they are not a matched attribution against the European study
and do not establish that aggregation generally works. Only the registered new
matched control and complete risk readout can decide the current hypothesis.
