# OpenDD: label location at identical pair-step coverage

Keep the same forecast weights, histories, queries, 8/12 clock and
train-location exclusions as the archived backbone readout. Retain all
queries with at least two agents and an originally complete 12-step
future for every agent. This is an already exposed development subset
and may differ from naturally incomplete queries.

In each retained query, hide the first or last three prediction steps
for every agent, or hide three uniformly sampled steps. Repeat at six
steps as a secondary stress test. The same mask applies to all arms,
all pairs and all seeds, and depends only on the query order, location
and fixed random seed. All three patterns at a given severity retain
exactly the same number of labeled pair-steps. Use 32 uniform masks
at each severity, not as extra independent data.

Compare the same three contrasts as in the backbone report:
EqMotion neighbors versus ego, Social-STGCNN neighbors versus ego,
and the completion-trained forecaster versus CV. Report the known
full-grid gain, the observed-only gain and its bias from the known
gain, and the two full-grid envelope endpoints. The bound must cover
the known gain for every mask. Average seeds within location and
locations equally, and bootstrap seven locations 3,000 times. Also
report the variation over uniform masks separately. This tests
controlled temporal dropout, not the missingness process of any
real dataset, and changes no external rule, role, forecast or threshold.
