# Single-Agent ADE Versus Explicit Relative-Path Supervision

Use the same nine nonempty training sources and 69,746 rows.
No exposed development, DroneCrowd calibration or confirmation
source enters fitting or preprocessing. Both arms share the same
frozen OLS4-relative backbone and local-neighbor adapter architecture.
Within seed, match initial weights, sampled rows, training steps
and all optimizer settings. Do not choose an epoch, weight or
threshold using development labels.

Reconstruct the training graph from current positions in each
training query, never from future coordinates. Use the same
four-nearest-neighbor undirected graph as the evaluation. For
each of 900 updates, draw 64 single rows and 32 edges
source-uniformly, then uniformly within source. Keep both
endpoints and duplicate draws in the 128-row individual ADE
term for both arms. Edges remain eligible regardless of future
label support. For the treatment add 0.25 times the error on
relative paths, averaged over co-observed steps for each edge
and divided by the geometric mean of the endpoint past RMS
speed scales. Unsupported edges have zero pair loss; their
missing points are not filled. Both arms compute the pair
quantity, but its weight is zero in the control.

Freeze all six 12-step forecasts before development scoring.
Hold fixed the original gain-policy recipients, exact per-query
switch counts and common substitution mask. Primary paired
contrasts: relative-path full-grid lower gain and conservative
single-agent utility for treatment vs matched control. Also
report available pair error and original recording gate. A
benefit requires positive conservative relative-path gain with
no evidence of aggregate utility or easy-risk deterioration.
Report the paired interval from 3,000 source resamples after
averaging the three trained seeds. No evaluation-rule changes.

Use native arm64 CPU with two Torch threads, zero loader workers,
atomic per-fit checkpoints and heartbeat. All coordinates are
image pixels; the horizon is annotation steps, not seconds.
Exposed development only. No deployment, Stage5C or SMC.
