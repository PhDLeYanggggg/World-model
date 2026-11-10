# OpenDD backbone controls: vehicle-pair relative trajectories

Keep the original 500 recordings, seven held training locations,
three fixed seeds, 8/12 grid, target validity, frozen model weights
and causal CV and OLS4 paths. Reproduce the archived individual
CV-relative results before scoring relative paths. No refit, new
recipient selection, risk threshold or external label access.

Within each query, form all unordered pairs before checking future
support. Compare the vector between the two agents' forecasts with
the corresponding label vector, where both positions are present.
Report available relative ADE, endpoint error and the 95th percentile
of query ADE. Give each supported query equal weight within location
and each location equal weight after averaging seeds.

For direct paired contrasts, on each missing pair-step charge the norm
of the difference between the two predicted relative vectors. This
gives lower and upper bounds by the reverse triangle inequality.
Average over all pair steps within each query, including queries
without simultaneous labels. Bounds need not be attainable jointly
across pairs that share a vehicle. Report the available contrast
separately; its denominator excludes unsupported pair steps. Use
3,000 paired location bootstrap draws after averaging seeds.

The two neighbor ablations isolate added observed neighbors within
each architecture at the matched training budget. This is a
retrospective OpenDD diagnostic of deterministic adapted backbones,
not a benchmark of their original multimodal tasks, a causal claim
about vehicle interactions, or a physical safety test.
