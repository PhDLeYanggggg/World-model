# Neighbor Association Limitation

This is an observation-only structural diagnostic added after registration
but before comparative outcome readout. It changes neither training nor the
primary forecast contrast. It does not measure accuracy or prove the cause
of a forecast failure.

The model flattens all neighbor position/time tokens and gives them the same
modality embedding. The supplied agent-track associations are not embedded
or used in a per-agent temporal encoder. Consequently, independent neighbor
reassignments at individual past times preserve the entire token multiset.
This is stronger than the desirable invariance to consistently renaming an
agent across its whole history.

On256 causally eligible source queries with at least two complete neighbor
histories, the probe exchanges two agents' positions at past slots1,3,5. It
preserves ego history, current neighbor positions, times, masks, baseline and
the observed spatial point set at every time. All256 constructed neighbor
paths change: mean summed neighbor path length146.52 becomes708.69 in source
coordinates. These deliberately reassigned tracks need not be physically
plausible; the probe tests representation identity, not a new data population.

The freshly trained first fixed producer, seed17, responds with a maximum
coordinate difference4.5776e-5 and mean5.9074e-7, consistent with floating-point
permutation effects. No future labels are read. A nonzero-output synthetic
regression test demonstrates the same structural invariance.

The model may still infer aggregate flow or plausible associations from
geometry. The result does not establish that tracking identity would improve
forecasting. It does show that this representation cannot directly distinguish
different supplied track associations with identical per-time observations.
An agent-wise temporal encoder followed by agent interaction is a specific
next hypothesis. A separate candidate now implements this two-layer topology
with identical parameter shapes and initial baseline output. Three structural
tests check association sensitivity, consistent-agent permutation invariance,
finite gradients and missing-agent robustness. This is not a trained repair
or predictive gain, and it is excluded from the registered nine-model comparison.
No deployment changes follow from this probe.
