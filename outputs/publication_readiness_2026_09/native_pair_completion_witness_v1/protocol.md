# Constructive Sign Check For The Frozen Joint Contrast

The overlapping-label envelopes cross zero. An envelope alone does not
prove both signs are attainable. Test whether explicit shared missing
positions achieve opposing signs for the registered joint vs additive
pair ADE contrast. Use the same 34 exposed recordings, 14 sources, three
frozen seed paths, query weights and released history and future labels.

Represent each recording, agent and absolute annotation step by one
coordinate. Check that every repeated released coordinate agrees and
that no future-missing occurrence is released in any eligible history.
The reference completion is piecewise linear between released positions,
with constant extension outside them. Missing points in each gap share
one bounded two-dimensional direction. Its per-coordinate displacement
is bounded by the agent's median released consecutive displacement
(floored at 0.25 image pixels) times distance to the nearest released
frame, capped at twelve annotation steps. Use this radius for the
primary witness check and double it for a second fixed sensitivity.
These image-coordinate perturbations are not a physical speed model.

For each radius and sign, use L-BFGS-B from zero and one seeded
perturbation, retaining only a checked feasible assignment. Optimize
the seed-averaged, query-weighted contrast. One coordinate assignment
must be shared across all queries, edges and seeds of a recording.
Recompute the witness contrast separately, verify the observed term
and interpolation reference against the original scorer, and compare
each witness with the earlier outer envelope. No global optimum is
claimed, and failure to find a sign would not exclude that sign.

This uses exposed labels for an evaluation sensitivity. It does not
recover actual labels, supply training input, change the original 2%
gate or official outcome, or read any external role. Bootstrap draws
describe the constructed scenarios after seed averaging, not inference
about real missing coordinates. No deployment, Stage5C or SMC.
