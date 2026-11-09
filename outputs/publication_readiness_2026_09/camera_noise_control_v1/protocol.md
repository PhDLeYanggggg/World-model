# Donor observation-noise control

Compare the same OLS4-anchored spline and loss with clean or perturbed
input histories. Seven OpenDD development locations are donors only.
Use seeds 17, 29 and 43 with paired initial weights, minibatches,
optimizer and 800 updates. The perturbation is a synthetic stressor,
not a fitted camera-error model: Gaussian AR(1) with coefficient
0.5, and per-row RMS amplitudes 0, 0.25 or 0.75 times its clean
past step RMS. Absolute targets stay unchanged; prediction origins
follow the perturbed final observation. Normalize both arms by
their observed prefix only.

Freeze all predictions on the previously exposed Convenience Store
WEPDTOF, GMU and CMU camera prefixes before scoring. Keep the same
8 observations, 12 prediction steps, online track roster and stable
past matching. Do not score Miraikan (prior history-support failure),
the 13 scored WEPDTOF external groups, or the sealed DroneCrowd groups.
Compare with CV, anchored OLS4, and anchored median pair slope.
This is a direct forecaster screen, not a selector or risk-threshold
search. Show observed and conservative paired gains using common
CV error mass, unavailable rows, and raw pixel errors by camera.
Do not pool pixel scales or treat seeds as independent cameras.
Use the existing risk cutoffs only on the WEPDTOF development pilot;
there is no fitted ACME easy threshold or ACME risk pass.
