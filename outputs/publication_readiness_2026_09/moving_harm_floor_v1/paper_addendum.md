# Negative Control Study: Harm-Weight Allocation

## Research Question

Can emphasizing lower-magnitude harmful labels improve intervention decisions
without relaxing a fixed safety contract? This is a component-level study of
neural risk heads, not a new trajectory forecasting method or world-model claim.

## Method

We compare three loss arms in three fitting contexts: the preceding joint
decision loss, uniform amplification of positive harm coefficients, and a
positive coefficient floor at the fitting-only weighted median. The latter
two match total positive coefficient mass under the exact planned query draws.
This controls overall mass but not realized gradient magnitudes. Each head has
64,260 parameters and receives 492 causal inputs. All arms retain the same
initialization, inputs, draws, conservative missing-label supervision, original
risk loss, 2,000-update budget, last-iterate rule and fixed inference thresholds.
No held labels determine the weighting. The floor is part of training only.

## Results

The floor lowers fitting false-safe cost by 43.53%, 77.24% and 40.28% relative
to the equal-mass uniform control, but raises total decision cost by 0.6261%,
0.2402% and 0.3512%. Posthoc fitting-only attribution shows that lost useful
interventions outweigh repaired harm plus recovered benefit in each context.

On seven already exposed development sources, known partial-outcome ADE gain
is 0.7926% for the floor versus 0.8809% for the uniform control. The primary
contrast is -0.0883 percentage points, with missing-label completion bounds
[-0.1004, -0.0762] pp. These bounds address incomplete outcomes, not sampling
uncertainty. Worst defined easy-net degradation is 8.0481%; only one of 17
recording views meets the complete original risk/support condition. The full
requirement therefore fails. No model is promoted.

## Limitations And Interpretation

All three contexts share one forecast-producer seed and exposed SOURCE data;
they are not three independent dataset replications. Same-data exact replay
does not provide statistical or independent confirmation. Missing support is
not waived, and the original 24-group primary remains not estimable. Obs8/pred12
coordinates are native image coordinates, not calibrated physical units or
elapsed seconds. No reserved confirmation data are opened.

The result rejects this particular allocation repair, not all cost-sensitive
learning. It supports a narrower conclusion: stronger suppression of harmful
training examples alone does not establish safe utility improvement. Future
work should test causal discrimination between useful and harmful proposals
under the same evaluation contract. This addendum records a negative component
experiment and is not a claim of submission readiness or an updated compiled
submission package.

## Reproducibility

Nine full fits, three exact predecessor controls, a full treatment refit,
interrupted-versus-continuous pilots, weight reconstruction and prediction/metric
replay are complete. Code, fixed configuration, registration, loss traces,
summary numbers and failure records accompany the study. Private data, caches
and weights remain outside the public repository. See [validation](validation_receipt.md)
and [method positioning](related_work_position.md); cited learning-to-defer
theory is not claimed to certify this loss or the per-recording safety rule.
