# Scoring Target and Method Positioning

The log score is a strictly proper categorical scoring rule; its sign-reversed
form motivates probability estimation by log loss. Reading scope: Section3,
Examples1 and3 and the surrounding definition, not a claim to have reviewed
the entire paper. [Gneiting and Raftery, JASA2007](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf).

Our conditional-expectation calculation is elementary, not a new scoring-rule
theorem. Weighting binary log loss by H changes its estimand from ordinary
easy membership to E[H E|X]/E[H|X]. Normalizing by a fitting-wide positive
constant does not change that population optimum. It does change gradient
variance and finite-sample optimization. Neither the cited paper nor this
identity establishes useful features, domain transfer, scene-level risk
control, or finite-sample calibration for this data.

The experiment is a controlled supervision repair. It preserves a direct
nested-cost readout and does not multiply auxiliary probabilities into costs.
A favorable result would support cost-target alignment under this source
protocol, not novelty from weighting BCE or combining JEPA/Transformer.
The larger contribution still requires transported relative gain/harm,
scene-level joint intervention, independent risk calibration and meaningful
matched controls. Historical exposed results remain exploratory.
