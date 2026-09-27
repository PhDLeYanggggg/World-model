# Method Positioning and Statistical Boundary

## Material Passport

Primary sources checked2026-09-27. This note motivates an experiment; it is not
new performance evidence or an originality claim for tail-weighted regression.

SelectiveNet jointly optimizes prediction and abstention at a desired coverage.
Its risk/coverage formulation makes clear why reduced risk after rejecting more
cases is not, by itself, a better selector. Our experiment keeps forecasters and
utility frozen, and compares risk-head objectives with exact same-query counts;
it does not reproduce SelectiveNet or establish superiority over it.
[Geifman and El-Yaniv, ICML2019](https://proceedings.mlr.press/v97/geifman19a.html).

Learn then Test calibrates a fixed predictive system using statistical testing
and family-wise error control. Its stated calibration setup uses independent,
identically distributed calibration units. Here we have opened development
localities and overlapping windows, not independent window-level calibration
data. A fixed empirical2% screen, tail weighting or locality bootstrap does not
constitute LTT or a risk certificate.
[Angelopoulos et al., Learn then Test](https://arxiv.org/abs/2110.01052).

## What the Weighted Head Estimates

For scalar harm H and a positive training-defined weight w(H), weighted squared
loss has conditional minimizer E[w(H)H|X]/E[w(H)|X]. This follows by taking the
derivative of E[w(H)(f-H)^2|X]. It generally differs from E[H|X]. Dividing weights
by their global fitting mean changes optimization scale, not that conditional
minimizer. The treatment therefore produces tilted risk scores, not a calibrated
conditional mean or a statistical upper bound. This is our elementary objective
analysis, not a new theorem or an attribution to either paper.

The method question remains relative intervention over a usable causal fallback,
with producer-chain exclusion and scene-query allocation. A neural architecture
or reweighted loss is not sufficient novelty. The proposed route still needs
useful harm ranking at matched coverage, independent-source calibration and
confirmation, and evidence that joint agent decisions add value.
