# What This Repair Tests

For a fixed baseline forecast b and a fixed neural forecast n, let R be the
baseline ADE and H=max(ADE(n)-R,0). Let E denote the existing easy label,
defined by0<CV_ADE<=the frozen easy cut. The easy-risk target is E*(H-.02R).
R,H and E are supervision only. Causal input x contains no realized future.

The conditional-expectation identity motivates

    q_easy(x) = P(E=1|x) * [ E(H|E=1,x) - .02 E(R|E=1,x) ].

This identity is elementary, not a new theoretical contribution. In this
experiment the fitted expectations refer to the known-label, source-balanced,
query-balanced fitting distribution, not automatically to a new deployment
population. Missing labels, source shift and repeated windows limit inference.

The marginal arm and the explicitly supervised arm share the same factorized
network. Only the latter receives easy-occurrence BCE and conditional positive
harm/reference-cost squared-error supervision. Both retain the same marginal
signed-risk loss. Thus the prespecified comparison tests an auxiliary-supervision
package; it does not identify separate causal effects of each auxiliary term.

The old risk head's four basis outputs had only two supervised combinations.
Calling a component an easy probability or identified conditional moment was
not justified. The new explicit supervision fixes that modeling omission, but
does not prove that estimated moments or probabilities are well calibrated.

Probability cancels the sign test for an individual with strictly positive
predicted occurrence. It can change how agents contribute to a query's summed
risk. Accordingly, independent and joint decisions are both reported, and the
main comparison uses a common feasible intersection anchor to match query-level
intervention counts exactly. Empty anchors remain empty and undefined realized
selected-risk denominators are not replaced with zero.

Proper-scoring motivation comes from the binary/finite examples in Section3
of Gneiting and Raftery (2007), [JASA](https://doi.org/10.1198/016214506000001437).
BCE/Brier alone provide neither a finite-sample risk certificate nor robustness
under domain shift. Conditional-cost fit, probabilistic quality and downstream
allocation must be checked separately. No neural improvement has been observed
for this repair before its frozen action readout.

Scope remains obs8/pred12, stride12raw frames, image-local detector-silver,
12opened development localities. Independent selection/calibration/confirmation
are closed. No metric/seconds, human-gold, true3D, foundation, physical-safety
or submission-ready claim; Stage5C/SMC disabled.
