# Failure Analysis: Context Availability Is Not Useful Risk Identification

## Implementation Defect, Then a Real Negative Result

The original copy inherited a frozen encoder from its parent constructor.
Only 130 output parameters trained. This was detected by parameter inspection,
reproduced with a failing regression test and fixed before held readout. Both
corrected arms trained 12,322 parameters, including actual encoder updates,
while the protected base remained exact. The corrected negative result cannot
be explained by that already repaired defect. The defective run is not deleted.

## Equal-Count Error Decomposition

All values below are normalized by full known reference mass within each view,
then averaged equally across localities. Raw units are not pooled across sources.
Positive harm delta is worse. Unknown penalty is an uncertainty deduction, not
an observed error. This is descriptive accounting of frozen actions, not a new
causal attribution or an inference feature.

| Contrast | Known benefit delta pp | Known harm delta pp | Unknown penalty pp | Paired lower utility pp |
| --- | ---: | ---: | ---: | ---: |
| Peer - self, full | -0.388418 | -0.182053 | 0.774011 | -0.980376 |
| Peer - self, matched | -0.067220 | +0.011536 | 0.160518 | -0.239274 |
| Peer - rational, full | +0.090533 | -0.065084 | 0.738601 | -0.582984 |
| Peer - rational, matched | +0.003665 | +0.021994 | 0.140156 | -0.158485 |

Relative to self at matched counts, known utility already falls by 0.078756pp.
The result is therefore not solely an unknown-outcome penalty. Full selection
also drops from 32,621 to 24,501 occurrences, but matching counts does not fix
the ordering. Relative to rational, small known full-policy benefit is insufficient
to cover uncertainty, and known matched utility is negative as well.

## Selected Harm Remains Underestimated

For peer's own known selected cohort, predicted/observed normalized harm mass
is 0.042456/1.271650pp; easy harm is 0.001550/0.033989pp. These are normalized
moment masses, not aggregate risk ratios. The analogous self easy-harm masses
are 0.000893/0.038159pp. The peer change increases predicted easy harm, but
selected-risk support still fails in every view. A better global MSE would not
establish calibration on a selected subgroup; here peer conditional MSE is
6.619546 versus self 6.537742, with difference CI crossing zero.

## Availability, Optimization and Scope

Peer context is available in most evaluated rows: for example locality008 has
27,082/27,496 multi-agent rows; locality074 has 28,254/28,640 and query size up
to72. Locality020 has none in its held recording, despite some in inner fit.
The latter is a genuine support shift, but it cannot explain failures everywhere.
The complete per-context availability table is in verification.json.

Final matched sampled primary losses average 0.986922 for peer and 1.054476
for self. These are one logged minibatch per fit, not full fitting risk or a
convergence proof. Recorded gradient norms exceed clip5 in 98.68% and 99.47%
of peer/self log points respectively; log points are not all updates. Seven
peer and twelve self final dual cells reach the cap across 144 cells per arm.
Optimization conditioning remains a plausible limitation, not a proven cause.

All three head seeds have negative peer-minus-self matched utility:
- seed17: -0.198777pp;
- seed29: -0.251443pp;
- seed43: -0.267602pp.

Mean-before-encoder pooling discards individual-agent structure. Its inputs
also contain ego-relative features, so a pooled mean is not a physical common
coordinate graph. This test rejects this fixed construction under this budget,
not all interaction modeling or the existence of useful causal information.
The pre-existing 380 features already contain up to eight neighbor histories.

## Decision

Do not deploy, tune thresholds on these views, add outcome availability to
inputs, or select a favorable locality/seed. Do not declare a new trajectory
or world-dynamics result. Keep the original TRAIN development boundary and
independent roles closed. Next work must separate estimation/optimization
failure from information loss before proposing another context architecture;
see project_gap.md. Stage5C and SMC remain off.
