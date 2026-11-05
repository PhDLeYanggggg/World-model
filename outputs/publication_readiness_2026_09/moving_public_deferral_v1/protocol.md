# Frozen public regression-deferral control

Status at registration: not_run. No scientific result is implied by this document.

## Question and prior evidence

The completed joint-event experiment improved average harm estimation but failed
the original risk and matched-utility requirements. This experiment tests a
published cost-sensitive routing objective instead of requiring a cost regression
head to produce the routing decision. It does not fix the primary evaluation
definition, add identity certification, or choose a favorable risk threshold.
The previous European-bank deferral experiment remains a distinct, exposed-source
experiment; it is not reused as a MovingDroneCrowd result.

## Public method

Mao, Mohri and Zhong, *Regression with Multi-Expert Deferral*, ICML 2024,
PMLR 235:34738-34759, Section 4, Equation (3):
https://proceedings.mlr.press/v235/mao24d.html

With frozen experts, minimize sum_k (sum_j c_j - c_k) times
negative log2 softmax(logit)_k. Cost is available-label ADE divided by a frozen
past-only scale. There is no one-hot oracle label, added deferral fee, cost cap,
per-row cost-weight normalization or learned threshold. The forecast bank is the
unchanged physical floor plus six frozen neural amplitudes. Floor is always legal;
other legal actions come only from causal geometry. Ties choose the floor.

This is a task-adapted public-method control, not a reproduction of the authors'
datasets, architecture or published numbers. Unbounded pixel costs, variable
legal-action sets, partial labels and dependent source queries do not inherit a
bounded-loss consistency theorem or any relative-risk guarantee automatically.
Past-scale normalization preserves the conditional optimal action but changes
finite-capacity weighting. Unknown futures are excluded, not observed zeros.

## Fixed implementation and fitting

Use the same 380 local causal fields and set the remaining 48 shared-latent fields
to zero. Use width64 and the same other-agent query mean trunk. A single logit
output per action has 27969 parameters, versus the original point cost head's
28229. Therefore the comparison is not a pure one-factor loss ablation.
No target or future mask is accepted at inference. Future labels enter loss only.

Fit 54 routers: 18 original producer/critic contexts, three original inner-C
parts each. Producer seeds are 17,29,43. Own-part preprocessing and source-balanced
complete-query sampling remain unchanged. Every model receives 2000 updates,
108000 primary updates total. Three real pilot cells use 50+50 resumed and 100
direct updates each (600 total); independently repeat the first full fit (2000).
Save optimizer, sampler, RNG, config and weights. Freeze all54 before validation.

## Unchanged source roles and controls

Only already exposed SOURCE inner-C partitions are used. All reserved Moving
groups, DUT, DroneCrowd and new-H confirmation remain unopened. For validation V,
fit router A on one other part and use original point critic B, replica0, from the
remaining part. Swap A/B. Never choose a preferred critic or seed after readout.
The point proposer comes from the same A bank as before. Both router and point
proposals receive exactly the same fixed B veto: positive all/easy reference,
predicted H/R <=2% and EH/ER <=2%. The veto cannot change amplitude or add actions.

Policies: physical floor; point proposer; point veto; point count-and-amplitude
matched control; public router; public veto; public count-and-amplitude matched
control. Matching uses the same query and number of each amplitude. Router
matching ranks causal selected-logit minus floor-logit, not a calibrated
probability. Empty required matched views stay NA, never pass or zero.

## Readout and interpretation

All 108 orientations, 276 source views and 696 recording views remain required.
Report direct public-minus-point and vetoed-public-minus-vetoed-point contrasts,
nonempty matched utility, separately labeled full-strategy secondary contrasts,
available-label ADE/FDE, tails, easy net degradation, support, intervention and
the original full recording risk bounds. Paired bootstrap uses 3000 source-unit
resamples after averaging repeated contexts/seeds/orientations within 23 sources.
The original24group primary remains not estimable, not passed. These are
predeclared development contrasts, not replacement confirmation or deployment.

No identity certificate. No risk relaxation. Prior failures remain failures.
Stage5C and SMC remain off. Native observation8/prediction12 annotation order;
image pixels, not metric or verified seconds. Not true3D or foundation evidence.
