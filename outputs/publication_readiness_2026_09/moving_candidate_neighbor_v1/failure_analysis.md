# Candidate-Neighbor Failure Analysis

## Observed Result

The fixed addition fails its prospective internal follow-up screen in all three
contexts. Relative MSE improvement is -6.2855%, -0.2226%, -5.1245%; joint sign-cost
improvement is -1.2896%, -0.4721%, +1.1641%. Only two of eight source folds reduce
MSE, and neither also reduces decision cost. This is a completed negative probe,
not a new neural training failure and not a revised deployment result.

## What Was Actually Tested

Sixteen ridge probes compare the existing 96-dimensional motion/history relation
basis against that basis plus 24 candidate-neighbor features. Eight own-B source
folds cover 62,466 causal rows, with 48,061 known proposed outcomes used in the
probe and 1,274 unknown proposed outcomes explicitly excluded from supervised
probe fitting. Other rows have no proposed intervention. Windows overlap; these
counts are not independent statistical sample sizes. The control has 194 fitted
coefficients and the treatment 242, including two intercepts. There is no model
or penalty search. This is not an equal-capacity ablation.

All rows have at least one neighbor aligned to the last two observed times.
Thus total absence of usable neighbor context does not explain this experiment.
The geometry is a constant-velocity proxy built from past observations. It is
not a measured future neighbor trajectory or a physical collision label.

## Error Slices

Frozen neural false-safe cost is largest in proximity quartile1,2,3 respectively
for critic0,1,2, not consistently in quartile0 where the candidate reduces nearest
distance most. Cuts use causal proposed rows, not outcome availability. Costs
are contributions under original fitting sampler mass; they are not prevalence
rates or source-independent risk guarantees. The absence of a monotonic pattern
does not prove absence of interaction effects.

## Interpretation And Limitations

- A simple explicit candidate-neighbor feature deficiency is not supported as
  the explanation to prioritize for another large neural run.
- Source-specific predictive relationships remain unstable. The control itself
  has source-macro target-RMS-normalized MSE above1 in every context, but that is
  not a comparison against a separately fitted constant predictor.
- A linear probe may miss nonlinear information. This result does not rule out
  learned interactions, better observation quality or different representations.
- Source065 has only44 known proposals; source support remains uneven. No source
  is excluded to improve the score. Only two or three fitting sources are
  available inside each context, limiting what any cross-source learning can
  establish.
- Frozen producer lineages and source data have prior development exposure.
  Leaving a source out of this new probe does not restore independent test status.
- The sign-cost helper is unchanged. Probe costs use signed errors in causal
  scale units, whereas the separate frozen-neural slice uses its original target
  RMS and sampler weights. Their absolute costs must not be compared as the same
  estimand.

## Model Action

Do not launch the proposed neighbor neural branch after its failed screen.
The next distinct model hypothesis is that mean fitting objectives hide
source-specific risk errors. Test a matched neural source-robust objective
against continued mean-objective training, retaining the current features,
initialization, query draws and original admission rule. Register that experiment
before training; neither its objective nor this probe changes the 2% evaluation
contract. This is a proposed mechanism, not a verified root cause.

Prior neural safety failure remains0/17 views; original24-group primary remains
not estimable. No new V/H, identity exception, deployment or submission-ready
claim. Native obs8/pred12 image coordinates only; Stage5C/SMC remain off.
