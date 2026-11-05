# Error Discovery And Failure Analysis

The comparison changes error discovery, not evaluation. Discovery teachers
have no full-B warm weights or normalization; their held source is excluded
from fitting, common normalization, target RMS and kinematic normalization.
The frozen A proposer remains foreign to all B sources.

## Supervised Priority Channels

| Context | In-source all/easy | Held-source all/easy |
|---|---:|---:|
| eqmotion_fold1_seed17_critic0 | [72, 34] | [106, 74] |
| eqmotion_fold0_seed17_critic1 | [267, 60] | [585, 144] |
| eqmotion_fold0_seed17_critic2 | [2812, 962] | [5934, 2638] |

## Complete Fitting Loss

Identical empirical query weighting; not minibatch loss. Priority masks
differ, so both mask-specific losses are shown for every student.

| Context | Model | Anchor MSE | In-source-priority MSE | Held-source-priority MSE |
|---|---|---:|---:|---:|
| eqmotion_fold1_seed17_critic0 | warm | 0.020248872 | 0.000109065 | 0.000186222 |
| eqmotion_fold1_seed17_critic0 | in_source_errors | 0.012577246 | 0.000011594 | 0.000141981 |
| eqmotion_fold1_seed17_critic0 | held_source_errors | 0.012443631 | 0.000076528 | 0.000034209 |
| eqmotion_fold0_seed17_critic1 | warm | 0.018960970 | 0.000170909 | 0.001409936 |
| eqmotion_fold0_seed17_critic1 | in_source_errors | 0.019316292 | 0.000123911 | 0.001135303 |
| eqmotion_fold0_seed17_critic1 | held_source_errors | 0.021941627 | 0.000286504 | 0.000279137 |
| eqmotion_fold0_seed17_critic2 | warm | 0.893710541 | 0.005748003 | 0.026974593 |
| eqmotion_fold0_seed17_critic2 | in_source_errors | 0.686580523 | 0.002736696 | 0.024965616 |
| eqmotion_fold0_seed17_critic2 | held_source_errors | 0.712852100 | 0.003998254 | 0.012247360 |

These are supervised output channels across repeated fitting contexts,
not independent people, complete admitted decisions or safety passes.
Unknown labels never receive priority; their bound-aware anchor remains.

## Prospective Context Results

| Context | Completion lower contrast pp | Expansion condition |
|---|---:|---|
| eqmotion_fold1_seed17_critic0 | 0.0898 | True |
| eqmotion_fold0_seed17_critic1 | -3.3152 | False |
| eqmotion_fold0_seed17_critic2 | 0.0862 | False |

## Interpretation Limits

A reduction in fitting error or a lower switch rate is not proof of safe
held utility. Negative or unsupported views remain in the result.
Teacher source scarcity is a plausible limitation, not a proven sole
cause. This comparison cannot disentangle every source-size/difficulty
effect. Mask cardinality and total priority-gradient mass can differ;
the experiment tests the fixed discovery procedure, not an isolated
equal-weight causal effect of individual error identities. Changing
error discovery does not add new observed motion
information. No new guarantee of calibrated expected harm is claimed.
The unchanged original support deficit remains separate from model
quality; training cannot make an absent reference denominator observed.
