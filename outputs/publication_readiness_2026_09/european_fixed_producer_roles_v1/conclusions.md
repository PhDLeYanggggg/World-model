# Fixed-Producer Controller Study: Conclusions

I completed the registered experiment, not just a runtime demonstration:
144 new Torch utility/risk heads,288,000 updates,72 fixed ridge fits, and180
frozen decision views. The original trajectory forecasters and motion floors
were reused with verified hashes and checkpoint replays. No new trajectory
forecaster was trained. The experiment remains opened-source development.

## What Improved

Keeping the stronger four-source producer/floor fixed and supervising the cost
head with that same producer gives a useful easy-preservation result. Across36
dependent role/seed/event views, the matched controller's worst locality easy
degradation is **0.23767%**, below the2% criterion. The OOF-supervision control
reaches7.85031%; fixed ridge reaches8.23348%. No source/view easy violations occur
for matched supervision. The unchanged four-source floors also have no such
violations, unlike the weaker two-source floors in the preceding experiment.

Matched supervision improves all-ADE over its own common four-source floor by
0.06143% to2.56760%, with36 positive conditional locality-bootstrap intervals.
This is not enough to replace the stronger original controller.

## What Did Not Pass

| Matched controller versus | All-ADE gain range | Positive / negative95% CI |
|---|---:|---:|
| Fresh OOF-supervision control, same final forecasts | -0.98010% to+1.12547% |21 /6|
| Fixed ridge cost controller, same final forecasts | -3.08454% to+2.76633% |14 /15|
| Unchanged stopping-protected controller | -0.90074% to+1.64708% |12 /3|

These are36 correlated development views per comparison, not independent trials.
The hard subset has5 negative intervals versus the original controller. Raw
neural deployment remains unsafe in this empirical sense: worst easy degradation
is81.29545%, and zero-CV rows are harmed. No branch is selected post hoc.

The learned2% risk gate is not calibrated:66/144 selected locality/views exceed
that positive-harm ratio in realized outcomes and122 underpredict it. This can
coexist with small net easy degradation because benefit offsets harm; the two
quantities must not be conflated.

## Why The Remaining Negative Cases Lose

Post-readout arithmetic on unchanged decisions separates newly added switches
from removed original switches. In all3 significantly negative all-ADE views,
new switches produce net benefit. The larger loss comes from removing original
switches that were, in aggregate, useful. This is an observed error accounting,
not proof of the causal reason the learned head made those decisions.

| Negative view | New-switch net change, pp | Removed-switch net change, pp | Total degradation | Gain95% CI |
|---|---:|---:|---:|---:|
| A0/B2, seed29, all target | -0.38589 |+0.85164 |+0.46575% |[-0.75716,-0.20344]%|
| A1/B2, seed29, all target | -0.38538 |+1.28611 |+0.90074% |[-1.23571,-0.56576]%|
| A1/B2, seed43, all target | -0.23926 |+0.84937 |+0.61010% |[-0.89940,-0.15647]%|

Each component uses the same original-controller denominator, with localities
weighted equally. Negative change helps; positive change hurts. The complete
[accounting](changed_action_accounting.json) retains all36 groups, not only these
three. It did not generate a new policy.

## Decision

Do not promote this controller. Keep the unchanged protected policy. Matching
producer supervision is a promising development repair, but stable superiority
over the stronger controller and calibrated risk remain unproven. The next
targeted experiment should preserve that controller as the reference policy and
learn the value/harm of overrides directly, instead of learning floor-versus-neural
selection afresh. It must be registered before fitting; no readout tuning here.

362 scoped tests in60 files pass.144 new checkpoint prefixes,72 full ridge score
banks,180 scalar/constant policies,144 coordinate arrays and2,376 metric reductions
verify. Training took795 seconds and evaluation172 seconds. A pretraining recovery
bug was fixed and recorded; there was no training resource failure or downgrade.
CREATE was queried read-only; no remote job was submitted.

All results are image-pixel obs8/pred12 at raw annotation stride12 from detector
tracks. They are not metric/seconds, human gold, physical safety, true3D,
foundation, independent confirmation or submission readiness. Stage5C and SMC
remain off. Registration8dd1dd2d and training milestonee629fc29 were pushed before
the new readout. The underlying raw data, caches and checkpoints remain local.
