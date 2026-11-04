# Decoder-Link Matched Continuation

## Material Passport

Fresh_run: 36 real Torch continuations from 18 fixed 2k parents, 72,000 new
primary updates, two complete first-job refits, full fitting-region check
and held-policy inference. Cached_verified: parent forecasts/weights and
separate new cost/policy scalar replay. Warm updates are not recounted.
This is exposed-source development, not independent final confirmation.
Original 24-group primary remains not estimable; 23 groups are secondary.

## Fixed Equal-Budget Contrast

Both arms continue identical weights, Adam/sampler/RNG, inputs and original
normalizers for 2,000 updates. The only factor is a fixed unit auxiliary
on coherent action cost fractions and harm-weighted EH/H gate logits.
It changes loss geometry and example weighting; it is not solely a
floating-point repair or a novel scoring rule. Earlier log-loss and
hurdle negative results are not reclassified.

Joint-free known auxiliary-minus-continuation: -9.3066 [-15.5876, -3.4391] pp.
Fixed paired completion-lower: -16.8573 [-27.2972, -7.6909] pp.

| Model | Policy | Known gain [95%CI] | Completion lower [95%CI] | Original risk support | Known all/easy violations | Worst known easy degradation (%) |
|---|---|---:|---:|---:|---:|---:|
| continue_original | floor | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0/348 | 0/0 | +0.0000 |
| continue_original | frame_uniform | +6.3871 [+2.3518, +11.1395] | +1.5521 [-4.1476, +6.3055] | 0/348 | 171/176 | +297.7324 |
| continue_original | independent | +10.2555 [+4.0446, +16.7437] | +3.9768 [-5.5663, +11.8216] | 7/348 | 286/286 | +428.9130 |
| continue_original | joint_free | +11.3861 [+4.5526, +18.5194] | +3.6727 [-5.9401, +11.5087] | 6/348 | 286/289 | +428.9130 |
| continue_original | joint_matched | +10.9308 [+4.3306, +17.8575] | +3.3748 [-5.9602, +11.0491] | 5/348 | 286/286 | +428.9130 |
| continue_original | uncontrolled_neural | +16.4074 [+7.4982, +25.5310] | -5.0187 [-16.4749, +5.0800] | 6/348 | 304/320 | +1742.9603 |
| decoder_auxiliary | floor | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0/348 | 0/0 | +0.0000 |
| decoder_auxiliary | frame_uniform | +0.0699 [-0.0350, +0.2466] | +0.0567 [-0.0605, +0.2376] | 0/348 | 14/9 | +184.1985 |
| decoder_auxiliary | independent | +0.9064 [+0.3461, +1.5426] | +0.8351 [+0.2620, +1.4833] | 6/348 | 233/184 | +198.8376 |
| decoder_auxiliary | joint_free | +2.0796 [+0.8154, +3.6303] | +1.9150 [+0.6494, +3.4509] | 4/348 | 236/192 | +227.8401 |
| decoder_auxiliary | joint_matched | +1.5188 [+0.6332, +2.5499] | +1.3873 [+0.4950, +2.4299] | 7/348 | 224/190 | +194.7780 |
| decoder_auxiliary | uncontrolled_neural | +16.4074 [+7.4982, +25.5310] | -5.0187 [-16.4749, +5.0800] | 6/348 | 304/320 | +1742.9603 |
| frozen_parent | floor | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0/348 | 0/0 | +0.0000 |
| frozen_parent | frame_uniform | +5.2071 [+2.2020, +9.1758] | +2.0541 [-1.9341, +5.8289] | 0/348 | 143/151 | +277.9125 |
| frozen_parent | independent | +9.7080 [+4.2118, +15.5267] | +5.1618 [-2.9968, +12.0370] | 7/348 | 285/280 | +309.5896 |
| frozen_parent | joint_free | +11.3326 [+5.0080, +17.9928] | +5.1034 [-3.3339, +12.3919] | 7/348 | 281/283 | +340.5715 |
| frozen_parent | joint_matched | +10.7543 [+4.6989, +17.1264] | +4.7033 [-3.3526, +11.6839] | 4/348 | 283/282 | +340.5715 |
| frozen_parent | uncontrolled_neural | +16.4074 [+7.4982, +25.5310] | -5.0187 [-16.4749, +5.0800] | 6/348 | 304/320 | +1742.9603 |

Required views repeat producers/seeds, not independent places. Unknown or
empty reference support is not measured safe zero. Same-count controls
remain utility-only and cannot replace the full original 2% risk contract.

## Intervention And Empty Controls

Switch rates average within source group over excluded producers and
actual seeds, then average source groups. They are not IID row estimates.

| Model | Joint-free switch (%) | Empty same-count group views | Empty same-count queries |
|---|---:|---:|---:|
| frozen_parent | 63.7546 | 4/138 | 2057 |
| continue_original | 69.9956 | 3/138 | 1501 |
| decoder_auxiliary | 11.8629 | 16/138 | 7322 |

None of these arms has a same-count utility estimate for the complete
required group scope. Empty comparisons stay not estimable; they are not
zero utility or safety evidence. Shrinking interventions alone does not
establish a better learned admission rule.

## Fitting Cost Reliability

These are pooled 18-job known-mass ratios on critic-fitting sources only,
not official macro results, independent tests or new policy selection.
All causal regions are fixed before labels are used. Compare each model
on all four regions, not just the region that looks best after selection.

| Scoring model | Fixed selection region | Known H / predicted H | Known EH / predicted EH | Known all/easy risk (%) |
|---|---|---:|---:|---:|
| frozen_parent | all_neural | +1.4670 | +0.6461 | +14.8823/+47.4642 |
| frozen_parent | frozen_parent_independent | +64.4607 | +653.0692 | +9.6794/+31.6837 |
| frozen_parent | continue_original_independent | +13.4070 | +40.8364 | +10.0809/+34.5897 |
| frozen_parent | decoder_auxiliary_independent | +34.1693 | +190.3055 | +5.0911/+16.8107 |
| continue_original | all_neural | +1.8178 | +0.8368 | +14.8823/+47.4642 |
| continue_original | frozen_parent_independent | +118.4714 | +498.1002 | +9.6794/+31.6837 |
| continue_original | continue_original_independent | +118.6566 | +912.9178 | +10.0809/+34.5897 |
| continue_original | decoder_auxiliary_independent | +106.7928 | +1583.1821 | +5.0911/+16.8107 |
| decoder_auxiliary | all_neural | +0.6966 | +0.2558 | +14.8823/+47.4642 |
| decoder_auxiliary | frozen_parent_independent | +0.9084 | +0.4852 | +9.6794/+31.6837 |
| decoder_auxiliary | continue_original_independent | +0.9044 | +0.4680 | +10.0809/+34.5897 |
| decoder_auxiliary | decoder_auxiliary_independent | +13.0514 | +19.8269 | +5.0911/+16.8107 |

## Actual Cost And Scope

Training/refit invocation: 2930.122 s, peak RSS 7,331,446,784 bytes.
Held evaluation: 358.364 s; separate replay: 183.093 s.
Native arm64 CPU4/interop1/workers0. Checkpoints/RNG/sampler support
identical resume. Model/cache arrays stay private. The loss figure uses
the common original fitting objective,not incomparable treatment totals.
No new trajectory/JEPA/Transformer fitting, latent rollout or visual-scene
contribution is established by a downstream cost-head continuation.
Pixels/native 8/12 steps are not metres/seconds or SDD raw t50.
No independent confirmation, deployment, submission-ready, true 3D or
foundation claim. Stage5C execution and SMC are off.
