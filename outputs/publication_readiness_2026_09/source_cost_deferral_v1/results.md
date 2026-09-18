# Training-Only Cost Deferral Results

## Material Passport

Six fresh continuations,48000 new updates; three dense controls and all ancestors cached_verified.
All15430 training rows. No held/main forecasts, threshold selection or deployment.
Three-seed mean and range describe optimization, not generalization uncertainty.

| Variant | Step | Output | Gain vsCV (%) [seed range] | Moving gain (%) | Hard gain (%) | Zero-target pixel harm | Actual intervention | Training-signal seeds |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| expected_cost | 2000 | proposal | -1.357696 [-1.486139,-1.289876] | -0.250282 | -0.028247 | 0.02249434 | 99.669% | 0 |
| expected_cost | 2000 | hard_action | +0.000000 [+0.000000,+0.000000] | +0.000000 | +0.000000 | 0.00000000 | 0.000% | 0 |
| expected_cost | 4000 | proposal | -1.225032 [-1.311576,-1.175567] | -0.165042 | +0.006645 | 0.02153104 | 99.669% | 0 |
| expected_cost | 4000 | hard_action | +0.000000 [+0.000000,+0.000000] | +0.000000 | +0.000000 | 0.00000000 | 0.000% | 0 |
| expected_cost | 6000 | proposal | -0.653298 [-0.701355,-0.597477] | +0.157745 | +0.105313 | 0.01647430 | 99.669% | 0 |
| expected_cost | 6000 | hard_action | +0.000000 [+0.000000,+0.000000] | +0.000000 | +0.000000 | 0.00000000 | 0.000% | 0 |
| expected_cost | 10000 | proposal | +0.195787 [+0.168134,+0.235332] | +0.736438 | +0.336153 | 0.01098197 | 99.669% | 1 |
| expected_cost | 10000 | hard_action | +0.000000 [+0.000000,+0.000000] | +0.000000 | +0.000000 | 0.00000000 | 0.000% | 0 |
| cost_supervised | 2000 | proposal | -1.357696 [-1.486139,-1.289876] | -0.250282 | -0.028247 | 0.02249434 | 99.669% | 0 |
| cost_supervised | 2000 | hard_action | +0.000000 [+0.000000,+0.000000] | +0.000000 | +0.000000 | 0.00000000 | 0.000% | 0 |
| cost_supervised | 4000 | proposal | -1.186083 [-1.353678,-1.050015] | -0.162078 | +0.003017 | 0.02080010 | 99.669% | 0 |
| cost_supervised | 4000 | hard_action | +0.009774 [+0.000000,+0.026231] | +0.009774 | +0.005059 | 0.00000000 | 0.076% | 2 |
| cost_supervised | 6000 | proposal | -0.661320 [-0.851957,-0.532954] | +0.143234 | +0.105170 | 0.01634250 | 99.669% | 0 |
| cost_supervised | 6000 | hard_action | +0.075918 [+0.054888,+0.090266] | +0.090247 | +0.048247 | 0.00029104 | 1.385% | 3 |
| cost_supervised | 10000 | proposal | +0.184359 [+0.144399,+0.230768] | +0.712599 | +0.325820 | 0.01072987 | 99.669% | 1 |
| cost_supervised | 10000 | hard_action | +0.233917 [+0.199395,+0.294471] | +0.578392 | +0.276383 | 0.00699713 | 61.002% | 3 |
| dense_control | 2000 | proposal | -1.357696 [-1.486139,-1.289876] | -0.250282 | -0.028247 | 0.02249434 | 99.669% | n/a |
| dense_control | 4000 | proposal | -1.206919 [-1.441927,-1.038764] | -0.176879 | -0.002435 | 0.02092267 | 99.669% | n/a |
| dense_control | 6000 | proposal | -0.704618 [-0.836864,-0.576333] | +0.137272 | +0.099888 | 0.01710088 | 99.669% | n/a |
| dense_control | 10000 | proposal | +0.187244 [+0.147848,+0.229792] | +0.717682 | +0.325144 | 0.01077451 | 99.669% | n/a |

## Checks and Limits

24exact prediction/score replays,3matched streams with dense controls,all milestones verified. 290 artifacts unchanged on completed resume,zero updates.
Hard action returns the exact baseline at score<=0. Expected soft-action risk is a training surrogate, not deterministic trajectory accuracy or a calibrated probability.
Easy percentage is undefined at zero baselineerror. Training-signal counts do not pass the original easy gate or an independent evaluation gate. All-baseline collapse does not count as predictive success.
The new head has44897parameters versus44864for the reused dense head. A synthetic dense-control execution is exact; the extra33gate parameters are the intentional architecture change.
Future errors supply loss targets only. Cost-supervised targets are detached, but computed from a jointly changing candidate; their in-sample fit is not independently calibrated risk.
Offline annotation inputs and source stride12/+144rawframes; no meters/seconds, human intentiongold, true3D or foundation claim. Stage5C/SMC remainoff.
