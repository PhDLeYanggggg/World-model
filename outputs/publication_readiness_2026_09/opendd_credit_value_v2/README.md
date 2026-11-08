# Does Learning When to Wait Help?

I kept the OpenDD EqMotion forecast bank, original held-site release
stream, two-query deadline and local caps fixed. At each query the
controller can spend credits on a prefix of the same causal CV/OLS
disagreement ranking. This ranking is shared by every arm here; it
differs from the learned recipient ranking in the previous timing
experiment. The immediate control spends all current releases.

## Training

Each legal action spends zero or more FIFO credits. The two age bins,
current ranked trajectories and previous query summary are inputs.
No policy sees a future row, realized error, or recording end. Offline,
I compute each action's advantage over spending zero, with and without
the optimal continuation under the same queue. The reward is lower
gain against CV minus positive all/easy/signed excess with weights
2/4/1 and a fixed motion-scaled expiry cost. This additive training
proxy does not certify the original recording-level constraints.
The two outputs share a small Torch network; both evaluation arms can
wait, but the myopic arm scores only the current action's return.

The donor trajectories and release heads exclude both that donor and
the outer evaluation site. Held-site release counts reproduce the
original 3,323 edits. The first run stopped on a donor with no legal
spend decision. The corrected version omits and reports those
donor-seed cells (23 of 126); it retains every evaluation recording,
including those with no releases. No model or threshold was selected
on the held outcomes.

## Results

| Policy | Edits | Unused | Undefined views | Risk pass | Joint pass |
| --- | ---: | ---: | ---: | ---: | ---: |
| Immediate | 3,323 | 0 | 438 | 11 | 5 |
| Myopic | 1,644 | 1,679 | 439 | 18 | 14 |
| Myopic, same credits used immediately | 1,644 | - | 498 | 9 | 3 |
| Continuation | 1,629 | 1,694 | 440 | 18 | 14 |
| Continuation, same credits used immediately | 1,629 | - | 499 | 9 | 3 |

The retrospective controls receive only the corresponding arm's used
credits. They match counts per recording, not recipients or radii;
they are not deployment policies or safety certifications.

| Direct lower gain (percentage points of observed CV error mass) | Estimate [95% interval] |
| --- | ---: |
| Myopic vs matched immediate | +0.149 [-0.198, +0.722] |
| Continuation vs matched immediate | +0.131 [-0.207, +0.711] |
| Continuation vs myopic, full policies | -0.015 [-0.081, +0.038] |
| Continuation vs CV | +0.049 [-0.187, +0.299] |

The 3,000 bootstrap draws resample seven sites after averaging seeds.
There are 202 exposed recordings and 606 recording-seed views, not
606 independent recordings. Of the 3,323 releases, 3,138 came from two
sites; one site never released a credit. Neither learned arm
establishes better utility or all-recording safety.

## What Is Lost by Fixing the Recipient Order?

I also solve the original outcome-aware risk and dual-gain problem
with the same total edits per recording, but with only ranked prefixes
available at each query. Each solution selects at most one prefix per
query. Every positive witness is rescored in the original scorer.

| Same-total menu | No delay | Up to two queries |
| --- | ---: | ---: |
| Arbitrary legal recipients, verified prior result | 88 | 223 |
| Fixed causal ranking, fresh exact solve | 5 | 84 |

No recording-seed case is unresolved. With the fixed order, 77 views
lack reachable easy support at delay two; another 217 never receive
credits. The 139-view difference from the unrestricted ceiling
isolates a restriction of this action menu under these caps.
It does not establish which recipient order can be learned online.
Learning when to spend alone is not enough to explain the larger
hindsight opportunity. I would next test joint recipient and timing
choices against the fixed rank, with the release clock unchanged.

Sequential deferral already exists for decisions that change
downstream outcomes ([Pre-emptive learning to defer](https://arxiv.org/abs/2109.06312)),
and online expert deferral has been studied with a query budget
([Online Decision Deferral](https://arxiv.org/abs/2409.20489)).
This is a different experiment: the forecasts do not change observed
dynamics; the queue couples interventions through expiring credits,
and I evaluate the original recording-level error and support rules.
I do not claim a novel general deferral algorithm from this experiment.

## Reproduce

```sh
.venv-pytorch/bin/python scripts/run_m3w_opendd_credit_value.py
.venv-pytorch/bin/python scripts/run_m3w_opendd_credit_value.py --replay
.venv-pytorch/bin/python scripts/run_m3w_opendd_prefix_capacity.py
```

`fresh_run`: 63 release fits, 21 value fits, 33,600 updates; 21.7
seconds on local arm64 Torch CPU with four threads, zero loader
processes. Checkpoints and heartbeat every 100 steps; the same
command resumes. The exact ranked-menu diagnostic is fresh.
`cached_verified`: existing trajectory bank and held release heads;
7.5-second zero-update replay matches the learned readout.
Twenty focused tests passed, covering queue state, exact capacity,
frozen credit comparisons and the independent source split.
Native 8/12 annotation steps and dataset-local coordinates.
No new CREATE job, deployment, Stage5C or SMC. DroneCrowd's 15/30
roles remain frozen and confirmation outcomes were not opened.

[Registration](registration.json) | [Site and seed results](analysis.json) |
[Prefix registration](prefix_registration.json) | [Exact prefix results](prefix_capacity.json)
