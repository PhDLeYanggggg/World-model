# Relative-Cost Deferral: Small Training Repair, No Deployment Claim

## What Was Actually Run

`fresh_run`: all six real Torch continuations, three seeds and 48,000 new
optimizer updates on all 15,430 registered source-training queries. The original
100-step pilot is included: the subsequent process performs 47,900 remaining
updates, not 48,000 extra ones. Full-run log span is 37.0237 minutes; summed
continuation work is 2227.77 seconds including inherited pilot time. Native
arm64 CPU4/inter-op1/workers0; observed resident memory about 2.7 GiB, not a
NumPy fallback. The fixed budget was completed without downgrading or a crash.

`cached_verified`: three 2,000-step parent models and three matched dense cosine
controls. The parents contribute 6,000 unique inherited updates. They are not
counted as fresh models or independent experiments. All same-seed branches
preserve the sampled rows and training budget exactly.

`not_run`: held-bookstore forecasting, main development/calibration/confirmation,
independent risk calibration and deployment evaluation. None was authorized by
this registration. Historical data-lineage restrictions remain in force.

## Final Training Results

These are means over seeds 17/29/43 at the fixed final step 10,000. Gains are
ADE reductions versus stationary CV on this training population, not t+50
metrics, whole-benchmark scores or held-scene confidence intervals.

| Output | All gain | Moving gain | Hard gain | Zero-target absolute harm (pixels) | Actual intervention |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cached dense control | +0.187244% | +0.717682% | +0.325144% | 0.01077451 | 99.669% |
| Expected-cost proposal | +0.195787% | +0.736438% | +0.336153% | 0.01098197 | 99.669% |
| Expected-cost hard action | 0.000000% | 0.000000% | 0.000000% | 0 | 0% |
| Cost-supervised proposal | +0.184359% | +0.712599% | +0.325820% | 0.01072987 | 99.669% |
| Cost-supervised hard action | +0.233917% | +0.578392% | +0.276383% | 0.00699713 | 61.002% |

The cost-supervised hard gains are +0.207886%, +0.294471% and +0.199395%.
Their improvements over the same-seed dense control are +0.023794, +0.064679
and +0.051547 percentage points. All three pass the narrowly registered
training-signal condition. That is not a research or deployment gate.

The mean native pixel ADE changes from 1.12554113 for the dense control to
1.12501482 for the cost-supervised action, only 0.00052631 pixels. Stationary-CV
ADE is 1.12765259 pixels. Practical magnitude must not be inflated by reporting
only a relative percentage. Three-seed ranges describe optimization variation.

## What Improved, and What Did Not

1. Exact baseline fallback is implemented and reproducible. Pure expected-cost
   training nevertheless rejects every query in all three seeds. It has zero
   harm and zero predictive gain, not a successful protected neural model.
2. Explicit cost supervision with a continuing proposal term avoids that complete
   rejection. It lowers zero-target harm by 35.06% relative to the dense model.
   Harm remains positive. Zero-CV error makes percentage easy degradation
   undefined, so the approved 2% easy-preservation gate is not passed here.
3. The aggregate benefit is a tradeoff, not stronger dynamics. Raw cost-supervised
   proposals are slightly worse than dense control. Hard gain falls by 0.048761
   percentage points and moving gain also falls after gating. The 95th and 99th
   normalized-error quantiles are slightly worse than the dense control.
4. Even training-site performance is not uniformly positive. Cost-supervised
   hard-action mean gains are coupa +0.413059%, deathCircle -0.175465%, gates
   +0.811246% and hyang +0.231769%. deathCircle loses in every seed, on 3,054
   training rows. No per-site threshold is adjusted in response.
5. No RGB, JEPA, Transformer, interaction or joint-scene mechanism contribution
   has been established by this geometry/coverage-only comparison.

## Failure Taxonomy

**Joint rejection and candidate learning:** the expected-cost head learns strongly
negative scores while its proposal is initially harmful. Proposals become
slightly useful late, but scores remain around -1.05 and never cross zero.
This temporal pattern is consistent with optimization lag under a decaying
learning rate. It does not isolate lag, initialization, gradient clipping or
feature insufficiency as the unique cause. Every new update clips gradients;
the earlier successful microfit shows clipping alone is not proof of failure.

**Supervised-cost partial repair:** final score RMSE is 0.03945/0.04290/0.03831,
below the corresponding constant-training-mean RMSE 0.04270/0.04717/0.04245.
This supports some in-sample discrimination, not independent risk calibration.
Mean sigmoid scores are near 0.5004. They are not confident success probabilities;
requested intervention is also distinct from actual changed predictions.

**Huber-target hypothesis checked, not confirmed as the main cause:** a Huber
location need not equal expected gain. Here only 3-4 supervised-branch labels
per 15,430 rows exceed the unit Huber radius. All global Huber locations remain
positive and close to mean gains. Their small difference cannot explain the
expected-cost branch's all-rejection result, and that branch does not even use
the Huber fit term. Conditional effects remain untested. See the explicitly
[post-hoc training diagnosis](score_target_diagnostic.md).

**Weak candidate class and heterogeneous costs:** most query-level signed gains
are negative, although their average is positive. The candidate's training-only
future oracle gain is just 0.818% for cost-supervised proposals. This is an
action-class diagnostic, not an inference input, a global learnability limit or
evidence that adding a larger model must work. deathCircle and zero-target harm
remain concrete negative slices.

## Verification and Limits

All 24 proposal/score snapshots replay exactly. Three three-way matched sampling
streams and every milestone are verified. Completed resume leaves 290 artifacts
unchanged and performs zero new updates; all runtime processes are terminal.
Thirty-two targeted tests pass. The full legacy suite was not rerun. The
[learning-curve figure](training_curves.svg) was visually checked.

Pre-fit registration is `e91a7a37`. The analysis consumer was frozen at `ac37cfc6`
during training, after first-branch training results, not pre-fit. The extra
score-target diagnosis is explicitly post-hoc and changes no model or threshold.
Its labels come only from the registered training rows.

The source task uses offline supplied annotations, eight observations and twelve
predictions at stride 12 / +144 raw frames. Pixel/past-normalized only; not
audited meters, seconds, human intention gold, true 3D or foundation modeling.
No Stage5C execution or SMC. No change to the approved main estimand or deployment.

## Next Decision

Freeze all six endpoints and all negative slices. The positive training signal
supports a separately registered, fixed source-site diagnostic against matched
controls, not threshold tuning or a best-seed release. Bookstore has already
been explored; such a check must remain explicitly exploratory and cannot
replace independent main confirmation. It has not been run in this experiment.

If rejection persists outside fitting rows, the smallest causal follow-up is
to freeze candidate trajectories and compare a newly fitted cost head against
joint fitting, preserving the approved roles. That would separate moving-target
optimization from head learnability; it is not yet registered or run. No larger
architecture or new test-set selection is justified by a 0.0467pp training gain.
