# Better Point Decisions Reduce Drift, Not the Cross-Scene Forecasting Gap

## Material Passport

Fresh fit-only point-decision experiment under cached, hash-verified forests.
No new neural or tree model was trained. All18 fixed ExtraTrees settings were
evaluated: two held physical fit scenes, three seeds and three feature arms.
The 365 stationary windows, labels, 8/12 native-step task, primary normalized
ADE, gate0.9 and exposure history remain unchanged. No development, calibration
or confirmation labels were opened; these folds are not independent test sites.

## What Changed

The previous squared-error regressors average the trajectories in their leaves.
We reconstruct that conditional training distribution exactly and change only
the point forecast: a per-step geometric median minimizes weighted Euclidean
distance rather than squared error. Query weights depend only on causal query
features and the frozen trees. The opposite fit scene supplies all support labels.

The reconstructed means replay the forest predictions within7.64e-17 in the
scene-normalized frame and the saved native predictions within1e-12. All solver
outputs have conditional training risk no worse than the original mean or zero
forecast. This is in-sample conditional risk, NOT a guarantee on another scene.
The algorithm is an established point estimator, not a new M3W contribution;
its bibliographic attribution and access limits are in the
[frozen decision](../conditional_ade_decision.md).

## Results

Three-seed means, native/parent-normalized ADE percentage gains coincide on
this exact-stationary subset. These are dataset-local coordinates, not meters.

| Held scene | Features | Original mean gain | Median gain | Median + fixed gate gain |
| --- | --- | ---: | ---: | ---: |
| ETH | pooled | -3.31% | 0.00% | 0.00% |
| ETH | static scene | -6.92% | 0.00% | 0.00% |
| ETH | scene + neighbors | -4.36% | 0.00% | 0.00% |
| Hotel | pooled | -346.70% | -90.38% | -65.13% |
| Hotel | static scene | -213.07% | -21.37% | -0.20% |
| Hotel | scene + neighbors | -252.23% | -21.25% | -0.36% |

All nine ETH median forecasts are exactly zero on every row and step. That
removes mean-induced false motion but merely equals CV, rather than predicting
the starts. Every unrestricted Hotel setting is still negative. No median or
guarded median setting has positive full-subset gain. Ten guarded settings equal
the floor; the other eight are worse. Zero is not positive transfer.

Still-row native harm is zero on ETH but remains positive on Hotel: seed means
0.02312 pooled,0.00600 static scene,0.00585 scene+neighbors before the fixed gate.
Percentage easy degradation is undefined because these rows have zero CV error;
we retain absolute harm and do not mark the ratio as a pass.

## Failure Mechanism

**The point-decision mismatch is real but insufficient.** Replacing the mean
substantially reduces Hotel damage under identical partitions. It does not make
the available features identify a useful displacement on the held scene.

**A safe zero decision can be optimal for the learned distribution and wrong
for the query distribution.** For all ETH queries, training leaf distributions
have enough mass at zero for every future step to make zero an optimal point
forecast. ETH nevertheless contains many changed labels. Optimizing the source
conditional distribution more accurately cannot remove that domain mismatch.

**Movement probability is not a direction forecast.** On Hotel, some conditional
zero optima arise even without a majority zero atom because nonzero directions
balance. The public table reports those counts without using held labels to
filter the forecasts. Start classification and useful trajectory prediction
remain separate requirements. This does not prove that all possible past-only
features lack information, only that these frozen learned distributions fail.

**Independent support is still missing.** The windows contain31 agents and45
runs at two sites. Repeated seeds, waypoints and numerical solves do not supply
additional independent scenes or validate a deployment rule.

## Numerical Qualification

The initial modified-Weiszfeld solver left23/39,420 waypoint computations above
its strict tolerance. Those results are preserved in `conditional_ade_probe`.
A separately registered training-only refinement found8 exact support-atom
solutions and certified14 further smooth solutions. One point remains above
the strict tolerance, with objective-gap bound2.293e-9 in scene-normalized
coordinates. It is explicitly approximate, not marked converged based on an
optimizer status. The refinement changes no positive/zero/negative conclusion;
maximum native-coordinate prediction change is3.122e-5. Do not claim that all
39,420 solutions have a strict convergence certificate.

All18 original receipts and predictions passed a resume/hash verification;
zero new fits or point recomputations occurred in that resume.23 focused tests
pass. The unchanged full legacy test suite was not rerun. The local point
computation took about16.1 seconds, followed by numerical refinement; there was
no long neural run or CREATE job, and none is claimed. Data/models remain local.

## Next Decision

Do not spend another architecture budget on the unchanged stationary feature
store or claim that a new loss alone repaired the model. The next candidate
needs verified past directional context and independent behavioral support, or
a different prospectively specified forecast hypothesis with sufficient data.
Past-aligned visual/body information cannot be admitted until its timing and
source lineage are established. The pending native-ADE/FDE-primary question is
separate; no metric change rescues these stationary-subset negative gains.

The broader M3W hypothesis remains unproven, no new policy is deployed, and the
submission goal is incomplete. Historical exposed results remain exploratory.
No true3D/foundation/metric/seconds/safety claim, Stage5C or SMC follows.
