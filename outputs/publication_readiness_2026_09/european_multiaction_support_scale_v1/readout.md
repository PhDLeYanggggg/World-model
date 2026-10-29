# Support-Unit Repair: Collapse Fixed, Risk and Neural Increment Still Fail

Fresh_run native Torch optimization and full inference,29October2026.
This is a single-factor training-unit experiment, not a changed risk budget.
The original failed policy, first inference error and weight-preserving shape
repair remain separate historical evidence.

## Execution

- Real recovery pilot: PID66366,exit0,60updates,17.767055s; all three banks
  reproduce interrupted5+5 versus direct10states exactly.
- Full unchanged-unit neutral: PID67103,exit0,500updates,53.785976s; all
  scientific state matches the old endpoint except declared runtime/identity.
- Primary training: PID69018,exit0,54fits/27,000full-gradient updates,
  992.385098s,peakRSS5,794,103,296B;673parameters per head.
- All endpoints frozen in f90f7cd5 before first readout.
- First readout: PID73720,exit0,134.149339s,peakRSS6,958,874,624B.
- Exact replay: PID74298,exit0,137.335583s,peakRSS7,074,840,576B;
  all144groups and summary reproduce exactly.
- The1,512,476-byte aggregate bundle preserves exact numerical JSON bytes,
  not raw trajectories, private caches or weights.

## What the Controlled Change Establishes

The old mixed-bank training monitors have18/18empty fits and444/444violated
positive-support constraints. The new unit gives0/18empty fits and12/444
support violations. All/easy violations remain122/182; full monitor support
is228/444. These monitors precede the final optimizer update, so they are not
the endpoint inference numbers below.

The first-order numerical-pressure diagnosis was actionable: changing only
the support unit changes the learned behavior and removes wholesale collapse.
It was not the sole obstacle. Risk feasibility and utility remain unresolved.
The feasible set,2%all/easy budgets,model,inputs,optimizer,steps and action
banks are unchanged. The [mathematical note](support_units_math.md) states
both the invariant constraint and the limits of the diagnosis.

## Fixed Endpoint Results

All utility below is equal-locality known-label ADE gain over the frozen floor.
These are exposed SOURCE-development records, not independent confirmation.

| Bank | Fitting gain % | Inner-held gain % [conditional95%CI] | Held risk-supported record/head views |
|---|---:|---:|---:|
| Physical only |0.030225|0.053946 [0.013604,0.115866]|96/186|
| Physical+frozen neural |0.023490|0.034796 [0.002944,0.084866]|92/186|
| Frozen neural only |0.126972|0.149501 [0.053194,0.269304]|56/186|

The mixed-bank head-seed gains are0.022996,0.048555and0.032837%. Each context
retains all three head seeds; the forecaster remains frozen at seed43. There
are31unique inner-held source/recordings, not186independent videos.

The mixed policy makes8,279interventions,including77unknown-outcome rows.
Its known all-risk violations are34/186,and easy-risk violations41/186.
Fifteen easy-risk views remain undefined. It supports24/72source/head views
and92/186record/head views,not the complete required roster.

The actual fitting endpoint supports227/444record/head views,not all444.
This failure is already present on fitting recordings, before transfer to the
exposed inner-held set. More independent testing cannot repair that learning
failure; it is not evidence that the data contain no feasible oracle actions.

## Does the Neural Candidate Add Utility?

Mixed minus physical-only known utility is-0.019150pp,conditional3,000-locality
CI[-0.037903,-0.004219]. Its completion-lower contrast is-0.043270pp,
CI[-0.104130,-0.005320];the completion-upper contrast is+0.004970pp. Bounds
describe uncertainty from missing outcomes, not realized labels for those rows.

Matched known utility is+0.003608pp,CI[+0.000222,+0.008673],but matched
completion-lower is+0.000831pp,CI[-0.000661,+0.002283]. These tiny matched
utility comparisons do not pass safety or prove a robust learned neural increment.

Relative to the preceding mixed moment policy, known gain falls by0.518991pp,
CI[-1.210879,-0.131761]. Risk support improves from20/186to92/186at substantial
utility cost. This is an optimizer/safety tradeoff, not a deployable upgrade.

## Failure Taxonomy and Next Action

1. **Repaired:** weak scaling of the positive-support dual and wholesale empty
   action collapse. Real matched training confirms a behavioral change.
2. **Unresolved fitting feasibility:** the discrete shared decision head does
   not satisfy all record constraints even on its own fitting population.
3. **Unproven causal identifiability:** feasible label-aware choices need not
   be recoverable from the compressed causal descriptors and noisy moments.
4. **Unresolved utility/risk tradeoff:** selecting fewer actions improves some
   required slices but sacrifices most measured gain and still misses others.
5. **No neural increment:** mixed utility is lower than physical-only under
   the fixed full-policy contrast; oracle headroom cannot substitute for learning.

The next controlled work should first separate decision expressivity, moment
error and constrained optimizer behavior on TRAIN. Preserve the fixed bank and
risk contract; compare the compressed head to richer existing causal history/
rollout information before any independent-role opening. Do not simply tune
an exposed-test threshold, enlarge the forecaster, or declare population safety.

No policy is promoted. Independent selection,calibration and confirmation remain
unopened. This is not scene-joint inference,a new trajectory producer,metric/
effective-second validation,true3D,foundation success or submission readiness.
Stage5C execution andSMC remain off.
