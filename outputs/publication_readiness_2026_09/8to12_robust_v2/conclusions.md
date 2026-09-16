# Robust-Loss Ablation: Repair, Not Method Success

Date: 2026-09-16. Source: `fresh_run`, real arm64 CPU Torch fits and completed
development evaluation. This is adaptive research on historically exposed data,
not independent confirmation. No metric, seconds, 3D or foundation claim.

## Completed Work

All three registered seeds (17/29/43) completed full and scene-held forecasters,
identical-row OOF ridge/neural benefit-harm heads, and the frozen five-arm
development comparison. Fifteen neural fits each completed 1,000 updates; training
elapsed totals 983.24 seconds, excluding OOF construction and evaluation.
Together with v1 this is 30 neural fits and 30,000 updates, not an import check.
Code version `9e592089` contains the bound v2 training implementation. Original
MSE artifacts remain unchanged and replay at `707d4017`.

## What The Change Helped

Only forecaster MSE changed to Smooth-L1(beta=1). Same training/development rows,
architecture, budget, seeds, risk-head losses, easy definition, candidate policies
and primary metric. Mean uncontrolled primary gain versus CV changes from
-7.414% to -0.247%; seed SD changes from 0.428 to 0.075 percentage points.
Every seed has lower primary prediction error than its paired MSE run. This
supports objective sensitivity as one failure mechanism, not proof that it is
the only cause. Different training loss scales are not compared.

## What Still Failed

- Primary gains are -0.231%, -0.329%, -0.181%. No seed beats CV overall.
- All policy selections remain the CV floor. Learned interventions still damage
  the easy slice, whose floor ADE is only 0.0002578. Uncontrolled easy ADE is
  0.23084, 0.30096, 0.16906; the large percentage penalties are not hidden.
- Joint and independent controls still give identical errors. Pairwise coupling
  is not an established contribution. Shared budget caps do not prove matched
  realized-count superiority; the dedicated count-matched control is pending.
- Two-candidate future-label oracle headroom is only 0.320%, 0.351%, 0.231% on
  the frozen primary metric. Even a perfect chooser within this candidate set
  cannot attain a 5% primary gain. The oracle is never an inference feature.
- Students03 native-coordinate gains over CV are 6.27%, 8.18%, 4.15%, but a
  damped baseline already captures most of that benefit. Against the lowest
  development causal ADE, the three gains are -1.66%, +0.41%, -3.96%.
  Students01 has no such advantage. This descriptive comparison does not change
  the prespecified floor, select a test winner, or justify a favorable metric swap.

## Evidence Gates

| Evidence item | Status | Interpretation |
| --- | --- | --- |
| Real Torch fitting, three seeds, checkpoints | pass, fresh_run | Fixed budgets completed |
| Past-only interface and held-fold producer checks | checked | No old teacher used; not independent-test certification |
| Improvement versus failed MSE predictor | pass, development only | Objective repair helps all three seeds |
| Improvement versus CV and easy-safe intervention | fail | All selections remain floor |
| Added joint-decision benefit | fail in this comparison | Same result as independent selection |
| Actual-count-matched and deferral comparison | not_run | Interfaces exist, real experiment remains needed |
| Real public forecaster comparison | not_run | EqMotion-core adapter is not a trained benchmark result |
| Independent scene confirmation / scene CI | unavailable | One exposed University development site |
| Raw-frame t+50 supplement | not_run | Separate task; not inferred from 12 steps |
| Submission-ready contribution | fail | Not yet supported |
| Stage5C / SMC | disabled | No execution |

## Next Evidence-Producing Actions

1. Audit causal scale handling and fit a credible public candidate under a new,
   explicit protocol, retaining native-unit and frozen-metric sensitivity tables.
   Verify convergence and observed-neighbor support rather than adding more gates.
2. Once candidate headroom exists, run identical-OOF cost-sensitive deferral and
   exact-intervention-count controls. If joint choice still adds nothing, remove
   the joint contribution claim instead of tuning against a final set.
3. Admit sufficiently independent new scenes after source, quality, unit and
   exposure review. Existing ETH/UCY windows cannot become untouched by renaming.

This is still a trajectory-based developmental study, not a completed multimodal
world model. No new deployment is justified. Missing legal/source admission,
public-comparator and independent-confirmation evidence remains explicit.
