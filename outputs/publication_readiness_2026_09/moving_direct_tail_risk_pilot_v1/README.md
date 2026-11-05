# Direct Tail Risk: Completed Model Repair Pilot

I kept the original evaluation rules and tested a model-side change: predict
signed all/easy harm-budget excess directly, with an unbounded linear output.
The two matched losses are mean squared error and fixed .9 quantile pinball.
Both models retain causal context, the source-excluded router and original 2%
budget. A predicted quantile is not a calibrated safety certificate.

## Evidence Scope

fresh_run: six real neural heads,12,000 primary optimizer updates;400 resume-
smoke updates and an independent2,000-update full refit are additional checks.
Training phase completed in456.69s, peak RSS6.408GB.
cached_verified: original forecasts, own-source preprocessing and frozen policy
controls. All three new inference/readout blocks and cached controls replay.
not_run: full108-context-direction comparison, additional seeds, independent H,
new trajectory training, or deployment. The three fixed contexts all use seed17;
they are not three independent training seeds. There are7SOURCE
units and7source views. The table is source-macro descriptive,
not an independent-test confidence interval. No favorable view is removed.

## Results

| Policy | Known gain vs floor (%) | Hard gain (%) | Switch (%) | Supported / required | All violations | Easy violations | Undefined selected easy |
|---|---:|---:|---:|---:|---:|---:|---:|
| public_router | 14.496012 | 18.306013 | 62.383468 | 0/17 | 15 | 17 | 0 |
| all_action_veto | 5.532699 | 6.013297 | 28.542124 | 1/17 | 14 | 15 | 0 |
| conditioned_veto | 5.331845 | 6.532826 | 31.794134 | 1/17 | 14 | 16 | 0 |
| mean_veto | 4.091202 | 5.035472 | 21.307681 | 0/17 | 13 | 15 | 0 |
| quantile90_veto | 0.151484 | 0.218176 | 1.519389 | 2/17 | 10 | 9 | 4 |

| Paired strategy | Known (pp) | Completion lower (pp) | Completion upper (pp) |
|---|---:|---:|---:|
| mean_minus_conditioned | -1.240643 | -2.354583 | -0.126703 |
| quantile_minus_conditioned | -5.180360 | -6.729201 | -3.631519 |
| quantile_minus_mean | -3.939718 | -5.244448 | -2.634988 |

The prespecified pilot expansion conditions fail. This loss variant is not promoted to a full study or deployment.
Nonempty matched primary unavailable: mean0,
quantile2source views. The old24group
primary remainsNA, not passed. Full-strategy contrasts in the table do not
replace that primary. Positive-harm/support and signed easy-net remain distinct.

See[protocol](protocol.md),[all machine-readable results](analysis.json),
[failure analysis](failure_analysis.md),[reproducibility](reproducibility.md),
and[fit audit](fit_audit.json). Deployment and submission readiness remain false.
obs8/pred12 native annotation order, local-image pixels, non-human-gold labels;
not metric, verified seconds, SDDt50, true3D or foundation. Stage5C/SMC off.
