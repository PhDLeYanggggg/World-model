# Frozen Harm Readout Results

## Material Passport
288 fresh readouts /576,000updates; frozen cached-verified encoders and original controls.
Previously opened source development; no new forecast, policy or independent confirmation.

| Comparison / pair / source assignment | Conditional easy-harm MSE gain (%) | 95% locality CI |
|---|---:|---:|
| fractional_vs_matched / full / producer0_controller1 | -0.14831616824575253 | [-1.8234862011429431, 0.9664718115967488] |
| fractional_vs_matched / full / producer0_controller2 | -0.14028075866645495 | [-5.76599104480932, 5.224694424298699] |
| fractional_vs_matched / full / producer1_controller0 | -0.04316283017823545 | [-1.20674596345345, 1.6329128696758277] |
| fractional_vs_matched / full / producer1_controller2 | 0.524835676325609 | [-0.5705253729425706, 2.0122383941395108] |
| fractional_vs_matched / full / producer2_controller0 | 2.3732787556425206 | [-0.2068195621013944, 5.707263327421174] |
| fractional_vs_matched / full / producer2_controller1 | -9.357219545427444 | [-24.61299182096342, 0.1286423863141815] |
| fractional_vs_matched / motion_only / producer0_controller1 | 5.80561253225394 | [-0.027755935705185246, 14.471316598354814] |
| fractional_vs_matched / motion_only / producer0_controller2 | -11.18938475043933 | [-28.225178907434426, -1.1100791408891728] |
| fractional_vs_matched / motion_only / producer1_controller0 | -14.505615196176086 | [-43.35010298350009, 0.13656146857155724] |
| fractional_vs_matched / motion_only / producer1_controller2 | -12.277242725562656 | [-36.29821902040625, 0.13725449777161092] |
| fractional_vs_matched / motion_only / producer2_controller0 | -2.6297416522354045 | [-7.737006305604476, -0.05889842604874107] |
| fractional_vs_matched / motion_only / producer2_controller1 | -2.6215894676485774 | [-9.14565207740558, 1.586746666325722] |
| fractional_vs_original / full / producer0_controller1 | -1.8322867587556506 | [-2.9374076327833722, -0.21416477337567358] |
| fractional_vs_original / full / producer0_controller2 | -39.78871491386628 | [-98.36790247977834, -0.49852599210087956] |
| fractional_vs_original / full / producer1_controller0 | -2.165148545924025 | [-4.173347784167255, -0.15694930768079415] |
| fractional_vs_original / full / producer1_controller2 | -3.334439149441885 | [-11.071491717157846, 0.8603155406769274] |
| fractional_vs_original / full / producer2_controller0 | 1.4111132475625827 | [-2.6728035814432527, 5.495030076568418] |
| fractional_vs_original / full / producer2_controller1 | -21.794626698746026 | [-64.37554558881003, 0.5192266144835618] |
| fractional_vs_original / motion_only / producer0_controller1 | -17.775353105757027 | [-46.30050250328662, -0.05537590156591324] |
| fractional_vs_original / motion_only / producer0_controller2 | -77.28768769658419 | [-217.79301141769798, -3.426341034023448] |
| fractional_vs_original / motion_only / producer1_controller0 | -54.41097399094268 | [-159.34168093791277, 0.34075159733085514] |
| fractional_vs_original / motion_only / producer1_controller2 | -51.653491189664024 | [-154.31435331246033, -0.08911398121439293] |
| fractional_vs_original / motion_only / producer2_controller0 | -13.276573322322783 | [-41.184412875940524, 1.10072856158709] |
| fractional_vs_original / motion_only / producer2_controller1 | -27.541442039728384 | [-64.97124601539178, -0.24000524543511195] |
| matched_vs_original / full / producer0_controller1 | -1.7273870360444028 | [-3.460827417466846, 0.00605334537804051] |
| matched_vs_original / full / producer0_controller2 | -44.4214624831376 | [-120.29335524331577, -0.6223498141176232] |
| matched_vs_original / full / producer1_controller0 | -2.138599745702282 | [-4.616008558377637, 0.20354893205419688] |
| matched_vs_original / full / producer1_controller2 | -3.8092522237071575 | [-12.932494128777712, 1.4296306117860305] |
| matched_vs_original / full / producer2_controller0 | -1.000658180816031 | [-2.5119533184090366, 1.1714404161472807] |
| matched_vs_original / full / producer2_controller1 | -5.7352691925732495 | [-19.49631979961228, 1.8905314149008503] |
| matched_vs_original / motion_only / producer0_controller1 | -32.17231700331407 | [-85.26855614143815, -0.028157512059557938] |
| matched_vs_original / motion_only / producer0_controller2 | -39.54372967690899 | [-112.75942204256685, -1.4541372707061293] |
| matched_vs_original / motion_only / producer1_controller0 | -29.206897653747465 | [-83.89873163712633, 0.2038809487786786] |
| matched_vs_original / motion_only / producer1_controller2 | -28.87599722365822 | [-86.45623909746689, 0.13217798568955444] |
| matched_vs_original / motion_only / producer2_controller0 | -9.290895048260227 | [-29.412367525769653, 1.2124644453173952] |
| matched_vs_original / motion_only / producer2_controller1 | -22.648235083461213 | [-50.450062221399584, 0.0658348956490461] |

Three seeds averaged within locality, then3,000 resamples of four localities.
Repeated roles/windows are not independent; intervals are exploratory, not multiplicity-adjusted.

## Gates

- training_complete: true
- reference_moments_preserved: true
- matched_primary_six_positive: false
- original_primary_six_positive: false
- tail_coverage_guards: true
- development_advance_gate: false
- new_policy_evaluated: false
- independent_confirmation: false
- deployment_changed: false
- submission_ready: false
- stage5c_executed: false
- smc_enabled: false

Full component errors, tail capture, coverage and membership partitions are retained in aggregate_metrics.json.
The nested easy-harm fraction is not a calibrated event probability. Reference-cost moments remain unchanged.
No metric/seconds, human-gold, physical safety, true3D or foundation claim. Stage5C/SMC off.
