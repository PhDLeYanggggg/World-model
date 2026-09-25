# Aligned Source-C Calibration Results

## Material Passport
Fresh source-C inference and calibration; cached_verified models; fresh frozen development readout.
Three seeds,72 maps,288 policy views. Six reused opened model-selection localities, not confirmation.

| Pair / scorer / rule | All gain vs own raw | Hard gain vs own raw | All gain vs full neural raw | Worst easy degradation | Easy passes | Complete observed risk passes | C risk passes | Abstention maps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full__neural__none | +0.000000% to +0.000000% | +0.000000% to +0.000000% | +0.000000% to +0.000000% | +0.000000% to +0.730345% | 18/18 | 0/18 | 5/18 | 0/18 |
| full__neural__population_rescale | -2.842483% to -0.252543% | -4.088038% to -0.294660% | -2.842483% to -0.252543% | +0.000000% to +0.000000% | 18/18 | 2/18 | 13/18 | 0/18 |
| full__neural__reference | -6.084049% to -4.245729% | -6.810174% to -4.638224% | -6.084049% to -4.245729% | +0.000000% to +0.000000% | 18/18 | 18/18 | 18/18 | 18/18 |
| full__neural__selected_risk_grid | -1.697488% to +0.000373% | -2.081725% to +0.000000% | -1.697488% to +0.000373% | +0.000000% to +0.379671% | 18/18 | 0/18 | 18/18 | 0/18 |
| full__ridge__none | +0.000000% to +0.000000% | +0.000000% to +0.000000% | -3.865930% to +0.269596% | +0.000000% to +0.000000% | 18/18 | 4/18 | 12/18 | 0/18 |
| full__ridge__population_rescale | -4.005953% to +0.000000% | -4.691332% to +0.000000% | -3.974741% to +0.139525% | +0.000000% to +0.000000% | 18/18 | 5/18 | 16/18 | 0/18 |
| full__ridge__reference | -6.143691% to -1.127701% | -7.113259% to -1.304649% | -6.084049% to -4.245729% | +0.000000% to +0.000000% | 18/18 | 18/18 | 18/18 | 18/18 |
| full__ridge__selected_risk_grid | -0.924980% to +0.000000% | -0.963537% to +0.000000% | -3.865930% to +0.116132% | +0.000000% to +0.000000% | 18/18 | 4/18 | 18/18 | 0/18 |
| motion_only__neural__none | +0.000000% to +0.000000% | +0.000000% to +0.000000% | -0.697843% to +0.818339% | +0.000000% to +0.000000% | 18/18 | 0/18 | 6/18 | 0/18 |
| motion_only__neural__population_rescale | -2.243627% to -0.244156% | -2.468379% to -0.293069% | -2.712110% to +0.281175% | +0.000000% to +0.000000% | 18/18 | 6/18 | 14/18 | 0/18 |
| motion_only__neural__reference | -5.814705% to -4.904682% | -6.465927% to -5.281185% | -6.292869% to -4.386211% | +0.000000% to +0.000000% | 18/18 | 18/18 | 18/18 | 18/18 |
| motion_only__neural__selected_risk_grid | -1.818019% to +0.000000% | -1.966712% to +0.000000% | -2.285182% to +0.252842% | +0.000000% to +0.000000% | 18/18 | 0/18 | 18/18 | 0/18 |
| motion_only__ridge__none | +0.000000% to +0.000000% | +0.000000% to +0.000000% | -2.315203% to +0.094026% | +0.000000% to +0.000000% | 18/18 | 3/18 | 12/18 | 0/18 |
| motion_only__ridge__population_rescale | -3.159385% to +0.000000% | -3.712734% to +0.000000% | -3.838905% to -0.001988% | +0.000000% to +0.000000% | 18/18 | 10/18 | 16/18 | 0/18 |
| motion_only__ridge__reference | -5.791640% to -3.597458% | -6.599089% to -3.871500% | -6.292869% to -4.386211% | +0.000000% to +0.000000% | 18/18 | 18/18 | 18/18 | 18/18 |
| motion_only__ridge__selected_risk_grid | -0.376237% to +0.000000% | -0.308238% to +0.000000% | -2.315203% to +0.094026% | +0.000000% to +0.000000% | 18/18 | 3/18 | 18/18 | 0/18 |

Complete observed risk means both positive-harm/R-error events, net CV-easy <=2%,
and no CV-zero harm. Missing support cannot pass. Structural fallback is not positive gain.
Ranges contain all source-role/seed groups, not chosen winners. Predicted ratios in group
JSON use all indexed rows and raw moments. moment_transport.json additionally reports
matched-label-support raw/adjusted moments so unknown future support is not silently mixed.

## Three-Seed All-ADE Contrasts

Seedwise gains averaged within locality, then3,000 paired locality resamples.
No row independence or multiplicity-adjusted discovery claim.

| Producer/controller | Contrast | Gain | 95% locality CI |
|---|---|---:|---:|
| producer0_controller1 | full__neural__population_rescale_vs_none__all | -2.371004% | -3.178838% to -1.597275% |
| producer0_controller1 | full__neural__selected_risk_grid_vs_none__all | -0.733988% | -1.013319% to -0.454658% |
| producer0_controller1 | full__neural_vs_ridge__none__all | -0.076226% | -0.380462% to +0.186595% |
| producer0_controller1 | full__neural_vs_ridge__population_rescale__all | +1.181473% | +0.680072% to +1.709766% |
| producer0_controller1 | full__neural_vs_ridge__selected_risk_grid__all | -0.470254% | -0.966736% to -0.003707% |
| producer0_controller1 | full__ridge__population_rescale_vs_none__all | -3.689009% | -5.210218% to -2.438570% |
| producer0_controller1 | full__ridge__selected_risk_grid_vs_none__all | -0.341911% | -0.434094% to -0.244587% |
| producer0_controller1 | motion_only__neural__population_rescale_vs_none__all | -1.990906% | -2.932264% to -1.169631% |
| producer0_controller1 | motion_only__neural__selected_risk_grid_vs_none__all | -1.023678% | -1.516308% to -0.603597% |
| producer0_controller1 | motion_only__neural_vs_ridge__none__all | -0.036198% | -0.194754% to +0.117178% |
| producer0_controller1 | motion_only__neural_vs_ridge__population_rescale__all | +0.909998% | +0.503262% to +1.326615% |
| producer0_controller1 | motion_only__neural_vs_ridge__selected_risk_grid__all | -0.884185% | -1.433277% to -0.381816% |
| producer0_controller1 | motion_only__ridge__population_rescale_vs_none__all | -2.975500% | -4.448772% to -1.770232% |
| producer0_controller1 | motion_only__ridge__selected_risk_grid_vs_none__all | -0.174224% | -0.277221% to -0.100848% |
| producer0_controller2 | full__neural__population_rescale_vs_none__all | -0.575852% | -0.736227% to -0.415515% |
| producer0_controller2 | full__neural__selected_risk_grid_vs_none__all | -0.623125% | -0.862215% to -0.416509% |
| producer0_controller2 | full__neural_vs_ridge__none__all | +2.700990% | +1.955858% to +3.567825% |
| producer0_controller2 | full__neural_vs_ridge__population_rescale__all | +2.141478% | +1.506608% to +2.864253% |
| producer0_controller2 | full__neural_vs_ridge__selected_risk_grid__all | +2.098397% | +1.506890% to +2.792969% |
| producer0_controller2 | full__ridge__population_rescale_vs_none__all | +0.000000% | +0.000000% to +0.000000% |
| producer0_controller2 | full__ridge__selected_risk_grid_vs_none__all | +0.000000% | +0.000000% to +0.000000% |
| producer0_controller2 | motion_only__neural__population_rescale_vs_none__all | -0.424275% | -0.616367% to -0.251147% |
| producer0_controller2 | motion_only__neural__selected_risk_grid_vs_none__all | -0.370702% | -0.594891% to -0.188552% |
| producer0_controller2 | motion_only__neural_vs_ridge__none__all | +1.468537% | +1.138014% to +1.906989% |
| producer0_controller2 | motion_only__neural_vs_ridge__population_rescale__all | +1.051599% | +0.870085% to +1.317471% |
| producer0_controller2 | motion_only__neural_vs_ridge__selected_risk_grid__all | +1.105430% | +0.934083% to +1.339019% |
| producer0_controller2 | motion_only__ridge__population_rescale_vs_none__all | +0.000000% | +0.000000% to +0.000000% |
| producer0_controller2 | motion_only__ridge__selected_risk_grid_vs_none__all | +0.000000% | +0.000000% to +0.000000% |
| producer1_controller0 | full__neural__population_rescale_vs_none__all | -2.378341% | -3.145981% to -1.615413% |
| producer1_controller0 | full__neural__selected_risk_grid_vs_none__all | -0.324044% | -0.490475% to -0.127878% |
| producer1_controller0 | full__neural_vs_ridge__none__all | +1.039894% | +0.492807% to +1.606978% |
| producer1_controller0 | full__neural_vs_ridge__population_rescale__all | +1.251562% | +0.678936% to +1.842251% |
| producer1_controller0 | full__neural_vs_ridge__selected_risk_grid__all | +0.931561% | +0.482824% to +1.420406% |
| producer1_controller0 | full__ridge__population_rescale_vs_none__all | -2.601460% | -3.679538% to -1.603348% |
| producer1_controller0 | full__ridge__selected_risk_grid_vs_none__all | -0.214295% | -0.291112% to -0.143845% |
| producer1_controller0 | motion_only__neural__population_rescale_vs_none__all | -1.031860% | -1.473517% to -0.622443% |
| producer1_controller0 | motion_only__neural__selected_risk_grid_vs_none__all | -0.036571% | -0.164475% to +0.109976% |
| producer1_controller0 | motion_only__neural_vs_ridge__none__all | +0.103199% | -0.275744% to +0.408112% |
| producer1_controller0 | motion_only__neural_vs_ridge__population_rescale__all | +1.020402% | +0.641654% to +1.384762% |
| producer1_controller0 | motion_only__neural_vs_ridge__selected_risk_grid__all | +0.067607% | -0.177158% to +0.253726% |
| producer1_controller0 | motion_only__ridge__population_rescale_vs_none__all | -1.969161% | -2.449501% to -1.415635% |
| producer1_controller0 | motion_only__ridge__selected_risk_grid_vs_none__all | +0.000000% | +0.000000% to +0.000000% |
| producer1_controller2 | full__neural__population_rescale_vs_none__all | -0.433005% | -0.587511% to -0.315501% |
| producer1_controller2 | full__neural__selected_risk_grid_vs_none__all | -1.378161% | -1.867201% to -0.992208% |
| producer1_controller2 | full__neural_vs_ridge__none__all | +3.099757% | +2.282385% to +3.935878% |
| producer1_controller2 | full__neural_vs_ridge__population_rescale__all | +2.681339% | +1.945655% to +3.457125% |
| producer1_controller2 | full__neural_vs_ridge__selected_risk_grid__all | +1.770001% | +1.209095% to +2.457075% |
| producer1_controller2 | full__ridge__population_rescale_vs_none__all | +0.000000% | +0.000000% to +0.000000% |
| producer1_controller2 | full__ridge__selected_risk_grid_vs_none__all | +0.000000% | +0.000000% to +0.000000% |
| producer1_controller2 | motion_only__neural__population_rescale_vs_none__all | -0.413906% | -0.683398% to -0.203473% |
| producer1_controller2 | motion_only__neural__selected_risk_grid_vs_none__all | -0.789833% | -1.312422% to -0.371916% |
| producer1_controller2 | motion_only__neural_vs_ridge__none__all | +1.364644% | +1.073870% to +1.802384% |
| producer1_controller2 | motion_only__neural_vs_ridge__population_rescale__all | +0.957966% | +0.738585% to +1.191955% |
| producer1_controller2 | motion_only__neural_vs_ridge__selected_risk_grid__all | +0.588515% | +0.317876% to +0.801745% |
| producer1_controller2 | motion_only__ridge__population_rescale_vs_none__all | +0.000000% | +0.000000% to +0.000000% |
| producer1_controller2 | motion_only__ridge__selected_risk_grid_vs_none__all | +0.000000% | +0.000000% to +0.000000% |
| producer2_controller0 | full__neural__population_rescale_vs_none__all | -0.639964% | -0.971305% to -0.387757% |
| producer2_controller0 | full__neural__selected_risk_grid_vs_none__all | +0.000124% | -0.003973% to +0.006484% |
| producer2_controller0 | full__neural_vs_ridge__none__all | -0.064680% | -0.146215% to -0.007911% |
| producer2_controller0 | full__neural_vs_ridge__population_rescale__all | -0.628532% | -0.997782% to -0.380622% |
| producer2_controller0 | full__neural_vs_ridge__selected_risk_grid__all | -0.064556% | -0.148215% to -0.004179% |
| producer2_controller0 | full__ridge__population_rescale_vs_none__all | -0.076221% | -0.120658% to -0.034341% |
| producer2_controller0 | full__ridge__selected_risk_grid_vs_none__all | +0.000000% | +0.000000% to +0.000000% |
| producer2_controller0 | motion_only__neural__population_rescale_vs_none__all | -0.604964% | -0.840417% to -0.419046% |
| producer2_controller0 | motion_only__neural__selected_risk_grid_vs_none__all | +0.000000% | +0.000000% to +0.000000% |
| producer2_controller0 | motion_only__neural_vs_ridge__none__all | +0.052787% | +0.010222% to +0.097965% |
| producer2_controller0 | motion_only__neural_vs_ridge__population_rescale__all | -0.517006% | -0.707812% to -0.330194% |
| producer2_controller0 | motion_only__neural_vs_ridge__selected_risk_grid__all | +0.052787% | +0.010222% to +0.097965% |
| producer2_controller0 | motion_only__ridge__population_rescale_vs_none__all | -0.034554% | -0.065474% to -0.009023% |
| producer2_controller0 | motion_only__ridge__selected_risk_grid_vs_none__all | +0.000000% | +0.000000% to +0.000000% |
| producer2_controller1 | full__neural__population_rescale_vs_none__all | -1.132940% | -1.470086% to -0.807393% |
| producer2_controller1 | full__neural__selected_risk_grid_vs_none__all | -0.546205% | -0.711853% to -0.386485% |
| producer2_controller1 | full__neural_vs_ridge__none__all | -0.145470% | -0.216996% to -0.076480% |
| producer2_controller1 | full__neural_vs_ridge__population_rescale__all | -1.029349% | -1.489208% to -0.608895% |
| producer2_controller1 | full__neural_vs_ridge__selected_risk_grid__all | -0.152176% | -0.461677% to +0.092806% |
| producer2_controller1 | full__ridge__population_rescale_vs_none__all | -0.249708% | -0.377387% to -0.133198% |
| producer2_controller1 | full__ridge__selected_risk_grid_vs_none__all | -0.540602% | -0.744133% to -0.369343% |
| producer2_controller1 | motion_only__neural__population_rescale_vs_none__all | -1.197869% | -1.734379% to -0.679972% |
| producer2_controller1 | motion_only__neural__selected_risk_grid_vs_none__all | -0.613921% | -0.916187% to -0.328113% |
| producer2_controller1 | motion_only__neural_vs_ridge__none__all | +0.005349% | -0.082620% to +0.086301% |
| producer2_controller1 | motion_only__neural_vs_ridge__population_rescale__all | -1.059291% | -1.656914% to -0.461821% |
| producer2_controller1 | motion_only__neural_vs_ridge__selected_risk_grid__all | -0.501075% | -0.896240% to -0.140366% |
| producer2_controller1 | motion_only__ridge__population_rescale_vs_none__all | -0.132579% | -0.170956% to -0.089772% |
| producer2_controller1 | motion_only__ridge__selected_risk_grid_vs_none__all | -0.107318% | -0.129643% to -0.084023% |

## Within-C Stability Diagnostic

| Pair/scorer | Three-source fit, held-C feasible | Abstentions |
|---|---:|---:|
| full__neural | 59/72 | 0 |
| full__ridge | 66/72 | 0 |
| motion_only__neural | 60/72 | 0 |
| motion_only__ridge | 66/72 | 0 |

These views overlap and never select the final four-C map. Twelve reserved calibration
and six confirmation localities remain closed. No deployment, metric/seconds/physical-safety,
true3D/foundation/human-gold claim, Stage5C or SMC.
