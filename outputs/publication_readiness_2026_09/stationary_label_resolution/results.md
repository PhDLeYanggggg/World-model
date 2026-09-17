# Stationary Label Resolution Audit

Fresh fit-only source/projection audit and frozen-forecast replay; no models refitted or labels changed.
Inferred pixels are computed under supplied H, not verified image/annotation synchronization.

## Source Precision and Support

| Fit source | Stationary windows | Changed | Agents | Runs | Changed compatible with text rounding | Returned to origin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| eth_eth | 81 | 59 | 5 | 11 | 0 | 0 |
| eth_hotel | 284 | 129 | 26 | 34 | 0 | 37 |

Closed decimal intervals represent printed precision only, not measurement uncertainty.
A non-overlap rules out a constant value explainable by that serialization precision alone.
It does not prove physical movement, annotation accuracy or identifiable human intent.

## Fixed Inferred-Image Sensitivity

Above means maximum displacement > threshold +0.001 inferred pixels. Boundary counts expose arithmetic sensitivity.
Cuts are label-side descriptions, not selected targets, physical thresholds or allowed deployment filters.

| Source | Threshold | Rows above | Agents | Runs | CV error share above | Boundary rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| eth_eth | 0.5 | 59 | 5 | 11 | 100.0000% | 0 |
| eth_eth | 1 | 56 | 5 | 11 | 99.9310% | 3 |
| eth_eth | 2 | 45 | 5 | 10 | 98.2450% | 1 |
| eth_eth | 5 | 35 | 3 | 7 | 94.2584% | 0 |
| eth_eth | 10 | 31 | 3 | 7 | 92.5779% | 0 |
| eth_hotel | 0.5 | 129 | 19 | 25 | 100.0000% | 0 |
| eth_hotel | 1 | 119 | 16 | 22 | 98.5653% | 10 |
| eth_hotel | 2 | 112 | 15 | 20 | 96.9591% | 2 |
| eth_hotel | 5 | 56 | 10 | 14 | 81.2861% | 17 |
| eth_hotel | 10 | 37 | 7 | 9 | 68.4603% | 0 |

## Every Frozen Regressor on Fixed Slices

Seed-mean native-ADE gain versus CV, computed separately by recording. Empty/zero-CV slices are null, not passes.
Still-row absolute native harm is retained. Returned-origin rows overlap the magnitude bins.

| Held | Features | Family | Arm | Still harm | <=0.5 px gain | 0.5--1.5 px gain | 1.5--5 px gain | >5 px gain |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| ETH | pooled | linear | unrestricted | +0.021830 | null | -84.263608 | -6.677905 | -2.917528 |
| ETH | pooled | linear | fixed_gate | +0.000000 | null | +0.000000 | +0.000000 | +0.000000 |
| ETH | pooled | extra_trees | unrestricted | +0.009794 | null | -30.940682 | -6.748115 | -1.530123 |
| ETH | pooled | extra_trees | fixed_gate | +0.000000 | null | +0.000000 | +0.000000 | +0.000000 |
| ETH | scene | linear | unrestricted | +0.052863 | null | -169.394177 | -37.478334 | -9.998446 |
| ETH | scene | linear | fixed_gate | +0.000000 | null | +0.000000 | -7.027999 | +0.000000 |
| ETH | scene | extra_trees | unrestricted | +0.018157 | null | -56.054119 | -13.108073 | -3.672254 |
| ETH | scene | extra_trees | fixed_gate | +0.000000 | null | +0.000000 | +0.000000 | +0.000000 |
| ETH | scene_neighbor | linear | unrestricted | +0.053805 | null | -179.847447 | -38.551535 | -5.189714 |
| ETH | scene_neighbor | linear | fixed_gate | +0.000000 | null | +0.000000 | +0.000000 | +0.000000 |
| ETH | scene_neighbor | extra_trees | unrestricted | +0.009748 | null | -30.766199 | -8.467563 | -2.587504 |
| ETH | scene_neighbor | extra_trees | fixed_gate | +0.000000 | null | +0.000000 | +0.000000 | +0.000000 |
| Hotel | pooled | linear | unrestricted | +0.317108 | null | -2644.691486 | -1583.437947 | -246.372569 |
| Hotel | pooled | linear | fixed_gate | +0.150311 | null | -905.064467 | -1035.699595 | -105.258828 |
| Hotel | pooled | extra_trees | unrestricted | +0.084912 | null | -684.803660 | -395.060988 | -55.845863 |
| Hotel | pooled | extra_trees | fixed_gate | +0.026213 | null | -282.241990 | -68.512896 | -10.361444 |
| Hotel | scene | linear | unrestricted | +0.370020 | null | -2890.666986 | -2136.429245 | -322.029236 |
| Hotel | scene | linear | fixed_gate | +0.084270 | null | -1224.037203 | -1197.911836 | -100.722230 |
| Hotel | scene | extra_trees | unrestricted | +0.050283 | null | -599.541998 | -245.479393 | -33.433483 |
| Hotel | scene | extra_trees | fixed_gate | +0.000516 | null | +0.000000 | +0.000000 | +0.000000 |
| Hotel | scene_neighbor | linear | unrestricted | +0.425076 | null | -3519.677100 | -2355.724643 | -495.495734 |
| Hotel | scene_neighbor | linear | fixed_gate | +0.114726 | null | -1594.186284 | -1137.611601 | -144.769409 |
| Hotel | scene_neighbor | extra_trees | unrestricted | +0.060347 | null | -605.863584 | -293.941416 | -40.034143 |
| Hotel | scene_neighbor | extra_trees | fixed_gate | +0.000382 | null | -3.005105 | +0.000000 | +0.144434 |

All72 corrected models were hash-verified and their predictions replayed within1e-12.
No settings or labels were selected. No independent-scene significance claim is made.
Detailed decimal precision, inferred lattice residuals, quantiles and all seed/slice metrics are in audit.json.
Raw source rows, per-row projections and checkpoints stay local. No metric/seconds, Stage5C or SMC claim.
