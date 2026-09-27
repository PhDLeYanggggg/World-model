# Coordinate-Unit Sensitivity

Uniformly rescale ONLY observed geometry and causal rollouts, then undo scaling on predictions.
The observed-context clamp is inactive in every chosen case. CV/selected baseline equivariance
is checked. No labels, model selection or physically verified scale are involved.
The unchanged wrapper computes B(x) + R(x) * squash(S(x) * f(x/S(x))).
For positive rescaling c, the squash argument becomes c*S(x)*f, so the composed
forecaster need not scale linearly even when the normalized core inputs match.
This is a design sensitivity, not proof that it caused held forecast error or that
removing scale restoration would improve learning. Current frozen models are not changed.

| Trial | Arm | Input factor | Mean output change after undo | Maximum change | Approximately equivariant |
|---|---|---:|---:|---:|---|
| single0_seed17 | grouped | 0.25 | 0.815094 | 54.5318 | False |
| single0_seed17 | grouped | 0.5 | 0.52492 | 28.2076 | False |
| single0_seed17 | grouped | 2.0 | 0.920681 | 38.096 | False |
| single0_seed17 | grouped | 4.0 | 2.5849 | 70.8309 | False |
| single0_seed17 | flat | 0.25 | 0.984549 | 48.6068 | False |
| single0_seed17 | flat | 0.5 | 0.648667 | 28.473 | False |
| single0_seed17 | flat | 2.0 | 1.21105 | 24.7341 | False |
| single0_seed17 | flat | 4.0 | 3.43135 | 55.8145 | False |
| single0_seed29 | grouped | 0.25 | 0.698409 | 51.8246 | False |
| single0_seed29 | grouped | 0.5 | 0.455031 | 29.2708 | False |
| single0_seed29 | grouped | 2.0 | 0.817746 | 21.7686 | False |
| single0_seed29 | grouped | 4.0 | 2.31125 | 40.3541 | False |
| single0_seed29 | flat | 0.25 | 0.674782 | 50.6817 | False |
| single0_seed29 | flat | 0.5 | 0.441843 | 29.1176 | False |
| single0_seed29 | flat | 2.0 | 0.804601 | 22.9805 | False |
| single0_seed29 | flat | 4.0 | 2.2698 | 52.566 | False |
| single0_seed43 | grouped | 0.25 | 0.849236 | 45.4566 | False |
| single0_seed43 | grouped | 0.5 | 0.559282 | 27.4375 | False |
| single0_seed43 | grouped | 2.0 | 1.03792 | 26.5728 | False |
| single0_seed43 | grouped | 4.0 | 2.9306 | 59.0029 | False |
| single0_seed43 | flat | 0.25 | 1.0258 | 37.8928 | False |
| single0_seed43 | flat | 0.5 | 0.681254 | 23.5947 | False |
| single0_seed43 | flat | 2.0 | 1.31944 | 28.5001 | False |
| single0_seed43 | flat | 4.0 | 3.79328 | 55.8334 | False |
| single1_seed17 | grouped | 0.25 | 4.79177 | 249.973 | False |
| single1_seed17 | grouped | 0.5 | 3.09109 | 158.791 | False |
| single1_seed17 | grouped | 2.0 | 5.2061 | 205.796 | False |
| single1_seed17 | grouped | 4.0 | 12.9126 | 369.547 | False |
| single1_seed17 | flat | 0.25 | 5.37827 | 253.888 | False |
| single1_seed17 | flat | 0.5 | 3.48324 | 158.732 | False |
| single1_seed17 | flat | 2.0 | 5.85097 | 202.503 | False |
| single1_seed17 | flat | 4.0 | 14.328 | 384.474 | False |
| single1_seed29 | grouped | 0.25 | 7.08238 | 423.029 | False |
| single1_seed29 | grouped | 0.5 | 4.51673 | 236.801 | False |
| single1_seed29 | grouped | 2.0 | 6.93426 | 171.018 | False |
| single1_seed29 | grouped | 4.0 | 15.6532 | 247.788 | False |
| single1_seed29 | flat | 0.25 | 6.47549 | 472.101 | False |
| single1_seed29 | flat | 0.5 | 4.13207 | 234.867 | False |
| single1_seed29 | flat | 2.0 | 6.62924 | 153.283 | False |
| single1_seed29 | flat | 4.0 | 15.7146 | 291.635 | False |
| single1_seed43 | grouped | 0.25 | 6.93347 | 322.998 | False |
| single1_seed43 | grouped | 0.5 | 4.48318 | 197.91 | False |
| single1_seed43 | grouped | 2.0 | 7.3299 | 206.588 | False |
| single1_seed43 | grouped | 4.0 | 17.1409 | 323.257 | False |
| single1_seed43 | flat | 0.25 | 6.72904 | 394.633 | False |
| single1_seed43 | flat | 0.5 | 4.34531 | 236.63 | False |
| single1_seed43 | flat | 2.0 | 7.08883 | 222.217 | False |
| single1_seed43 | flat | 4.0 | 16.5139 | 334.688 | False |
| single2_seed17 | grouped | 0.25 | 5.1018 | 66.924 | False |
| single2_seed17 | grouped | 0.5 | 3.37521 | 42.3385 | False |
| single2_seed17 | grouped | 2.0 | 6.21164 | 53.7352 | False |
| single2_seed17 | grouped | 4.0 | 15.8249 | 113.464 | False |
| single2_seed17 | flat | 0.25 | 4.71676 | 79.4751 | False |
| single2_seed17 | flat | 0.5 | 3.12421 | 49.5406 | False |
| single2_seed17 | flat | 2.0 | 5.82454 | 56.906 | False |
| single2_seed17 | flat | 4.0 | 15.159 | 110.938 | False |
| single2_seed29 | grouped | 0.25 | 5.31426 | 54.8411 | False |
| single2_seed29 | grouped | 0.5 | 3.514 | 35.7134 | False |
| single2_seed29 | grouped | 2.0 | 6.43164 | 56.1703 | False |
| single2_seed29 | grouped | 4.0 | 16.2384 | 114.107 | False |
| single2_seed29 | flat | 0.25 | 5.77082 | 58.1066 | False |
| single2_seed29 | flat | 0.5 | 3.80821 | 37.6861 | False |
| single2_seed29 | flat | 2.0 | 6.84421 | 57.2236 | False |
| single2_seed29 | flat | 4.0 | 16.866 | 112.353 | False |
| single2_seed43 | grouped | 0.25 | 4.85983 | 80.0512 | False |
| single2_seed43 | grouped | 0.5 | 3.21386 | 50.0276 | False |
| single2_seed43 | grouped | 2.0 | 5.89662 | 58.3526 | False |
| single2_seed43 | grouped | 4.0 | 14.991 | 117.553 | False |
| single2_seed43 | flat | 0.25 | 4.73778 | 73.5181 | False |
| single2_seed43 | flat | 0.5 | 3.13499 | 46.5354 | False |
| single2_seed43 | flat | 2.0 | 5.78554 | 58.8703 | False |
| single2_seed43 | flat | 4.0 | 14.8415 | 121.354 | False |
