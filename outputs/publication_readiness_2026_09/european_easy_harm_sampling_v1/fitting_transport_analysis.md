# Fitting Versus Transport: Fixed-Action Diagnosis

These posthoc reductions do not alter fitting, actions, thresholds or model
selection. B inference is fresh_run from frozen heads; C reductions use
cached_verified frozen predictions. Data are in replay_receipt.json (B) and
matched_transport_audit.json (C). Cost and RMS scales remain B-only. C
equal-locality supported-row weights are evaluation weights, not fitted
preprocessing statistics.

Rows below summarize 18 dependent role/seed settings per forecast pair.
Ratios are corrected-sampling MSE divided by uniform-control MSE; below one
is better. The four outputs are D_all, H_all, D_easy, H_easy.

| Pair / population | Median D_all ratio | Median H_all ratio | Median D_easy ratio | Median H_easy ratio | H_easy improved /18 |
|---|---:|---:|---:|---:|---:|
| Full / B | 1.02060 | 1.08103 | 1.01152 | 0.99394 | 13 |
| Full / C | 1.00690 | 0.99134 | 1.00791 | 1.00877 | 4 |
| Motion / B | 1.02629 | 1.08284 | 1.01447 | 0.95746 | 18 |
| Motion / C | 1.00842 | 1.00929 | 1.01091 | 1.01539 | 3 |

The entire B population, not just a small diagnostic batch, shows the tradeoff.
Both reference components worsen in all 18 settings for each pair. Full H_all
worsens in all 18. Importance correction preserves the expected objective,
not the finite-budget optimization path, gradient variance or clipped Adam
updates. These results are consistent with a fitting tradeoff, not proof of
a unique gradient-conflict mechanism.

## Same Old-Raw Actions for Both Heads
To avoid comparing different selected populations, the existing raw-neural
action mask is fixed and shared. Median fitted / actual easy-harm mass:

| Pair / source | Uniform | Corrected |
|---|---:|---:|
| Full / B | 0.61847 | 0.60214 |
| Full / C | 0.58180 | 0.57828 |
| Motion / B | 0.57164 | 0.64236 |
| Motion / C | 0.44430 | 0.49566 |

Better population easy-harm MSE is not necessarily better selected-mass
estimation. In full C, fixed-action H_easy MSE improves in only 4/18 settings,
median ratio 1.00316. In motion C it improves in 3/18, median ratio 1.00571.
Motion mean-mass coverage improves, yet squared error and deployment risk do
not consistently improve. No guarantee follows from a single aggregate bias
measure.

On corrected-joint's own action sets, supported easy-harm prediction / actual
harm has median 0.43419 for full and 0.37326 for motion. Those action sets
differ from the fixed raw actions above; the ratios must not be presented as
a paired calibration gain. Underprediction occurs in 62/72 full and 54/72
motion dependent locality views.

## Failure Locality
Full fold1 / seed29 / controller2 at locality074 has actual selected easy
harm 1,742.36 against predicted supported harm 307.66. The observed ratio to
actual easy reference mass is 5.3292%; the predicted whole-query ratio is
0.8764%. Unknown futures supply only 1.4091% of predicted easy reference mass.
Missing support cannot plausibly explain this magnitude on its own. It remains
an observational diagnosis, not permission to use future availability in a
deployment rule.

These are image-pixel, annotation-step, detector-derived source-development
results. No independent certification, physical safety, metric/seconds,
human-gold, true3D or foundation claim. Stage5C and SMC remain off.
