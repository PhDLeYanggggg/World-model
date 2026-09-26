# Direct Easy-Membership Results

## Material Passport
288 fresh Torch classifiers; cached_verified source/producer lineage. Source development only.
No new forecast, cost model, policy evaluation or independent confirmation.

| Pair / roles / arm | Conditional Brier skill % [95% CI] | AUROC minus 0.5 [95% CI] | Log-loss gain [95% CI] |
|---|---:|---:|---:|
| full / producer0_controller1 / linear | 21.033624 [19.513012, 23.334484] | 0.312680 [0.292847, 0.325874] | 0.102262 [0.095678, 0.108845] |
| full / producer0_controller1 / mlp | 27.849972 [25.312354, 29.925117] | 0.346420 [0.322179, 0.364328] | 0.123658 [0.096486, 0.150831] |
| full / producer0_controller2 / linear | 32.140345 [25.478398, 38.697333] | 0.322662 [0.284689, 0.362449] | 0.090307 [-0.031118, 0.164958] |
| full / producer0_controller2 / mlp | 37.131762 [28.815327, 46.494629] | 0.345945 [0.322703, 0.372988] | 0.122629 [0.017284, 0.198788] |
| full / producer1_controller0 / linear | 29.445415 [26.928540, 32.022238] | 0.336742 [0.296922, 0.372425] | 0.140063 [0.121105, 0.166525] |
| full / producer1_controller0 / mlp | 35.702694 [32.639273, 38.766114] | 0.355109 [0.315312, 0.387001] | 0.180051 [0.133563, 0.217210] |
| full / producer1_controller2 / linear | 32.612614 [26.853022, 38.372206] | 0.329991 [0.296042, 0.363939] | 0.145255 [0.126550, 0.165960] |
| full / producer1_controller2 / mlp | 40.273574 [33.335507, 47.211641] | 0.358088 [0.334432, 0.381744] | 0.157008 [0.077121, 0.214074] |
| full / producer2_controller0 / linear | 34.955055 [29.322299, 42.025480] | 0.325883 [0.288313, 0.373935] | 0.150063 [0.113043, 0.215116] |
| full / producer2_controller0 / mlp | 46.308318 [42.674880, 49.144104] | 0.338194 [0.290594, 0.383475] | 0.218180 [0.171575, 0.260667] |
| full / producer2_controller1 / linear | 40.311934 [35.103880, 45.519989] | 0.388555 [0.368661, 0.409691] | 0.162232 [0.127464, 0.202056] |
| full / producer2_controller1 / mlp | 52.603800 [47.763243, 58.444727] | 0.396374 [0.374846, 0.413208] | 0.219712 [0.194141, 0.242216] |
| motion_only / producer0_controller1 / linear | 41.104461 [29.764708, 52.444214] | 0.141391 [0.051560, 0.207217] | 0.106572 [0.067882, 0.138611] |
| motion_only / producer0_controller1 / mlp | 56.112451 [35.163449, 77.061454] | 0.231497 [0.151778, 0.311217] | 0.152938 [0.080663, 0.225213] |
| motion_only / producer0_controller2 / linear | 44.424626 [11.411298, 62.284324] | 0.142027 [0.118189, 0.167541] | 0.134166 [-0.020074, 0.214437] |
| motion_only / producer0_controller2 / mlp | 51.540169 [10.480852, 75.012619] | 0.125836 [0.087203, 0.168536] | 0.116570 [-0.161108, 0.263955] |
| motion_only / producer1_controller0 / linear | 34.081064 [13.727629, 45.029626] | 0.175548 [0.100374, 0.246828] | 0.087684 [-0.057902, 0.172071] |
| motion_only / producer1_controller0 / mlp | 48.786166 [25.327570, 68.547516] | 0.191671 [0.155840, 0.235153] | 0.180529 [0.081889, 0.287119] |
| motion_only / producer1_controller2 / linear | 39.840285 [2.545129, 61.040268] | 0.136181 [0.113213, 0.161424] | 0.106238 [-0.081582, 0.203722] |
| motion_only / producer1_controller2 / mlp | 52.145566 [13.018564, 75.027454] | 0.146891 [0.101010, 0.198414] | 0.118525 [-0.159337, 0.267093] |
| motion_only / producer2_controller0 / linear | 22.524351 [14.033826, 31.282185] | 0.193778 [0.132225, 0.251610] | 0.087678 [0.045378, 0.116024] |
| motion_only / producer2_controller0 / mlp | 36.525786 [23.503221, 49.548350] | 0.215033 [0.168565, 0.261501] | 0.143846 [0.106019, 0.200847] |
| motion_only / producer2_controller1 / linear | 18.548699 [15.563160, 21.773626] | 0.199071 [0.160668, 0.244359] | 0.059470 [0.046437, 0.070578] |
| motion_only / producer2_controller1 / mlp | 41.782965 [30.408581, 53.157349] | 0.219359 [0.190157, 0.248353] | 0.128675 [0.082881, 0.174468] |

Three seeds are averaged within locality; 3,000 resamples of four localities per assignment.
Repeated source roles/windows are dependent. Exploratory CIs, no multiplicity adjustment.
The reference-cost ratio is not a calibrated event probability; its proper-score diagnostics do not make it one.
Brier skill compares with a train-prevalence constant, not a constant refitted on held labels.
The inherited easy definition excludes exact-zero CV errors.

A separately declared pre-readout sensitivity also uses training prevalence conditional on positive disagreement.
It is a stronger prevalence control, not a revised primary gate or held-label fit. Its complete contrasts are in aggregate_metrics.json.

## Gates

```json
{
  "arms": {
    "linear": {
      "Brier_skill_percent": {
        "positive": 6,
        "negative": 0,
        "not_estimable": 0
      },
      "AUROC_above_chance": {
        "positive": 6,
        "negative": 0,
        "not_estimable": 0
      },
      "log_loss_gain": {
        "positive": 5,
        "negative": 0,
        "not_estimable": 0
      },
      "membership_signal_gate": false
    },
    "mlp": {
      "Brier_skill_percent": {
        "positive": 6,
        "negative": 0,
        "not_estimable": 0
      },
      "AUROC_above_chance": {
        "positive": 6,
        "negative": 0,
        "not_estimable": 0
      },
      "log_loss_gain": {
        "positive": 6,
        "negative": 0,
        "not_estimable": 0
      },
      "membership_signal_gate": true
    }
  },
  "membership_signal_established": true,
  "policy_advance": false,
  "deployment_changed": false,
  "independent_confirmation": false,
  "submission_ready": false,
  "stage5c_executed": false,
  "smc_enabled": false
}
```

Image pixels and annotation steps, detector-derived labels; no metric/seconds, human-gold, physical safety, true3D or foundation claim.
Independent calibration/confirmation remain closed. Stage5C/SMC off.
