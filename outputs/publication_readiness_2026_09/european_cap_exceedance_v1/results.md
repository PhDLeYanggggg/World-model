# Cap-Event Learnability Results

## Material Passport
fresh_run:288 native-Torch heads,576000 fixed updates and144 source-held readouts.
cached_verified:frozen source forecasters, inner OOF risk teachers and outer risk estimators.
not_run:new trajectory training, policy evaluation and independent selection/calibration/confirmation.

| Inputs / arm / contrast | Positive / negative / overlap / missing intervals | Point range |
|---|---|---|
| full/linear/AP_gain_envelope | [6, 0, 0, 0] | [0.22772985288090036, 0.3265085687987778] |
| full/linear/AP_gain_frozen_easy_fraction | [6, 0, 0, 0] | [0.034418568127993106, 0.27851177982814423] |
| full/linear/AP_gain_frozen_harm_fraction | [6, 0, 0, 0] | [0.058585859215022615, 0.2946589307258512] |
| full/linear/AP_gain_linear | [0, 0, 6, 0] | [0.0, 0.0] |
| full/linear/BCE_gain_linear | [0, 0, 6, 0] | [0.0, 0.0] |
| full/linear/BCE_gain_prior | [6, 0, 0, 0] | [0.03458262804307585, 0.12157075278734951] |
| full/linear/Brier_gain_prior | [6, 0, 0, 0] | [0.006571689927998331, 0.026170972232354366] |
| full/linear/capture_gain_envelope | [4, 0, 2, 0] | [0.10318335604690848, 0.4784536718572653] |
| full/mlp/AP_gain_envelope | [6, 0, 0, 0] | [0.22279868175317857, 0.3426018281302655] |
| full/mlp/AP_gain_frozen_easy_fraction | [6, 0, 0, 0] | [0.02948739700027133, 0.25990679281030266] |
| full/mlp/AP_gain_frozen_harm_fraction | [6, 0, 0, 0] | [0.05365468808730084, 0.27626506085436703] |
| full/mlp/AP_gain_linear | [2, 1, 3, 0] | [-0.018604987017841634, 0.022291871130160023] |
| full/mlp/BCE_gain_linear | [0, 0, 6, 0] | [-0.04962939938060447, 0.0012830676918819091] |
| full/mlp/BCE_gain_prior | [5, 0, 1, 0] | [0.014161524569106593, 0.11609854162471996] |
| full/mlp/Brier_gain_prior | [6, 0, 0, 0] | [0.004888787441740812, 0.02951581782883142] |
| full/mlp/capture_gain_envelope | [5, 0, 1, 0] | [0.1532436575136974, 0.47380995107915025] |
| motion_only/linear/AP_gain_envelope | [5, 0, 0, 1] | [0.011180747729040538, 0.038490079981902126] |
| motion_only/linear/AP_gain_frozen_easy_fraction | [4, 0, 1, 1] | [0.010365696442504166, 0.028657792909931812] |
| motion_only/linear/AP_gain_frozen_harm_fraction | [3, 0, 2, 1] | [0.0022635314959184305, 0.027404055626791077] |
| motion_only/linear/AP_gain_linear | [0, 0, 5, 1] | [0.0, 0.0] |
| motion_only/linear/BCE_gain_linear | [0, 0, 6, 0] | [0.0, 0.0] |
| motion_only/linear/BCE_gain_prior | [2, 0, 4, 0] | [-0.0018457134557026249, 0.0395360101826723] |
| motion_only/linear/Brier_gain_prior | [0, 3, 3, 0] | [-0.0004811072699891579, 0.0019108446675418468] |
| motion_only/linear/capture_gain_envelope | [4, 0, 1, 1] | [0.025363211109184443, 0.15940556025112054] |
| motion_only/mlp/AP_gain_envelope | [5, 0, 0, 1] | [0.01098141193503771, 0.05954635527357719] |
| motion_only/mlp/AP_gain_frozen_easy_fraction | [5, 0, 0, 1] | [0.01016636064850134, 0.03764092657528944] |
| motion_only/mlp/AP_gain_frozen_harm_fraction | [4, 0, 1, 1] | [-0.00015073735671472074, 0.04593522892512799] |
| motion_only/mlp/AP_gain_linear | [2, 0, 3, 1] | [-0.0024378540332656413, 0.026751754832878467] |
| motion_only/mlp/BCE_gain_linear | [0, 4, 2, 0] | [-0.05990196512622786, -0.011869309585700068] |
| motion_only/mlp/BCE_gain_prior | [1, 2, 3, 0] | [-0.06174767858193048, 0.024189627285510353] |
| motion_only/mlp/Brier_gain_prior | [0, 4, 2, 0] | [-0.0013870267312641026, 0.0030185971515234583] |
| motion_only/mlp/capture_gain_envelope | [4, 0, 1, 1] | [0.05413120104409554, 0.14478657596970512] |

Positive means improvement. Ranges are six point estimates, not one confidence interval.
Each interval averages three seeds within locality before3000 paired resamples of four localities.
The six assignments overlap. These source-development intervals are exploratory and not multiplicity-adjusted.
Source outcomes have prior development exposure. No independent scene or window-count significance claim.
The target transports a two-locality inner producer to a three-locality outer producer.
The top10% is a fixed ranking diagnostic, not a deployment threshold. Event probability is not expected harm.

```json
{
  "cap_event_transfer_signal": false,
  "magnitude_calibration_proven": false,
  "policy_utility_proven": false,
  "deployment_changed": false,
  "independent_confirmation": false,
  "stage5c_executed": false,
  "smc_enabled": false
}
```

Obs8/pred12 annotation steps, detector pixels only; no metric/seconds/physical-safety/foundation claim.
