# Causal reference bridge with lead-specific support

Use the frozen v155 Moving fitting, calibration and held source roles.
Train three matched architectures with the same 800 updates, 256 draws
per within-record minibatch and three seeds. The shared forecast first
interpolates CV and OLS4 by lead. The bridge arm allows no transverse
or additional residual. The lead-mean arm gates a bounded residual with
training-only mean observation availability by lead. The support arm
gates the same residual with a prediction of observation availability
from past motion and current neighbors. All arms learn the same auxiliary
binary support target. Detach the probability in the forecast gate so
the trajectory objective cannot directly alter the support prediction.
Support means label availability in this dataset, not visibility or
physical presence. Do not use an outcome or an observed future mask at
inference.

Optimize conservative gain versus OLS4, penalize a negative conservative
gain versus CV, and retain the original recording-risk surrogate. This
is a training loss, not a coverage or risk guarantee. Compare the raw
paths, fixed CV/OLS4 and the frozen v155 OLS-only and dual-risk controls.
Select strict options from CV interpolants of the three trained paths
and OLS4 on calibration only: all recordings must pass the original 2%
rule, with positive source-macro lower gain against both references.
Otherwise use CV without counting undefined support as a pass.

On the held development roster, compare lead-mean and causal support
both raw and after each row is radially shrunk to the smaller of the
two CV-relative peaks. This post-hoc paired control holds recipient
and peak fixed without amplification and cannot select a deployed
policy. Report paired conservative gain, observed ADE/FDE, original
risk and paired source-bootstrap intervals after seed averaging.
Evaluate support prediction with Brier score versus the training-only
lead mean. Do not use Moving A or DroneCrowd independent confirmation.
