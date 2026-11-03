# Arithmetic Precision Is Not The Main Admission Failure

## Material Passport

Fresh_run:36frozen cost heads,critic-fitting sources only,zero optimizer updates.
Original float32 forward output is checked bitwise against the original API;
float64 uses the same already-normalized float32 inputs and identical weights.
PID12050exits0 in207.181s,peakRSS7,167,311,872bytes. Separate PID12332exits0
in207.704s and reproduces every saved scalar. Three focused synthetic checks
pass1.54s,including exact original forward,forbidden label-field rejection and
link derivatives versus direct autograd. No new held-role metric is read.

This is post-readout mechanism diagnosis,not a preregistered independent test.
Counts below sum repeated producer/critic/seed views,not independent samples.

## Precision And Link Results

|Feature arm|Fitting row views|Float32/float64 independent-choice changes|Selected known EH mass with float32-underflow recovered in64bit|
|---|---:|---:|---:|
|Local history|1,171,716|0|0.0961%|
|Shared motion|1,171,716|0|0.6540%|

Exact-zero float32 EH occurs in996and2,573row views. Its recovery in64bit
accounts for little of the true selected EH mass and changes no choice.
Maximum absolute moment differences are0.007585/0.003616pixel-mass units.
These are not trajectory ADE or a risk guarantee. Merely changing precision
does not repair admission on this complete fitting population.

The dimensionless link derivatives are dH/dHlogit divided by its causal
envelope, and dEH/dgate-logit divided by H. They are not full parameter gradients.

|Derivative threshold|Selected known EH mass with either link below threshold: local|Shared|
|---|---:|---:|
|1e-2|98.5793%|99.3023%|
|1e-4|71.2543%|70.7090%|
|1e-6|43.4914%|39.4260%|

On all-neural fitting rows,the1e-4fractions are38.0150%/38.9444%,rather than
71.2543%/70.7090%on the fixed independent selected region. This measures weak
decoder link sensitivity in a relevant failure region; it does not prove that
gradient attenuation is the sole training cause or that a new loss will work.
The three thresholds were coded before this diagnostic; none selects a policy.

## Consequence For The Next Experiment

Do not spend another run on float64 as a supposed safety repair. Do not claim
generic log loss is novel:the old SDD two-cost log-loss trial and European
sampling,hurdle,conditional-cost and regional-gradient controls remain failed.

Test one matched continuation of the current coherent5-moment head:retain its
original moment objective and add stable supervision at the existing cost-
fraction logits. Compare against exactly the same additional original-loss
updates,not the weaker frozen2kmodel alone. Both use local-history inputs;
the prior shared-motion ablation did not establish useful incremental signal.
Any new continuation is source development on already exposed groups,not a
new final test. No new architecture,information,normalizer or risk threshold.

## Reproduction

```sh
.venv-pytorch/bin/python -u scripts/diagnose_m3w_moving_decoder_signal.py evaluate
.venv-pytorch/bin/python -u scripts/diagnose_m3w_moving_decoder_signal.py verify
```

Large causal packets/weights remain private. Original24-group primary remains
not estimable;23supported groups remain secondary. No calibration/confirmation,
Stage5Cexecution,SMC,metres/seconds,true3D,foundation or submission claim.
