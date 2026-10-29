# Measured Arithmetic Failure and Repair Scope

The preserved float32 precursor's real failure is reproduced from its saved
1600-update checkpoint. Autograd traces NaN to the bounded odds update.
The repaired diagnostic fork completes400 further updates without disabling
nonfinite-gradient checking. That diagnostic is not one of the54 new finals.

A minimal scalar graph isolates a sufficient mechanism. At zero envelope and
zero harm, a branch preactivation2 gives expm1(5*tanh(2/5))=5.6841888. The retained
float32 denominator floor is1.17549435e-38. The forward value is0, but backward
produces numerator derivative8.50705917e37, preceding-product derivative inf,
and harm derivative NaN. The repaired float64 intermediates avoid that overflow
and return float32 predictions. Large-envelope extremes are also regression tested.

The registered model comment mentions squared-denominator underflow as a generic
numerical risk; this is too specific for the measured minimal failure. The
demonstrated path is intermediate gradient overflow followed by zero multiplication.
The bound source is retained unchanged for reproducibility; this note corrects
the causal explanation without silently amending the registered experiment.

The repair does not change risk thresholds, objectives, model size, source roles
or optimization budget. All54 registered repaired fits start anew from the same
original anchors;37 completed precursor fits are preserved but excluded. This
fix is engineering evidence, not scientific improvement or independent validation.
