# Quantile Coverage Is Not Mean-Harm Control

This is a mathematical limit, not an observed Moving result or an explanation
already proved for its failures. It was noted while the registered training
was running, before the pilot's validation readout. No objective, threshold,
or gate is changed in response.

Let every selected example have reference error R=1. Suppose H=0 with
probability .95 and H=1 with probability .05. For Z=H-.02R, the exact population
.9 quantile is -.02, but E[Z]=.03 and E[H]/E[R]=.05. Even a perfectly learned
.9 quantile would admit these examples under Z<=0 while failing the original
2% mean-harm condition. A finite learned quantile has additional estimation
and source-shift errors. This is why the pilot retains the observed original
gate and never labels quantile output a safety certificate.

The direct mean head targets the right conditional first moment in population,
but empirical estimation on the admitted tail can still be poor. The pilot
therefore asks an empirical, limited question: whether fixed tail emphasis
improves that decision in the preselected SOURCE contexts. It does not supply
a theoretical guarantee of the requested risk contract.

A possible later magnitude-sensitive objective is asymmetric squared error,
whose population minimizer is an expectile. For an exact conditional upper
expectile, the first-order equation weights positive residual magnitude more
heavily than negative magnitude; this differs from quantile coverage. Finite
neural predictions under source shift are still not certified bounds. This
classical idea is not an M3W invention: see Newey and Powell's original
[Asymmetric Least Squares Estimation and Testing manuscript](https://www.princeton.edu/~erp/ERParchives/archivepdfs/M325.pdf).
No expectile model has been trained in this pilot. It is a potential next
model test, not a retrospective replacement for either frozen arm.

No independent test access, identity certification, rule revision, metric,
seconds, physical-safety, true3D, foundation, Stage5C or SMC claim follows.
