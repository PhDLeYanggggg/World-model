# Why Doubling The Critic Width Is Not A Repair

## What Was Actually Tested

Nine new Torch cost critics use the same causal inputs, original SOURCE-only
training rows and complete budgets as the nine width-64 controls. The only
configuration change is width 64 to 128. The same initial predictions, query
draws, SOURCE preprocessing and complete fixed-endpoint exposure were verified.
An actual interrupted/resumed pilot and the first full refit match exactly.
This is a cost-model capacity control, not new trajectory or world-dynamics
training and not a matched-compute comparison.

## What Failed

The complete SOURCE objective falls from 0.721499644 to 0.718380440, only about
0.43%. Development known-label gain changes from 1.503747% to 1.431216%.
The paired known-label difference is -0.072531 pp with interval
[-0.666519, +0.390598]; it does not establish either direction of improvement.

The registered full paired completion-lower contrast is a different quantity:
-0.636338 pp, interval [-1.178580, -0.278679]. It fails. The upper contrast is
also reported; the lower result is not a 0.636 pp observed-ADE loss.

The width-128 full policy supports only 9/252 required recording/pipeline views.
There are 202 known-all and 207 known-easy positive-harm violations, 13 undefined
selected-easy views and 10.165838% worst signed easy-net degradation. The width-64
control already failed; improving some counts does not make width 128 safe.
No isolated favorable seed, scene, hard subset or arm replaces this result.

## What The Two New Diagnostics Rule Out

The preceding SOURCE decomposition finds that shared-reference decoding and
the original output projection slightly improve every one of 18 frozen forest
objectives. Its fitting gap is already present before those transformations.
That observed result rejects decoder damage as the explanation for this forest
failure; synthetic examples where projection can hurt are not observed causes.

The neural width control shows that this particular doubling, with the original
data, loss and budget, does not solve safe routing. It does not show that all
larger networks fail, that causal features are irreducibly insufficient, or that
one particular loss, missing-label process or domain shift is the unique cause.

## Research Consequence

Keep the original deployment state. Do not expand to width 256 automatically or
retune thresholds on exposed calibration outcomes. A major unfilled paper gap
is the matched public forecasting comparison on the current EuropeanSquares
protocol. The existing pinned EqMotion source is a usable asset, but its older
SDD scores are not that comparison. A fixed-head K=1 adaptation must be named
as such, rather than presented as reproduction of published best-of-20 results.

Six development localities are exposed. Exact numerical replay is engineering
reproducibility, not independent confirmation. SOURCE labels are training or
diagnostic supervision only. Calibration private arrays, final confirmation and
DroneCrowd were not accessed by these studies. The historical calibration is
completed and unsuccessful, not newly unused data. Stage5C and SMC remain off.

## Scope

EuropeanSquares, eight observed steps and twelve requested steps, raw stride 12,
image-local pixels and released detector/tracker silver. Available-label ADE is
not fully observed 12-step ADE; bounds for wholly unknown rows do not fill missing
steps within partial futures. No metric, seconds, physical-safety, true-3D,
foundation or submission-readiness claim follows.
