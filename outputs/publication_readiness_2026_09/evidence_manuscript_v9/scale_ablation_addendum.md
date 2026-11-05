# Exploratory Retrained Absolute-Scale Ablation

On the already exposed MovingDroneCrowd++ SOURCE development partition, we
removed the risk critic's explicit log pixel-scale input while retaining
normalized trajectory and neighbor features. Three source-separated contexts
were trained under both MSE and signed-cost objectives, with identical warm
initialization, optimizer budget, proposals and labels to their unmasked
controls. Six new fits completed12,000updates; a2,000-update full refit and
400-update resume checks reproduced exactly. Original2%risk/support rules
and the independent data reservation were not changed.

For the prespecified primary MSE contrast, equal-source known ADE gain over
the floor increased from3.6006%to4.1531%, a+0.5525percentage-point change.
The missing-label completion interval was[+0.0944,+1.0106]pp; this is not a
sampling confidence interval. Hard gain increased from4.6257%to5.4549%, but
worst defined easy-net degradation increased from38.9046%to47.5913%. No
recording view passed the full original risk/support requirements(0/17).
The secondary signed-cost contrast also improved average known utility but
worsened worst easy-net degradation. Neither prospective expansion screen
passed. The original24group primary outcome remainsnotestimable.

The intervention shows that explicit scale affects scene-specific admission,
but does not establish robust harm discrimination or safe generalization.
The proposer and support heads retain original scale information, and the
warm model previously used that feature; this is not a full invariance result.
These repeatedly exposed, single-producer-seed findings belong in exploratory
negative-result analysis, not an independent-test or main safety claim.
No new trajectory dynamics, deployment, Stage5C execution or SMC is claimed.
Coordinates are nativeimagepixels with observation8/prediction12, not verified
metric units or seconds. The project remainsnotyet submission-ready.
