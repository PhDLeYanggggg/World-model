# Frozen Pretrained Appearance And Temporal Dynamics Comparison

## Material Passport

New training-side hypothesis after completed source_motion_quality_v1, commit
d1fd617c. The prior static-gradient intervention worsened actual forecasting;
past-box probability probes did not repair transfer. All historical explored
site status remains. This registration is not independent confirmation.

## Hypothesis And Matched Arms

Frozen ImageNet-pretrained appearance across the eight observed frames may
provide motion/context information unavailable to small from-scratch crop CNNs
or box shape alone. Test an actual twelve-waypoint trajectory head, not just
motion probabilities or a future oracle. No assurance of useful visual signal:
the existing32x32crops may be insufficient and upsampling adds no detail.

Three matched neural arms: geometry/coverage only; geometry plus current frozen
appearance repeated across eight tokens; geometry plus all eight frozen past
appearance tokens. Same geometry MLP, visual projection, GRU and zero-initialized
bounded trajectory head, seed and sampler. Coverage history remains in every
arm. Primary paired contrasts: sequence minus geometry, sequence minus current,
current minus geometry, all reported. No best-arm or best-seed deployment.

ResNet18 ImageNet1K V1 is only a fixed feature extractor, not a new method or
video foundation model. Use torchvision0.27.0 local implementation; official
256resize/224center crop, ImageNet normalization, frozen/eval BatchNorm, final
classifier removed,512dimensional L2-normalized features. Per-image normalization
uses no cohort statistics. Unsupported pixels are zeroed and no-image features
zeroed. Source crop coverage is explicit. Do not claim high-resolution input.

Original reference: [He et al., Deep Residual Learning](https://arxiv.org/abs/1512.03385).
Implementation/weights: [official torchvision](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html).
Weights URL https://download.pytorch.org/models/resnet18-f37072fd.pth;
SHA256 f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec.
Downloaded46,830,571bytes to ignored external_data/pretrained; do not redistribute.
Pretraining is publicly documented as ImageNet1K; an independent image-level
overlap audit against all SDD imagery is not performed, so no blanket pretraining
non-exposure certificate is claimed. No SDD labels train the frozen encoder.

## Fixed Data, Training And Readout

Keep all15,430stationary-history source queries,29recordings,545scopedagents,
four sites(coupa/deathCircle/gates/hyang). Each fold trains on the other three
sites. Bookstore/main/sealed queries are not fitted or forecast. Existing corpus
identity verification loads broader assets, so this is not zero-file-access.
Eight observed/twelve future annotation steps,stride12rawframes,not main full
benchmark or t50 rerun. Supplied historical labels may involve later annotation
controls; retain the approved offline, not sensor-as-of, observation contract.

Four folds x seeds17/29/43 x three arms =36fresh heads,10,000updates each,
360,000total. Full-row uniform sampler, batch64,AdamW lr.0003,weight_decay.0001;
first2,000constant rate then cosine to.01initial rate at10,000. All-target ADE,
including zero targets, divided by training-complement mean CV ADE. Observed
context radius/rotation and bounded local output unchanged. Exact common sample
draws checked against the frozen cross-fit controls. No teacher or future input.
No reweighting, no early outcome-selected stop, no held-loss checkpoint selection.

Primary: equal-site ratio of mean normalized ADE against stationary CV. Also
report every site/seed, training loss, hard slice using training error quantile,
zero-target absolute pixel harm (percentage undefined), tail error, raw output
rate, ADE/FDE and fixed binary future-oracle diagnostic. A favorable future bin
is not an input policy. Mean errors across seeds, not deployed ensembling.
Shared2,000physical-site bootstrap draws seed38113; per-site recording intervals.
Four historically explored sites and shared train folds limit inference; not
independent calibration. All arms/contrasts are conditional development results.

## Execution And Boundaries

Native arm64, CPU4threads/inter-op1,workers0. Cache25,300unique past crops once
(~49.4MiBembeddings), immutable chunk receipts/resume; no silent image filtering.
Run a fixed two-chunk extraction pilot and100-update named-head pilot for speed,
not held performance; these count toward full budgets. Checkpoints/heartbeat
every200updates. Exact uninterrupted/resumed synthetic replay before fitting;
exact real model replay after. Local65GiBfree observed. CurrentCREATE access
not available after fresh publickey denial in prior turn; no jobs submitted.
If resources are inadequate, preserve checkpoints and change execution resources,
not the registered prediction objective or population.

Legacy immutable optimizer uses the literal `mask_only` engine argument; the new
model's bound `arm` controls actual visual ablation explicitly. Tests must ensure
current ignores earlier appearance, geometry ignores all appearance, and
sequence can respond to past appearance. This is not a hidden mask-only run.

Main metric/splits/easy budget unchanged. Raw data, visual features, images,
checkpoints and third-party weights stay outsideGit. No new policy, seconds,
metric, human-gold, foundation, Stage5C execution or SMC claim. Positive feature
contrast alone is insufficient: actual forecasting and safety must improve.
