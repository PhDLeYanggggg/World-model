# Development Addendum: Conditional Neural Risk Transport

## Method And Experiment

We tested whether correcting local reference outcomes for their causal feature
mismatch improves intervention risk prediction. A source-only regressor f
produces a plain risk prediction; a second readout adds the weighted residual
of its frozen source-excluded references. Both readouts share the same model.
The references, original 120 features, two signed risk targets, fitting-only
scaling and original risk contract remain unchanged. Ridge regression provides
a linear control; shared64 and track32 are verified existing controls.

Eight actual Torch networks (9,890 parameters each) trained for 32 complete epochs,
with 5,792 primary updates. Eight ridge models were also fitted. All 48,061 known
own-B proposals were evaluated, while 1,274 unknown outcomes remained audited
but unsupervised. This is exposed source-held development, not independent test.
All 16 models reproduced by full refitting; checkpoint recovery was verified.

## Result

Neural transport reduces cost relative to its plain network by 1.769%, 6.434%,
0.893%, but is worse than shared64 by 6.914%, 54.522%, 5.517%. Harmful admissions
increase from 0/1/0 to 94/353/1288. The registered information screen passes 0/3.
Risk-MSE improves 32.192% in one context without safe decision improvement.
Thus neither a learned nonlinear regressor nor this reference adjustment
establishes the desired intervention contribution. No model is promoted.

Fitting losses decrease in every fold, but source-held errors remain larger.
Source076 has 0.526 fitting MSE versus 7.738 source-held MSE after only 64 updates
over 716 donor rows. Late fitting loss continues to fall. The result cannot
separate inadequate optimization from source-conditional mismatch conclusively.
The next fixed-budget convergence check must not select epochs on held outcomes.

## Related Methods And Novelty Boundary

Retrieval-augmented regression is established. TabR combines feed-forward
learning with a learned nearest-neighbor-like component; our fixed-reference
residual readout is not a reproduction of TabR and cannot claim its empirical
results. [Gorishniy et al., ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/4ef594af0d9a519db8fb292452c461fa-Abstract-Conference.html).

DNNR addresses feature scaling and local function variation with estimated
gradients and Taylor approximations. Our adjustment instead uses differences
of a global fitted risk function; we neither implement DNNR nor inherit its
theoretical bounds. [Nader et al., ICML 2022](https://proceedings.mlr.press/v162/nader22a.html).

Local residual correction is also discussed in the appendix of Igel and
Oehmcke's regression-bias study. Its discussion reinforces that adding a local
correction is not itself a novel contribution or a calibration guarantee.
Our donors were used in fitting f, so their residuals are not independent
calibration data. [Igel and Oehmcke, 2023](https://link.springer.com/article/10.1007/s13218-023-00801-0).

The intended M3W contribution remains useful scene-joint neural intervention
under the original relative-harm and support contract. This component study
does not establish it. No novelty-by-module-combination claim is made.

## Limitations

One seed, overlapping windows, few donor sources, partial/silver supervision,
in-sample donor residuals and already-exposed development roles limit inference.
No formal significance or independent confirmation follows from exact replay.
No original all-view safety pass is established; empty reference support is
NA, with no identity exemption. The original 24-group primary remains NA. Full-policy
evaluation and independent three-seed main evidence are not supplied here.

Native obs8/pred12 image-coordinate prediction is not metric, seconds-level,
true3D or foundation modeling. This is risk-head training, not a new trajectory
generator. Stage5C/SMC remain off. The compiled paper is unchanged; this separate
addendum records a negative development result. Submission is not ready.
