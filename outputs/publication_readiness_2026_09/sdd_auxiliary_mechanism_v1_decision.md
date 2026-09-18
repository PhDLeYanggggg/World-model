# Auxiliary Source Mechanism: Fixed Exposure and Supervision Controls

## Scope and Hypothesis

The previous goal turn completed 54 real fits and identified a source-exposure
confound: SDD plus main training performed less badly than 6,000 main updates,
but still lost to CV and harmed easy cases. It was progress, not a successful
method. This follow-up tests whether correct source supervision contributes
beyond fewer main-domain updates and nonspecific source initialization.

This is an authorized fit-only repair under the existing train-40 SDD admission.
No new dataset, split, primary, risk limit, sampling stride or sealed role is
introduced. No training on held fit rows and no held-score threshold/model
selection. Every registered result is reported, including negative results.

## Fixed Design

Reuse all 54 prior no_aux and sdd_aux fits as cached_verified controls, after
source/config/checkpoint/prediction/row identity checks. Add 54 fresh fits:
two new arms x geometry/mask_only/past_rgb x seeds 17/29/43 x ETH/Hotel/grouped
Zara folds. No new architecture, image resolution or loss weight.

- main4k: original random initialization, zero first-phase updates, 4,000 main
  updates on the exact same main minibatch stream as the old second phase.
- sdd_permuted: 2,000 SDD updates with shuffled baseline-relative correction
  labels, followed by the same 4,000 main updates as every source arm.

Total new optimizer updates: 270,000. The 4,000-update control intentionally
matches main-domain exposure, not total compute. The permuted source arm matches
source draws and total compute. Old controls are not relabeled fresh.
AdamW, batch 64, learning rate 0.0003, weight decay 0.0001, clip norm 5,
main-fold-only normalization, constant-column mask, final checkpoints and
8-to-12 interface remain unchanged. Optimizer and main sampler reset at the
same phase boundary. No dropout or early-stopping change.

## Label-Only Source Ablation

For each seed, create one deterministic row permutation within original source
recording and exact 12-point target-support mask. Use random cyclic derangement
inside each stratum with at least two rows; retain singleton rows and report them.
Replace only the source loss label by:

    own causal CV rollout + donor (future label - donor causal CV rollout)

Inputs, source mask, query membership, normalization and baseline stay unchanged.
This preserves the intended residual-label marginal within each stratum up to
floating-point arithmetic. It breaks row-specific correction correspondence,
not every dependency: record-level motion priors, same-agent donors and overlapping
windows may remain. Report same-agent/singleton counts. Stratification uses only
admitted source-training label support; it is not an inference feature or a
future-based row exclusion. Main training and held evaluation use original labels.

All-zero-support rows remain in the source population, never zero-error evidence.
Future targets remain loss/evaluation labels, never prediction inputs. No test
endpoint goals, central velocity or source validation/test raw data access.

## Contrasts and Interpretation

Report all four schedules for each modality, including primary ADE/FDE vs CV,
easy relative and absolute harm, tail errors, train/held gaps, fixed event and
recording slices, sampler counts, losses and runtime. Main cohort remains 11,966.
Primary aggregation remains equal physical scene; comparisons never pool native
pixels and unverified dataset-local coordinates as metric units.

1. real SDD vs main4k: benefit beyond the same number of main-domain updates.
2. permuted SDD vs main4k: effect of nonspecific source initialization/training.
3. real SDD vs permuted SDD: value of correct row-specific source supervision.
4. no_aux6k vs main4k: effect of extra main-domain optimization.
5. past RGB vs same-schedule mask: contribution beyond image support.

Use all three seeds and 2,000 paired physical-site bootstrap draws. Only three
already exposed sites are available; all intervals are descriptive, not independent
confirmation or guarantees. No hypothesis is declared proved solely by one
favorable slice. Distinct contrasts do not isolate every domain/view/label effect.

## Runtime and Verification

Reuse the complete 2.61 GB source cache. The previous real arm64 CPU run took
about three hours for 324,000 updates; this 270,000-update matrix is reasonable
locally, without new HPC transfer or duplicate jobs. Four compute threads,
one interop thread, workers zero, checkpoints every 200 updates, logged PID,
exact resume and real 100-update pilot followed by continuation of the same fit.
Wait for slow progress; no automatic budget shrink or early-stopping based on loss.

Before real fitting, freeze config/code and test label isolation, source/support
preservation, donor determinism and zero-pretraining resume/main-stream identity.
After fitting, replay every new checkpoint and verify completed resume performs
zero updates. Keep private data/checkpoints out of Git. Update public results
and author-voice README with all outcomes. Stage5C and SMC remain disabled.

This study does not establish metric/seconds alignment, true 3D, foundation or
submission readiness. SDD uses raw stride 12 and a +144-frame final target;
offline interpolation is disclosed. Independent confirmation remains unresolved.
