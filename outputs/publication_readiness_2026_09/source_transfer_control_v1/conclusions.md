# Matched Modality Continuation: Training Repair Does Not Transfer

## Result and Provenance

The registered comparison is complete and negative for a useful held-source
forecast gain. Six new coverage-only Torch continuations completed48,000updates
on all15,430 source-training rows outsidebookstore. The100-step timing pilot is
included. Full-run log span36.70minutes; summed continuation/evaluation work
2208.49seconds. Six completed RGB continuations and six inherited2kparent states
were `cached_verified`, not retrained in this run. All18fixed states were freshly
evaluated on6,944bookstore queries,7recordings and181scopedagent IDs.

Training registration71d0f520 and analysis-code freeze fbb0517a were pushed
before new held-source scoring. Every model/seed/endpoint remains in the table.
No model, threshold or deployment policy was selected from these results.
Bookstore has been explored before, so this is not independent confirmation.

## Fixed Comparisons

Three-seed means; gain is ADE reduction versus the same stationaryCV. Positive
is better. The intervals below use2,000paired recording-block resamples within
one previously explored physical source site. They are not scene-generalization
confidence intervals and must not be reported as untouched-test evidence.

| Final10k condition | Training gain | Held-source gain | Conditional95% CI | Fixed0.9guard held gain |
| --- | ---: | ---: | --- | ---: |
| Coverage-only, constantLR | -1.3081% | -4.2320% | [-11.4091%,-2.3410%] | -0.0311% |
| RGB, constantLR | -1.0310% | -5.7629% | [-15.9421%,-3.1406%] | -0.2310% |
| Coverage-only, cosineLR | +0.1872% | -1.7438% | [-5.8996%,-0.8108%] | -0.0187% |
| RGB, cosineLR | +0.2533% | -2.0806% | [-6.9848%,-0.9895%] | -0.1155% |

All18individual states lose toCV in both evaluation modes. All12arm/schedule/mode
mean intervals are negative. Equal-scoped-agent sensitivity also remains negative.
Decay improves the held point estimate relative to its parent by0.3349pp(mask)
and0.3334pp(RGB), but both paired intervals crosszero. This is neither stable
parent improvement nor baseline superiority. Continuing with constantLR is worse.

The uncontrolledRGB-minus-mask contrast is-1.5309pp underconstantLR and-0.3368pp
undercosineLR; both conditional intervals are negative. The latter is
[-1.0851pp,-0.1669pp]. The earlier training gain therefore cannot be attributed
to demonstrated transferable RGB value. The coverage-only control retains past
geometry and coverage, so this does not prove that all scene information is useless.

Native annotationpixelCV ADE is0.77394765. Cosinemask/RGB ADE is0.78744408 /
0.79005073. Changing the reported unit does not turn the negative result positive.
Zero-target mean harm remains0.02243060 /0.02675066annotationpixels without the
guard. Its percentage isundefined because the baseline error iszero. There is no
valid2%easy-preservation pass. The fixedguard changes0.288%/2.021%of rows for
mask/RGB, reducing harm without creating positive aggregate gain. Guarded RGB
contrasts also change the inherited classifier, not only the trajectory predictor.

## Failure Taxonomy

| Hypothesis | Evidence | Conclusion |
| --- | --- | --- |
| Broken numerical learning path | Earlier microfits succeed; current cosine arms fit slightly better; exact replays pass | Not a universal inability to optimize, but convergence is not proved. |
| Only insufficient update count |41.48mean training draws perrow; both modalities improve training, yet every held state loses | More updates alone did not resolve this comparison. |
| RGB supplies transferable state-change information |Matched-budget uncontrolledRGB contrasts are negative | No demonstrated RGB benefit here; not a universal impossibility claim. |
| Unsafe output on exact-stationary targets | Post-hoc decomposition assigns75.51-80.75%of positive error increase to zero-target rows | A major observed failure component; moving rows are also harmed. |
| Existing classifier is a useful deployment guard | Fixed0.9guard remainsnegative in all18states | Lower intervention is damage reduction, not calibrated improvement. |
| Threshold search can rescue strong candidates | Future-informed oracle over all18complete paths plusCV reaches only1.2752% gain; hard ceiling0.5267% | This frozen action set offers limited headroom. This supplementary post-hoc bound is not a causal selector result. |
| Dataset labels fully represent visible intention | Earlier source audit reports many tiny annotationcenter changes and generated histories | Observability and label semantics remain hypotheses, not causally established failure mechanisms. |

The candidate ceiling applies only to choosing one saved completepath orCV per
query on these already-labeled rows. It excludes blending, scaling, new forecasts
and different populations. It is not a global theorem limiting world modeling.
See [post-hoc ceiling](candidate_ceiling.md), [all fixed scores](results.md),
[paired contrasts](paired_contrasts.csv) and [recording/agent summaries](held_group_metrics.csv).

## Verification

All42saved forecasts replay exactly:18held outputs plus24newmasktraining
milestones. The analysis also checks all48training milestones across modalities,
three four-way sample streams, finite bounded outputs and exact zero fallback
outside pastcontext support.202registered artifacts, including reused parents,
remain unchanged on completed resume and repeated evaluation;0newupdates.
Forty-three focused tests pass. The full legacy suite was not rerun. The figure
was visually checked; all training/evaluation/replay/analysis processes finished.
[Figure](matched_results.svg), [reproduction](reproducibility.md),
[Chinese operations](operation_zh.md), [model/data card](model_data_card.md).

## Next Decision

Do not deploy these heads, retune their heldsite gate, or launch a blind larger
network grid. Preserve the completed negative evidence. The next useful repair
must create genuinely better candidate paths and exact-baseline behavior on
unsupported/simple contexts, using the training complement only. A bounded
training-side comparison of a zero-capable baseline-relative head against the
same decoder is more informative than another bookstore threshold search.
Its scope, loss and held-use rules require a separate prospective registration;
it has not been implemented or run by this report. Scene-level joint intervention
and independent risk calibration remain research requirements, not achievements.

Main native8-to12 rules and sealed selection/calibration/confirmation stay
unchanged. This source diagnostic uses8-to12 atstride12/+144rawframes. No seconds,
metric, true3D, foundation, world-model success or submission-ready claim.
No new deployment, Stage5C execution or SMC. The long-term objective remains active.
