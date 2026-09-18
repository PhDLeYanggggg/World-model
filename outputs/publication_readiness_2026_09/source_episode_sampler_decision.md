# Fixed Equal-Episode Exposure Repair

## Hypothesis

The fresh current-cohort audit maps 15,430 windows to 1,457 annotation plateau
groups and 545 scoped tracks. Repeated static segments receive more uniform-row
training exposure. Large half-box excursions occupy 207 windows but only 55
groups/47 tracks. Almost all have moving neighbors. Reweighting existing errors
by episode does not reverse the negative result, so no weighting is promoted as
a better evaluation protocol.

Test a narrower causal intervention: does equal exposure per past-defined
episode during training improve actual held-site forecasts under the unchanged
primary metric? This cannot create new independent events, cure annotation
interpolation or prove that the existing images contain departure intent.

## Fixed Comparison Before Fitting

Train 24 fresh models: four source sites, seeds 17/29/43, geometry and centered
temporal appearance. Both retain the 63,960-parameter architecture, seeded initial
state, all-target ADE including zero targets, baseline skip, optimizer, schedule,
batch size and 10,000-update budget. The frozen image encoder is unchanged.

Only the sampler changes. Within each training complement, choose an annotation
episode uniformly, then choose one of its windows uniformly. Equivalently each
row gets probability 1/(number of training episodes * rows in its episode).
All rows have positive weight. Group keys derive from past plateau starts, not
future change labels; held-site counts do not enter probabilities. Normalizer,
loss scale, hard cutoff and evaluation remain those of the original uniform
control. No sample deletion, horizon change or static-gradient suppression.

Old geometry and centered fits are `cached_verified` controls. New arms have the
same weighted sampling stream within each site/seed; they intentionally do not
have the old uniform row draw counts. Record weights, episode counts and draw
hashes. Check exact checkpoint resume and exact forecast replay.

No adaptive stopping, best seed/checkpoint, threshold search or deployment
selection. Run all 240,000 updates. A 100-update named pilot counts in budget.
CPU four threads, inter-op one, workers zero; checkpoint every 200 steps. Prior
matched runs take about 13 minutes locally; 64 GiB free observed. No HPC job.

## Readout And Boundaries

Primary remains equal-site ratio of mean normalized ADE over all original rows,
three-seed error averaging, against stationary CV. Report actual gain, every
site/seed, hard slices, native absolute zero-target harm, nonzero-target gain,
and each fixed candidate's future oracle (diagnostic only). Fixed contrasts:
weighted geometry minus uniform geometry, weighted centered minus uniform
centered, and weighted centered minus weighted geometry. Use the same 2,000
paired physical-site bootstrap draws, seed 38113, conditional on four already
explored sites and shared fitting folds. No independent confirmation claim.

All 15,430 queries remain. Bookstore/main/outer roles remain closed. Zero-target
percentage degradation is undefined, not a 2% safety pass. The full experiment
may still fail; lower harm or a positive training score alone is not success.
Offline supplied annotations may use later controls. Dataset-local annotation
pixels and raw frames only. No new deployment, Stage5C execution or SMC.
