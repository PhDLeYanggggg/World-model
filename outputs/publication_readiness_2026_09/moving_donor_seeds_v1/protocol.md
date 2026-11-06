# Risk-Head Seed Replication On Frozen Donor Data

The previous seed17 donor comparison improved matched neural information in
three contexts, but did not meet the stronger forest harm comparison. Replicate
the same treatment using risk-head seeds17/29/43. Seed17 is cached_verified;
train48new models for29/43 and repeat all48 exactly before readout. Do not
relabel cached seed17 as fresh. The forecaster/OOF teacher seed stays17.

Keep the original120features,64/32MLP,4,096updates, optimizer, source/query
sampling,50/50pool mixture, donor candidates,8source-held folds, normalization
and zero threshold fixed. No hyperparameter, donor or seed selection. Reuse the
same original fitting-only MSE evaluation scale for every arm and seed.

Before primary fitting, prove the new explicit-seed engine equals the frozen
engine at seed17 on one complete4,096-update real-data fit, then prove
128+128vs256recovery at seed29. The pilot is separate and not reused. The frozen
historical code is left intact. A copy of the fixed training algorithm is used
to make the seed an explicit parameter; a regression test prevents engine drift.

Primary replication screen: the existing information criterion (at least1%
cost reduction versus matched own-only, nonworse false-safe cost and lower MSE)
holds in at least2of3contexts separately for each of three risk seeds. This
does not replace or relax the original full-policy safety contract. Always
report the forest comparison, harmful admissions and negative source slices.

Report seed mean/SD/range and3000paired source-cluster bootstrap replicates,
sampling2/3/3source components within the three contexts. Use the same sampled
component indices for all arms/seeds and average seeds before the effect.
Windows are never bootstrap units. Intervals are exploratory, conditional on
these exposed sources, fixed teachers and fixed partitions, not independent
confirmation or a large-domain guarantee. The three risk seeds are not three
forecaster seeds, three independent datasets or a complete main experiment.

No new outer C/H or reserved dataset is evaluated. No test/held threshold search,
future endpoint input, central velocity, endpoint goals, identity exemption,
metric/seconds, human-gold, true3D or foundation claim. Original2% all/easy harm,
signed easy-net, support conditions, empty-control NA, old failures and24-group
main NA remain. No deployment/submission upgrade, Stage5C or SMC execution.

Use arm64 .venv-pytorch, CPU4/inter-op1/workers0. Atomic checkpoints every128
updates, heartbeat every256, fixed-final checkpoints only. Keep5GiBfree, new
private artifacts below1GiB, allow healthy runs up to12h. Resume identical jobs;
do not duplicate a live process. Pilot cost decides whether local execution is
reasonable; no new HPC job is needed unless the measured resource demand changes.

fresh_run: new seeds, full repeats and development evaluation. cached_verified:
seed17models/results, donor arrays, own-B inputs and pinned source lineage.
not_run: new independent confirmation, full-policy deployment, teacher-seed study.
