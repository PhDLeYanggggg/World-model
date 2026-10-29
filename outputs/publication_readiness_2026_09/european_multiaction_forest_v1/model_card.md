# Model Card: Seven-Action Conventional Cost Control

## Purpose

This is a conventional expected-cost estimator for deciding whether to replace
a frozen motion floor. It tests whether the failure of the preceding neural
cost heads is specific to that estimator family. It does not train a new
trajectory forecasting network, latent dynamics model or generative model.

## Model And Inputs

The fixed design is six SOURCE contexts, three controller seeds and three
candidate banks, giving 54 ExtraTrees regressors with 128 trees each. Settings:
maximum depth 12, minimum leaf size 64, one third of features per split,
no bootstrap, four fitting threads, checkpoint every 16 trees. Actual installed
scikit-learn is 1.8.0.

Each action receives 398 causal history/rollout/static-region features, seven
action-identity coordinates and one causal envelope coordinate. The 18 static
region features are not verified walkable geometry. Future endpoints, future
label masks, evaluation easy/hard labels, recording IDs and oracle choices
are not inference inputs.

Five continuous cost moments are regressed using the original fitting query
draw multiplicities and query/action weights. Shared normalization is fitted
on physical fitting data. Nonnegative moment projection and the fixed predicted
all/easy 2% gate follow regression. This differs from the neural head's
asymmetric loss and constrained decoder; compute is not matched.

## Training And Evaluation Status

The immutable registration, pilot, full-128 replay, final fitting receipts and
readout records are the authoritative execution evidence. Do not infer full
completion from the model design or this card. `readout.md`, `summary.json` and
`gates.json` report the measured result when the registered sequence completes.

The primary comparator is the frozen static-scene neural cost head, not the
incompatible older binary forest. All three banks and fixed harm sensitivities
remain in the report. No seed or checkpoint is selected after readout. This
SOURCE inner-held set is exposed development, not independent confirmation.

## Intended Use And Restrictions

Use for controlled research on forecast replacement and continuous risk-cost
estimation. It is not a real-world agent controller or a physical collision
safety model. Empty selected risk is not_estimable. Known average gains do not
override unknown-completion or per-recording risk failures. Count matching
addresses utility only. No deployment promotion follows from this experiment.

Three controller seeds are not three independent forecasting pipelines. Weak
detector-track labels and static region proxies are not human gold. The current
task is pixel/raw-frame obs8/pred12/stride12, not verified meters, seconds,
true 3D or foundation-model evidence. Independent roles and DroneCrowd remain
closed. Stage5C execution and SMC remain disabled.

Model weights and working matrices are private local artifacts, excluded from
Git. See [reproduction instructions](reproduction_zh.md) for required assets and
the distinction between fresh training, resume and inference replay.
