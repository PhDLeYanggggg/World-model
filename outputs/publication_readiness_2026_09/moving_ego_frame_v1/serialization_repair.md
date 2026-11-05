# Pilot Serialization Repair, No Scientific Change

Pilot PID43180 exited1 after saving its first history121 forest checkpoint.
The failure was `TypeError: Object of type ndarray is not JSON serializable`
while exporting target_scale, not a training/runtime failure. No completed
pilot or held-source result receipt was written. No score was selected.

The repair exports the identical scale vector using `.tolist()`. A synthetic
cell regression test verifies JSON roundtrip and unchanged values supplied to
the metric helper. Original registration.json and the saved pilot checkpoint
remain intact. registration_v2.json records the serialization-only code repair.
Scientific checkpoint bindings continue to use the unchanged original design
hash, with the execution registration recorded separately. Resume reuses the
existing pilot forest and performs its scheduled exact full-refit comparison.

No feature, target, model, seed, split, normalizer, threshold, budget, loss or
scientific rule is changed. The failed process is retained as an engineering
failure, not hidden or mislabeled as completed training validation.
