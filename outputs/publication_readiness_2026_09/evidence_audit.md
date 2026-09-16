# M3W Submission Evidence Audit

Fresh source/artifact audit and synthetic diagnostics. No model retraining or real-data prediction replay.

- Historical run: 2026-06-04T06:14:32.083873+00:00; mode: small; rows: {'train': 12000, 'val': 5000, 'test': 8000}
- Recorded input hash matches present caches: True
- Checkpoints matching recorded SHA256: 7/7
- Retrospective validation best: hybrid_no_scene; recorded test-selected best: hybrid_no_scene
- Equal validation/test winners do not restore independent confirmation.
- Reported gains are scale-normalized four-waypoint ADE vs an endpoint-interpolated floor, not directly Stage37 FDE or community-standard ADE.

## Findings

- test_selected_variant: Final architecture ranking reads test_eval; it is not confirmatory model selection.
- test_driven_repair: Repair trigger reads test_eval; the current recorded run contains seven base variants and no repair variants.
- proxy_semantics: Interaction label is hard OR failure; validity label is waypoint completeness; density label is historical density.
- gain_harm_target_mismatch: Gain uses candidate oracle vs strongest; harm uses easy OR small oracle margin, not realized neural-vs-floor harm.
- nonstandard_comparator: Floor waypoints are linear interpolation of a selected endpoint; errors are per-row scale-normalized over four waypoints.
- asserted_leakage_gate: The Stage44 no-leakage gate reads declared flags, not a complete data-lineage or test-selection audit.
- target_encoder_not_updated: The randomly initialized future target encoder is detached and has no EMA update in this module.

## Synthetic Checks

```json
{
  "status": "fresh_run",
  "data": "synthetic_only_not_prediction_evidence",
  "runtime": {
    "machine": "arm64",
    "torch": "2.12.0",
    "threads": 4
  },
  "target_encoder_parameters_with_gradient": 0,
  "context_encoder_parameters_with_gradient": 48,
  "target_encoder_unchanged_after_optimizer_step": true,
  "best_variant_changes_when_only_test_metrics_change": true
}
```

## Scope

No training, new benchmark gain, complete no-leakage certification, or deployment promotion is claimed.
Current caches must not be called cached_verified replay unless input identities and complete lineage match.
Dataset-local/raw-frame only; no metric, seconds-level, true-3D or foundation claims. Stage5C and SMC remain disabled.
