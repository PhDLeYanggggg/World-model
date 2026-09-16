# Fixed-forecast supplementary analysis

Status: specified while the conditioned-context v6 primary study is running, before any v6 development results. This diagnostic does not modify the frozen v6 primary protocol or select any additional models.

Parent protocol: `53be3aafbda47ddf8d60891e779896f6685fcd59222a1fe4ea867e689f8914de`.

## Scope

Run all registered seeds (17, 29, 43), both predictor families (conditioned EqMotion and conditioned Transformer), both fitted cost heads, and both already-frozen policy settings. Reuse the primary checkpoint, candidate family, computation device, recording population, past-only scene queries and full 12-step forecast. Complete the primary study first. This is development-only evidence, not an independent test or calibrated safety claim.

1. Raw-frame t+50: score the exact prefix ending at offset 50 only when that offset is on the native 12-step forecast grid. Do not interpolate, reinterpret steps as seconds, refit the normalization scale, or request a newly conditioned short-horizon prediction. Preserve agents with a valid prefix but unavailable later labels. Missing endpoints are not replaced by last available positions. Easy/hard membership is inherited from the full-path baseline label; unknown membership stays unknown. Retain per-recording native coordinate errors separately.
2. Matched intervention count: obtain the independent decision using only past inputs and its predicted-risk constraints. Solve a joint decision under the same constraints and exactly that intervention count. Report count-matched nonzero queries separately from zero-count queries. Retain infeasible or time-limited cases in the full summary; report their count and do not present them as matched evidence. Equal counts do not imply equal realized harm.

No new thresholds are searched. No supplementary arm is eligible for deployment or primary model selection. Report ADE and FDE, all/easy/hard slices, scene-level paired uncertainty when independent scene count allows it, and all seed results including negative results. Two recordings at one physical site do not supply two independent scene observations. The full-horizon pair-proximity proxy is not reinterpreted as a short-prefix physical-safety measure.

## Reproduction

After both registered primary families finish, run `scripts/evaluate_m3w_forecast_supplement.py` with the v6 protocol, completed study directory, a new ignored output-cache directory and the same primary device. Use `--resume` only when the exact identity matches. Cache receipts bind checkpoints, code, policy plans and this decision. Large row-level diagnostic outputs remain local and ignored by Git.

No Stage5C execution. No SMC. No metric or seconds-level claims.
