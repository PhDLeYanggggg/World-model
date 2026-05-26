# Stage42-BE Local T100 Conversion Readiness

- source: `fresh_local_conversion_readiness`
- generated_at_utc: `2026-05-26T12:30:34.975942+00:00`
- git_commit: `f6e0265`
- input_hash: `8a4f5287b20b2a3f161e8885992c6b531906544ef2c8b2fe042e878fa7ff1666`
- gate: `12 / 12`
- verdict: `stage42_be_local_t100_conversion_readiness_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-BE 是 local t100 conversion-readiness / no-leakage audit，不训练模型、不写大 feature store。
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- 本步骤不把 local candidate 写成 official converted dataset；full conversion / source-CV 仍需后续执行。
- t100 仍是 raw-frame diagnostic / blocker，不能写成 seconds-level。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Summary

- candidate_files: `4`
- schema_conversion_ready_files: `4`
- estimated_t10_windows: `51061`
- estimated_t25_windows: `32451`
- estimated_t50_windows: `15813`
- estimated_t100_windows: `6257`
- domains_with_source_cv_after_conversion: `UCY`
- full_feature_store_written: `False`
- stage42_bf_actual_conversion_recommended: `True`

## Source Readiness

| source | domain | rows | agents | t50 | t100 | max track | common step | gap ratio | schema ready |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ETH/seq_eth/biwi_eth_10fps.txt` | ETH_UCY | 5492 | 360 | 82 | 14 | 114 | 10 | 0.000 | True |
| `UCY/students01/students001.txt` | UCY | 21813 | 415 | 6445 | 1949 | 352 | 10 | 0.000 | True |
| `UCY/students03/obsmat_px.txt` | UCY | 21859 | 428 | 6493 | 3415 | 540 | 10 | 0.002 | True |
| `UCY/students03/students003.txt` | UCY | 17953 | 434 | 2793 | 879 | 289 | 10 | 0.000 | True |

## Source-CV Readiness Plan

| domain | t100-capable sources | estimated t100 windows | source-CV feasible after conversion |
| --- | ---: | ---: | ---: |
| `ETH_UCY` | 1 | 14 | False |
| `UCY` | 3 | 6243 | True |

## Interpretation

- Stage42-BE verifies that the local novel candidates can be mapped to the external row schema, but it does not write the full feature store.
- UCY has enough novel local t100-capable sources for a source-CV readiness plan after actual conversion.
- ETH_UCY gains one small t100-capable source but remains insufficient for independent t100 support by itself.
- t100 remains raw-frame diagnostic / blocker until Stage42-BF actual conversion and train-only source-CV pass.
