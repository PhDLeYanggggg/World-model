# Stage42-AF Weak-Slice Validation-Margin Guard Repair

- source: `fresh_run_from_stage42x_cache_and_stage42r_validation_margin`
- generated_at_utc: `2026-05-26T06:18:30.743922+00:00`
- git_commit: `d1e0c68`
- input_hash: `587e741f5483ef968a0942c051b64ee1f655f734154b8395074dfeafda8826dc`
- gate: `13 / 13`
- verdict: `stage42_af_weak_slice_guard_repair_pass_with_eth_t50_limitation`

## Current Claim Boundary

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 dataset-local / raw-frame 2.5D 多智能体轨迹世界状态模型。
- Stage42-AF 是 validation-margin weak-slice guard repair，不重新训练大模型，不读取/提交 raw data。
- Guard rule 只使用 Stage42-R validation score 和预设 margin，不用 test 调阈值。
- Future waypoints/endpoints 只作为 labels/eval，不作为 inference input。
- t+50 / t+100 是 raw-frame horizons，不能说成 seconds-level。
- External coordinates remain dataset-local / unverified weak metric diagnostic。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Guard Rule

- rule: `validation_margin_guard`
- val_margin_threshold: `0.02`
- fallback_action: force floor for non-UCY domain|horizon source choices with validation score below threshold
- uses_test_metrics_for_threshold: `False`

## Summary

- ADE all: `0.09068229132363603`
- ADE t50: `0.06109367671246102`
- ADE t50 CI low: `0.05367075264893123`
- ADE t100 raw-frame diagnostic: `0.08153326024168321`
- ADE hard/failure: `0.09464861943836626`
- easy degradation CI high: `0.006232934803070727`
- switch_rate: `0.22896556692119294`

## Repair Effect

- horizon25 ADE before: `-0.004781149088858072`
- horizon25 ADE after: `0.0`
- horizon25 delta: `0.004781149088858072`
- ETH_UCY t50 ADE before: `0.017092525274558956`
- ETH_UCY t50 ADE after: `0.017092525274558956`
- ETH_UCY t50 CI low after: `-0.013218100958604987`
- ETH_UCY FDE@50 CI low after: `-0.04199023614248535`
- ETH_UCY t50 limitation remaining: `True`

## Per-Horizon Stress After Guard

| horizon | rows | ADE all | ADE all low | hard | hard low | switch rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `10` | 16726 | 0.201453 | 0.157234 | 0.212007 | 0.165025 | 0.372813 |
| `25` | 15208 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| `50` | 13689 | 0.061094 | 0.053671 | 0.061094 | 0.053671 | 0.273309 |
| `100` | 9905 | 0.081533 | 0.052781 | 0.081533 | 0.052781 | 0.276325 |

## Guarded Keys By Seed

- pair `0` guarded `ETH_UCY|25, TrajNet|25` rows={'ETH_UCY|25': 6823, 'TrajNet|25': 5685}
- pair `1` guarded `ETH_UCY|25, TrajNet|25` rows={'ETH_UCY|25': 6823, 'TrajNet|25': 5685}
- pair `2` guarded `ETH_UCY|25, TrajNet|25` rows={'ETH_UCY|25': 6823, 'TrajNet|25': 5685}

## Conclusion

Stage42-AF repairs the Stage42-AE horizon=25 weak slice by a validation-only low-margin guard: horizon=25 moves from negative to non-harm/floor. Global all/t50/hard remain positive and easy degradation stays under 2%. This is a real safety improvement, but ETH_UCY t50/FDE@50 lower-bound weakness remains; it must stay in the paper limitations rather than being overclaimed.
