# Stage42-BM ETH-Person Terms / Official-Use Audit

- source: `fresh_eth_person_terms_audit`
- generated_at_utc: `2026-05-26T13:46:36.954815+00:00`
- git_commit: `a715fa1`
- input_hash: `6bf21f5760d51a02e590623910244f7b02713f448c2b783e11d0bdbe14f1d802`
- gate: `14 / 14`
- verdict: `stage42_bm_eth_person_terms_audit_pass_claim_blocked`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-BM 是 ETH-Person XML terms / official-use audit，不训练模型，不下载数据。
- Stage42-BL 的 ETH-Person XML t100 result 是 technical dry-run，terms 未确认前不能升级为 official converted/evaluated result。
- OpenTraj 根目录 MIT license 适用于 toolkit/software，不能自动覆盖第三方 trajectory datasets。
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- t100 仍是 raw-frame diagnostic，不能写成 seconds-level。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Stage42-BL Technical Result Preserved

- BL verdict: `stage42_bl_eth_person_xml_t100_dry_run_pass`
- technical_t100_all_folds_safe_positive: `True`
- technical_t100_mean_improvement_vs_fallback: `0.6835494566410742`
- technical_t100_min_improvement_vs_fallback: `0.4964237263017553`
- technical_t100_max_easy_degradation: `-0.014155078403600982`
- BL t100 windows total: `1485`

## Local Terms / License Audit

- OpenTraj license path: `external_data/OpenTraj/LICENSE.txt`
- OpenTraj license name: `MIT`
- OpenTraj license scope classification: `software_toolkit_only`
- OpenTraj toolkit license can cover ETH-Person dataset: `False`
- ETH README has no-license statement: `True`
- ETH-Person local terms files found: `[]`
- ETH-Person official URL from OpenTraj README: `https://data.vision.ee.ethz.ch/cvl/aess/`
- official terms verified: `False`

## Claim Boundary

- license_terms_confirmed: `False`
- official_converted_dataset_claim_allowed: `False`
- deployable_t100_claim_allowed: `False`
- global_t100_positive_claim_allowed: `False`
- next_stage_official_conversion_allowed: `False`

## Interpretation

- Stage42-BL remains useful technical evidence: the XML loader/source-CV path works and is strongly positive under dry-run conditions.
- The local repository does not include ETH-Person-specific license or terms files.
- The OpenTraj MIT license is treated as toolkit/software license only; it is not accepted as permission for the underlying ETH-Person dataset.
- Therefore ETH-Person XML cannot be counted as official converted/evaluated data until the user confirms official terms or provides an official permission/terms link.
- This audit intentionally blocks deployable/global t100 claims despite the positive technical result.
