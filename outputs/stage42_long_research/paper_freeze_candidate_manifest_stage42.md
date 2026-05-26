# Stage42-CZ Paper Freeze Candidate Manifest

- source: `fresh_freeze_candidate_manifest_from_cx_cy`
- generated_at_utc: `2026-05-26T20:33:18.405328+00:00`
- git_commit: `16fed1c`
- manifest_hash: `bb6689cae4a1deb1ba2684d5b4814952267726fbec94767868510d4cfa5a3cce`
- gate: `14 / 14`
- verdict: `stage42_cz_paper_freeze_candidate_manifest_pass`
- freeze_status: `candidate_clean`
- final_immutable_release: `True`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-CZ 是 paper freeze candidate manifest，不重新训练，不调 threshold。
- manifest 冻结的是证据包候选，不是 broad foundation/3D/metric/seconds claim。
- 若仍有 metadata-only caveats，状态必须写 candidate_with_metadata_caveats，而不是 final immutable release。
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 不能写成 global metric。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Summary

- files_total: `74`
- paper_files: `9`
- evidence_json_files: `21`
- evidence_md_files: `20`
- runner_files: `21`
- files_with_git_caveat: `0`

## Files

| role | path | size | sha256 | git status |
| --- | --- | ---: | --- | --- |
| `paper_file` | `outputs/stage42_long_research/paper_outline_stage42.md` | 8120 | `297405c91e2d85f15435807b40eaee4076f6282daf3e8213d7422cbe0b2304cb` | `clean` |
| `paper_file` | `outputs/stage42_long_research/method_draft_stage42.md` | 8428 | `eda8264c964ae30672f744f00a5b07789fc70010f623d45bc263ecb813746311` | `clean` |
| `paper_file` | `outputs/stage42_long_research/experiment_tables_stage42.md` | 19032 | `75dd9032bdc62bde575de1b572ea67553ff28cb40252987ffadb7ea8c7a2b039` | `clean` |
| `paper_file` | `outputs/stage42_long_research/ablation_tables_stage42.md` | 26430 | `0e087615e5587d93f2d416b1c6c75fa422b97e0d45c179007091d7444138678a` | `clean` |
| `paper_file` | `outputs/stage42_long_research/failure_taxonomy_stage42.md` | 9179 | `1728e27fe9c8b0c4d549bc9fd92cd76f851d5cfd553250f86bf69f67327f75a6` | `clean` |
| `paper_file` | `outputs/stage42_long_research/model_card_stage42.md` | 16747 | `9ba9d4736267802ac875d70ed24dd629370ba44565b06e6e7598ec066be9c428` | `clean` |
| `paper_file` | `outputs/stage42_long_research/data_card_stage42.md` | 5949 | `73a574b452323ec81a924c9add81bd10d41680eef0c33413e7721214e51c84dc` | `clean` |
| `paper_file` | `outputs/stage42_long_research/reproducibility_stage42.md` | 10224 | `ac43411a752c79ef8a10eb57a29d664f4e10e4774bb45da19d09d2a79cc557dc` | `clean` |
| `paper_file` | `outputs/stage42_long_research/a_journal_gap_stage42.md` | 35389 | `92a65874c7dd19c4455364766f0d13f4964ffafb28cf223837baaf37588ed054` | `clean` |
| `frozen_runtime_policy_artifact` | `outputs/stage42_long_research/frozen_proximity_guard_composer_policy_stage42_policy.json` | 3414 | `8cdfcc939a0f6c8261f2155e1a9a28efab1a691a7603859779d813e810a63707` | `clean` |
| `evidence_json:data_calibration` | `outputs/stage42_long_research/data_calibration_stage42.json` | 74705 | `9603ce75ba49f7cdb6c0abeaafe6c1b5bb9940262dec3999792ceb806e0c94c0` | `clean` |
| `evidence_md:data_calibration` | `outputs/stage42_long_research/data_calibration_stage42.md` | 9490 | `0e1716b81a6a2760aa7a750fad74f187ee455b4fd7e91365c4f5d6504d5612de` | `clean` |
| `runner:data_calibration` | `run_stage42_data_calibration.py` | 136 | `2f9afd769b91493e32ac5f39925b83946bb3b7a9bab207bf47db47a39b823895` | `clean` |
| `evidence_json:external_validation` | `outputs/stage42_long_research/external_validation_stage42.json` | 3841495 | `739691e8db5421106581d375792e3c67ec15df5b167dd63b516d9b12d7a39136` | `clean` |
| `evidence_md:external_validation` | `outputs/stage42_long_research/external_validation_stage42.md` | 3189 | `7fbd6b71ad5c89cde21f8ee5827037027195ae4eb690354a9740196b651b81e0` | `clean` |
| `runner:external_validation` | `run_stage42_external_validation.py` | 143 | `755c28007e800ee59bf85ed590835542b26f3091f2c89be40b171edcab22257e` | `clean` |
| `evidence_json:full_waypoint_dynamics` | `outputs/stage42_long_research/full_waypoint_dynamics_stage42.json` | 39554 | `80c16a7ec02d7e9cee0c9be02792e57d5dc272f67478734fb7ac1c048129cf89` | `clean` |
| `evidence_md:full_waypoint_dynamics` | `outputs/stage42_long_research/full_waypoint_dynamics_stage42.md` | 3789 | `885382dccced48c51569dc34160238f917350fb7224a094f63dfd962de5250ae` | `clean` |
| `runner:full_waypoint_dynamics` | `run_stage42_full_waypoint_dynamics.py` | 152 | `e5e18802641864facf37cb6559d19299bf3dafeb9f543a0f6c19eb1827c6600d` | `clean` |
| `evidence_json:causal_ablation` | `outputs/stage42_long_research/causal_ablation_stage42.json` | 16738 | `4dba156313d7bf1d6c3d416ffb6984192cec21dfc0fd794ecf7b19c14c29fb18` | `clean` |
| `evidence_md:causal_ablation` | `outputs/stage42_long_research/causal_ablation_stage42.md` | 8282 | `0d9332f299ea1073c7fec4c3b4531356a18095415b3b7de865b7595b78f61522` | `clean` |
| `runner:causal_ablation` | `run_stage42_causal_ablation.py` | 131 | `0aa61ac98393998ec6ca904516f641bbc1afbc81f19f062eef594278e302f481` | `clean` |
| `evidence_json:safety_floor` | `outputs/stage42_long_research/safety_floor_stage42.json` | 27310 | `5b73af2deefff3d5110005ed62b93353f6756380db14d44182774114057182c4` | `clean` |
| `evidence_md:safety_floor` | `outputs/stage42_long_research/safety_floor_stage42.md` | 5503 | `b93041e0131ca61cccd4da98486e7a9d389fb4305f484067d54e4b04140f93f9` | `clean` |
| `runner:safety_floor` | `run_stage42_safety_floor.py` | 122 | `5259ebd2dbeb5a64b7aff53a3f78eb192d90d66f1a54caf35cd7e65f96d25c50` | `clean` |
| `evidence_json:paper_package` | `outputs/stage42_long_research/paper_package_stage42.json` | 9537 | `4ec71440d909cd2beda738b57a7b8ca82329e4089da8333778e6b52156449b9e` | `clean` |
| `runner:paper_package` | `run_stage42_paper_package.py` | 125 | `e4197e224712883040dcac5244e14c071b250279e6b473733fe8c7bc95353f61` | `clean` |
| `evidence_json:strict_time_geometry_calibration` | `outputs/stage42_long_research/source_time_geometry_calibration_stage42.json` | 15612 | `7a5d8a525a008545c95c9adb729de789773c309884231cefee416a72c72a8a40` | `clean` |
| `evidence_md:strict_time_geometry_calibration` | `outputs/stage42_long_research/source_time_geometry_calibration_stage42.md` | 4205 | `c650770f0020121b497e01c8150c86dd306966462289891c46c7c574f99eb985` | `clean` |
| `runner:strict_time_geometry_calibration` | `run_stage42_source_time_geometry_calibration.py` | 295 | `92e4a6fcddadd10339c4e5ef2aa15a975a5557dbdb072cb8f68cec4e1e0b59e0` | `clean` |
| `evidence_json:metric_time_claim_guard` | `outputs/stage42_long_research/metric_time_claim_guard_stage42.json` | 7724 | `21acab6b57b200e224d3d5c75dce34d86d1541f34814ac10451a930db0f0cb4c` | `clean` |
| `evidence_md:metric_time_claim_guard` | `outputs/stage42_long_research/metric_time_claim_guard_stage42.md` | 3057 | `13ddb703c4b5c2953b16d657bf7f68f46afe919f5f504c2a1b09e55267f06f05` | `clean` |
| `runner:metric_time_claim_guard` | `run_stage42_metric_time_claim_guard.py` | 155 | `0fa564ef5fad34202ba8ec5f48b87b2a97c28cf676a0083d8a1680a02c883fda` | `clean` |
| `evidence_json:source_terms_validation` | `outputs/stage42_long_research/source_terms_validation_stage42.json` | 14852 | `3712d080967d228e062952973614690aff69406e0ad8bb9baf6dd805c6a8b8db` | `clean` |
| `evidence_md:source_terms_validation` | `outputs/stage42_long_research/source_terms_validation_stage42.md` | 3060 | `55dd6cbe95f3f3575eb15eb9683524ba7275e0582a139dbd632718160b55487a` | `clean` |
| `runner:source_terms_validation` | `run_stage42_source_terms_confirmation_validator.py` | 191 | `91fcab4829f5defb379c5c405300df64b8143a0131ab1963ce27101f1dc1ef2c` | `clean` |
| `evidence_json:context_contribution_forensics` | `outputs/stage42_long_research/context_contribution_forensics_stage42.json` | 8022 | `2640333f20e52405de58920d65aef77200ec0938186b6bf88e786217809ed0dd` | `clean` |
| `evidence_md:context_contribution_forensics` | `outputs/stage42_long_research/context_contribution_forensics_stage42.md` | 4915 | `dbbea434935b3b561345e8121bb16af88bed7325c62aecd9d3d57e6291186bf2` | `clean` |
| `runner:context_contribution_forensics` | `run_stage42_context_contribution_forensics.py` | 176 | `0a62177d9c3166315e1b0aed2261742c9c2661ea765eb532b561b730da018cb3` | `clean` |
| `evidence_json:goal_scene_gated_expert` | `outputs/stage42_long_research/goal_scene_gated_expert_stage42.json` | 42002 | `d80b5cf912bc44ccfcd9b89f51ac37e315a98bf49bc67a9ff9ccb67becc15f71` | `clean` |
| `evidence_md:goal_scene_gated_expert` | `outputs/stage42_long_research/goal_scene_gated_expert_stage42.md` | 4628 | `8183dfa140d739f28fb2d8d81f1b9f338dacb6ba55e416ea3a631656d81c2d1f` | `clean` |
| `runner:goal_scene_gated_expert` | `run_stage42_goal_scene_gated_expert.py` | 155 | `20b9767154ff1fd708b1034dd19eb885ff0781c0a139a6020431e3f45c627e62` | `clean` |
| `evidence_json:neighbor_interaction_gated_expert` | `outputs/stage42_long_research/neighbor_interaction_gated_expert_stage42.json` | 85532 | `883118db5f5758e59f2248bf4783f739b9e8628b1df1d0da2d25894af2b2f1d4` | `clean` |
| `evidence_md:neighbor_interaction_gated_expert` | `outputs/stage42_long_research/neighbor_interaction_gated_expert_stage42.md` | 7025 | `aa978fabc031362f9c92dd529b4b5b40b108135d06744818cb2d0d0139d03a17` | `clean` |
| `runner:neighbor_interaction_gated_expert` | `run_stage42_neighbor_interaction_gated_expert.py` | 185 | `26b4d095ee96f3a333534197827418f10e11d7ebb0a2ff0d76272bd4bcbd0fbe` | `clean` |
| `evidence_json:common_validation_bridge_shape_composer` | `outputs/stage42_long_research/common_validation_bridge_shape_composer_stage42.json` | 91625 | `e98658f341e1985741f4e26f938c30ee89955b6c99836fccc3c6ea8e33866340` | `clean` |
| `evidence_md:common_validation_bridge_shape_composer` | `outputs/stage42_long_research/common_validation_bridge_shape_composer_stage42.md` | 2074 | `fcef7fbff6891e30db1c7e9e8a1c920611f7e68f14773b2c21fc96f86c644992` | `clean` |
| `runner:common_validation_bridge_shape_composer` | `run_stage42_common_validation_bridge_shape_composer.py` | 261 | `1b6de9d33c89389597bc45121038a84e10f9a34addebf5d8f4419d653e7a1d8f` | `clean` |
| `evidence_json:composer_safety_bootstrap` | `outputs/stage42_long_research/common_validation_composer_safety_stage42.json` | 112593 | `03c2eb0528e17b1e8600a8450d9f0481af542c9855f3de9a7f4adade1db37b13` | `clean` |
| `evidence_md:composer_safety_bootstrap` | `outputs/stage42_long_research/common_validation_composer_safety_stage42.md` | 2277 | `a28cc9d830b397426abcca74a81e3bc08ef1aded24b0fe9f531dd3436a18927d` | `clean` |
| `runner:composer_safety_bootstrap` | `run_stage42_common_validation_composer_safety.py` | 244 | `9dd1e1c391da62aa4b8c8c9187f0ea88a76c51f765112e15f53fd02bfc2f0457` | `clean` |
| `evidence_json:proximity_aware_composer_guard` | `outputs/stage42_long_research/proximity_aware_composer_guard_stage42.json` | 274975 | `9d15e7338270fb43ef1eb1c8be7c87995054a9113f12f4d1a660393064258c79` | `clean` |
| `evidence_md:proximity_aware_composer_guard` | `outputs/stage42_long_research/proximity_aware_composer_guard_stage42.md` | 2740 | `f8ea21a56524e491e5956d7a705f57647f1310cba6311a0c37bd5f97452ed84f` | `clean` |
| `runner:proximity_aware_composer_guard` | `run_stage42_proximity_aware_composer_guard.py` | 248 | `9c99ab4100a7940bea2e20f2925cdced56b0c6cd8f739083009fdca470067a28` | `clean` |
| `evidence_json:proximity_guard_ablation` | `outputs/stage42_long_research/proximity_guard_ablation_stage42.json` | 536421 | `6b290ecd9858c7c86dd39542f6759ab82072f4605be97638c5ae6ef95b38798c` | `clean` |
| `evidence_md:proximity_guard_ablation` | `outputs/stage42_long_research/proximity_guard_ablation_stage42.md` | 2620 | `973db8c29ecea85dd52be1ec05a8edfc84e66f3ad6d9a806f1092b2c3eb2f527` | `clean` |
| `runner:proximity_guard_ablation` | `run_stage42_proximity_guard_ablation.py` | 242 | `6a6f70bdce9fd4e11e680aa9399fc945c885a2810606934c7f0a7bbaefe9a646` | `clean` |
| `evidence_json:frozen_proximity_guard_policy` | `outputs/stage42_long_research/frozen_proximity_guard_composer_policy_stage42.json` | 8534 | `30fad57044f009a6b01b5092d73020764998eceb776680e31df68e97802633e8` | `clean` |
| `evidence_md:frozen_proximity_guard_policy` | `outputs/stage42_long_research/frozen_proximity_guard_composer_policy_stage42.md` | 2678 | `66b0b75f410d15960aab28c4c655aafe02196c831feb6820e4802e68a1ff3d0a` | `clean` |
| `runner:frozen_proximity_guard_policy` | `run_stage42_freeze_proximity_guard_policy.py` | 173 | `44d87377587f595963858a859b4626b77cac095eb878be3f87a37f06a1477e77` | `clean` |
| `evidence_json:frozen_policy_replay` | `outputs/stage42_long_research/frozen_proximity_guard_policy_replay_stage42.json` | 11827 | `f4e0e86d119139f9df234be9647541fe1919420ff2cfdf4b0f19e161e03caf4a` | `clean` |
| `evidence_md:frozen_policy_replay` | `outputs/stage42_long_research/frozen_proximity_guard_policy_replay_stage42.md` | 2162 | `4fd18daab4eb6696eef377e588f55adc7bdbffd62a9e5027190e494e05f67442` | `clean` |
| `runner:frozen_policy_replay` | `run_stage42_replay_proximity_guard_policy.py` | 173 | `03d19a195208680faf460d3af0d2368bdd1927604c4075526189ffe01fd13cab` | `clean` |
| `evidence_json:runtime_policy_api` | `outputs/stage42_long_research/proximity_guard_runtime_policy_stage42.json` | 11386 | `72f51caaaf17020f781bb9cec42ad02150c3741bb72ccf422df0583487337de9` | `clean` |
| `evidence_md:runtime_policy_api` | `outputs/stage42_long_research/proximity_guard_runtime_policy_stage42.md` | 2566 | `157e24311de9682ac41a66ee02753243da643aeeed09f068da719862904e5861` | `clean` |
| `runner:runtime_policy_api` | `run_stage42_runtime_proximity_guard_policy.py` | 176 | `75982b8bc544a1dfe2dbc82783b519e67d2f94aac9040e68083d248feb3a68dd` | `clean` |
| `evidence_json:batch_runtime_replay` | `outputs/stage42_long_research/proximity_guard_batch_replay_stage42.json` | 25316 | `c688f59e4c563b91c68be27310edd40ac014646603f536d0ec48d96fbc1873ed` | `clean` |
| `evidence_md:batch_runtime_replay` | `outputs/stage42_long_research/proximity_guard_batch_replay_stage42.md` | 2475 | `5a7790865bea9850bea1df32321fa3da4653dd2a2789d8be171f4e7ecd6df70f` | `clean` |
| `runner:batch_runtime_replay` | `run_stage42_batch_replay_proximity_guard_policy.py` | 184 | `962578e2628b7e3c117decde8c63298d25a691c15fb7aed103b0fea71644155e` | `clean` |
| `evidence_json:runtime_replay_paper_refresh` | `outputs/stage42_long_research/runtime_replay_paper_refresh_stage42.json` | 60102 | `4daf68dcf18ab429052b9f94ec35b8001c3070b9492c62fbfd5a40bd2ff58828` | `clean` |
| `evidence_md:runtime_replay_paper_refresh` | `outputs/stage42_long_research/runtime_replay_paper_refresh_stage42.md` | 2826 | `8ace17b2d1d2a7ad6d8e18894aaa3c39057a5c027faa76599a78b694d3786e3a` | `clean` |
| `runner:runtime_replay_paper_refresh` | `run_stage42_runtime_replay_paper_refresh.py` | 170 | `47568296ec4774c160d2b75f351837d70b4f230969896ce7fa86488b08c4071b` | `clean` |
| `provenance_verifier_json` | `outputs/stage42_long_research/evidence_provenance_stage42.json` | 23656 | `1fc0a9030d5b38046fb507d0e49586661fd327bcbcef723ebf7c26f8d2db1019` | `clean` |
| `worktree_caveat_classifier_json` | `outputs/stage42_long_research/worktree_caveat_classifier_stage42.json` | 4151 | `ac69d14565cd4f4d9b13873f1f6c80a21f4735efeb8b7ab984a18dcab91fb567` | `clean` |

## Interpretation

- Stage42-CZ creates a hash manifest for the current paper evidence candidate.
- Because CY records metadata-only Stage42 caveats, this is a candidate manifest, not a final immutable release.
- The supported claim remains protected dataset-local/raw-frame 2.5D multi-agent world-state evidence.
