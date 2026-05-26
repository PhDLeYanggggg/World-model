# Stage42-BJ User Action Required

- source: `fresh_post_bi_t100_source_package`
- purpose: obtain legal independent t100-capable sources for domains still blocking global t100 support.

## ETH_UCY

- priority: `critical`
- additional_independent_t100_sources_needed: `2`
- next_action: `provide_or_approve_legal_independent_t100_sources_then_rerun_conversion_and_source_cv`

### ucy_crowd_original / UCY Crowd

- official_url: `http://graphics.cs.ucy.ac.cy/research/downloads/crowd-data`
- local_path_found: `True`
- found_paths: `['external_data/OpenTraj/datasets/UCY']`
- auto_download_allowed: `False`
- blocked_reasons:
  - requires user confirmation of official terms/login/challenge access
  - auto_download_allowed is false by policy
- expected_t100_role: adds the missing independent UCY t100-capable original-train source if legally available

### trajnetpp_epfl_aicrowd / TrajNet++

- official_url: `https://www.epfl.ch/schools/enac/about/open-science/open-research/trajnet/`
- local_path_found: `True`
- found_paths: `['external_data/OpenTraj/datasets/TrajNet', 'external_data/OpenTraj/datasets/TrajNet++']`
- auto_download_allowed: `False`
- blocked_reasons:
  - requires user confirmation of official terms/login/challenge access
  - auto_download_allowed is false by policy
- expected_t100_role: adds/validates independent trajectory sources for TrajNet and possibly ETH/UCY-style splits

### opentraj_toolkit / OpenTraj toolkit

- official_url: `https://github.com/crowdbotp/OpenTraj`
- local_path_found: `True`
- found_paths: `['external_data/OpenTraj', '/Users/yangyue/Downloads/World/external_data/OpenTraj']`
- auto_download_allowed: `False`
- blocked_reasons:
  - auto_download_allowed is false by policy
- expected_t100_role: source discovery/loader hub; may expose additional legal source files already local

### eth_ucy_original_sources / ETH/UCY original pedestrian sources

- official_url: `ETH/BIWI and UCY original dataset pages; source-specific terms must be manually verified`
- local_path_found: `True`
- found_paths: `['external_data/OpenTraj/datasets/ETH', 'external_data/OpenTraj/datasets/ETH-Person', 'external_data/OpenTraj/datasets/UCY']`
- auto_download_allowed: `False`
- blocked_reasons:
  - requires user confirmation of official terms/login/challenge access
  - official source not fully resolved in Stage42-BC
  - auto_download_allowed is false by policy
- expected_t100_role: highest-priority source-level repair target for ETH_UCY and UCY t100 support

## TrajNet

- priority: `critical`
- additional_independent_t100_sources_needed: `3`
- next_action: `provide_or_approve_legal_independent_t100_sources_then_rerun_conversion_and_source_cv`

### trajnetpp_epfl_aicrowd / TrajNet++

- official_url: `https://www.epfl.ch/schools/enac/about/open-science/open-research/trajnet/`
- local_path_found: `True`
- found_paths: `['external_data/OpenTraj/datasets/TrajNet', 'external_data/OpenTraj/datasets/TrajNet++']`
- auto_download_allowed: `False`
- blocked_reasons:
  - requires user confirmation of official terms/login/challenge access
  - auto_download_allowed is false by policy
- expected_t100_role: adds/validates independent trajectory sources for TrajNet and possibly ETH/UCY-style splits

### opentraj_toolkit / OpenTraj toolkit

- official_url: `https://github.com/crowdbotp/OpenTraj`
- local_path_found: `True`
- found_paths: `['external_data/OpenTraj', '/Users/yangyue/Downloads/World/external_data/OpenTraj']`
- auto_download_allowed: `False`
- blocked_reasons:
  - auto_download_allowed is false by policy
- expected_t100_role: source discovery/loader hub; may expose additional legal source files already local

## Non-Claims

- UCY local t100 support does not establish global t100 success.
- Registry-only datasets, missing paths, gated sources, or failed downloads must not be counted as converted/evaluated data.
- Dataset-local/raw-frame horizons must not be reported as metric or seconds-level trajectories.
