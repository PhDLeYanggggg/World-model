# Stage42-BC User Action Required For T100 Source Acquisition

- source: `fresh_synthesis_from_stage42_bb_plus_official_web_pages`
- purpose: official-source and local-path actions needed to repair t100 source support without bypassing terms.

## ETH_UCY

- priority: `high`
- action: Provide or approve legal source-specific t100-capable train data, then rerun source-CV without using test metrics.
- minimum_extra_t100_sources_needed: `2`
- candidate_source_ids: `['ucy_crowd_original', 'trajnetpp_epfl_aicrowd', 'opentraj_toolkit', 'eth_ucy_original_sources']`

## TrajNet

- priority: `high`
- action: Provide or approve legal source-specific t100-capable train data, then rerun source-CV without using test metrics.
- minimum_extra_t100_sources_needed: `1`
- candidate_source_ids: `['trajnetpp_epfl_aicrowd', 'opentraj_toolkit']`

## UCY

- priority: `high`
- action: Provide or approve legal source-specific t100-capable train data, then rerun source-CV without using test metrics.
- minimum_extra_t100_sources_needed: `1`
- candidate_source_ids: `['ucy_crowd_original', 'trajnetpp_epfl_aicrowd', 'opentraj_toolkit', 'eth_ucy_original_sources']`

## ucy_crowd_original

- priority: `high`
- action: verify local path and official terms; do not auto-download raw/gated/restricted data
- dataset_name: UCY Crowd
- official_url: `http://graphics.cs.ucy.ac.cy/research/downloads/crowd-data`
- local_paths_found: `['external_data/OpenTraj/datasets/UCY']`
- blocked_reasons:
  - requires user confirmation of official terms/login/challenge access
  - auto_download_allowed is false by policy

## trajnetpp_epfl_aicrowd

- priority: `high`
- action: verify local path and official terms; do not auto-download raw/gated/restricted data
- dataset_name: TrajNet++
- official_url: `https://www.epfl.ch/schools/enac/about/open-science/open-research/trajnet/`
- local_paths_found: `['external_data/OpenTraj/datasets/TrajNet', 'external_data/OpenTraj/datasets/TrajNet++']`
- blocked_reasons:
  - requires user confirmation of official terms/login/challenge access
  - auto_download_allowed is false by policy

## opentraj_toolkit

- priority: `medium`
- action: verify local path and official terms; do not auto-download raw/gated/restricted data
- dataset_name: OpenTraj toolkit
- official_url: `https://github.com/crowdbotp/OpenTraj`
- local_paths_found: `['external_data/OpenTraj', '/Users/yangyue/Downloads/World/external_data/OpenTraj']`
- blocked_reasons:
  - auto_download_allowed is false by policy

## eth_ucy_original_sources

- priority: `medium`
- action: verify local path and official terms; do not auto-download raw/gated/restricted data
- dataset_name: ETH/UCY original pedestrian sources
- official_url: `ETH/BIWI and UCY original dataset pages; source-specific terms must be manually verified`
- local_paths_found: `['external_data/OpenTraj/datasets/ETH', 'external_data/OpenTraj/datasets/ETH-Person', 'external_data/OpenTraj/datasets/UCY']`
- blocked_reasons:
  - requires user confirmation of official terms/login/challenge access
  - official source not fully resolved in Stage42-BC
  - auto_download_allowed is false by policy

## aerialmpt_dlr

- priority: `medium`
- action: verify local path and official terms; do not auto-download raw/gated/restricted data
- dataset_name: AerialMPT
- official_url: `https://www.dlr.de/en/eoc/about-us/remote-sensing-technology-institute/photogrammetry-and-image-analysis/public-datasets/aerialmpt-a-dataset-for-pedestrian-tracking-in-aerial-imagery`
- local_paths_found: `['data/aerialmpt/DLR_AerialMPT_Dataset.zip', 'data/stage11_scene_packs/aerialmpt']`
- blocked_reasons:
  - license restricts derivative/commercial/redistribution use; keep manual review
  - auto_download_allowed is false by policy

## sdd_stanford

- priority: `medium`
- action: verify local path and official terms; do not auto-download raw/gated/restricted data
- dataset_name: Stanford Drone Dataset
- official_url: `https://cvgl.stanford.edu/projects/uav_data/`
- local_paths_found: `['external_data/StanfordDroneDataset', '/Users/yangyue/Downloads/World/external_data/StanfordDroneDataset', 'data/stage21_sdd_world_state']`
- blocked_reasons:
  - requires user confirmation of official terms/login/challenge access
  - auto_download_allowed is false by policy

## Non-Claims

- Do not call AerialMPT/SDD/TGSIM evidence a repair for ETH_UCY/TrajNet/UCY t100 unless a separate source-level protocol proves it.
- Do not claim metric or seconds-level pedestrian prediction from local homography/FPS hints alone.
- Do not auto-download or redistribute gated/restricted/raw third-party data.
