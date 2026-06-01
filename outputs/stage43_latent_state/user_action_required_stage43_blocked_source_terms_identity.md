# User Action Required: Stage43 Blocked Source Terms / Identity

These rows are local technical candidates only. Fill the template after confirming official source identity and terms; do not treat this file as permission.

## Town-Center

- local_path: `external_data/OpenTraj/datasets/Town-Center`
- preferred_official_url_hint: `http://www.robots.ox.ac.uk/ActiveVision/Research/Projects/2009bbenfold_headpose/project.html`
- source_confidence: `low`
- terms_status: `manual_terms_required_high_risk`
- blockers: `['terms_not_confirmed_by_user', 'source_identity_not_confirmed_by_user', 'conversion_scope_not_confirmed_by_user', 'not_converted_into_stage43_feature_store']`

## Wild-Track

- local_path: `external_data/OpenTraj/datasets/Wild-Track`
- preferred_official_url_hint: `https://www.epfl.ch/labs/cvlab/data/data-wildtrack/`
- source_confidence: `high`
- terms_status: `manual_terms_or_download_page_review_required`
- blockers: `['terms_not_confirmed_by_user', 'source_identity_not_confirmed_by_user', 'conversion_scope_not_confirmed_by_user', 'not_converted_into_stage43_feature_store']`

## PETS-2009-S2L1

- local_path: `external_data/OpenTraj/datasets/PETS-2009/data`
- preferred_official_url_hint: `http://www.cvg.reading.ac.uk/PETS2009/a.html`
- source_confidence: `medium`
- terms_status: `manual_terms_review_required_before_conversion`
- blockers: `['terms_not_confirmed_by_user', 'source_identity_not_confirmed_by_user', 'conversion_scope_not_confirmed_by_user', 'not_converted_into_stage43_feature_store']`

## TrajNet_biwi

- action: locate or acquire an independent biwi-like source before repair training.
- reason: current useful support is entangled with the held-out source.
