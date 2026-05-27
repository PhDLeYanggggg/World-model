# User Action Required: Stage42-GO Official Source / Terms Live Verifier

No automatic download or conversion is allowed yet. The agent did not accept terms.

Please confirm these fields manually for the top-priority sources if you want conversion to proceed:

## 1. ucy_crowd_original

- official_url: `https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data`
- live_status: `official_url_known_but_page_unavailable_in_live_audit`
- terms_status: `not_verified_by_agent`
- access_status: `manual_terms_or_credit_review_required`
- user_must_confirm: official terms URL/version or access date, allowed use, redistribution policy, derived-data policy, local path/source identity
- suggested next action: user fills official terms/path/source identity; rerun validator -> contract -> guarded harness

## 2. eth_biwi_original

- official_url: `https://vision.ee.ethz.ch/datsets.html`
- live_status: `official_page_reachable_with_dataset_download_links`
- terms_status: `not_verified_by_agent`
- access_status: `manual_terms_or_credit_review_required`
- user_must_confirm: official terms/version or access date, allowed use, redistribution policy, derived-data policy, local path/source identity, annotation frame rate / H-matrix convention if claiming restricted metric/time subset
- suggested next action: user fills official terms/path/source identity; rerun validator -> contract -> guarded harness

## 3. opentraj_toolkit

- official_url: `https://github.com/crowdbotp/OpenTraj`
- live_status: `official_github_reachable_toolkit_license_only`
- terms_status: `toolkit_mit_not_underlying_dataset_terms`
- access_status: `toolkit_usable_under_mit_but_dataset_terms_separate`
- user_must_confirm: which underlying dataset source is being used, underlying dataset official terms, allowed use, local path/source identity
- suggested next action: user fills official terms/path/source identity; rerun validator -> contract -> guarded harness

After manual confirmation, rerun validator -> source conversion contract -> guarded conversion harness.
