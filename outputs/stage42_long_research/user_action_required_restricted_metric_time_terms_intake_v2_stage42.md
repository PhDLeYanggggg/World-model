# User Action Required: Stage42-HM Restricted Metric/Time Terms Intake v2

To unlock a future restricted metric/time conversion, manually fill:

- `outputs/stage42_long_research/restricted_metric_time_terms_intake_v2_template_stage42.json`

Required fields per source:

- `terms_accepted_by_user`
- `terms_acceptance_date`
- `official_terms_url`
- `accepted_terms_version_or_access_date`
- `allowed_use`
- `redistribution_allowed`
- `derived_data_allowed`
- `local_path`
- `source_identity`
- `confirmed_by_user`

Important:

- The agent cannot accept terms for you.
- The current template is not permission, not conversion, not evaluation, and not metric/seconds evidence.
- After filling, run `.venv-pytorch/bin/python run_stage42_restricted_metric_time_terms_intake_v2.py --validate-only`.
