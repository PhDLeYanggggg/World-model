# Secondary accuracy and geometry on the calibration-only release

Run only after the registered calibration's full inference, score and exact replay have verified. Use its frozen past-only forecasts, official 40-recording, 15-group roster and all three seeds. Do not alter the primary outcome, thresholds, eligibility, support requirement, role assignment or confirmation access. No selection, training, or model change.

Compute available-label individual ADE (mean over each agent's observed prediction steps) and available step-12 FDE only where the final point is observed. Report all, positive-easy, and hard as defined by the original fixed CV ADE thresholds. For each statistic average eligible agent rows per query, then merge query means within source/seed. Average seeds within source and macro-average sources. Bootstrap 3,000 paired source draws. Keep any missing source-seed cell explicit instead of dropping it. Available-label ADE is not complete-grid ADE.

On the same causal current-position graph as the decision controls, compute joint-minus-independent, joint-minus-scene and joint-minus-additive relative-path ADE gain. Verify matched switch counts in every query. Report observed pair gain on co-observed steps and full-grid lower/upper contrasts using the reverse-triangle forecast-disagreement debit on unknown pair steps. Preserve official switch totals and report support counts; counts represent overlapping windows, not independent observations. These geometry measures do not establish physical safety. Use image pixels and eight observation/twelve prediction annotation steps only.

This secondary analysis applies to calibration, not independent confirmation. It cannot promote a threshold, alter deployment, recover missing labels, execute Stage5C or enable SMC.
