# Past Forecast Error Within The Fixed Action Shortlist

Hypothesis: a short forecast backtest inside the observation window
improves action choice where 12-step observed forecast feedback may not
yet be available. Use exposed Moving A development only (eight observed
and twelve predicted annotation steps, native image coordinates).

Keep the eight-action family-diverse shortlist, candidate forecasts,
original direct interventions, source roles (7 fit / 6 calibrate / 1
held), three forecaster seeds, loss, ridge penalty, opening timing, and
original 2% risk/support rules fixed. Compare the exact unaugmented
risk-penalized choice to separate fits that add either:

- Past-window CV backtest: one-step, two-step, and smoothed one-step
  errors inside the eight observed positions. For each, use average
  magnitude and the recent residual projected along/across the
  candidate's third-step departure from the floor. No target array
  enters these features.
- Already-resolved forecast feedback: exact prior same-agent 12-step
  forecast errors and past recording summaries from the verified
  geometry-only cache. No outcome from a later observation enters.

Fit only on fitting source moments. Calibrate the unchanged opening
threshold grid on the six donor sources. Freeze decisions before
scoring held-source outcomes. Compare calibrated policies and, for
choice isolation, unthresholded policies on exactly the same queries
and switch counts. The matched comparison is not a safety certificate.
Report first/first-two shortlist recall, final choice recall, and
availability of resolved feedback in the 18 original zero-direct views.

Primary policy evidence: more full observed passes, no increase in
signed easy violations, and a positive 95% paired completion-lower
interval versus the unaugmented calibrated selector (3,000 source
resamples with the three seeds averaged within source). Recall and
matched-switch contrasts are secondary. No identity-fallback pass, new
thresholds from held outcomes, independent outcomes, deployment,
Stage5C, or SMC.
