# Zara Past-Media Source Adapter

This is an adaptive fit-only follow-up to the source lineage audit, not an
independent confirmation or a new forecasting experiment. Inspection has already
shown that Zara02's H.txt discrepancy is approximately a constant translation.
The follow-up tests that hypothesis rather than claiming it was unknown.

## Frozen Checks

- Keep the supplied H.txt, image row/column convention and stored-frame-minus-one
  annotation index. Do not fit a new matrix, time lag or scale.
- Zara01 keeps zero origin correction. Zara02 uses exactly the first canonical
  source row and its matching raw control to define a two-coordinate translation.
  Check all remaining source rows and consecutive displacement differences;
  do not use them to estimate the translation.
- This repairs numerical coordinate lineage only. No physical scale, original
  calibration accuracy, capture clock or online identity availability is proved.
- Select the first complete eight-step history for each of the first 12 eligible
  agent IDs in each recording. Do not require future survival or future motion.
- Decode only those current/past indexed frames. Keep out-of-bounds crops visible
  and count them. Save private contact sheets and row-level mappings locally.
- Preserve sparse-control provenance. An annotation time at or before the query
  is not sufficient to establish that its interpolating control was available then.
- The first raw anchor must be an exact control, available no later than every
  inspected query. No future endpoint or future target is used by the adapter.

## Scope and Pending Decisions

The existing eight-to-twelve parent metric, split, rows and model results remain
unchanged. Students development/calibration/confirmation data remain unopened.
No formal image-training cohort is admitted by this audit. Standard offline
annotation forecasting versus strict sensor-as-of observations is pending user
decision, as is the separate prospective primary-metric question. This diagnostic
does not silently filter interpolation-dependent histories to choose a new task.

Raw images, annotations and row-level caches remain local and ignored by Git.
Stage5C and SMC remain disabled.
