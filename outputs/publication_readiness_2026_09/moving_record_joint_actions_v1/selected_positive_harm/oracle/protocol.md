# What Does The Original Filter Leave For A Perfect Ranker?

Freeze all 102 A recording views, three seed banks, original raw choices,
pre-screened eligibility, labels and the original 2% rule. Compare exact
label-aware decisions within (1) screened candidates and raw query counts,
(2) the full bank at those same counts, and (3) screened candidates with
free count up to the unchanged per-query cap. These contrasts diagnose
screening and count limits; the earlier unrestricted full-bank oracle is
not a matched cap control. Empty raw query counts stay empty in both
matched arms. No labels enter any learned selector or forecast.

Use one action per row, positive selected easy reference, all/easy positive
harm including the unknown completion radius, and signed known easy net.
Maximize query-balanced completion gain. An exact lower bound from the
separate additive constraints can prove infeasibility; otherwise use
integer optimization with a fixed five-second limit per view and arm.
Re-evaluate each returned solution using the original scorer. A verified
witness may establish feasibility without optimality; timeout or failed
postcheck does not establish infeasibility. No selected oracle row choices
are written. Not a new model result, deployment or independent evidence.
