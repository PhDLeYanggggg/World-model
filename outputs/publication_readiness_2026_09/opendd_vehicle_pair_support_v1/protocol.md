# Two vehicles, one query, the same edit radii

Reuse the prior seven OpenDD vehicle locations, 8/12 steps, three
seeds, 63 double-excluded forecasts, factorized cost heads and
actor-support classifiers. Each outer/calibration direction uses
models that saw neither site. No new fitting and no Songdo or
reserved DroneCrowd access. This is development evidence.

For each recording choose the query with the most legal actors
(earliest tie) that has at least three observed actors and a legal
same-family pair. Hold the query, the three path families, and
exactly two equal-radius edits fixed for all methods. A recording
without such a query is not in the matched diagnostic. CV with zero
edits remains the deployment fallback, not a count-matched control.

Convert value forecasts to native units before summing actors. Replace
the factorized head's internal support probability with the dedicated
actor probability when reconstructing conditional easy harm and mass.
All methods use the same values. Each actor's scalar merit is the
smaller of its two gains, minus positive predicted all/easy excesses,
minus its radius times predicted unsupported probability.

* Independent: maximize the sum of actor merits with distinct actors.
* Uniform: same scalar objective, both edited actors must use one family.
* Joint: take the smaller total gain, subtract the positive excesses
  only after summing the two risk moments, and penalize missing easy
  support by the sum of both radii times one minus the greater
  actor support probability. The last factor uses the union lower
  bound, not an assumption that the actors are independent. Marginal
  support is estimated, so this is a ranking score, not a calibrated
  probability bound or risk guarantee.

The exact enumerator considers distinct-actor pairs. For each method
calibrate its own rank threshold on the other excluded site using
the longest tied-score prefix where all recordings pass the original
all/easy/signed 2% checks and have positive conservative gain versus
both CV and OLS4. Each recording contributes zero or two edits; no
qualifying prefix means fallback. Freeze all thresholds and proposals
before outer scoring. Compare the calibrated policies separately from
the two-edit matched mechanism study.

Primary matched comparison is joint minus independent. Compute the
direct paired lower difference in accuracy on the full recording
roster, not the difference of separately bounded gains. Also measure
direct paired relative-motion lower gain on all pairs in the same
selected query, with missing pair-step debit and query weighting.
Report CV-relative gain, matched risk and dual pass counts, plus
separate held-site release counts. Three thousand paired bootstrap
draws over the seven locations, averaging repeated seeds and
calibration directions within each site. No threshold or safety
definition changes. A joint ranking score does not model physical
collision or guarantee safe deployment.
