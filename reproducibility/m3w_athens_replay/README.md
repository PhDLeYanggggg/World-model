# Frozen external comparisons

Run `python3 verify.py` after extracting this archive into an empty directory.
Python's standard library is sufficient; no network or dataset access is needed.

This extension checks the frozen four-date Athens zero-shot roles, the primary
and secondary readout bindings, date-level missing-label envelopes, and the
paired 3,000-draw source bootstrap for the equal-step sensitivity. The inherited
archive replays the previously reported development, WEPDTOF and Songdo results.
The cross-cohort comparison keeps each cohort separate.

The replay uses released aggregate scalars. It does not reconstruct annotations,
train models, reopen labels, or independently establish that a timestamp preceded
label access. The hashes check internal file identity; the original freeze is
part of the experiment record. Three seeds do not add independent source dates.

Athens has no calibration or deployment test. The lower sample bound against CV
is positive, but joint selection has no established advantage over independent
selection. Missing-label envelopes and source-bootstrap intervals are different
quantities. No operating threshold or original 2% criterion is changed here.

Data source: pNEUMA - open-traffic.epfl.ch.
E. Barmpounakis and N. Geroliminis, Transportation Research Part C 111:50-71,
2020, doi:10.1016/j.trc.2019.11.023. Dataset version 1, 2024,
doi:10.5281/zenodo.10491409.
