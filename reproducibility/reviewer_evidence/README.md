# Trajectory decision evidence

Extract this archive and run `python -I -B -S verify.py` in the
extracted directory. The standard library is sufficient. The verifier
uses local files only, checks all member hashes and prints a JSON report.

| Check | Evidence included | Scope |
| --- | --- | --- |
| Headline counts | Anonymous source summaries and frozen bound values | OpenDD 330/606 hindsight and 57/606 past-only; MovingDroneCrowd++ 67/72 nonpositive relaxed OLS4 bounds; WEPDTOF 0/45 fixed-joint dual passes |
| WEPDTOF | Recording/seed scalars and the original source resampling indices | Reapply the three 2% checks, then require gains against both references. Undefined denominators do not pass. |
| Budget diagnostics | Source summaries and original resampling indices | Recompute the matched-budget tables and location intervals. |
| OpenDD decision follow-up | Seven anonymous source means per paired contrast | Recompute empirical source bootstrap intervals for the joint and easy-support comparisons. |

The follow-up intervals enumerate the empirical distribution of
seven-source resampling; they are not exact population confidence
intervals. Location means already average the relevant seed and
calibration directions. The 8,874 views come from 493 recordings,
not 8,874 independent examples. A conservative gain debits the
paired prediction disagreement on missing steps; its sign does not
reveal the missing ground truth.

These checks replay reported statistics, not inference, label
generation, selection from raw data or model training. The OpenDD
and MovingDroneCrowd++ evidence is development or hindsight evidence.
The frozen WEPDTOF evaluation is external and does not support
deployment. No Songdo outcomes or new independent confirmation
results are included.
