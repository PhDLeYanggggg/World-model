# Replaying the frozen external comparison

Run `python verify.py` in this directory. It needs only Python's standard
library. No repository, video, model, label, or network access is needed.

The input contains recording-seed summary numbers, grouped by original
source, with identifying paths and source names removed. The verifier
recomputes the original three-way 2% risk test from its saved ratios. A
view passes both tests only if its risk passes and both conservative
reference gains are positive. It does not import any saved pass flag.

For the paired comparison, average clips and seeds within a source and
retain a source only if all of its views are estimable. The bundle includes
the original 3,000 bootstrap source indices. It also enumerates the exact
empirical bootstrap distribution for the matched contrast, using the
multiplicity of each unordered draw. This is a check on finite resampling,
not an exact population confidence interval. No intervention is not a
risk pass, and an empty paired comparison is not a zero effect.

The inputs are already-scored summaries. This does not verify tracking,
compute risk ratios from raw trajectories, rerun inference or establish an
independent deployment result.
