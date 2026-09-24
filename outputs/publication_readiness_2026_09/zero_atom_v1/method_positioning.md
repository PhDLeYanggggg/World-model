# Method Scope and Prior Work

Primary-source reading, 2026-09-24. This note does not change the registered fit
or policy. Leaf-weighted conditional distribution estimation is established:
Meinshausen's [Quantile Regression Forests, JMLR 2006, Sections2--3](https://jmlr.org/papers/volume7/meinshausen06a/meinshausen06a.pdf)
describes using forest-derived observation weights to estimate more than a
conditional mean. Our weighted event readout on fixed ExtraTrees partitions is
a simple related construction, not a replication of that entire algorithm or
its consistency assumptions. No novelty is claimed for counting labels in leaves.

The local question is narrower: does separately representing the exact-zero-CV
component improve baseline-protected intervention beyond a shared easy net-risk
mean, at matched per-query intervention count? A mean easy loss can hide the
component even when its training labels are included. The six original moments,
forecasts, native cutoff and risk tolerance remain fixed.

The zero-reference veto is strict only in the empirical model: a positive
frequency forbids switching. Zero frequency is not an upper confidence bound
or proof of no unseen events. Trees share data, overlapping windows share tracks,
and four physical sites are already development-exposed. Neither128 trees nor
768000 training draws are independent calibration samples. No conformal or
individual-safety guarantee follows, even if the observed harms disappear.

The selected reference solution is a valid incumbent for same-count controls.
Retaining it when a numerical proposal fails prevents coverage mismatch but does
not establish the best achievable objective. Such controls remain explicit about
optimality. All source utility/protection tradeoffs and unknown future support
must be reported before any broader contribution claim.
