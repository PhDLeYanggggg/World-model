# Matched-Recipient Forecast Geometry

On the same exposed Moving A recordings, does geometry improve recording
capacity even if the intervention recipients and per-query counts cannot
change? The diagnostic retains all four whole-query options, then adds
alternatives on the exact same recipients: OLS4, and an OLS4 forecast
translated by the uniform mixture of OLS4, CV and self-kinematic query
mean displacements. The common shift preserves OLS4 relative geometry
between replaced agents, not between replaced and fallback agents.
If any listed recipient would get CV exactly, drop that alternative.

The original CV fallback, easy/hard cut, all/easy positive-harm caps,
signed easy cap, easy support, missing-outcome bound and query weighting
remain unchanged. For each of 102 development views (34 recordings,
14 sources, three seeds) optimize conservative utility over OLS4 for four
nested slates: frozen, plus OLS4, plus common, and both. A solver may
abstain or select one intact option per query. Each witness is checked
again by the original risk scorer. Allow four seconds per solve; report
unresolved cases and never treat a timed incumbent as an upper bound.

Primary readout: verified positive conservative OLS4 witnesses, including
transitions from proved nonpositive or infeasible frozen views and the
number of their distinct sources. Secondary: risk-feasibility transitions
and gain relative to the proved frozen optimum where both have witnesses.
No learned selector or forecast update; the solver uses exposed labels.
No calibration or confirmation inputs/labels are accessed, and no witness
predictions, selections or row data are exported.
