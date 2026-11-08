# Fixed time-profile diagnosis

The fitted local model failed on the complete prediction grid. The
unlearned neighbor candidate shows a known-label gain over OLS4 but not
a positive lower interval on that grid. Test one fixed explanation:
does carrying an early displacement offset through step 12 expose
the forecast to more unsupported error than the short-term gain merits?

Take the existing fixed neighbor forecast and preserve its offset
through native step 6. Then apply a cubic smoothstep from full weight
at step 6 to zero at step 12, retaining the OLS4 endpoint. The schedule
is fixed by the midpoint and endpoint; there is no horizon sweep.
Contrast it with OLS4, the original fixed neighbor forecast, and the
identical rule using the disjoint same-count outer neighborhood. Compare
near vs outer on the common nonidentity roster as a direct control.
Use the same nine fitting sources and group-bootstrap reducer. This is
a follow-up diagnostic motivated by an earlier outcome, not a new
independent experiment, safety pass, or endpoint gain claim.
