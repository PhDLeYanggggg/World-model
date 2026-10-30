# Individual utility is stricter than joint utility

## Material Passport

Mathematical implementation check,not a real-data result or a new statistical
guarantee. The following numbers are predicted conditional cost moments,not
individual realized mutually-exclusive benefit/harm labels. No real output is
used to construct this example or choose a threshold.

The fixed decoder maximizes sum(B-H) subject to the original signed budget
inequalities sum(H-.02R)<=0 and sum(EH-.02ER)<=0. It additionally filters every
candidate to B>H. That extra filter is not implied by positive joint utility.

For one candidate on each of four different agents,consider:

|Agent|B|H|R|ER|EH|Net utility|
|---|---:|---:|---:|---:|---:|---:|
|A|10|0.04|20|0.1|0.04|9.96|
|D|0|0.01|4|3|0|-0.01|
|S1|1|0|2|1|0|1|
|S2|1|0|2|1|0|1|

All moments are nonnegative and respect B<=R,ER<=R,EH<=H. With exactly two
interventions,the original positive-action solver selects S1,S2:utility2.
A plus either safe S has easy risk0.04/1.1>0.02. The enlarged solver can choose
A,D:utility9.95,all positive-harm/reference0.05/24=0.0020833 and easy positive-
harm/reference0.04/3.1=0.0129032. It never subtracts benefit from either harm
numerator. D is not a no-op;its predicted harm is positive. Both controls use
two interventions,so the difference is not explained by the count.

Executable exhaustive/counterexample tests check this and strict feasibility,
actual canonical postchecks,failed-solver anchors and random fixed-count optima.
The small example establishes only that the feasible families can differ. It
does not show useful differences on the SOURCE data or calibrated predictions.

At a zero matched count,both policies must abstain identically. This limitation
is preserved,not repaired by adding label-aware anchors or changing the count.
The free-count secondary contrast can expose new support but must separately
pass full recording risk;it cannot override a failed matched primary.

Mean prediction nesting after enlargement is not observed trajectory safety.
The output still depends on biased cost estimates. Label-side positive-action
oracle restriction is a separate diagnostic,not the same conditional-moment
example and not a causal policy. This is established discrete optimization,
not a novelty theorem or evidence of neural multi-agent dynamics.

Obs8/pred12/rawstride12,local-image detector-silver context. No metric/seconds,
human-gold,true3D,foundation,independent efficacy or deployment claim. Independent
roles,Stage5C execution and SMC remain closed.
