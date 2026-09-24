# Correction: Signed Easy Risk, Not Overall Net Gain

The frozen four-output net-moment head stores easy harm, easy benefit, easy
reference denominator and easy probability. Thus the prior numerical checks
comparing column0 > column1 measure whether **signed easy risk is positive**.
They do not measure overall predicted gain. The immutable JSON field named
`net_gain_sign_changed` is retained for exact artifact replay but must be read
with this correction. Its values and the unit-sensitivity finding are unchanged.

For the isolated100x two-column control, signed-easy-risk signs change in
411 damping, 369 Transformer and 325 EqMotion windows out of754. Neither these
counts nor the original report establish a complete deployment decision or
observed future error. No original numerical artifact or checkpoint was edited.

The new matched dimensionless-risk experiment explicitly predicts separate
overall benefit/harm outputs alongside the four easy-risk moments, preventing
the two quantities from being conflated in the decision code or report.
