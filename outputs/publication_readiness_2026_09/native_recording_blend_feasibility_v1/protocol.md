# Recording-Constrained Affine Slate Diagnostic

Use the existing 34 exposed development recordings, three frozen seeds,
9,675 original switch recipients and unchanged CV floor. For every recipient
allow an individual coefficient between the two frozen paths, but no new
recipient or path direction. No external role is read.

For known rows, minimize the original masked ADE independently over each
coefficient. For wholly unknown rows, minimize maximum step disagreement
to CV. All three original recording risk numerators are nondecreasing
functions of these scalar costs, so their componentwise minima occur at
the same vector. When recipients stay fixed their three denominators are
constant. Zero denominators remain undefined. Allowing identity during
minimization is only a relaxation for exclusion; report feasibility only
with the exact original nonidentity recipient mask and scorer. A numerical
lower bound follows from the final scalar minimizer bracket and a
Lipschitz constant. Unknown disagreement alone is also compared against
the easy allowance, without changing the denominator or marking it safe.

On the checked feasible subset only, start from this individual optimum
and test a single recording-wide ray toward the previous query-local pair
witnesses. Re-score quarter-spaced fractions of its checked safe extent,
selecting the largest pair lower gain versus the start, including zero.
Report contrasts to the starting witness, original individual path, CV,
and OLS4. This checks existence on one ray, not global pair optimality;
the feasible subset is selected using labels and is not a performance
estimate for unseen recordings. Neither coefficients nor forecasts are
exported. No deployment, Stage5C or SMC.
