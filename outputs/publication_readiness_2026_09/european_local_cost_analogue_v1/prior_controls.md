# Why Test This Objective, and What Is Already Known

- european_cost_harm_newton_v1: 72 fixed-forest positive, mean-preserving heads
  converged after exact-curvature repair, but signed cost error and safety did
  not beat the original. Thus objective alignment is not assumed sufficient.
  That model uses seven quality descriptors, global leaf fits and projection.
  This test uses frozen 380-feature nearest histories, selected donor population,
  no positive-exponential/mean-preserving constraint and no solver. It is a
  local conditional-information test, not the same Newton model renamed.
- european_easy_hurdle_v1: better easy-occurrence Brier did not yield better
  signed risk or matched utility. No fresh event classifier is proposed here.
- european_risk_decoupling_v1: better average TRAIN error did not fix selected
  easy error. The original selected cohort stays frozen throughout this test.
- european_selection_conditioned_analogue_v1: reference matching reduced easy
  MSE 50.10% against full bank but remained 7.65x original, with 0/18 paired
  wins. Reference coverage is complete, so missing donors do not explain that
  failure. Event-positive harm allocation remains poor despite average event
  prevalence agreement. No query-derived scalar rescaling is allowed.

These reports and the scored signed-risk formula were read before registration.
Local quadratic regression is a conventional control, not a novelty claim.
Reusing current references removes changes in feature retrieval, population,
query selection and support; it does not remove upstream training exposure.
If the local optimum still fails, neither extra loss tuning nor a favorable
secondary control may silently replace the stated hypothesis or primary result.
