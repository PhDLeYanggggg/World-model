# Calibration Is Not a Transfer Safety Certificate

Fresh primary-source reading on1 October2026:
[Romano, Patterson and Candes, Conformalized Quantile Regression](https://arxiv.org/html/1905.03222v1),
Sections1,3,4 and Theorem1. The method separates proper training from calibration,
corrects regression intervals with held-out conformity scores, and establishes
marginal coverage under exchangeability. Conditional quantile regression alone
does not supply the same finite-sample guarantee.

Implication for this experiment: our recording-level residual adjustments are
not CQR. Cross-locality transfer, post-score selection and correlated windows
do not satisfy an established exchangeability argument here. Covering a cost
component also does not by itself control a selected harm/reference ratio.
We therefore use the explicit label empirical component calibration, preserve
the2% risk objective, and test utility, risk and abstention separately. Neither
this construction nor combining it with a Transformer is a novelty claim.

CQR public-method reproduction: not_run. This note records what was actually
read and the limitation it imposes, not a claim that a formal baseline has run.
