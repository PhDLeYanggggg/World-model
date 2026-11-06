# Estimand, Prior Work And What The Model Does Not Guarantee

## Material Passport

Method note written during the fixed experiment, before its policy readout.
Fresh derivation and primary-source checking; existing source experiment results
are cached_verified. This does not add an outcome, threshold or main claim.

## Complete-Contribution Target

Let X contain only the permitted current/past inputs. Let M indicate whether
the released future window contains any valid evaluation labels. For a candidate
action a, let C_a=(B,H,R,ER,EH) denote observed available-label contributions and
U_a(X)=(0,D_a,0,0,D_a) the original causal unknown-envelope contribution. D_a is
the maximum candidate/floor path disagreement, not observed future error.

The evaluator's complete-contribution variable is

    Z_a = M C_a + (1-M) U_a(X).

Writing q(X)=P(M=1|X), the elementary conditional-expectation identity is

    E[Z_a|X] = q(X) E[C_a|X,M=1] + (1-q(X)) U_a(X).

This identity concerns the evaluator's bound contribution. It does not recover
unreleased physical trajectories, identify missing outcomes, require a
missing-at-random assumption or justify calling partial-label ADE full12 ADE.
Label availability is not physical visibility, failure probability or human gold.

The observed cost and label-support networks are separate. The treatment's
conditional loss has zero gradient on bound-only rows; its BCE has no gradient
to the conditional/reference networks. The comparison arm instead backpropagates
complete composed-cost error. A common reference branch prevents action features
from changing predicted floor R/ER. Both arms share every other declared factor.

The implemented harm-channel underprediction weight is4, inherited as a fixed
risk-oriented choice. An asymmetric squared objective is not an unbiased
conditional-mean estimator in general. The identity defines an ideal estimand;
finite, constrained, risk-oriented fitted outputs are approximations, not a
proof that true conditional moments or calibrated upper bounds are recovered.

## A Subset Cannot Create Reference Support

For a recording, the original selected easy-reference denominator is

    ER(S) = sum_{i in S} 1{observed_i and easy_i} R_i.

All terms are nonnegative. If ER(S)=0 and T is a subset of S, then ER(T)=0.
An extra post-veto of the frozen mean/unanimity accepted set therefore cannot
repair their6/12unsupported easy views. This is an algebraic support limit,
not a statement that every new learned proposal veto is impossible: the broader
frozen proposer has observed easy reference in all24views. Proposal-only versus
full-action decisions must determine whether action-bank expansion is useful.

Actual future easy/observed labels appear only in this analysis and supervision,
never as inputs to admission or to a support-based query filter.

## Predicted Joint Feasibility Is Not A Realized Recording Guarantee

The allocator enforces two sums of predicted excesses per current frame:

    sum selected (H_hat - .02 R_hat) <= 0,
    sum selected (EH_hat - .02 ER_hat) <= 0.

At true conditional expectations, measurable causal selection and integrability
would relate these sums to expected excess. That ideal statement does not give
the experiment's finite-record realized ratios, ensure a nonzero actual
denominator, prove source-shift calibration or control pairwise trajectory
conflicts. Model errors remain, and its asymmetric losses are not that ideal
expectation oracle. Hence every policy still faces the original full recording
contract on all rows. No inequality is relaxed by solver feasibility, empty
support or zero intervention.

## Relevant Primary Work

[Mao, Mohri and Zhong, Regression with Multi-Expert Deferral, ICML2024](https://proceedings.mlr.press/v235/mao24d.html)
studies regression routing, including frozen predictors with learned deferral
and surrogate consistency. Our existing router is an adaptation, not a new
multi-expert-deferral invention. Those results do not by themselves certify
our additional selected-reference recording contract. The regression setup was
also checked in the [authors' draft](https://arxiv.org/pdf/2403.19494).

[Angelopoulos et al., Conformal Risk Control, ICLR2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/f3549ef9b5ff520a7e41ff3cc306ab2b-Abstract-Conference.html)
provides expected-risk calibration under stated conditions. Theorem1 in the
[authors' v4 text, June2025](https://arxiv.org/pdf/2208.02814) uses exchangeable,
bounded, monotone loss functions and a calibration construction. Ordinary
cost fitting is not that construction and does not inherit its guarantee.
We make no general impossibility claim about non-monotone risk; the existing
manuscript also records later extensions. OpenReview archival PDF access was
challenged, so the checked v4 text is not misrepresented as the archived PDF.

These primary sources were checked6November2026. No quoted passages are used.
The decomposition and mixed-integer solver are not, alone, novel contributions.
The hypothesis under test is whether gradient-separated risk learning and
current-frame allocation improve useful intervention under the retained
contract, with matching and source-separated evidence. If the registered primary
or complete risk fails, this comparison is not a new main paper method.

Independent calibration/confirmation, multiple forecaster seeds, interaction
consistency evidence and the original24-group primary are still separate gaps.
No new C/H, Stage5C, SMC, deployment, metric/seconds, true3D or foundation claim.
