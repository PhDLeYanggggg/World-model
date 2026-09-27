# Method Position and Submission Constraints

Checked 27 September2026. This note records primary-source reading, not a new
theorem, calibration run, submission or change to the approved risk tolerance.

## Risk-Control Literature

**Conformal Risk Control**, Angelopoulos et al., arXiv2208.02814v4,
Sections1.1,2.1 and2.4: the central guarantee concerns expected loss and requires
exchangeable loss functions, a finite upper bound and monotonicity in the
conservativeness parameter. The paper gives a counterexample for applying the
same algorithm to general non-monotone losses. It also describes a monotone
envelope construction. These conditions are not established by a small neural
ADE gain. [Original paper](https://arxiv.org/pdf/2208.02814).

**Learn then Test**, Angelopoulos et al., arXiv2110.01052v5, Sections1.1 and2,
Theorem1: independent identically distributed calibration units support valid
risk tests, combined through family-wise error control. This gives a high-
probability guarantee over calibration randomness and need not assume a
one-dimensional monotone risk curve. A family of configurations can be checked
only with valid tests and the associated multiplicity control; ordinary
threshold sweeping plus a bootstrap interval is not the same procedure.
[Original paper](https://arxiv.org/pdf/2110.01052).

## Consequences for M3W (Our Inference)

The paper contribution cannot be named simply "conformal fallback": established
methods already calibrate model post-processing. The open question is useful
relative gain/harm learning and scene-joint intervention under realistic weak
forecast gains, locality shifts, dependent agents and unsupported contexts.

A scene-joint optimizer can change several actions non-monotonically as a
threshold varies. Error ratios with outcome-dependent small denominators also
need a valid bound/estimand treatment. We cannot import either theorem without
checking these conditions. Calibration units must reflect the independent
scene/locality protocol, not thousands of overlapping windows.

No current source-development comparison supplies those independent calibration
units. The next policy experiment must retain equally protected simple motion
controls, no intervention, independent-agent, whole-scene and scene-joint rules,
with the same forecast bank and explicit risk/rate comparisons. A better
predictor is a prerequisite improvement, not a substitute for this experiment.

## Official Calendar

CVPR2027 lists registration10November2026, full paper16November2026 and
supplement23November2026, all Anywhere on Earth. [Official dates](https://cvpr.thecvf.com/Conferences/2027/Dates).

All authors need current OpenReview profiles; the call specifies dual-submission
restrictions and reviewer-service obligations. LLM policy details remain under
development. [Official call](https://cvpr.thecvf.com/Conferences/2027/CallForPapers).
The linked [Author Guidelines](https://cvpr.thecvf.com/Conferences/2027/AuthorGuidelines)
still returned404 on this check. Exact current template/page rules remain
unverified; do not silently substitute last year's requirements.

Keep the existing evidence freeze/writing buffer. No registration, profile
change or submission occurred. Author confirmation is required for submission.
Current predictor work does not establish CVPR submission readiness.
