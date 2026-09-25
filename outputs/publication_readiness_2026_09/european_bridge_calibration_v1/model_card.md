# Model Card: Reference-Aligned Calibration Maps

## Intended Use
Development comparison of whether source-held calibration transports to opened
selection localities. This is not a released deployment policy or a physical
safety controller. Stage5C and SMC remain off.

## Fixed Models and New Fits
Eighteen ordered source A/B/seed settings use seeds 17, 29 and 43. For each,
full and motion-only forecast pairs have matched neural and ridge cost scorers.
The 144 utility/risk scoring models are restored from verified checkpoints;
this study does not update neural weights. Source-C inference is fresh.
Seventy-two new empirical maps yield 288 frozen policy views.

Reference R is the conservative easy-policy delivery; P is the all-policy
delivery. Motion-only counterparts use causal CV/damping policies without
neural trajectory candidates or neural policy bits. Future labels are absent
from the gate. Positive predicted utility, current motion and distinct rollouts
are prerequisites for switching.

The four rules are reference-only fallback, unchanged 0.02 moment-ratio gating,
conservative population rescaling, and a C-selected grid from 0 to 0.020.
Only tighter rules are allowed. Grid feasibility checks all/easy positive harm
relative to R, net easy degradation relative to CV and no zero-CV harm in every
C locality. No positive feasible gain causes structural fallback. A zero cutoff
does not itself mean fallback. Three-of-four C fits are diagnostic only.

## Limitations
A fitted score ratio is not realized risk, and four source localities are not
an independent large calibration sample. Scene shift and low-denominator events
can invalidate population moment transport. Reference-only selection has zero
additional R-to-P harm, but cannot guarantee R itself satisfies every CV-based
constraint. No claim of neural superiority, independent risk certification,
metric prediction, seconds-level dynamics, true 3D or foundation modeling is
supported merely by completing this experiment. Results and negative transport
are reported without choosing the best seed or source assignment.
