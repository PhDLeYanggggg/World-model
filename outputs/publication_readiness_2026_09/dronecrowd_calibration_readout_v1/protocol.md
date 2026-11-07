# DroneCrowd Calibration-Only Readout

Use the approved 15-group, 40-recording calibration role. The 30-group,
72-recording confirmation role is not an input to this runner. Keep
native observation-8/prediction-12 at stride one, image pixels, the
training-derived CV cuts, and the original 2%/support/undefined rules.

Freeze the exact development-only controller coefficients and three
seed-specific banks (six causal paths including CV and OLS4, plus the
same three neural predictor families). Independent, shared-family,
pair-additive and joint proposals use identical counts per query.
They are matched controls, not alternative threshold-search policies.

Screen the joint proposal only if its predicted gain is positive,
predicted all/easy excesses are nonpositive, and estimated easy
reference is positive. The four fixed variants additionally require
normalized gain sum greater than 0, 0.01, 0.03 or 0.10. Inference is
strictly current-prefix only; freeze predictions for all 40 recordings
before any calibration label access. No refitting of the predictor
bank or cut derivation on DroneCrowd.

A threshold is eligible only when the original recording rule passes
on all 120 recording-seed views, the policy switches at least once,
and the paired source-bootstrap lower-utility interval against CV has
a strictly positive lower endpoint. Select the eligible threshold with
maximum lower-utility point estimate, then fewer switches, then fixed
arm order. Report OLS4 separately as the strong accuracy reference.
These calibration estimates are not confirmation or a guarantee of
new-source risk. If no threshold qualifies, record that result and
leave all confirmation outcomes sealed.

Aggregate by the 15 exclusion groups after averaging three seeds.
Bootstrap 3,000 paired groups, not trajectory windows. The 40 original
recordings and 120 repeated views are not verified independent sites.
