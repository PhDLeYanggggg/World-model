# Local velocity innovations instead of position-anchor edits

The same-budget anchor oracle is too small to justify another retrieval
pass. Test a different forecast: a learned short-lived and a sustained
velocity innovation in the ego-heading basis, with the observed endpoint
fixed. Each of the two velocity vectors is bounded to one quarter of the
past RMS step size. Train on ADE difference from OLS4, debit unobserved
steps by the actual prediction disagreement, and give positive harm twice
the loss weight. The model starts at OLS4.

Use the original 69,746 fitting windows, eight observed and 12 predicted
native annotation steps. Run the same fixed 800 updates for each of the
three seeds and nine held fitting sources. The three matched arms use the
nearer half of the complete-history nearest-eight bank, its disjoint outer
half, or no neighbors. The two peer arms have the same count per row and
both receive past relative offsets and velocities. If there are fewer than
two eligible neighbors, neither peer arm gets context. The outer control
tests proximity, not the absence of interactions. This is not a
full-trajectory scene decoder or an attribution of physical forces.

Primary contrasts are near vs OLS4, ego, and outer. Report the full-roster
observed-label score, historical wholly-unknown-row lower score and a
separate full-grid lower score using per-step bounds. Include CV and a
fixed bounded neighbor-acceleration comparator. Report paired source
intervals (3,000 replicates, seeds averaged before resampling), ADE, FDE,
known hard gain, known easy positive harm and unsupported source cells.
A favorable near-versus-OLS4 result alone does not show that local context
helps; paired near-versus-ego and near-versus-outer lower intervals must
also support positive differences. No threshold or arm is selected on
these scores. These are component comparisons, not matched-count
deployment interventions or a certification under the original 2% rule.

Use no development, calibration or confirmation outcomes. No change to
the 15/30 split, no metric or seconds claim, no deployment, no Stage5C or
SMC. Previous fixed social acceleration controls remain separate evidence.
For the motivation to distinguish marginal and joint evidence, see
[Weng et al., ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Weng_Joint_Metrics_Matter_A_Better_Standard_for_Trajectory_Forecasting_ICCV_2023_paper.html).
