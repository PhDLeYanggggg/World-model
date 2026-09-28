# Readout Verification Preflight

The registered training and action code is unchanged: all91 decision source/control
bindings still match. New reporting and verification adapters are separate from
training, action selection and the registered scientific contrast. They cannot
select a threshold or promote a checkpoint from held outcomes.

Thirty-three targeted tests in six files passed locally in8.29seconds. New cases
reject changed per-query counts, invalid probability/risk factorization, reversed
contrast signs and replacement of undefined selected risk with zero. They also
check that a favorable ADE contrast with a failed screen cannot change deployment.
Even a passed development screen cannot establish independent confirmation.

The actual readout is **not_run**. Synthetic fixtures and implementation checks
are not a real evaluation. Once training, the first-pair training replay, causal
action freeze, action replay and numerical readout replay complete, the adapters
will independently check every query constraint, selected/fixed-denominator cost
view, probability/cost view, contrast, locality reduction and per-seed reduction.
All11 policies and both favorable and unfavorable contrasts will be reported.
The original selected-risk primary is not replaced by all-floor harm accounting.

## Resources

Full CREATE job37576457 was freshly observed RUNNING at elapsed00:06:09, with a
train-phase heartbeat and no stderr. The last observed group was
single0_seed43_controller1_dimensionless_pair4, risk_priority step500. This is
progress, not completion. The original CPU4/16GiB/two-hour allocation and all108
pairs remain unchanged. Do not duplicate the job or treat an SSH timeout as exit.

Local disk headroom fluctuates. It was below the existing10GiB reserve plus800MB
restore allowance during this preflight. The source experiment occupies roughly
89MB of checkpoints and273MB of action records; this is not a zero-cost copy.
Recheck before restoration. Do not delete old data, lower the reserve, or reduce
the registered experiment to fit local disk. Remote fitting is unaffected.

## Evidence Boundaries

Only12 already-opened development localities. Independent selection/calibration/
confirmation remain closed. Image-local detector silver, raw-frame obs8/pred12;
no metric/seconds, human-gold, physical-safety, true3D, foundation or submission
claim. Stage5C and SMC remain disabled. No deployment changed.
