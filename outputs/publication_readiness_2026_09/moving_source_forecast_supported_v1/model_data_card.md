# Model And Data Card

Task: image-plane native8/12forecasting on fit-admitted MovingDroneCrowd++.
Population:64processed recordings,58with queries,24declared/23supported groups.
All195,286past-eligible rows remain,including partial and unknown labels.
Architectures: existing PCT and pinned-author fixed-head EqMotion controls.
Initialization: random seeds17/29/43,one fitting fold per producer,other two excluded.
Source normalization/baseline/loss factors follow fitting groups only.
Inputs: past positions,valid past masks,selected complete-past neighbors and
causal rollout/time requests. Labels and target-validity masks never enter inputs.
Raw current-visible input construction is preserved; incomplete-past agents
are not neural neighbor tokens under the matched comparator rule.
Source snapshots/role and dependency hashes are in registration.json/job_roster.json.
The 48 shared-motion fields are unused prospective critic features, not image modality.
Training uses native arm64 TorchCPU4,atomic checkpoints and resume,no workers.
All weights/predictions remain private; Git contains code/scalars/source hashes.
Source use: attributed local non-commercial CC-BY-NC4.0 release; no raw redistribution.
The fixed-head adapter is not published best-of20reproduction; no new EqMotion novelty.
Original24-groupprimary not estimable;23-group results are secondary development evidence.
No independent geographic-site certification or transport of risk guarantees.
No deployable M3W upgrade,full2%gate,metric/seconds,true3D or foundation claim.
Stage5Cexecution andSMCremain off.
