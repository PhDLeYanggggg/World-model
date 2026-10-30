# Current-query allocation: model and data card

## Purpose and status

Research-only allocation over fixed forecasts. No new neural model is trained.
The complete policy fails the original risk contract and is not deployed.
Independent confirmation and submission readiness are false.

## Frozen learned components

18 cached_verified unit-admission cost heads across six contexts and head
seeds17/29/43. Each has30,405 parameters,width64 and462 causal input dimensions.
The actual neural forecast producer seed43 and all six physical actions remain
unchanged. No new preprocessing,features,latent extraction or training targets
are fitted in this allocation experiment.

The head predicts benefit,harm,reference,easy reference and easy harm for each
of seven actions. The learned reference can vary by action,although the true
fallback reference for a target is common; predicted budget feasibility is not
observed-risk calibration. Inputs retain their earlier past-only/fitting-only
lineage. No future endpoint,central velocity,test endpoint goal or test statistic
is added by this controller.

## New decision component

Groups use source/recording/current raw frame and canonical row IDs. At most one
real action per target. Joint matched keeps the original independent count in
each query and maximizes predicted net gain under both predicted2% signed
budgets. Uniform and free-count arms are separately retained. A solver failure
or failed raw postcheck retains the original independent decision.

This is multiple-choice budget coupling,not a new dynamics network,pairwise
interaction learner or a reproduction of JFP. The query roster is not guaranteed
to include all visible agents. Physical collision or safety claims are unsupported.

## Data and observation

149,510 unique allowed SOURCE queries from74 recordings and12 localities.
112,596 reused query views and897,060 reused target-row views arise across
contexts/seeds,not independent new samples. The full cohort retains unknown
outcomes. All data are exposed fitting/resubstitution,not whole-pipeline OOF.

Eight observed and12 requested steps at raw stride12. Supplied detector-silver
positions remain image-local/pixel/raw-frame observations,not human gold,
verified seconds,metric coordinates,true3D or foundation-model evidence.
ADE uses available requested positions;FDE needs the final position. A required
locality lacks endpoints,so complete-roster FDE is not estimable. Completion
sensitivity covers wholly unknown rows,not missing interior positions of
partially observed ADE labels.

## Execution and checks

Native arm64 CPU4/interop1/workers0,SciPy1.17.1/HiGHS. No GPU or HPC job is
required by the measured local workload. Pilot128queries/3,267rows has exact
interrupted recovery. Main PID19894 completes in729.54s at10.92GB peak RSS;
independent process PID22148 reproduces all72 groups in741.43s at9.81GB.
All parent prediction/action/full-score and record-oracle checks match.

Action progress is atomically saved every250queries;explicit resume preserves
input bindings and processed decisions.12h runtime,10GiB free-disk and256MiB
private progress guards apply. Eight primary postcheck rejections are not
claimed as exact-optimal solutions. Code and aggregate metrics are public;
raw data,features,individual action caches,weights and images remain private.

## Results and limits

Known ADE gain2.552950%,versus independent1.904562%. Primary+0.648388pp,
conditional locality95%CI[+0.181208,+1.272930];completion-lower+0.506055pp,
CI[+0.071211,+1.107688].3,000 bootstrap draws follow context/head-seed averaging.
Only43/444 required recording views pass;324 known easy violations remain.
Fitting gates5/7. Exact replay verifies calculation,not independent science.
No deployment,Stage5C execution or SMC.
