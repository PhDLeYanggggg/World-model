# Observation-Support Cost Factorization On SOURCE

## Material Passport

Prospective real Torch comparison following the complete bound-target and
same-set diagnoses. EqMotion forecasts, direct bound-aware control and causal
SOURCE packets are cached_verified. New cost training/prediction/scoring are
fresh_run. SOURCE is exposed development, not an independent final test.

## Hypothesis And Distinction From Prior Failures

The existing direct bound regressor understates selected EH already on its own
fitting sites. Explicit unknown supervision improves fitting but does not pass
the registered paired primary or recording risk. A previous generic easy-event
hurdle also improved regression and lost utility. Neither is relabeled success.

Test a different decomposition: let nu mean at least one released requested
future label exists. Predict q=P(nu=1|past packet), conditional known five-moment
cost m, and compose

    m_total = q*m + (1-q)*(0,D,0,0,D).

D is exact candidate/floor maximum path disagreement. Zero reference on an
unknown row is absent known denominator contribution, not zero future error.
q is not physical visibility or full12-label availability. This is classical
conditional-mixture algebra, not a new theorem or a safety guarantee.

Both new arms use the same width64 query-context conditional cost network plus
one shared observation logit:29,318parameters versus the old29,253,an increase
of65. Registration additionally records both executable counts. Both initial logit weights are zero,
bias log(.95/.05);.95is a fixed prior,not test or fitting prevalence. Conditional
head initialization, data normalization, RMS, source-balanced query draws,
optimizer, complete budgets and all actual seeds match within the two new arms.
The legacy conditional decoder/envelope remains; only the missing contribution
uses exact causalD. There is no new input, classifier threshold or learned prior.

Marginal_support minimizes the same full-bound five-moment MSE plus half
absolute-gain MSE as the previous direct-bound arm. Identified_support adds
unweighted observation BCE on all legal fitting rows and normalized conditional
five-moment MSE on observed fitting rows, both coefficient1. Query averages
retain the existing supported-query semantics. Targets/masks enter these losses
only; they never enter inference. Unweighted BCE is intentional: a balanced
classifier score would not automatically be the required probability.

## Controlled Comparisons

Primary: identified_support bounded-independent minus the frozen direct-bound
bounded-independent policy, using direct paired completion-lower gain pp with
the same original known floor reference. This is a family/objective repair,
not a parameter- or initialization-matched isolated loss effect versus the old
head. Old direct control retains its complete training exposure and budgets.

Fixed secondary contrasts: identified minus marginal (isolates the auxiliary
identification objective within matched new architecture); marginal minus direct
(composition/family control). Neither can rescue a failed primary. Binary
admission, full neural substitution and no-intervention floor are secondary.
No arm, seed, hyperparameter, confidence threshold or amplitude is selected
after scores. Lower fitting loss or better availability AUROC is not policy lift.

## Data, Runtime And Decisions

Retain all149510SOURCEwhitelist windows,74recordings,twelve localities and
actual seeds17/29/43. Use the parent's18nested forecaster/critic jobs: four sites
for the producer, four disjoint critic-fitting sites, remaining four evaluation
sites; rotate both critic choices. Evaluation is excluded from both fits.
All fitted preprocessing and original floor/easy/hard choices retain those roles.

Per arm, budgets stay2000/3584/4096according to the actual fitting-group coverage:
116160primary updates across36newfits,4000first-job full-refit updates,1200real
pilot updates. Each arm/group has50->100/direct100recovery, then both first full
jobs are independently refit. Record every actual budget rather than the estimate.
Freeze all models, then all36cost/observation-probability/amplitude arrays before
reading held outcomes. Separate full prediction and metric replay is required.

Inference retains exactly the parent's bounded2%predicted guard. Only the neural
proposal can intervene in the primary. Low predicted benefit/support falls back;
no new q threshold is introduced. Require the original2%all/easypositive-harm
and signed easy-net contract plus finite support in every recording. Empty
selected reference is not estimable, not a safe zero; no epsilon intervention.

Report known/lower/hard/easy accuracy, available endpoint FDE, tails, switch and
amplitude, predicted/actual selected-record costs, support availability AUROC/
AUPRC/Brier/ECE, and all risk failures. All probability targets mean wholly
missing released-label support, not pedestrian visibility. Use3000locality
bootstrap draws after excluded-view/seed averaging. The444recording/pipeline
views repeat74recordings and twelve localities,not444independent scenes.

Use local nativearm64 Torch CPU4/inter-op1/workers0, atomic checkpoints,
heartbeat and unchanged resume. A real pilot estimates full time/RSS. Disk reserve
10GiB and private1GiBcap are maintained; no unrelated data are deleted. A12hphase
boundary preserves checkpoints for same-budget resume,not automatic downgrading.
No new HPC job is needed if the pilot remains within this local envelope.

No calibration private arrays, independent final confirmation or DroneCrowd are
opened. Partial-mask ADE is not full12ADE; wholly unknown bounds do not fill
partial future steps. Pixels/rawstride12/silver are not metric,seconds,gold,
physical safety,true3D or foundation. No new trajectory/JEPA/Transformer training,
deployment, submission-readiness, Stage5C execution or SMC.
