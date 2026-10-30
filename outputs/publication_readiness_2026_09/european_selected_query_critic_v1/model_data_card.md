# Selected-query risk critic: model and data card

## Intended role

A controlled source-fitting experiment on when to switch among seven frozen
forecasts. This is a risk-cost head, not a newly trained trajectory forecaster,
generative model or deployable world model. No new candidate is promoted.
Historical Stage26/37 results concern other protocols and are not independent
comparators for this obs8/pred12 experiment.

## Architecture and outputs

Each head has 30,726 parameters: a 475-to-64 SiLU encoder and a 130-to-2 output.
Inputs are 462 frozen normalized causal fields, five original moment estimates,
seven action indicators and a forecast-envelope value. The output layer sees
the focal embedding, either that same embedding or the mean ORIGINAL selected
embedding, and two selected-count descriptors. Both arms have identical size.

The rational anchored decoder changes only positive harm and positive easy harm.
Benefit, reference and easy-reference moments are exactly frozen. Harm capacity
uses the parent envelope-minus-benefit with its numerical anchor protection;
easy harm cannot exceed all harm. These are model-output constraints, not
verified physical constraints or calibration guarantees.

## Data and label access

European source-development material: 149,510 unique agent-target rows, 18,766
current-frame queries, 74 recordings and 12 exposed localities. Six context
rotations and three critic seeds produce 897,060 reused row views and 112,596
query views. They are not additional independent examples.

The feature API admits only registered causal keys. Future endpoints, future
remaining track length, central velocity and future/test goal labels are not
inputs. Original selection uses past-derived model predictions. Future costs
and availability masks supervise loss/evaluation only; unknown rows remain in
query draws/context and never supervise. Input causality is distinct from
whole-pipeline producer independence; the latter is not established here.

The labels are released detector-silver tracks in local image coordinates.
ADE uses available future points. A missing required endpoint makes full-roster
FDE not estimable. Wholly-unknown completion bounds do not validate missing
interior points of partially observed trajectories. No metric/time claim.

## Actual training and reproduction

Native arm64 Torch CPU4/interop1/workers0; 36 matched fits at 2,000 updates each,
plus two separate exact 2,000-update refits. AdamW lr0.001, clip5, 16 source-
balanced current queries per update, checkpoint100 and heartbeat50. Final step
is fixed, not validation-selected. The real 400-update pilot checks recovery
and matched draws. Actual training/optimization traces are published separately.

Private `.pt.gz` weights, source packets, row caches and trajectories are not
in Git. Public code/config/registrations, aggregate metrics and receipts are
available; this is not a verified raw-data-free cold reproduction package.
An exact local replay is engineering reproducibility, not scientific confirmation.

## Observed limitations

Selected-set context loses to the matched row-local control. The point-risk MSE
improvement does not pass selected-policy protection; complete-recording risk
fails. Original selected-set training and new optimized selection differ.
No risk guarantee, scene/goal contribution, learned pair dynamics, independent
generalization, deployment, submission readiness, true3D or foundation claim.
Stage5C execution and SMC remain disabled.
