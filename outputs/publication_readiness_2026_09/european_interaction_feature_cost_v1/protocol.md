# Frozen Interaction Features For Intervention Cost Learning

## Material Passport

SOURCE-only matched feature experiment. At initial implementation: full frozen
cache and experiment registration pending;no new head trained or outcome scored.
Keep all previous observation/support/capacity/loss negative results.

## Specific Hypothesis

The same-set cost diagnosis finds large selected easy-harm underprediction even
on fitting queries;reweighted targets,observation factorization and generic
capacity controls did not solve it. Test whether the frozen stronger trajectory
forecaster's causal interaction state adds useful information to the critic.
This is a feature-utility hypothesis,not a claim that a hybrid architecture is
new or that latent distance reduction proves prediction gain.

## Matched Feature Intervention

Use all nine frozen EqMotion producers,three actual seeds17/29/43,the same
149510SOURCE windows,74recordings and12localities. ProducerFfits four sites,
criticGfits another four,and scoringHuses the remaining four;rotate both
critic choices exactly as before. Full immutable cache must reproduce every
parent forecast and all897060retained h64rows before training begins.

Two heads per original18nested jobs:zero64andinteraction64. Append the ego h64
from the producer-excluded original-packet cache to every action's existing
444causal descriptors. Fit equal-locality mean/std on criticGonly,never all
cache or held sites. Both arms share the same508feature normalizer;zero64sets
the final64standardized inputs to zero at training/inference. Both use width64,
the same initialization,query draws,RNG,targets,optimizer and full original
coverage budgets. Dimension508gives33349parameters per head versus29253in the
frozen444control. The primary is within the matched new capacity;old-direct
contrasts are secondary and cannot isolate latent information by themselves.

The h64extractor uses its original complete-neighbor packet while the existing
444descriptors use the later partial-neighbor packet. A positive feature
contrast therefore establishes utility of this frozen feature bundle,not an
isolated learned-encoding effect under identical raw neighbor coverage. If it
succeeds,a matched raw-original-context control is required before that stronger
attribution. This distinction is recorded before any head fit or held readout.

No future-target latent,endpoint,future masks,central velocity,eventual track
length,test goals,domain/record ID input or held normalization. Existing IDs
group queries and enforce roles,not neural covariates. Forecaster features are
from learned past-context interactions,not JEPA or new dynamics training.

## Objective,Policy And Primary

Reuse the complete direct-bound5moment regression plus.5net-gain objective,
including causal disagreement upper targets for entirely unknown future rows.
Partial-label ADE remains available-label ADE,not an identified full12ADE.
Same2%all/easy positive-harm bounds and signed easy-net contract. Same bounded
interpolation using predicted benefit/harm/reference/easy-reference/easy-harm;
no new threshold,alpha grid,seed selection or epsilon activation.

Primary:paired bounded-completion lower gain,interaction64minuszero64,with
equal-locality aggregation and3000paired locality bootstraps. Also known-label
gain,upper gain,easy/hard,all required per-record risk/support,amplitude/switch
rates and fitting objective. Secondary interaction64minusfrozen direct and
zero64minusfrozen direct,plus binary deployment controls,floor and full neural.
Freeze all36heads and all36cost/amplitude arrays before held outcomes. No
best-seed/model choice;report all fixed jobs and seeds. Complete risk failure
cannot be hidden by positive mean gain or an easier/undefined reference.

## Runtime And Reproduction

Actual six-arm/group50->100versusdirect100recovery pilot,total1200updates;
all36primary fits116160updates;first full refit for each arm4000updates.
Nativearm64CPU4/inter-op1/workers0,checkpoint every100updates and heartbeat,
explicit unchanged resume,12hboundary,1GiBprivate cap,10GiBdisk reserve. Separate
replay of every new cost/amplitude array and72metric groups,frozen direct
control exact. Full learning endpoints,not a quick-probe substitute.

No independent confirmation/calibration/DroneCrowd access. SOURCE is already
exposed development;these exclusions do not make a new independent benchmark.
No deployment/submission/paper-readiness promotion solely from this result.
Pixels/rawstride12/silver/detector-label/support limitations remain. No metric,
seconds,physical safety,true3D,foundation,Stage5Cexecution orSMC.
