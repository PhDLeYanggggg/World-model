# Cyclic Complete-Region Cost Learning

Prospective29October2026. The registered54-state TRAIN diagnosis measures a
material gradient-estimator mismatch. Mixed-head median64-minibatch versus
complete-region cosine is0.685630 and relative vector difference0.940755.
Neural-only has5,500of36,000original steps with no auxiliary region. This finite
probe includes Monte Carlo noise,region inclusion weighting and nonlinearity;
it is not proof of an infinite-sample bias or a unique cause of held failure.

## One Primary Training Repair

Keep current398causal features,26,309parameters,initial weights,preprocessing,
original128-row source-balanced base draws,AdamW,schedule,seeds and fixed2%
inference unchanged. Keep the same old scene teacher choices and known-label
H/R/ER/EH region centroids. Do not refresh teacher membership or tune weights.

Instead of applying nonlinear divergence to whichever teacher rows appear in
each ordinary minibatch,cycle over all nonempty fixed recording regions. Use a
seeded NumPy permutation independent of the original Torch sampler. At each
optimizer update,add the exact gradient of one complete region's moment
divergence with fixed weight1. A no-grad moment pass and a second chunked
chain-rule pass use the same unchanged weights. Accumulate with the ordinary
base loss,then apply the same clipping and optimizer step. This is exact within
one region,not an exact full-all-region update or independent calibration.

The order balances regions over complete cycles but changes their weighting
relative to the old represented-minibatch objective. This is an optimization/
conditional-objective test,not a claimed software bugfix. The additional known
teacher rows and two-pass forward row counts must be reported separately:
same base draws and optimizer steps do not imply same FLOPs or data visits.

## Matched Controls And Runtime

Sixcontexts,threehead seeds,threebanks:54fresh fits/108,000primary updates.
Chunk512,batch128,width64,2,000steps,lr.001,underharm4,ranking.5,gradientclip5.
Compare to the previous54stochastic-region endpoints,which are cached-verified
controls. Require all-bank10+10/direct20real recovery,full2,000-step disabled
neutral matching old scene scientific state,and all initialization/base draw/
RNG/normalizer/teacher-region hashes. Bind new region order and extra counters
in checkpoints. Preserve the frozen original forecaster.

First run local arm64CPU4/interop1/workers0,checkpoint100/heartbeat50. Pilot
estimates full cost. Healthy training up to12h is allowed,not a reason to claim
quick completion. Preserve10GiBdisk reserve and256MiBprivate checkpoint cap.
If resources are genuinely insufficient,retain the same scientific roster and
consider CREATE after current scheduler/environment audit;never duplicate jobs.

## Readout And Nonclaims

Commit all54endpoints before first exposed inner-held readout. Report all144
groups,paired exact-minus-stochastic and mixed-minus-physical utility/completion
intervals,all required-recording all/easy2%risks,undefined support,unknown
outcomes and fixed1/2/4sensitivities. No best multiplier,seed or subgroup
promotion after readout. Empty selected risk is not_estimable. Count matching
is utility-only. Exact replay is required,not independent scientific evidence.

This is SOURCE development on obs8/pred12/rawstride12 webcam detector-track
pixel coordinates,not SDD gold,metric/seconds,true3D or foundation modeling.
Independent selection/calibration/confirmation and DroneCrowd stay closed.
No future endpoint,input future-validity,central velocity,test normalization or
test-built goals. Supervised future costs are fitting labels only. No Stage5C
execution,SMC or deployment promotion follows from fitting convergence alone.
