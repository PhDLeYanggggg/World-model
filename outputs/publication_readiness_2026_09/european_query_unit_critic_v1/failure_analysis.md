# Query units repaired; complete-policy protection still fails

## Controlled result

Both 36-head fits and the full72-group readout are complete. A separate full
replay exactly reproduces every group,144 fixed-critic policy controls and the
canonical summary. These are exposed SOURCE fitting/resubstitution results,
not independent confirmation or a newly trained trajectory forecaster.

Only the query-score loss denominator changes. Both arms retain the same
30,726-parameter network,0.5/0.5 weights,2,000 updates,source/query draws,
initialization,point loss,original selected-set context,forecasts and allocator.
The changed scale is the source-balanced RMS of actual original-query signed
targets,floored1e-6. It is a fitting-label statistic,not an inference feature.

Selected-set joint known ADE gain increases6.574842%->8.215347%:
+1.640505pp,conditional locality CI[+1.101933,+2.251535]. The known increment
is positive in every12 exposed locality and all three critic-seed means.
The prespecified wholly-unknown completion-lower increment is -2.597889pp,
CI[-7.642133,+0.293356]. It fails the required positive lower CI;the interval
does not establish a strictly negative population effect either.

Complete-recording risk remains decisive:1/444 supported views,367 known
all-risk violations,383 known easy-risk violations and13 undefined easy views.
Four of eight fitting checks pass. No policy is deployed or promoted.

## What actually improved

1. The actual gradient manipulation works. Same-batch query/point gradient
   medians rise from about0.0008/0.0007 to1.758/1.635. Unknown draws still have
   zero supervised loss. This is not generic evidence that larger gradients
   are always better or that another multiplier would succeed.
2. The comparable fixed16-query fitting monitors improve in ORIGINAL units:
   row0.000443910->0.000144110,set0.000448284->0.000153624. In contrast,the
   all-action point-risk fit worsens. Different new/old total loss values are
   not compared as if their units were unchanged.
3. The complete read-only score audit confirms more than a favorable monitor:
   on the same original selections,all-channel set-score MSE falls
   0.000974966->0.000648158,and easy MSE0.000193034->0.000099323. Support is
   the same66/72 views and11/12 localities;the empty locality is not zero-imputed.
4. Known full-policy utility improves despite worse all-action MSE
   (set0.108521->0.127527). Marginal MSE,selected utility and risk support are
   distinct endpoints. None replaces the prespecified primary requirement.

## What remains wrong

The new set-joint policy differs substantially from the ORIGINAL selected set
used in its query loss:200,594 introduced row views,55,690 removed row views,
126,725 changed retained actions. The score audit evaluates the original and
final selections separately;their populations differ,so differences do not
causally identify selection shift as the unique failure mechanism.

On each model's final selected set,conditional all-channel score MSE increases
0.002657289->0.003562664,while easy-channel MSE improves
0.000474468->0.000154542. The new final-set mean easy-score underprediction is
0.001615704 = harm underprediction0.001659358 + reference contribution
-0.000043655. Harm underprediction persists even when easy-reference bias has
the opposite sign. Earlier exact reference-sharing repair also failed;neither
reference incoherence nor unit scaling is a sufficient demonstrated explanation.

Unknown support is not repaired by fewer interventions. Selected rows decrease
432,796->410,575,but unknown selected rows increase8,010->8,430. Locality082
has only+0.014824pp known increment but -28.610064pp completion-lower increment.
That is uncertainty from missing outcomes,not an observed negative trajectory
effect. It cannot be removed from the full policy to claim safety. Known-risk
violations also remain,so unknown labels are not the only problem.

Selected-set pooling still loses to local risk learning with corrected units:
known -0.347678pp CI[-0.617066,-0.028657],lower -5.657865pp
CI[-13.058285,-1.481855]. The explicit common-count known contrast crosses zero,
but its lower contrast stays negative. Better normalized learning does not
establish useful selected-context representation or a context/units interaction.

## Next decisive repair

A separate registered experiment changes only query supervision from the fixed
original actions to the CURRENT decoder's detached choices. It keeps context
inputs,normalizers,weights,model,sampler and final decoder unchanged. This tests
the moving-selection explanation;it is not an already successful fix,an exact
policy-gradient method or a calibration theorem. The real600-update pilot
passes fixed-control and interrupted-recovery checks;full fitting is running.
Do not use source partial outcomes to choose thresholds or claim final success.

Missing-label support,reference bias and protected-output interpolation limits
remain separate constraints. The current query change does not solve them by
definition. Any further action must follow the completed comparison,not a new
generic architecture or threshold sweep. Independent roles remain closed.

## Claim boundary

Three critic seeds share one forecaster seed;overlapping windows/reused contexts
are not independent samples. Intervals are nominal exploratory locality
bootstraps,not adaptive-search-adjusted confirmation. Complete-label accounting
means available ADE-derived costs,not every future point or endpoint.
No new physical-only reoptimization with these critics is run,so the earlier
bank result cannot be reused to attribute all8.215% to the neural forecast.
No new dynamics,metric,seconds,human-gold,true3D,foundation,Stage5C execution,
SMC,deployment or submission-ready claim.
