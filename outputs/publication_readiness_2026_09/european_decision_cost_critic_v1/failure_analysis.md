# What Improved And What Still Fails

## Observed Result

Known-label full-joint ADE gain rises7.672120%->9.138437%;paired gain is
+1.466317pp [conditional95%CI +0.529223,+2.342938]. Ten exposed localities
improve and two worsen. This positive SOURCE result is retained,not hidden.
Known benefit increases1.493403pp and known harm increases0.027086pp.
Their difference exactly reconstructs the paired known gain in each source.

The registered completion-lower contrast is-2.744940pp [-6.587075,+0.210798].
It does not pass. Its interval crosses0,so this is not a significantly negative
paired effect at the nominal95%level. The own floor-relative lower gain is
+8.653152pp [+5.387204,+12.817439],but it cannot replace the paired endpoint
or original recording-risk requirements. No promotion or deployment.

## Failure Taxonomy

1. **Soft-policy support mismatch,demonstrated.** Bounded net-cost logits and
   a fixed zero fallback score force high soft intervention mass at the frozen
   global RMS temperature. Median rowwise minimum switch probability is
   0.752903-0.797614 across contexts. An independently checked relaxed logit-box
   optimum still has positive expected constraints in all444 recording views,
   for all three constraints. This proves this soft training family cannot
   satisfy those SOURCE constraints;more of the identical training cannot fix
   representability. It is not an impossibility proof for the actual hard decoder.
2. **Cost calibration versus decision pressure,measured association.** The same
   outputs serve as moments and softmax scores. Normalized harm MSE rises
   0.106970->0.575948;benefit MSE rises0.349550->0.379746. Easy-harm MSE improves
   slightly0.151156->0.144618,but known easy-risk violations rise362->375.
   Lower average moment error is not final-selection calibration;these figures
   do not uniquely identify a causal contribution from every objective component.
3. **Soft/hard and context/selection gap,not removed.** Inference discards training
   temperature/duals and applies the unchanged eligibility and joint MILP. The
   original selected-query context need not equal the final selected set. The
   support proof identifies a mismatch,not a complete explanation of each wrong action.
4. **Unobserved outcome disagreement,measured.** The paired unknown penalty is
   4.211257pp,exceeding known gain1.466317pp. Unknown selected views fall7256->5722,
   but counts alone do not bound changed-forecast disagreement. This cannot be
   repaired by imputing labels,dropping sparse localities or using held labels.
5. **Utility is not recording safety.** Only9/444record-risk views are supported,
   versus6/444previously. Known all-risk violations300,easy375;12 selected-easy
   denominators undefined. Worst known easy degradation grows26.477426%->28.329410%,
   with53record views above2%(previous28). Hard pooled gain barely changes
   9.372667%->9.386142%;easy pooled gain falls4.648667%->3.549522%.
   All original required views remain;mean gain cannot replace their risk test.
6. **Evidence boundary.** These are exposed SOURCE resubstitution results with
   one forecast producer and three critic seeds. No independent generalization,
   world-dynamics improvement or submission readiness is established.

## Engineering,Not Scientific Failure

The initial readout directory error is repaired by creating own output directories;
no loss,model or registration changes. The full resumed readout exits0.
Five new full-joint numerical postchecks retain their feasible independent parent;
all are scored. Six new matched-control postchecks also retain parents. There is
no claim every MILP achieved an optimum,no dropping difficult queries,and no
counterfactual assertion that these fallbacks caused the aggregate result.
Exact full refit and all72 archived-action inference/score checks pass. They do
not turn SOURCE exposure into independent evidence. No Stage5C or SMC.
