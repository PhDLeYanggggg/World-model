# Detector-History Failure Analysis

## What The Matched Repair Actually Changed

The previous controls varied losses,local retrieval and static geometry. This
repair adds genuine supplied past box/confidence inputs to the same cost task,
with unchanged forecast producers,targets,splits,sampler,known-only loss and
2,000updates.144real heads are trained;36disabled endpoints exactly reproduce
the old states. The first full history/mixed refit is exact. Initial startup
and internal-pilot-resume defects are reproduced and fixed before formal work;
they are not a reason to discard the successful complete experiment.

## Findings And Limits

1. **The new feature block carries useful fitting information.** Mixed known
   ADE gain rises from0.580799%disabled to1.968657%history. The paired known
   change is+1.387858pp,conditional locality CI[+0.413289,+2.887799]. History
   also exceeds current-only by+0.865586pp,CI[+0.253824,+1.878385],and noise
   by+1.331149pp,CI[+0.370384,+2.721853]. This is fitting evidence,not an
   independent test. The block mixes shape,confidence,body-relative motion and
   IoU; it does not isolate detector confidence as the cause.

2. **Greater average utility does not repair selected risk.** The mixed
   history policy selects259,875repeated row views,versus80,414disabled;
   unknown selected views rise3,445versus932. Risk support is42/444,with
   163known all-risk and307known easy-risk violating views and33undefined
   easy views. These overlap and repeat74recordings; they are not444new
   independent recordings. There are already many violations on observed
   outcomes,so missing labels alone do not explain failure.

3. **A positive pooled easy gain is not a safety contradiction.** The
   supplementary pooled known easy ADE gain is4.414932%. That is a net
   average over repeated views. The approved gate separately checks positive
   harm/reference on selected all/easy cases and retains every required
   recording. Benefits elsewhere cannot cancel those harms. Neither pooled
   net utility nor undefined empty risk overrides that contract.

4. **The physical bank captures nearly all observed mean improvement.** Its
   history gain is1.946059%,against mixed1.968657%. The mixed-minus-physical
   paired known change is only+0.022598pp,CI[-0.032990,+0.090980]. Its full
   completion-lower contrast is-0.301435pp,CI[-0.520784,-0.131315]; matched
   lower is-0.123156pp,CI[-0.244661,-0.012344]. A negative lower-bound CI is
   not proof of a negative unidentified true effect. It means a robust positive
   neural contribution is not established; the known-outcome CI also crosses0.

5. **History's primary comparison still has incomplete-outcome uncertainty.**
   History-minus-disabled full completion-lower is+0.972969pp,CI
   [-0.108698,+2.535644]; matched lower is+0.610170pp,CI
   [-0.064770,+1.640963]. All three primary control comparisons fail their
   prespecified completion-lower CI gates. No better-looking control replaces
   the primary. Physical history-minus-disabled full lower has a positive
   fitting CI,but its matched lower crosses0and its own risk support is42/444.

6. **Locality dependence and fitting optimism remain.** History mixed known
   gain reaches11.072281%at locality110,but0%at082and0.088086%at124.
   All12localities remain in the primary bootstrap. Seeds17/29/43 yield
   2.071689%,1.857457%,1.976825%; those are controller seeds on frozen
   trajectory producers,not three whole-pipeline replications. No test set is
   opened to decide which locality or model to keep.

7. **Lower objective is not a useful-policy certificate.** The mixed median
   minibatch loss falls from1.89560to0.324785for history. Noise ends still
   lower at0.301041,but its known ADE gain is only0.637508%,against history's
   1.968657%. These are fixed-step training minibatches,not validation losses
   or evidence of generalization. All144actual traces remain public.

## Failure Taxonomy

- Input alignment: repaired and exactly verified; no evidence of join failure.
- Runtime/optimization execution: full real training succeeds; no hidden fallback.
- Observable input block: positive known fitting signal,not a proven generalization lift.
- Safety after action selection: failed,with known violations as well as missing support.
- Neural proposal/routing contribution: not established; the added block mainly helps
  choosing within existing physical options at this level of evidence.
- Sensor/data quality: supplied detector-silver,acquisition-as-of uncertified.
- Geometry: no verified physical calibration; box features do not resolve it.
- Independent evidence: not_run by design; no submission/deployment promotion.

## Next Repair

Do not run another confidence/margin sweep on these scores. First examine the
frozen trajectory producer's input and candidate opportunity: the new history
was supplied only to the controller,not to trajectory dynamics. A matched
past-box branch in the forecast producer,with an exact disabled control,is a
concrete next source-development hypothesis. It must improve the proposal bank
and then earn a safe learned policy on the original protocol,not merely add
another positive oracle. A follow-up block ablation would be needed to separate
shape/confidence from the alternate body-relative motion encoding. Neither
repair nor new forecaster training is claimed completed in this experiment.

Before that repair,the producer's actual row coverage must be audited: the
current store admits only the74original controller-fitting recordings and
rejects out-of-cache IDs. It cannot silently stand in for every producer input
in the larger packed SOURCE store. Any necessary deterministic past-input
extension must preserve the appropriate producer fitting roles and closed
independent data.

Deployment remains unchanged. Independent roles remain closed. No Stage5C,
SMC,metric/seconds,true3D,foundation or human-gold claim.
