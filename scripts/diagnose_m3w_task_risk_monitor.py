"""Descriptive feasibility check of existing fixed TRAIN monitors, not tuning."""
import argparse
import json
import statistics

from scripts import read_m3w_task_risk_final_heads as heads
from scripts import report_m3w_task_risk as report

HOME=heads.manager.PUBLIC


def summarize(documents):
    output={}
    for arm in ('quadratic','task_utility','task_risk'):
        ds=[d for d in documents if d['arm']==arm]
        if len(ds)!=72 or any(d['trace'][-1]['step']!=2000 for d in ds):
            raise ValueError('All fixed final TRAIN monitors required')
        ms=[d['trace'][-1]['monitor'] for d in ds]
        output[arm]=dict(views=len(ds),
            positive_soft_all_excess=sum(m['soft_all_excess']>0 for m in ms),
            positive_soft_easy_excess=sum(m['soft_easy_excess']>0 for m in ms),
            median_soft_all_excess=statistics.median(m['soft_all_excess'] for m in ms),
            median_soft_easy_excess=statistics.median(m['soft_easy_excess'] for m in ms),
            median_duals=[statistics.median(d['dual'][i] for d in ds) for i in range(2)],
            maximum_duals=[max(d['dual'][i] for d in ds) for i in range(2)],
            cap_reached_views=sum(any(v>=100 for v in d['dual']) for d in ds))
    return output


def artifacts(documents,freeze,summary,diagnostics,bindings):
    rows=summarize(documents);risk=rows['task_risk']
    safety=summary['safety']['task_risk'];decision=diagnostics['task_risk']
    text=f'''# Why the Matched Decision-Risk Trial Does Not Advance

## Evidence Source

This is a post-readout descriptive check of already saved TRAIN monitors, not
another model evaluation or a hyperparameter search. The fixed128-query monitor
uses known TRAIN labels and the registered sampling seed. It is not all TRAIN,
independent calibration, a new dataset or an independent risk guarantee.
Every receipt and checkpoint is checked against the complete216-fit freeze.

## What Is Established

At the fixed2,000-update endpoint, task-risk still has positive soft easy-risk
constraint excess in **{risk['positive_soft_easy_excess']}/72** monitors and
positive all-risk excess in **{risk['positive_soft_all_excess']}/72**. Its median
soft easy excess is{risk['median_soft_easy_excess']:.9f} in normalized cost units.
Its median duals are{risk['median_duals']}; maximum duals are{risk['maximum_duals']}.
No fit reaches the registered100 dual cap. Thus cap saturation is not an observed
explanation; the empirical constraints were not solved even on most fixed TRAIN
monitors. More epochs or different normalization have not yet been tested.

The separate frozen development readout supports utility improvement over
quadratic, including same-count matching, but not over the task-utility control.
Known easy-risk violations remain{safety['known_easy_violations']}/72 and conservative upper-bound violations
{safety['easy_upper_violations']}/72. All{decision['selected_easy_harm_underprediction_views']} candidate views underestimate selected easy harm, with median
predicted/actual ratio about{decision['median_predicted_actual_selected_harm_ratio']:.6f}. This is not repaired calibration.

The mean soft action is{decision['mean_soft_action']:.6f} while mean hard intervention is{decision['mean_hard_intervention']:.6f} across
views. A0.5 diagnostic gives{decision['mean_soft_hard_disagreement']:.6f} disagreement. These soft values are not
calibrated probabilities; this comparison suggests a surrogate/deployment
mismatch, not a proved unique cause. It does not authorize tuning that cutoff.

## Interpretation and Next Falsifiable Step

- Decision-oriented training has a genuine development utility signal under the
  fixed forecast producers. This is not new trajectory dynamics or independent
  cross-domain evidence.
- Adding the current dual penalty gives up utility relative to the unconstrained
  task objective without enforcing the required risk budget. It is not a safety
  solution simply because some violation counts decrease.
- Missing-outcome bounds remain an additional obstruction, but known-label
  violations show that missingness alone cannot explain failure.
- Next isolate TRAIN constraint optimization: verify each constraint's scale,
  gradient and soft/hard agreement on the same fixed TRAIN queries, then register
  one matched repair before training. Candidate repairs include constraint-scale
  normalization or a better hard-rule surrogate. Their effectiveness is not yet
  established; do not choose them using independent labels or a new test sweep.

Keep the current deployment unchanged. Do not promote a utility-only arm after
seeing these results, weaken the2%definition, remove unsupported views, or open
independent confirmation to rescue a failed development policy. Stage5C and SMC
remain off; no metric, seconds, true3D or foundation claim is added.
'''
    receipt=dict(result_source='fresh_run_descriptive_recalculation_from_verified_TRAIN_traces',
        training_freeze_sha256=freeze,arms=rows,monitor_queries=128,
        known_TRAIN_only=True,full_TRAIN_risk_claim=False,
        new_inference=False,new_training=False,new_model_selection=False,
        unique_causal_explanation_established=False,independent_roles_read=False,
        deployment_changed=False,stage5c_executed=False,smc_enabled=False)
    receipt.update(bindings)
    return {'train_monitor_feasibility.json':json.dumps(receipt,indent=2,allow_nan=False)+'\n',
            'failure_analysis.md':text}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');a=p.parse_args()
    docs,inv=heads.local_admission()
    _,_,summary,groups=report.validate()
    bindings=dict(readout_summary_sha256=report.run.sha(report.HOME/'summary.json'),
                  readout_complete_sha256=report.run.sha(report.HOME/'complete.json'))
    for name,raw in artifacts(list(docs.values()),inv['training_freeze_sha256'],
                              summary,report.diagnostics(groups),bindings).items():
        path=HOME/name
        if a.check or path.exists():
            if path.read_bytes()!=raw.encode():raise ValueError('Preserve conflicting diagnostic')
        else:
            with path.open('xb') as f:f.write(raw.encode())
    print(json.dumps(dict(status='checked' if a.check else 'reported',new_training=False)))


if __name__=='__main__':main()
