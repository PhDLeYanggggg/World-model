"""Diagnose frozen choices; never alter predictions, admission or the primary."""
import csv
import io
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_moving_role_separated_cost as run
from scripts.report_m3w_moving_complete_donor_policy import category


def ratio(n,d):
    return float(n/d) if d>0 else None


def moment_summary(predicted,actual):
    p,t=map(run.np.asarray,(predicted,actual))
    if p.shape!=(5,) or t.shape!=(5,) or (p<0).any() or (t<0).any():
        raise ValueError('Nonnegative selected predicted/contract contributions required')
    return dict(predicted_all_risk=ratio(p[1],p[2]),predicted_easy_risk=ratio(p[4],p[3]),
        actual_completion_all_risk=ratio(t[1],t[2]),actual_completion_easy_risk=ratio(t[4],t[3]),
        harm_underprediction_factor=ratio(t[1],p[1]),easy_harm_underprediction_factor=ratio(t[4],p[4]),
        reference_overprediction_factor=ratio(p[2],t[2]),easy_reference_overprediction_factor=ratio(p[3],t[3]),
        predicted_harm=float(p[1]),actual_bound_harm=float(t[1]),
        predicted_reference=float(p[2]),actual_observed_reference=float(t[2]),
        predicted_easy_harm=float(p[4]),actual_bound_easy_harm=float(t[4]),
        predicted_easy_reference=float(p[3]),actual_observed_easy_reference=float(t[3]))


def amplitudes(choice):
    c=run.np.asarray(choice)
    if not ((c==-1)|((c>=1)&(c<=6))).all():raise ValueError('Original six amplitudes or genuine floor only')
    a=run.np.zeros(len(c));at=c>=1;a[at]=run.np.asarray(run.donor.FRACTIONS)[c[at]-1]
    return a


def main():
    cfg,reg=run.register();a=run.read(run.PUBLIC/'analysis.json');data=run.full.load_data()
    records=[];usage=[];primary=[];conditional_probe=[]
    for node in reg['jobs']:
        own,old=run.own_and_floor(node);job=node['job'];key=job['key'];ix=old['local_ids']
        target=data['target_eval'][ix]-data['origin'][ix,None];valid=data['valid'][ix]
        packet=dict(paths=old['paths'],floor=old['floor'],cv=old['cv'],
                    causal=dict(envelope=run.np.linalg.norm(old['paths']-old['floor'][:,None],axis=-1).max(-1)))
        truth=run.previous.method.amplitude.targets(packet,target,valid,job['easy_cut'])['moments']
        for seed in (17,29,43):
            bank={arm:run.load_cache(run.read(run.PUBLIC/'inference'/f'{key}_{arm}_seed{seed}.json')['cache'])
                  for arm in run.method.ARMS}
            paths={arm:run.full.method.paths_for(old['paths'],old['floor'],b['joint_free']) for arm,b in bank.items()}
            for site in node['part']['names']:
                take=old['sites']==site
                primary.append(dict(source=site,seed=seed,**run.full.method.original.paired(
                    paths['composed_anchor'][take],paths['role_separated'][take],old['floor'][take],target[take],valid[take])))
                for arm,b in bank.items():
                    original=next(r for r in a['rows'] if r['job']==key and r['source']==site and r['arm']==arm and r['seed']==seed)
                    active=old['active'][(old['sites'][old['active']]==site)&valid[old['active']].any(1)]
                    lookup=run.np.full((len(old['ids']),7),-1,int)
                    lookup[own['row'],own['action']]=run.np.arange(len(own['ids']))
                    pos=lookup[active,old['proposal_self'][active]]
                    if (pos<0).any():raise ValueError('Frozen proposal lost from diagnostic cohort')
                    yy=run.torch.from_numpy(own['moments'][pos]);y=(yy[:,[1,4]]-.02*yy[:,[2,3]]).numpy()
                    old_state=run.neural.rel.base.read_checkpoint(run.checked(run.full.risk_checkpoint(key,site,seed)))
                    sigma=old_state['preprocess']['target_scale'];w=own['weights'][pos];w=w/w.sum()
                    preds={}
                    for kind in ('conditional','moments'):
                        p=b[kind][active,old['proposal_self'][active]]/own['scales'][pos,None]
                        preds[kind]=p[:,[1,4]]-.02*p[:,[2,3]]
                    preds['frozen_known_only']=old['risk'][(17,29,43).index(seed),active]
                    conditional_probe.append(dict(source=site,seed=seed,arm=arm,rows=len(pos),
                        normalized_signed_MSE={k:float(run.np.sum(w[:,None]*((v-y)/sigma)**2)/2) for k,v in preds.items()}))
                    for name in run.POLICIES:
                        selected=take&(b[name]>=0);alpha=amplitudes(b[name][take])
                        usage.append(dict(job=key,source=site,seed=seed,arm=arm,policy=name,
                            rows=int(take.sum()),selected=int(selected.sum()),
                            mean_amplitude_all=float(alpha.mean()),mean_amplitude_selected=float(alpha[alpha>0].mean()) if (alpha>0).any() else None,
                            outside_frozen_proposer_selected=int((selected&(old['proposal_self']<0)).sum()),
                            changed_amplitude_within_proposer=int((selected&(old['proposal_self']>=0)&(b[name]!=old['proposal_self'])).sum()),
                            action_counts={str(k):int((b[name][take]==k).sum()) for k in range(1,7)}))
                        for recording,v in original['policies'][name]['recordings'].items():
                            at=run.np.flatnonzero(selected&(old['recordings'].astype(str)==recording))
                            p=b['moments'][at,b[name][at]].sum(0);t=truth[at,b[name][at]].sum(0)
                            summary=moment_summary(p,t)
                            for lhs,rhs in (('actual_completion_all_risk','completion_all_risk_upper'),
                                            ('actual_completion_easy_risk','completion_easy_risk_upper')):
                                if v[rhs] is None:assert summary[lhs] is None
                                else:run.np.testing.assert_allclose(summary[lhs],v[rhs],rtol=1e-10,atol=1e-12)
                            records.append(dict(job=key,source=site,seed=seed,arm=arm,policy=name,recording=recording,
                                reason=category(v),unknown_selected=v['unknown_selected'],**summary))
    for k in ('known_pp','completion_lower_pp','completion_upper_pp'):
        run.neural.rel.base.exact(run.interval(primary,k,cfg),a['primary'][k])
    run.once(run.PUBLIC/'primary_source_seed_rows.json',dict(rows=primary,summary_reproduces_registered_primary_exactly=True))
    run.once(run.PUBLIC/'amplitude_usage.json',dict(rows=usage,amplitude_or_count_change_not_pure_membership_effect=True))
    run.once(run.PUBLIC/'conditional_risk_probe.json',dict(rows=conditional_probe,
        identical_known_proposal_cohort_and_normalizer=True,
        composed_and_conditional_predictions_have_different_statistical_targets=True,
        posthoc_diagnostic_not_model_selection=True))
    header=list(records[0]);out=io.StringIO();writer=csv.DictWriter(out,fieldnames=header,lineterminator='\n')
    writer.writeheader();writer.writerows(records);(run.PUBLIC/'selection_calibration.csv').write_text(out.getvalue())
    summaries={}
    for arm in run.method.ARMS:
        summaries[arm]={}
        for name in run.POLICIES:
            rows=[r for r in records if r['arm']==arm and r['policy']==name]
            summaries[arm][name]=dict(record_views=len(rows),
                categories={key:sum(r['reason']==key for r in rows) for key in sorted(set(r['reason'] for r in rows))},
                easy_violations_without_unknown_selected=sum(r['unknown_selected']==0 and
                    r['actual_completion_easy_risk'] is not None and r['actual_completion_easy_risk']>.02 for r in rows),
                harm_underpredicted_views=sum(r['actual_bound_harm']>r['predicted_harm'] for r in rows),
                easy_harm_underpredicted_views=sum(r['actual_bound_easy_harm']>r['predicted_easy_harm'] for r in rows),
                reference_overpredicted_views=sum(r['predicted_reference']>r['actual_observed_reference'] for r in rows),
                easy_reference_overpredicted_views=sum(r['predicted_easy_reference']>r['actual_observed_easy_reference'] for r in rows))
    by_seed={str(seed):{k:float(run.np.mean([r[k] for r in primary if r['seed']==seed]))
             for k in ('known_pp','completion_lower_pp','completion_upper_pp')} for seed in (17,29,43)}
    by_source={site:{k:float(run.np.mean([r[k] for r in primary if r['source']==site]))
             for k in ('known_pp','completion_lower_pp','completion_upper_pp')} for site in sorted(set(r['source'] for r in primary))}
    run.once(run.PUBLIC/'selection_diagnostics.json',dict(by_policy=summaries,primary_by_risk_seed=by_seed,
        primary_by_source=by_source,posthoc_diagnosis_of_frozen_choices=True,used_for_policy_tuning=False,
        original_rules_unchanged=True,source_primary_unmodified=True))
    print('Frozen selection calibration, amplitudes and primary source/seed contrasts verified.')


if __name__=='__main__':main()
