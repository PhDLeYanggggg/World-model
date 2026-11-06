"""Attribute frozen selected easy-reference predictions; never tune decisions."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_moving_expert_reference as run


def attribution(predicted, easy_reference, known):
    p,r,k=map(run.np.asarray,(predicted,easy_reference,known))
    if p.shape!=r.shape or r.shape!=k.shape or k.dtype!=bool or (p<0).any() or (r<0).any():
        raise ValueError('Aligned nonnegative prediction/label masses and evaluation-only known mask')
    easy=k&(r>0);noneasy=k&~easy
    total=float(p.sum());wrong=float(p[noneasy].sum())
    return dict(selected=len(p),known_easy_selected=int(easy.sum()),known_noneasy_selected=int(noneasy.sum()),
        unknown_selected=int((~k).sum()),predicted_ER=total,actual_ER=float(r.sum()),
        predicted_ER_known_easy=float(p[easy].sum()),predicted_ER_known_noneasy=wrong,
        predicted_ER_unknown=float(p[~k].sum()),known_noneasy_credit_fraction=wrong/total if total>0 else None)


def main():
    _,reg=run.register();run.read(run.PUBLIC/'inference_freeze.json');data=run.parent.full.load_data();rows=[]
    for node in reg['jobs']:
        _,old=run.parent.own_and_floor(node);key=node['job']['key'];ix=old['local_ids']
        valid=data['valid'][ix];target=data['target_eval'][ix]-data['origin'][ix,None]
        packet=dict(paths=old['paths'],floor=old['floor'],cv=old['cv'],
                    causal=dict(envelope=run.np.linalg.norm(old['paths']-old['floor'][:,None],axis=-1).max(-1)))
        truth=run.parent.previous.method.amplitude.targets(packet,target,valid,node['job']['easy_cut'])['moments']
        for seed in (17,29,43):
            banks={a:run.parent.load_cache(run.read(run.PUBLIC/'inference'/f'{key}_{a}_seed{seed}.json')['cache']) for a in run.method.ARMS}
            banks['frozen']=run.parent.load_cache(run.read(run.parent.PUBLIC/'inference'/f'{key}_role_separated_seed{seed}.json')['cache'])
            for arm,b in banks.items():
                for cohort,choice in (('own_joint',b['joint_free']),('identical_frozen_joint',banks['frozen']['joint_free'])):
                    for rec in sorted(set(old['recordings'])):
                        at=run.np.flatnonzero((old['recordings']==rec)&(choice>=0));c=choice[at]
                        site=run.np.unique(old['sites'][old['recordings']==rec])
                        if len(site)!=1:raise ValueError('One recording belongs to one source component')
                        rows.append(dict(source=str(site[0]),recording=str(rec),seed=seed,arm=arm,cohort=cohort,
                            **attribution(b['moments'][at,c,3],truth[at,c,3],valid[at].any(1))))
    summaries={}
    for arm in (*run.method.ARMS,'frozen'):
        summaries[arm]={}
        for cohort in ('own_joint','identical_frozen_joint'):
            rr=[r for r in rows if r['arm']==arm and r['cohort']==cohort]
            total=sum(r['predicted_ER'] for r in rr);wrong=sum(r['predicted_ER_known_noneasy'] for r in rr)
            summaries[arm][cohort]=dict(repeated_record_views=len(rr),empty_predicted_reference_views=sum(r['known_noneasy_credit_fraction'] is None for r in rr),
                pooled_known_noneasy_credit_fraction=wrong/total if total>0 else None,
                pooled_unknown_credit_fraction=sum(r['predicted_ER_unknown'] for r in rr)/total if total>0 else None,
                pooled_repeated_view_diagnostic_not_risk_gate=True)
    run.once(run.PUBLIC/'easy_reference_attribution.json',dict(rows=rows,summary=summaries,
        uses_evaluation_labels_only_after_choices_frozen=True,used_for_inference_or_tuning=False,
        empty_support_is_not_a_safety_pass=True))
    print(summaries)


if __name__=='__main__':main()
