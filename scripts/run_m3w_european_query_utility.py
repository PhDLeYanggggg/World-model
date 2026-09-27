"""Freeze and evaluate source-separated query utility allocation."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_european_selection_exchange as source
from src.world_model import m3w_query_utility as api

parent=source.parent
PUBLIC=source.PUBLIC.parent/'european_query_utility_v1'
PRIVATE=source.PRIVATE.parent/'european_query_utility_v1'
CONFIG='configs/m3w_european_query_utility_v1.json'
FILES=[CONFIG,'src/world_model/m3w_query_utility.py','scripts/run_m3w_european_query_utility.py',
    'tests/test_m3w_query_utility.py',str(PUBLIC.relative_to(ROOT)/'protocol.md')]


def beat(state,**kw):
    r=dict(state=state,pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw)
    parent.base.inter.json_write(PRIVATE/'heartbeat.json',r)
    with (PRIVATE/'events.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
    print(json.dumps(r),flush=True)


def load(register=False):
    cfg=json.loads((ROOT/CONFIG).read_text()); sp=source.PUBLIC/'verification.json'
    assert parent.base.digest(sp)==cfg['parent_seal_sha256']
    seal=json.loads(sp.read_text())
    for p,h in seal['source_bindings'].items():assert parent.base.digest(ROOT/p)==h,p
    for p,h in seal['artifacts'].items():assert parent.base.digest(source.PUBLIC/p)==h,p
    _,data,jobs,oid,previous=source.load()
    assert cfg['groups']==108 and cfg['risk_budget']==.02
    assert not any(cfg[k] for k in ('new_training','threshold_search','independent_roles_read','deployment_changed','stage5c_executed','smc_enabled'))
    identity=dict(parent_seal=parent.base.artifact(sp),bindings={p:parent.base.digest(ROOT/p) for p in FILES},
        localities=previous['localities'],source_rows=len(data['sites']),new_training=False,independent_roles_read=False)
    if register:parent.base.immutable_json(PUBLIC/'registration.json',identity)
    else:
        assert json.loads((PUBLIC/'registration.json').read_text())==identity
        parent.base.inter.committed(PUBLIC/'registration.json')
    return cfg,data,jobs,oid,identity


def decisions(cfg,data,jobs,oid,identity,*,resume=False,replay=False,limit=None):
    # Deliberately remove all future fields before real predictor/context execution.
    causal={k:data[k] for k in ('sites','history','origin','geometry','recordings','frames')}
    freeze=json.loads((parent.PUBLIC/'decision_freeze.json').read_text())
    refs={Path(r['path']).parent.name:r for r in freeze['heads']}; outputs=[]
    for c in parent.base.floor_api.contexts(causal,jobs,oid):
        for pair in range(6):
            name=c['name']+f'_pair{pair}'; home=PRIVATE/'groups'/name; dest=home/'complete.json'
            ref=refs[name]; assert parent.base.artifact(ROOT/ref['path'])==ref
            rec=json.loads((ROOT/ref['path']).read_text())
            if dest.exists() and resume and not replay:
                doc=json.loads(dest.read_text()); assert doc['identity']==identity and doc['parent_head']==ref
                for r in doc['artifacts'].values():assert parent.base.artifact(ROOT/r['path'])==r
            else:
                if dest.exists() and not replay:raise ValueError('Use --resume for completed decisions')
                if shutil.disk_usage(PRIVATE).free<10*2**30:raise OSError('Keep10GiB and completed decisions')
                for r in rec['artifacts'].values():assert parent.base.artifact(ROOT/r['path'])==r
                with np.load(ROOT/rec['artifacts']['decisions']['path'],allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
                state=parent.api.read_checkpoint(ROOT/rec['artifacts']['checkpoint']['path']); pr=state['preprocess']
                pos=np.searchsorted(c['ids'],a['ids']);np.testing.assert_array_equal(c['ids'][pos],a['ids'])
                u=parent.api.descriptors(causal['geometry'][a['ids']],c['floor'][pos],c['prediction'][pos],c['x'][pos],pr)
                p=parent.api.predict(parent.model_from(state),c['x'][pos],u,c['env'][pos],pr,state['descriptor_preprocess'])
                assert parent.base.inter.array_hash(p)==rec['prediction_sha256']
                ridge=json.loads((parent.base.floor_api.PRIVATE/'fits'/name/'complete.json').read_text())
                for r in ridge['artifacts'].values():assert parent.base.artifact(ROOT/r['path'])==r
                with np.load(ROOT/ridge['artifacts']['scores']['path'],allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['ids'],a['ids'])
                    utility=z['scores'][:,5].astype(float)-z['scores'][:,6].astype(float)
                q=parent.base.api.signed(p.astype(float))/pr['cost_scale']
                keys=np.array([str(s)+'|'+str(r) for s,r in zip(causal['sites'][a['ids']],causal['recordings'][a['ids']])])
                choices,info=api.grouped(utility/pr['cost_scale'],q,a['eligible'],a['descriptor'],keys,
                    causal['frames'][a['ids']],a['ids'],node_limit=cfg['node_limit'])
                arrays=dict(ids=a['ids'],independent=a['descriptor'],control_matched_count=a['control_matched_count'],
                    eligible=a['eligible'],**choices)
                parent.write_arrays(home/'decisions.npz',arrays,replay)
                doc=dict(identity=identity,parent_head=ref,queries=info,future_fields_removed=True,
                    parent_prediction_hash_verified=True,held_outcomes_used=False,
                    artifacts=dict(decisions=parent.base.artifact(home/'decisions.npz')))
                parent.base.immutable_json(dest,doc)
            outputs.append(parent.base.artifact(dest));beat('decisions_replayed' if replay else 'decisions_frozen',group=name,complete=len(outputs))
            if limit and len(outputs)>=limit:return
    assert len(outputs)==108
    parent.base.immutable_json(PUBLIC/'decision_freeze.json',dict(identity=identity,groups=outputs,
        new_training=False,held_outcomes_used=False,future_fields_removed=True))
    if replay:parent.base.immutable_json(PUBLIC/'decision_replay.json',dict(groups=108,exact=True,future_fields_removed=True))


def evaluate(cfg,data,jobs,oid,identity,*,replay=False):
    parent.base.inter.committed(PUBLIC/'decision_freeze.json')
    freeze=json.loads((PUBLIC/'decision_freeze.json').read_text());assert freeze['identity']==identity
    refs={Path(r['path']).parent.name:r for r in freeze['groups']};rows=[];solver=[]
    for c in parent.base.floor_api.contexts(data,jobs,oid):
        cv,cf,(f,ff),(n,nf)=parent.base.floor_api.costs(c,data,np.arange(len(c['ids'])))
        for pair in range(6):
            name=c['name']+f'_pair{pair}';r=refs[name];assert parent.base.artifact(ROOT/r['path'])==r
            doc=json.loads((ROOT/r['path']).read_text());solver.append(doc['queries'])
            for a in doc['artifacts'].values():assert parent.base.artifact(ROOT/a['path'])==a
            with np.load(ROOT/doc['artifacts']['decisions']['path'],allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
            pos=np.searchsorted(c['ids'],a['ids']);np.testing.assert_array_equal(c['ids'][pos],a['ids'])
            for site in c['pairs'][pair][1]:
                at=data['sites'][a['ids']]==site;ix=pos[at]
                for policy in api.POLICIES:
                    take=np.zeros(len(ix),bool) if policy=='floor' else a[policy][at]
                    metric=parent.base.floor_api.metric(cv[ix],f[ix],n[ix],cf[ix],ff[ix],nf[ix],
                        np.where(take,n[ix],f[ix]),np.where(take,nf[ix],ff[ix]),take,
                        data['valid'][c['ids'][ix]],c['job']['design']['easy_cut'],c['job']['design']['hard_cut'])
                    rows.append(dict(group=name,site=site,seed=c['job']['old_identity']['seed'],policy=policy,metric=metric))
        beat('held_scored',group=c['name'])
    def reduce(rr):
        return {p:{k:parent.base.inter.paired_localities([r for r in rr if r['policy']==p],identity['localities'],k,
            cfg['bootstrap_resamples'],cfg['bootstrap_seed']) for k in rr[0]['metric']} for p in sorted({r['policy'] for r in rr})}
    index={(r['group'],r['site'],r['policy']):r['metric'] for r in rows};contrasts=[]
    selected=[r for r in rows if r['policy']=='joint_utility']
    for row in selected:
        x=row['metric']
        for old in ('independent','control_matched_count','utility_topk'):
            y=index[(row['group'],row['site'],old)]
            harm=None if x['selected_positive_harm_ratio'] is None or y['selected_positive_harm_ratio'] is None else 100*(y['selected_positive_harm_ratio']-x['selected_positive_harm_ratio'])
            contrasts.append(dict(group=row['group'],site=row['site'],seed=row['seed'],policy=old,
                metric=dict(ADE_gain_percent=100*(1-x['error_sum']/y['error_sum']) if y['error_sum']>0 else None,
                    intervention_difference_pp=100*(x['intervention_rate']-y['intervention_rate']),positive_harm_reduction_pp=harm)))
    summary=reduce(rows);paired=reduce(contrasts);m=[r['metric'] for r in selected]
    gates=dict(primary_ADE_advantage=paired['independent']['ADE_gain_percent']['ci95'][0]>0,
        every_view_defined_risk_within_budget=all(v['selected_positive_harm_ratio'] is not None and v['selected_positive_harm_ratio']<=.02 for v in m),
        every_view_easy_preserved=all(v['easy_gain_CV'] is not None and v['easy_gain_CV']>=-2 for v in m),
        no_zero_CV_harm=all(v['zero_CV_harmed']==0 for v in m),
        same_intervention_count=all(v['metric']['intervention_difference_pp']==0 for v in contrasts),
        nonzero_each_locality=all(v is not None and v>0 for v in summary['joint_utility']['intervention_rate']['by_site'].values()))
    gates['exploratory_joint_screen_pass']=all(gates.values())
    gates.update(independent_confirmation=False,calibration_certificate=False,deployment_changed=False,stage5c_executed=False,smc_enabled=False)
    out=dict(identity=identity,result_source='fresh_causal_query_utility_allocation_cached_verified_estimators',
        summary=summary,paired=paired,gates=gates,solver={k:sum(x[k] for x in solver) for k in solver[0]},
        by_seed={str(s):reduce([r for r in rows if r['seed']==s]) for s in (17,29,43)},
        worst_views={p:dict(worst_easy_gain_CV=min(r['metric']['easy_gain_CV'] for r in rows if r['policy']==p and r['metric']['easy_gain_CV'] is not None),
            risk_violating_views=sum(r['metric']['selected_positive_harm_ratio'] is not None and r['metric']['selected_positive_harm_ratio']>.02 for r in rows if r['policy']==p),
            undefined_risk_views=sum(r['metric']['selected_positive_harm_ratio'] is None for r in rows if r['policy']==p)) for p in summary},new_training=False)
    parent.base.immutable_json(PRIVATE/'details.json',dict(rows=rows,contrasts=contrasts))
    parent.base.immutable_json(PUBLIC/'summary.json',out);parent.base.immutable_json(PUBLIC/'gates.json',gates)
    if replay:parent.base.immutable_json(PUBLIC/'evaluation_replay.json',dict(exact=True,
        summary=parent.base.artifact(PUBLIC/'summary.json'),details=parent.base.artifact(PRIVATE/'details.json')))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['register','decide','replay','evaluate','replay_evaluate'],required=True)
    p.add_argument('--resume',action='store_true');p.add_argument('--limit',type=int);args=p.parse_args()
    PUBLIC.mkdir(parents=True,exist_ok=True);PRIVATE.mkdir(parents=True,exist_ok=True)
    parent.base.torch.set_num_threads(4);parent.base.torch.set_num_interop_threads(1)
    with (PRIVATE/'run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);beat('started',phase=args.phase)
        values=load(args.phase=='register')
        if args.phase in ('decide','replay'):decisions(*values,resume=args.resume,replay=args.phase=='replay',limit=args.limit)
        elif args.phase in ('evaluate','replay_evaluate'):evaluate(*values,replay=args.phase=='replay_evaluate')
        beat('complete',phase=args.phase)


if __name__=='__main__':main()
