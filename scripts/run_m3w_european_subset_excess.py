"""Registered anchored subset-risk repair on the frozen development roles."""
import argparse
import fcntl
import json
import os
import platform
from pathlib import Path
import shutil
import sys
import time
if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 .venv-pytorch required before Torch import')
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_european_query_excess_refit as previous
from src.world_model import m3w_subset_excess as api
base=previous.base
PUBLIC=previous.PUBLIC.parent/'european_subset_excess_v1'
PRIVATE=previous.PRIVATE.parent/'european_subset_excess_v1'
CONFIG='configs/m3w_european_subset_excess_v1.json'
FILES=[CONFIG,'src/world_model/m3w_subset_excess.py','scripts/run_m3w_european_subset_excess.py',
       'tests/test_m3w_subset_excess.py',str(PUBLIC.relative_to(ROOT)/'protocol.md')]
CAUSAL_KEYS=previous.CAUSAL_KEYS
CACHED={**{f'cached_{a}_{p}':f'{a}_{p}' for a in ('pointwise','query') for p in ('joint','rank')},
        'parent_joint':'parent_joint'}
POLICIES=('floor',*CACHED,*[f'{a}_{p}' for a in api.ARMS for p in ('independent','joint','rank')])
CONTRASTS=(('subset_aggregate_rank','subset_pointwise_rank'),
           ('subset_aggregate_rank','cached_pointwise_rank'),
           ('subset_aggregate_joint','subset_pointwise_joint'),
           ('subset_aggregate_joint','cached_pointwise_joint'))


def beat(state,**kw):
    row=dict(state=state,pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw)
    base.inter.json_write(PRIVATE/'heartbeat.json',row)
    with (PRIVATE/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)


def load(register=False):
    cfg=json.loads((ROOT/CONFIG).read_text());sp=previous.PUBLIC/'verification.json'
    assert base.digest(sp)==cfg['parent_seal_sha256']
    seal=json.loads(sp.read_text())
    for p,h in seal['source_bindings'].items():assert base.digest(ROOT/p)==h,p
    for p,h in seal['artifacts'].items():assert base.digest(previous.PUBLIC/p)==h,p
    pc,data,jobs,oid,pid,pbound,bound,di,pi=previous.load()
    assert cfg['head_training']==pc['head_training'] and cfg['new_heads']==216 and cfg['groups']==108
    assert cfg['arms']==list(api.ARMS) and cfg['anchor_weight']==cfg['subset_weight']==.5 and cfg['risk_budget']==.02
    assert not any(cfg[k] for k in ('new_forecasters','threshold_search','independent_roles_read','deployment_changed','stage5c_executed','smc_enabled'))
    identity=dict(parent_seal=base.artifact(sp),bindings={p:base.digest(ROOT/p) for p in FILES},
        source_rows=len(data['sites']),localities=sorted(set(data['sites'])),causal_context_keys=list(CAUSAL_KEYS),
        new_heads=216,groups=108,independent_roles_read=False,formal_primary_replaced=False)
    if register:base.immutable_json(PUBLIC/'registration.json',identity)
    else:
        assert json.loads((PUBLIC/'registration.json').read_text())==identity
        base.inter.committed(PUBLIC/'registration.json')
    return cfg,data,jobs,oid,pid,pbound,bound,di,pi,identity


def masks(c,data):
    ids=c['ids']
    return api.subset_masks(data['sites'][ids],data['recordings'][ids],data['frames'][ids],
        ids,c['moving'],c['neural_mask'],c['env'])


def prepare(c,data,pair,pid,pbound,bound,di,pi,identity):
    name,fit,held,y,pr,u,old,parent_id=previous.prepare(c,data,pair,pid,pbound,bound,di,pi)
    path=previous.PRIVATE/'heads'/name/'complete.json';rec=json.loads(path.read_text())
    assert rec['identity']==parent_id
    for r in rec['artifacts'].values():assert base.artifact(ROOT/r['path'])==r
    bank=masks(c,{k:data[k] for k in CAUSAL_KEYS})
    ident=dict(experiment=identity,parent_fit=base.artifact(path),roles_and_targets=parent_id['roles_and_targets'],
        fitting_subset_hash=base.inter.array_hash(bank[fit]),fitting_descriptor_hash=base.inter.array_hash(u[fit]))
    return name,fit,held,y,pr,u,old,bank,ident,rec


def preflight(cfg,data,jobs,oid,pid,pbound,bound,di,pi,identity):
    rows=json.loads((previous.PRIVATE/'details.json').read_text())['rows']
    reference=[r for r in rows if r['policy']=='parent_independent']
    assert len(reference)==216
    empty=[dict(group=r['group'],site=r['site']) for r in reference if r['metric']['intervention_rate']==0]
    floor=[r['metric']['floor_error_sum'] for r in reference]
    assert len(empty)==10 and min(floor)>0
    available=shutil.disk_usage(PRIVATE).free
    assert available>10*2**30+400_000_000,'Retain10GiB plus400MB checkpoint/action allowance'
    base.immutable_json(PUBLIC/'preflight.json',dict(identity=identity,
        fixed_parent_zero_action_views=empty,original_selected_risk_primary_still_undefined=True,
        all_reference_diagnostic_denominator_positive=True,minimum_full_reference_error=min(floor),
        formal_primary_replaced=False,selected_risk_tolerance=.02,
        diagnostic_is_not_2percent_certificate=True,free_disk_bytes=available,
        stage5c_executed=False,smc_enabled=False))


def training(cfg,data,jobs,oid,pid,pbound,bound,di,pi,identity,*,resume=False,pilot=False,replay=False):
    assert json.loads((PUBLIC/'preflight.json').read_text())['identity']==identity
    refs=[];causal={k:data[k] for k in CAUSAL_KEYS}
    for c in base.floor_api.contexts(causal,jobs,oid):
        for pair in range(6):
            name,fit,held,y,pr,u,old,bank,ident,parent=prepare(c,data,pair,pid,pbound,bound,di,pi,identity)
            home=PRIVATE/'heads'/name;record=home/'complete.json';states={};infos={};hashes={}
            if record.exists() and resume and not replay:
                doc=json.loads(record.read_text());assert doc['identity']==ident
                for r in doc['artifacts'].values():assert base.artifact(ROOT/r['path'])==r
            else:
                if record.exists() and not replay:raise ValueError('Use --resume for completed fits')
                if shutil.disk_usage(PRIVATE).free<10*2**30:raise OSError('Preserve10GiB and existing fits')
                ids=c['ids'][fit]
                for arm in api.ARMS:
                    directory=(PRIVATE/'fit_replay'/name if replay else home)/arm
                    model,info=api.fit(c['x'][fit],u[fit],y,data['sites'][ids],data['recordings'][ids],data['frames'][ids],
                        c['env'][fit],pr,bank[fit],arm=arm,seed=old['seed'],settings=cfg['head_training'],identity=ident,
                        directory=directory,heartbeat=lambda **kw:beat(group=name,arm=arm,**kw),
                        resume=resume and not replay,stop_at=cfg['pilot_updates'] if pilot else None)
                    state=api.head.read_checkpoint(directory/'checkpoint.pt.gz');states[arm]=state;infos[arm]=info
                    if replay:
                        original=api.head.read_checkpoint(home/arm/'checkpoint.pt.gz')
                        for k in ('model','optimizer','sampler_rng','torch_rng','draws','query_draws','query_keys','trace','subsets'):
                            api.exact(original[k],state[k])
                    p=api.head.predict(model,c['x'][held],u[held],c['env'][held],pr,state['descriptor_preprocess'])
                    hashes[arm]=base.inter.array_hash(p)
                    if not pilot:
                        control=api.head.read_checkpoint(ROOT/parent['artifacts']['pointwise']['path'])
                        for k in ('initial_model','draws','query_draws','query_keys','sampler_rng','torch_rng','step','settings','preprocess'):
                            api.exact(control[k],state[k])
                api.assert_matched(states[api.ARMS[0]],states[api.ARMS[1]])
                if pilot:
                    base.immutable_json(PRIVATE/'pilot.json',dict(fits=infos,matched=True,
                        checkpoint_bytes=sum((home/a/'checkpoint.pt.gz').stat().st_size for a in api.ARMS),
                        free_disk_bytes=shutil.disk_usage(PRIVATE).free));return
                if replay:
                    assert json.loads(record.read_text())['prediction_hashes']==hashes
                    base.immutable_json(PUBLIC/'fit_replay.json',dict(group=name,heads=2,updates=4000,exact=True));return
                doc=dict(identity=ident,fits=infos,prediction_hashes=hashes,
                    matched_queries_rows_and_initialization=True,subsets_built_without_future_fields=True,
                    held_outcomes_used=False,artifacts={a:base.artifact(home/a/'checkpoint.pt.gz') for a in api.ARMS})
                base.immutable_json(record,doc)
            refs.append(base.artifact(record));beat('paired_fit_complete',group=name,complete=len(refs))
    assert len(refs)==108
    base.immutable_json(PUBLIC/'training_freeze.json',dict(identity=identity,groups=refs,heads=216,updates=432000,
        held_outcomes_used=False,independent_roles_read=False))


def predictions(c,data,record,identity):
    assert base.artifact(ROOT/record['path'])==record
    rec=json.loads((ROOT/record['path']).read_text());assert rec['identity']['experiment']==identity
    held=np.flatnonzero(np.isin(data['sites'][c['ids']],rec['identity']['roles_and_targets']['held_sites']))
    scores={};states={}
    for arm,ref in rec['artifacts'].items():
        assert base.artifact(ROOT/ref['path'])==ref
        s=api.head.read_checkpoint(ROOT/ref['path']);assert s['step']==2000 and s['identity']==rec['identity']
        model=previous.descriptor.model_from(s);pr=s['preprocess']
        u=api.head.descriptors(data['geometry'][c['ids'][held]],c['floor'][held],c['prediction'][held],c['x'][held],pr)
        p=api.head.predict(model,c['x'][held],u,c['env'][held],pr,s['descriptor_preprocess'])
        assert base.inter.array_hash(p)==rec['prediction_hashes'][arm]
        scores[arm]=p;states[arm]=s
    api.assert_matched(states[api.ARMS[0]],states[api.ARMS[1]])
    return held,scores,states[api.ARMS[0]]['preprocess']


def actions(c,data,held,scores,pr,old_record):
    assert base.artifact(ROOT/old_record['path'])==old_record
    old=json.loads((ROOT/old_record['path']).read_text())
    assert base.artifact(ROOT/old['arrays']['path'])==old['arrays']
    ancestor=old['parent_action'];assert base.artifact(ROOT/ancestor['path'])==ancestor
    dd=json.loads((ROOT/ancestor['path']).read_text());ref=dd['artifacts']['decisions']
    assert base.artifact(ROOT/ref['path'])==ref
    with np.load(ROOT/ref['path'],allow_pickle=False) as z:p={k:z[k].copy() for k in z.files}
    out,stats=previous.actions(c,data,held,scores,pr,p)
    with np.load(ROOT/old['arrays']['path'],allow_pickle=False) as z:
        np.testing.assert_array_equal(out['ids'],z['ids'])
        for new,key in CACHED.items():out[new]=z[key].copy()
    return out,stats


def decisions(cfg,data,jobs,oid,pid,pbound,bound,di,pi,identity,*,resume=False,replay=False):
    base.inter.committed(PUBLIC/'training_freeze.json')
    frozen=json.loads((PUBLIC/'training_freeze.json').read_text());assert frozen['identity']==identity
    fits={Path(r['path']).parent.name:r for r in frozen['groups']}
    pf=json.loads((previous.PUBLIC/'decision_freeze.json').read_text())
    parents={Path(r['path']).stem:r for r in pf['groups']};outputs=[]
    causal={k:data[k] for k in CAUSAL_KEYS}
    for c in base.floor_api.contexts(causal,jobs,oid):
        for pair in range(6):
            name=c['name']+f'_pair{pair}';dest=PRIVATE/'decisions'/(name+'.json');arr=dest.with_suffix('.npz')
            if dest.exists() and resume and not replay:
                d=json.loads(dest.read_text());assert d['identity']==identity and d['fit']==fits[name]
                assert base.artifact(ROOT/d['arrays']['path'])==d['arrays']
            else:
                if dest.exists() and not replay:raise ValueError('Existing actions require --resume')
                if shutil.disk_usage(PRIVATE).free<10*2**30:raise OSError('Preserve10GiB and completed actions')
                held,scores,pr=predictions(c,causal,fits[name],identity)
                out,solver=actions(dict(c,pair=pair),causal,held,scores,pr,parents[name])
                previous.descriptor.write_arrays(arr,out,replay)
                d=dict(identity=identity,fit=fits[name],parent_action=parents[name],arrays=base.artifact(arr),solver=solver,
                    future_fields_removed=True,held_outcomes_used=False)
                base.immutable_json(dest,d)
            outputs.append(base.artifact(dest));beat('actions_replayed' if replay else 'actions_frozen',group=name,complete=len(outputs))
    base.immutable_json(PUBLIC/'decision_freeze.json',dict(identity=identity,groups=outputs,held_outcomes_used=False))
    if replay:base.immutable_json(PUBLIC/'prediction_replay.json',dict(groups=108,heads=216,exact=True,future_fields_removed=True))


def harm_accounting(floor,neural,known,take):
    use=known & take;den=float(floor[known].sum())
    harm=float(np.maximum(neural[use]-floor[use],0).sum())
    benefit=float(np.maximum(floor[use]-neural[use],0).sum())
    return dict(positive_harm_sum=harm,benefit_sum=benefit,selected_known_count=int(use.sum()),
        selected_reference_error=float(floor[use].sum()),positive_harm_over_all_floor=harm/den if den>0 else None)


def evaluate(cfg,data,jobs,oid,pid,pbound,bound,di,pi,identity,*,replay=False):
    base.inter.committed(PUBLIC/'decision_freeze.json')
    f=json.loads((PUBLIC/'decision_freeze.json').read_text());assert f['identity']==identity
    refs={Path(r['path']).stem:r for r in f['groups']};rows=[];quality=[];solver=[]
    for c in base.floor_api.contexts({k:data[k] for k in CAUSAL_KEYS},jobs,oid):
        cv,cf,(floor,ff),(neural,nf)=base.floor_api.costs(c,data,np.arange(len(c['ids'])))
        bank=masks(c,{k:data[k] for k in CAUSAL_KEYS})
        for pair in range(6):
            name=c['name']+f'_pair{pair}';ref=refs[name];assert base.artifact(ROOT/ref['path'])==ref
            d=json.loads((ROOT/ref['path']).read_text());solver.append(d['solver'])
            assert base.artifact(ROOT/d['arrays']['path'])==d['arrays']
            with np.load(ROOT/d['arrays']['path'],allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
            held,p,pr=predictions(c,data,d['fit'],identity);np.testing.assert_array_equal(a['ids'],c['ids'][held])
            for site in c['pairs'][pair][1]:
                at=data['sites'][a['ids']]==site;ix=held[at]
                common=dict(group=name,site=site,seed=c['job']['old_identity']['seed'])
                for policy in POLICIES:
                    take=np.zeros(len(ix),bool) if policy=='floor' else a[policy][at]
                    m=base.floor_api.metric(cv[ix],floor[ix],neural[ix],cf[ix],ff[ix],nf[ix],
                        np.where(take,neural[ix],floor[ix]),np.where(take,nf[ix],ff[ix]),take,data['valid'][c['ids'][ix]],
                        c['job']['design']['easy_cut'],c['job']['design']['hard_cut'])
                    m.update(harm_accounting(floor[ix],neural[ix],np.isfinite(cv[ix]),take))
                    rows.append(dict(**common,policy=policy,metric=m))
                y=base.floor_api.api.targets(cv[ix],floor[ix],neural[ix],c['job']['design']['easy_cut'])[:,[2,1,3,4]]/pr['cost_scale']
                for arm,pred in p.items():
                    quality.append(dict(**common,policy=arm,metric=api.quality(pred[at]/pr['cost_scale'],y,
                        data['recordings'][c['ids'][ix]],data['frames'][c['ids'][ix]],bank[ix])))
        beat('held_scored',group=c['name'])
    def reduce(rr):
        return {p:{k:base.inter.paired_localities([r for r in rr if r['policy']==p],identity['localities'],k,
            cfg['bootstrap_resamples'],cfg['bootstrap_seed']) for k in rr[0]['metric']} for p in sorted({r['policy'] for r in rr})}
    indexed={(r['group'],r['site'],r['policy']):r['metric'] for r in rows};contrasts=[]
    for new,old in CONTRASTS:
        for r in (v for v in rows if v['policy']==new):
            x=r['metric'];y=indexed[(r['group'],r['site'],old)]
            harm=None if x['selected_positive_harm_ratio'] is None or y['selected_positive_harm_ratio'] is None else 100*(y['selected_positive_harm_ratio']-x['selected_positive_harm_ratio'])
            fixed=None if x['positive_harm_over_all_floor'] is None or y['positive_harm_over_all_floor'] is None else 100*(y['positive_harm_over_all_floor']-x['positive_harm_over_all_floor'])
            contrasts.append(dict(group=r['group'],site=r['site'],seed=r['seed'],policy=new+'_vs_'+old,metric=dict(
                ADE_gain_percent=100*(1-x['error_sum']/y['error_sum']) if y['error_sum']>0 else None,
                selected_harm_reduction_pp=harm,all_reference_harm_reduction_pp=fixed,
                intervention_difference_pp=100*(x['intervention_rate']-y['intervention_rate']))))
    summary=reduce(rows);paired=reduce(contrasts);chosen=[r['metric'] for r in rows if r['policy']=='subset_aggregate_joint']
    comparison=paired['subset_aggregate_rank_vs_subset_pointwise_rank']
    gates=dict(equal_count_ADE_advantage=comparison['ADE_gain_percent']['ci95'][0]>0,
        equal_count_all_reference_harm_reduction=comparison['all_reference_harm_reduction_pp']['ci95'] is not None and comparison['all_reference_harm_reduction_pp']['ci95'][0]>0,
        ranks_count_matched=all(r['metric']['intervention_difference_pp']==0 for r in contrasts if '_rank_vs_' in r['policy']),
        joint_ADE_advantage=paired['subset_aggregate_joint_vs_subset_pointwise_joint']['ADE_gain_percent']['ci95'][0]>0,
        every_view_defined_selected_risk_within_2percent=all(v['selected_positive_harm_ratio'] is not None and v['selected_positive_harm_ratio']<=.02 for v in chosen),
        every_view_easy_preserved=all(v['easy_gain_CV'] is not None and v['easy_gain_CV']>=-2 for v in chosen),
        no_zero_CV_harm=all(v['zero_CV_harmed']==0 for v in chosen))
    gates['exploratory_joint_screen_pass']=all(gates.values())
    gates.update(formal_primary_replaced=False,independent_confirmation=False,calibration_certificate=False,
        deployment_changed=False,stage5c_executed=False,smc_enabled=False)
    out=dict(identity=identity,result_source='fresh_anchored_subset_training_and_readout',summary=summary,paired=paired,
        quality=reduce(quality),by_seed={str(s):reduce([r for r in rows if r['seed']==s]) for s in (17,29,43)},
        solver={a:{k:sum(v[a][k] for v in solver) for k in solver[0][a]} for a in api.ARMS},gates=gates,
        worst_views={p:dict(risk_violating_views=sum(r['metric']['selected_positive_harm_ratio'] is not None and r['metric']['selected_positive_harm_ratio']>.02 for r in rows if r['policy']==p),
            undefined_selected_risk_views=sum(r['metric']['selected_positive_harm_ratio'] is None for r in rows if r['policy']==p),
            abstaining_views=sum(r['metric']['intervention_rate']==0 for r in rows if r['policy']==p),
            worst_easy_gain_CV=min(r['metric']['easy_gain_CV'] for r in rows if r['policy']==p and r['metric']['easy_gain_CV'] is not None)) for p in summary})
    base.immutable_json(PRIVATE/'details.json',dict(rows=rows,contrasts=contrasts,quality=quality))
    base.immutable_json(PUBLIC/'summary.json',out);base.immutable_json(PUBLIC/'gates.json',gates)
    if replay:base.immutable_json(PUBLIC/'evaluation_replay.json',dict(exact=True,summary=base.artifact(PUBLIC/'summary.json'),details=base.artifact(PRIVATE/'details.json')))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',required=True,choices=['register','preflight','pilot','train','replay_fit','decide','replay','evaluate','replay_evaluate'])
    p.add_argument('--resume',action='store_true');a=p.parse_args()
    PUBLIC.mkdir(parents=True,exist_ok=True);PRIVATE.mkdir(parents=True,exist_ok=True)
    base.torch.set_num_threads(4);base.torch.set_num_interop_threads(1)
    with (PRIVATE/'run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);beat('started',phase=a.phase)
        values=load(a.phase=='register')
        if a.phase=='preflight':preflight(*values)
        elif a.phase in ('pilot','train','replay_fit'):training(*values,resume=a.resume,pilot=a.phase=='pilot',replay=a.phase=='replay_fit')
        elif a.phase in ('decide','replay'):decisions(*values,resume=a.resume,replay=a.phase=='replay')
        elif a.phase in ('evaluate','replay_evaluate'):evaluate(*values,replay=a.phase=='replay_evaluate')
        beat('complete',phase=a.phase)


if __name__=='__main__':main()
