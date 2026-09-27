"""Paired pointwise/query-risk training on the existing four/four/two/two roles."""
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
from scripts import run_m3w_european_query_utility as previous
from src.world_model import m3w_query_excess as api
descriptor=previous.parent
base=descriptor.base
PUBLIC=previous.PUBLIC.parent/'european_query_excess_refit_v1'
PRIVATE=previous.PRIVATE.parent/'european_query_excess_refit_v1'
CONFIG='configs/m3w_european_query_excess_refit_v1.json'
FILES=[CONFIG,'src/world_model/m3w_query_excess.py','scripts/run_m3w_european_query_excess_refit.py',
    'tests/test_m3w_query_excess.py',str(PUBLIC.relative_to(ROOT)/'protocol.md')]
POLICIES=('floor','parent_independent','parent_joint','pointwise_independent','pointwise_joint',
          'query_independent','query_joint','pointwise_rank','query_rank','utility_topk_parent')
CAUSAL_KEYS=('sites','history','origin','geometry','recordings','frames')


def beat(state,**kw):
    r=dict(state=state,pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw)
    base.inter.json_write(PRIVATE/'heartbeat.json',r)
    with (PRIVATE/'events.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
    print(json.dumps(r),flush=True)


def load(register=False):
    cfg=json.loads((ROOT/CONFIG).read_text());sp=previous.PUBLIC/'verification.json'
    assert base.digest(sp)==cfg['parent_seal_sha256']
    seal=json.loads(sp.read_text())
    for p,h in seal['source_bindings'].items():assert base.digest(ROOT/p)==h,p
    for p,h in seal['artifacts'].items():assert base.digest(previous.PUBLIC/p)==h,p
    _,data,jobs,oid,pid,pbound,bound,di=descriptor.load()
    assert cfg['groups']==108 and cfg['new_heads']==216 and cfg['arms']==['pointwise','query']
    assert cfg['risk_budget']==.02 and cfg['head_training']['steps']==2000
    assert not any(cfg[k] for k in ('new_forecasters','threshold_search','independent_roles_read','deployment_changed','stage5c_executed','smc_enabled'))
    identity=dict(parent_seal=base.artifact(sp),bindings={p:base.digest(ROOT/p) for p in FILES},
        localities=sorted(set(data['sites'])),source_rows=len(data['sites']),causal_context_keys=list(CAUSAL_KEYS),
        groups=108,new_heads=216,independent_roles_read=False)
    if register:base.immutable_json(PUBLIC/'registration.json',identity)
    else:
        assert json.loads((PUBLIC/'registration.json').read_text())==identity
        base.inter.committed(PUBLIC/'registration.json')
    return cfg,data,jobs,oid,pid,pbound,bound,di,identity


def prepare(c,data,pair,pid,pbound,bound,di,identity):
    name,fit,held,y,pr,old,mse,cs,control,u,ident,rec=descriptor.prepare(c,data,pair,pid,pbound,bound,di)
    path=descriptor.PRIVATE/'heads'/name/'complete.json'
    d=descriptor.done(path,ident)
    s=api.head.read_checkpoint(ROOT/d['artifacts']['checkpoint']['path'])
    api.head.assert_matched(s,cs)
    roles=ident['roles_and_targets']
    base.floor_api.api.assert_roles(roles['producer_sites'],roles['controller_sites'],roles['training_sites'],roles['held_sites'])
    new=dict(experiment=identity,parent_head=base.artifact(path),roles_and_targets=roles,
        fitting_descriptors_sha256=base.inter.array_hash(u[fit]),fitting_frames_sha256=base.inter.array_hash(data['frames'][c['ids'][fit]]))
    return name,fit,held,y,pr,u,s,new


def training(cfg,data,jobs,oid,pid,pbound,bound,di,identity,*,resume=False,pilot=False,replay_fit=False):
    refs=[];causal={k:data[k] for k in CAUSAL_KEYS}
    for c in base.floor_api.contexts(causal,jobs,oid):
        for pair in range(6):
            name,fit,held,y,pr,u,old,ident=prepare(c,data,pair,pid,pbound,bound,di,identity)
            home=PRIVATE/'heads'/name;record=home/'complete.json';states={};infos={};hashes={}
            if record.exists() and resume and not replay_fit:
                doc=json.loads(record.read_text());assert doc['identity']==ident
                for r in doc['artifacts'].values():assert base.artifact(ROOT/r['path'])==r
            else:
                if record.exists() and not replay_fit:raise ValueError('Use --resume to retain completed groups')
                if shutil.disk_usage(PRIVATE).free<10*2**30:raise OSError('Preserve10GiB and completed paired checkpoints')
                ids=c['ids'][fit]
                for arm in cfg['arms']:
                    directory=(PRIVATE/'fit_replay'/name if replay_fit else home)/arm
                    model,info=api.fit(c['x'][fit],u[fit],y,data['sites'][ids],data['recordings'][ids],
                        data['frames'][ids],c['env'][fit],pr,arm=arm,seed=old['seed'],settings=cfg['head_training'],
                        identity=ident,directory=directory,heartbeat=lambda **kw:beat(group=name,arm=arm,**kw),
                        resume=resume and not replay_fit,stop_at=cfg['pilot_updates'] if pilot else None)
                    states[arm]=api.head.read_checkpoint(directory/'checkpoint.pt.gz');infos[arm]=info
                    if replay_fit:
                        original=api.head.read_checkpoint(home/arm/'checkpoint.pt.gz')
                        for k in ('model','optimizer','sampler_rng','torch_rng','draws','query_draws','trace','query_keys'):
                            api.exact(original[k],states[arm][k])
                    p=api.head.predict(model,c['x'][held],u[held],c['env'][held],pr,states[arm]['descriptor_preprocess'])
                    hashes[arm]=base.inter.array_hash(p)
                api.assert_matched(states['pointwise'],states['query'])
                if pilot:
                    base.immutable_json(PRIVATE/'pilot.json',dict(fits=infos,matched=True,
                        checkpoint_bytes=sum((home/a/'checkpoint.pt.gz').stat().st_size for a in cfg['arms']),
                        disk_free_bytes=shutil.disk_usage(PRIVATE).free));return
                if replay_fit:
                    doc=json.loads(record.read_text());assert doc['prediction_hashes']==hashes
                    base.immutable_json(PUBLIC/'fit_replay.json',dict(group=name,heads=2,full_updates=4000,exact=True));return
                doc=dict(identity=ident,fits=infos,prediction_hashes=hashes,matched_query_draws=True,
                    future_fields_removed_from_inference=True,held_labels_used=False,
                    artifacts={a:base.artifact(home/a/'checkpoint.pt.gz') for a in cfg['arms']})
                base.immutable_json(record,doc)
            refs.append(base.artifact(record));beat('paired_fit_complete',group=name,complete=len(refs))
    assert len(refs)==108
    base.immutable_json(PUBLIC/'training_freeze.json',dict(identity=identity,groups=refs,heads=216,updates=432000,
        held_labels_used_for_selection=False,independent_roles_read=False))


def predictions(c,data,name,record,identity):
    assert base.artifact(ROOT/record['path'])==record
    doc=json.loads((ROOT/record['path']).read_text());assert doc['identity']['experiment']==identity
    roles=doc['identity']['roles_and_targets'];held=np.flatnonzero(np.isin(data['sites'][c['ids']],roles['held_sites']))
    models={};scores={};states={}
    for arm,r in doc['artifacts'].items():
        assert base.artifact(ROOT/r['path'])==r
        s=api.head.read_checkpoint(ROOT/r['path']);assert s['identity']==doc['identity']
        assert s['step']==2000 and s['arm']==arm
        model=descriptor.model_from(s);pr=s['preprocess']
        u=api.head.descriptors(data['geometry'][c['ids'][held]],c['floor'][held],c['prediction'][held],c['x'][held],pr)
        p=api.head.predict(model,c['x'][held],u,c['env'][held],pr,s['descriptor_preprocess'])
        assert base.inter.array_hash(p)==doc['prediction_hashes'][arm]
        scores[arm]=p;states[arm]=s
    api.assert_matched(states['pointwise'],states['query'])
    return held,scores,states['pointwise']['preprocess']


def actions(c,data,held,scores,pr,parent):
    ids=c['ids'][held];np.testing.assert_array_equal(ids,parent['ids'])
    eligible=parent['eligible'];keys=np.array([str(s)+'|'+str(r) for s,r in zip(data['sites'][ids],data['recordings'][ids])])
    name=c['name']+f"_pair{c['pair']}"
    ridge=json.loads((base.floor_api.PRIVATE/'fits'/name/'complete.json').read_text())
    for r in ridge['artifacts'].values():assert base.artifact(ROOT/r['path'])==r
    with np.load(ROOT/ridge['artifacts']['scores']['path'],allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'],ids)
        utility=(z['scores'][:,5].astype(float)-z['scores'][:,6].astype(float))/pr['cost_scale']
    out=dict(ids=ids,eligible=eligible,parent_independent=parent['independent'],parent_joint=parent['joint_utility'],
        utility_topk_parent=parent['utility_topk']);info={}
    for arm,p in scores.items():
        q=base.api.signed(p.astype(float))/pr['cost_scale']
        anchor=eligible & (q<=0).all(1)
        allocated,stats=previous.api.grouped(utility,q,eligible,anchor,keys,data['frames'][ids],ids,node_limit=256)
        out[arm+'_independent']=anchor;out[arm+'_joint']=allocated['joint_utility'];info[arm]=stats
        out[arm+'_rank']=base.parent.api.match_counts(parent['independent'],eligible,q.max(1),keys,data['frames'][ids],ids)
    return out,info


def decide(cfg,data,jobs,oid,pid,pbound,bound,di,identity,*,resume=False,replay=False):
    base.inter.committed(PUBLIC/'training_freeze.json')
    frozen=json.loads((PUBLIC/'training_freeze.json').read_text());assert frozen['identity']==identity
    refs={Path(r['path']).parent.name:r for r in frozen['groups']}
    old=json.loads((previous.PUBLIC/'decision_freeze.json').read_text())
    oldrefs={Path(r['path']).parent.name:r for r in old['groups']};outputs=[]
    causal={k:data[k] for k in CAUSAL_KEYS}
    for c in base.floor_api.contexts(causal,jobs,oid):
        for pair in range(6):
            name=c['name']+f'_pair{pair}';dest=PRIVATE/'decisions'/(name+'.json');arr=dest.with_suffix('.npz')
            if dest.exists() and resume and not replay:
                doc=json.loads(dest.read_text());assert doc['identity']==identity and doc['fit']==refs[name]
                assert base.artifact(ROOT/doc['arrays']['path'])==doc['arrays']
            else:
                if dest.exists() and not replay:raise ValueError('Existing decisions require --resume')
                if shutil.disk_usage(PRIVATE).free<10*2**30:raise OSError('Preserve10GiB and completed actions')
                held,p,pr=predictions(c,causal,name,refs[name],identity)
                r=oldrefs[name];assert base.artifact(ROOT/r['path'])==r
                od=json.loads((ROOT/r['path']).read_text())
                for a in od['artifacts'].values():assert base.artifact(ROOT/a['path'])==a
                with np.load(ROOT/od['artifacts']['decisions']['path'],allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
                out,solver=actions(dict(c,pair=pair),causal,held,p,pr,a)
                descriptor.write_arrays(arr,out,replay)
                doc=dict(identity=identity,fit=refs[name],parent_action=r,arrays=base.artifact(arr),solver=solver,
                    future_fields_removed=True,held_outcomes_used=False)
                base.immutable_json(dest,doc)
            outputs.append(base.artifact(dest));beat('actions_replayed' if replay else 'actions_frozen',group=name,complete=len(outputs))
    base.immutable_json(PUBLIC/'decision_freeze.json',dict(identity=identity,groups=outputs,held_labels_used=False))
    if replay:base.immutable_json(PUBLIC/'prediction_replay.json',dict(groups=108,heads=216,exact=True,future_fields_removed=True))


def evaluate(cfg,data,jobs,oid,pid,pbound,bound,di,identity,*,replay=False):
    base.inter.committed(PUBLIC/'decision_freeze.json')
    fr=json.loads((PUBLIC/'decision_freeze.json').read_text());assert fr['identity']==identity
    refs={Path(r['path']).stem:r for r in fr['groups']};rows=[];quality=[];solver=[]
    for c in base.floor_api.contexts({k:data[k] for k in CAUSAL_KEYS},jobs,oid):
        cv,cf,(f,ff),(n,nf)=base.floor_api.costs(c,data,np.arange(len(c['ids'])))
        for pair in range(6):
            name=c['name']+f'_pair{pair}';ref=refs[name];assert base.artifact(ROOT/ref['path'])==ref
            doc=json.loads((ROOT/ref['path']).read_text());solver.append(doc['solver'])
            assert base.artifact(ROOT/doc['arrays']['path'])==doc['arrays']
            with np.load(ROOT/doc['arrays']['path'],allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
            held,p,pr=predictions(c,data,name,doc['fit'],identity)
            np.testing.assert_array_equal(a['ids'],c['ids'][held])
            for site in c['pairs'][pair][1]:
                at=data['sites'][a['ids']]==site;ix=held[at];common=dict(group=name,site=site,seed=c['job']['old_identity']['seed'])
                for policy in POLICIES:
                    take=np.zeros(len(ix),bool) if policy=='floor' else a[policy][at]
                    metric=base.floor_api.metric(cv[ix],f[ix],n[ix],cf[ix],ff[ix],nf[ix],np.where(take,n[ix],f[ix]),
                        np.where(take,nf[ix],ff[ix]),take,data['valid'][c['ids'][ix]],c['job']['design']['easy_cut'],c['job']['design']['hard_cut'])
                    rows.append(dict(**common,policy=policy,metric=metric))
                y=base.floor_api.api.targets(cv[ix],f[ix],n[ix],c['job']['design']['easy_cut'])[:,[2,1,3,4]]/pr['cost_scale']
                for arm,pred in p.items():
                    quality.append(dict(**common,policy=arm,metric=api.quality(pred[at]/pr['cost_scale'],y,
                        data['recordings'][c['ids'][ix]],data['frames'][c['ids'][ix]])))
        beat('held_scored',group=c['name'])
    def reduce(rr):
        return {p:{k:base.inter.paired_localities([r for r in rr if r['policy']==p],identity['localities'],k,
            cfg['bootstrap_resamples'],cfg['bootstrap_seed']) for k in rr[0]['metric']} for p in sorted({r['policy'] for r in rr})}
    idx={(r['group'],r['site'],r['policy']):r['metric'] for r in rows};contrasts=[]
    for new,old in [('query_rank','pointwise_rank'),('query_joint','pointwise_joint'),('query_joint','parent_joint')]:
        for row in [r for r in rows if r['policy']==new]:
            x=row['metric'];y=idx[(row['group'],row['site'],old)]
            risk=None if x['selected_positive_harm_ratio'] is None or y['selected_positive_harm_ratio'] is None else 100*(y['selected_positive_harm_ratio']-x['selected_positive_harm_ratio'])
            contrasts.append(dict(group=row['group'],site=row['site'],seed=row['seed'],policy=new+'_vs_'+old,
                metric=dict(ADE_gain_percent=100*(1-x['error_sum']/y['error_sum']) if y['error_sum']>0 else None,
                    harm_reduction_pp=risk,intervention_difference_pp=100*(x['intervention_rate']-y['intervention_rate']))))
    summary=reduce(rows);paired=reduce(contrasts);selected=[r['metric'] for r in rows if r['policy']=='query_joint']
    primary=paired['query_rank_vs_pointwise_rank'];risk=primary['harm_reduction_pp']
    gates=dict(primary_equal_count_harm_reduction=risk['ci95'] is not None and risk['ci95'][0]>0,
        same_count_ADE_advantage=primary['ADE_gain_percent']['ci95'] is not None and primary['ADE_gain_percent']['ci95'][0]>0,
        all_rank_counts_matched=all(r['metric']['intervention_difference_pp']==0 for r in contrasts if r['policy']=='query_rank_vs_pointwise_rank'),
        joint_ADE_advantage=paired['query_joint_vs_pointwise_joint']['ADE_gain_percent']['ci95'][0]>0,
        floor_ADE_advantage=summary['query_joint']['all_gain_floor']['ci95'][0]>0,
        every_view_defined_risk_within_budget=all(m['selected_positive_harm_ratio'] is not None and m['selected_positive_harm_ratio']<=.02 for m in selected),
        every_view_easy_preserved=all(m['easy_gain_CV'] is not None and m['easy_gain_CV']>=-2 for m in selected),
        no_zero_CV_harm=all(m['zero_CV_harmed']==0 for m in selected),
        nonempty_localities=all(v is not None and v>0 for v in summary['query_joint']['intervention_rate']['by_site'].values()))
    gates['exploratory_joint_screen_pass']=all(gates.values())
    gates.update(independent_confirmation=False,calibration_certificate=False,deployment_changed=False,stage5c_executed=False,smc_enabled=False)
    out=dict(identity=identity,result_source='fresh_paired_query_objective_training_and_readout',summary=summary,paired=paired,
        quality=reduce(quality),by_seed={str(s):reduce([r for r in rows if r['seed']==s]) for s in (17,29,43)},gates=gates,
        solver={arm:{k:sum(v[arm][k] for v in solver) for k in solver[0][arm]} for arm in cfg['arms']},
        worst_views={p:dict(worst_easy_gain_CV=min(r['metric']['easy_gain_CV'] for r in rows if r['policy']==p and r['metric']['easy_gain_CV'] is not None),
            risk_violating_views=sum(r['metric']['selected_positive_harm_ratio'] is not None and r['metric']['selected_positive_harm_ratio']>.02 for r in rows if r['policy']==p),
            undefined_risk_views=sum(r['metric']['selected_positive_harm_ratio'] is None for r in rows if r['policy']==p)) for p in summary})
    base.immutable_json(PRIVATE/'details.json',dict(rows=rows,contrasts=contrasts,quality=quality))
    base.immutable_json(PUBLIC/'summary.json',out);base.immutable_json(PUBLIC/'gates.json',gates)
    if replay:base.immutable_json(PUBLIC/'evaluation_replay.json',dict(exact=True,summary=base.artifact(PUBLIC/'summary.json'),details=base.artifact(PRIVATE/'details.json')))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',required=True,choices=['register','pilot','train','replay_fit','decide','replay','evaluate','replay_evaluate'])
    p.add_argument('--resume',action='store_true');args=p.parse_args()
    PUBLIC.mkdir(parents=True,exist_ok=True);PRIVATE.mkdir(parents=True,exist_ok=True)
    base.torch.set_num_threads(4);base.torch.set_num_interop_threads(1)
    with (PRIVATE/'run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);beat('started',phase=args.phase)
        values=load(args.phase=='register')
        if args.phase in ('pilot','train','replay_fit'):training(*values,resume=args.resume,pilot=args.phase=='pilot',replay_fit=args.phase=='replay_fit')
        elif args.phase in ('decide','replay'):decide(*values,resume=args.resume,replay=args.phase=='replay')
        elif args.phase in ('evaluate','replay_evaluate'):evaluate(*values,replay=args.phase=='replay_evaluate')
        beat('complete',phase=args.phase)


if __name__=='__main__':main()
