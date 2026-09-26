"""Frozen recording-level cost and fitting-gradient accounting, resumable by group."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
if platform.system()=='Darwin' and platform.machine()!='arm64': raise RuntimeError('Use native arm64 Python')
from scripts import run_m3w_european_severity_auxiliary as parent
from src.evaluation import m3w_severity_transport as method
from src.evaluation.m3w_task_gradients import fitting_batch
import numpy as np
import torch
PUBLIC=ROOT/'outputs/publication_readiness_2026_09/european_severity_transport_v1'
PRIVATE=ROOT/'data/stage_cvpr2027_experiments/european_severity_transport_v1'
CONFIG='configs/m3w_european_severity_transport_v1.json'
artifact,digest,immutable_json=parent.artifact,parent.digest,parent.immutable_json
FILES=[CONFIG,'src/evaluation/m3w_severity_transport.py','tests/test_m3w_severity_transport.py',
       'scripts/run_m3w_european_severity_transport.py',str(PUBLIC.relative_to(ROOT)/'protocol.md')]


def registration(create=False):
    cfg=json.loads((ROOT/CONFIG).read_text()); _,pid=parent.registration()
    v=json.loads((parent.PUBLIC/'verification.json').read_text()); assert v['all_passed']
    for p,h in v['artifacts'].items(): assert digest(parent.PUBLIC/p)==h
    for p,h in v['source_bindings'].items(): assert digest(ROOT/p)==h
    assert cfg['frozen_heads']==144 and cfg['new_persisted_updates']==0
    assert (cfg['radial_training_quantile'],cfg['recording_concentration_threshold'],cfg['radial_excess_share_gap'])==(.95,.5,.2)
    assert not any(cfg[k] for k in ('threshold_refit','selection_access','reserved_calibration_access',
                                  'confirmation_access','deployment_changed','stage5c_executed','smc_enabled'))
    identity=dict(parent=pid,parent_verification=artifact(parent.PUBLIC/'verification.json'),bindings={p:digest(ROOT/p) for p in FILES})
    path=PUBLIC/'registration_lock.json'
    if create: immutable_json(path,identity)
    else:
        assert json.loads(path.read_text())==identity; parent.base.previous.require_committed(path)
    return cfg,identity


def beat(**kw):
    r=dict(pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw)
    parent.base.base.cross.json_write(PRIVATE/'heartbeat.json',r)
    with (PRIVATE/'events.jsonl').open('a') as f: f.write(json.dumps(r)+'\n')
    print(json.dumps(r),flush=True)


def compute(cfg,identity,verify=False,pilot=False):
    parent.checked_training(identity['parent']); parent.parent.checked_training(identity['parent']['parent']['parent'])
    refs=[]; started=time.monotonic()
    for g,data,pairs in parent.base.contexts(identity['parent']['source']):
        name=g['group']; bi=pairs['B']['ids']; sites=data['sites'][bi]; cv=data['baseline_ade'][bi,1]
        for pair in cfg['pairs']:
            path=PUBLIC/'groups'/(name+'_'+pair+'.json'); receipt=PRIVATE/'receipts'/path.name
            if path.exists() and not verify:
                assert artifact(path)==json.loads(receipt.read_text()); refs.append(artifact(path)); continue
            if shutil.disk_usage(PRIVATE).free<10*2**30: raise OSError('Preserve10GiB; completed groups resume safely')
            bx,be,by,*_=parent.base.pair_inputs(g,data,pairs,pair)
            old=json.loads((parent.PUBLIC/'groups'/path.name).read_text()); records=[]
            for held in sorted(set(sites)):
                x,env,y,e,ss,ids,pr=parent.diagnosis.fitting_inputs(g,data,pairs,pair,held)
                tr,te=sites!=held,sites==held; tag=name+'_'+pair+'_'+held
                directory=parent.PRIVATE/'heads'/tag; cp=artifact(directory/'checkpoint.pt')
                r=json.loads((directory/'complete.json').read_text()); model,state=parent.method.restore(directory)
                for k,a in (('train_x_sha256',x),('train_y_sha256',y),('train_easy_sha256',e),('train_ids_sha256',ids)):
                    assert parent.array_hash(a)==r['input'][k]
                batch=fitting_batch(x,env,y,e,ss,held,state)
                grad=method.gradient_concentration(model,batch,data['recordings'][ids[state['fixed_ids']]],state['mean_harm'])
                np.testing.assert_allclose(grad['weighted_BCE']['value'],state['trace'][-1]['severity_BCE'],rtol=2e-6,atol=1e-8)
                outside,support=method.radial_support(x,bx[te],pr,ss,held)
                target=parent.tail.diagnostic.event_targets(by[te],cv[te],pr['positive_easy_cut'])
                f=next(v for v in old['folds'] if v['held']==held)
                assert f['target_sha256']==parent.array_hash(target) and f['held_ids_sha256']==parent.array_hash(bi[te])
                easy=parent.parent.parent.parent.method.labels(cv[te],pr['positive_easy_cut'])
                with np.load(directory/'scores.npz',allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['ids'],bi[te]); pred=z['scores'].copy()
                comparisons={}
                for comp,arm in (('original','cost_only'),('ordinary_aux','membership_aux')):
                    with np.load(parent.parent.PRIVATE/'heads'/tag/arm/'scores.npz',allow_pickle=False) as z:
                        np.testing.assert_array_equal(z['ids'],bi[te]); ref=z['scores'].copy()
                    np.testing.assert_array_equal(pred[:,[0,2]],ref[:,[0,2]])
                    comparisons[comp]={}
                    for subset,mask in (('all',np.ones(te.sum(),bool)),('positive_disagreement',be[te]>0)):
                        d=method.error_accounting(pred[mask,3],ref[mask,3],target[mask,3],
                            data['recordings'][bi[te]][mask],data['agents'][bi[te]][mask],easy[mask],outside[mask])
                        previous_subset='all' if subset=='all' else 'envelope_positive'
                        if d['status']=='computed':
                            np.testing.assert_allclose(d['partitions']['all']['new_MSE'],f['metrics']['severity_aux'][previous_subset]['harm_MSE'],rtol=1e-10)
                            np.testing.assert_allclose(d['partitions']['all']['old_MSE'],f['metrics'][arm][previous_subset]['harm_MSE'],rtol=1e-10)
                        comparisons[comp][subset]=d
                assert artifact(directory/'checkpoint.pt')==cp
                records.append(dict(held=held,checkpoint=cp,fitting_ids_sha256=parent.array_hash(ids),
                    held_ids_sha256=parent.array_hash(bi[te]),target_sha256=parent.array_hash(target),
                    gradients=grad,radial_support=support,error_accounting=comparisons))
                beat(state='replayed_view' if verify else 'computed_view',group=name,pair=pair,held=held)
            row=dict(group=name,pair=pair,producer=g['producer'],controller=g['controller'],
                seed=int(name.split('_seed')[1].split('_')[0]),records=records,new_updates=0,
                result_source='fresh_run_frozen_accounting_and_gradients_cached_verified_source_models_and_labels')
            if verify: assert json.loads(path.read_text())==row
            else: immutable_json(path,row); immutable_json(receipt,artifact(path))
            refs.append(artifact(path))
            if pilot:
                immutable_json(PRIVATE/'pilot.json',dict(group=artifact(path),elapsed_seconds=time.monotonic()-started,
                    frozen_views=4,new_updates=0,projection_groups=36,includes_startup_and_hashing=True)); return
    assert len(refs)==36
    parent.checked_training(identity['parent'])
    immutable_json(PUBLIC/('replay_receipt.json' if verify else 'completion_checks.json'),dict(identity=identity,
        groups=refs,all_passed=True,frozen_heads=144,new_updates=0,parent_checkpoints_unchanged=True))
    beat(state='replay_complete' if verify else 'complete',elapsed_seconds=time.monotonic()-started)


def summarize(rows,cfg):
    populations={}
    def distribution(v):
        a=[x for x in v if x is not None]
        return dict(defined=len(a),median=float(np.median(a)) if a else None,
                    minimum=float(min(a)) if a else None,maximum=float(max(a)) if a else None)
    for pair in cfg['pairs']:
        rs=[f for r in rows if r['pair']==pair for f in r['records']]; out=dict(dependent_views=len(rs),comparisons={})
        out['gradient_recording_top1']={k:distribution([r['gradients'][k]['recording_gradient_norm_mass']['top1_share'] for r in rs])
                                      for k in ('ordinary_BCE','weighted_BCE','easy_harm_cost')}
        out['weighted_gradient_top1_at_least_half']=sum(r['gradients']['weighted_BCE']['recording_gradient_norm_mass']['top1_share'] is not None and
            r['gradients']['weighted_BCE']['recording_gradient_norm_mass']['top1_share']>=.5 for r in rs)
        for comp in cfg['comparators']:
            stats=[r['error_accounting'][comp]['positive_disagreement'] for r in rs]
            valid=[v for v in stats if v['status']=='computed']; bad=[v for v in valid if v['partitions']['all']['signed_excess_MSE']>0]
            out['comparisons'][comp]=dict(estimable_views=len(valid),worsening_views=len(bad),
                positive_excess_recording_top1=distribution([v['groups']['recording']['positive_row_excess']['top1_share'] for v in valid]),
                positive_excess_track_top1=distribution([v['groups']['track']['positive_row_excess']['top1_share'] for v in valid]),
                worsening_recording_top1_at_least_half=sum(v['groups']['recording']['positive_row_excess']['top1_share']>=.5 for v in bad),
                worsening_radial_excess_enriched=sum(v['partitions']['radial_outside']['positive_excess_share'] is not None and
                    v['partitions']['radial_outside']['positive_excess_share']-v['partitions']['radial_outside']['row_share']>.2 for v in bad),
                radial_outside_row_share=distribution([v['partitions']['radial_outside']['row_share'] for v in valid]),
                radial_outside_positive_excess_share=distribution([v['partitions']['radial_outside']['positive_excess_share'] for v in valid]))
        populations[pair]=out
    full=populations['full']; original=full['comparisons']['original']; n=original['worsening_views']
    ce=n>0 and original['worsening_recording_top1_at_least_half']>n/2
    cg=full['weighted_gradient_top1_at_least_half']>full['dependent_views']/2
    radial=n>0 and original['worsening_radial_excess_enriched']>n/2
    return dict(populations=populations,decision=dict(concentrated_error_flag=ce,concentrated_gradient_flag=cg,
        recording_influence_investigation_motivated=ce and cg,radial_association_flag=radial,
        causal_root_cause_proven=False,method_improvement_proven=False,new_training_updates=0,
        deployment_changed=False,independent_confirmation=False,stage5c_executed=False,smc_enabled=False))


def report(cfg,identity):
    d=json.loads((PUBLIC/'completion_checks.json').read_text()); assert d['identity']==identity
    rows=[]
    for r in d['groups']:
        assert artifact(ROOT/r['path'])==r; rows.append(json.loads((ROOT/r['path']).read_text()))
    a=summarize(rows,cfg); immutable_json(PUBLIC/'aggregate_metrics.json',a)
    lines=['# Frozen Severity Transport Results','','## Material Passport',
        'Fresh accounting and fitting gradients; cached_verified frozen models and source-development rows.',
        '144 frozen heads; zero training updates; independent roles unopened. No causal attribution or deployment claim.','',
        '| Pair / comparator | Worsening views | Worsening with >=50% excess in one recording | Worsening radial enrichment >20pp |',
        '|---|---:|---:|---:|']
    for pair,v in a['populations'].items():
        for comp,c in v['comparisons'].items(): lines.append(f"| {pair} / {comp} | {c['worsening_views']} | {c['worsening_recording_top1_at_least_half']} | {c['worsening_radial_excess_enriched']} |")
    lines+=['','Recording gradient concentration uses a saved final fitting batch, not the full training path.',
        'Positive and negative row-excess are retained separately; concentration ESS is not independent sample size.',
        'Radial scores use only fitting normalization/cut; they cannot prove conditional support or explain failure alone.','',
        '```json',json.dumps(a['decision'],indent=2),'```','',
        'Detector-derived pixels and annotation steps only. No metric/seconds, human gold, physical safety, true3D or foundation claim. Stage5C/SMC off.']
    (PUBLIC/'results.md').write_text('\n'.join(lines)+'\n'); print(json.dumps(a,indent=2))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--phase',required=True,choices=['register','pilot','run','verify','report']); args=ap.parse_args()
    PUBLIC.mkdir(parents=True,exist_ok=True); PRIVATE.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    with (PRIVATE/'lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); cfg,identity=registration(args.phase=='register')
        if args.phase in ('pilot','run','verify'): compute(cfg,identity,args.phase=='verify',args.phase=='pilot')
        elif args.phase=='report': report(cfg,identity)


if __name__=='__main__': main()
