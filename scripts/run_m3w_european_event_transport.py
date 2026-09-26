"""Registered common-event residual repair; no reserved roles or policy tuning."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import time
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts import run_m3w_european_nested_residual as parent
from src.world_model import m3w_event_transport as method
import numpy as np
import torch
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_event_transport_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_event_transport_v1'
CONFIG = 'configs/m3w_european_event_transport_v1.json'
FILES = [CONFIG,'src/world_model/m3w_event_transport.py','tests/test_m3w_event_transport.py',
         'scripts/run_m3w_european_event_transport.py',str(PUBLIC.relative_to(ROOT)/'protocol.md')]
artifact,digest,immutable_json,array_hash = parent.artifact,parent.digest,parent.immutable_json,parent.array_hash
residual,base,source = parent.residual,parent.base,parent.source


def registration(create=False):
    cfg = json.loads((ROOT/CONFIG).read_text()); _,pid = parent.registration()
    verification = json.loads((parent.PUBLIC/'verification.json').read_text())
    assert verification['all_passed']
    for p,h in verification['artifacts'].items(): assert digest(parent.PUBLIC/p)==h
    for p,h in verification['source_bindings'].items(): assert digest(ROOT/p)==h
    assert cfg['residual_variants']==list(parent.method.VARIANTS) and cfg['probe_arms']==list(residual.ARMS)
    assert (cfg['outer_views'],cfg['closed_form_fits'],cfg['ridge'],cfg['new_neural_updates'])==(144,864,.1,0)
    assert not any(cfg[k] for k in ('threshold_refit','selection_access','reserved_calibration_access',
                                  'confirmation_access','deployment_changed','stage5c_executed','smc_enabled'))
    identity = dict(parent=pid,parent_verification=artifact(parent.PUBLIC/'verification.json'),
                    bindings={p:digest(ROOT/p) for p in FILES})
    path = PUBLIC/'registration_lock.json'
    if create: immutable_json(path,identity)
    else:
        assert json.loads(path.read_text())==identity
        base.previous.require_committed(path)
    return cfg,identity


def beat(state,**kw):
    row = dict(pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),state=state,**kw)
    base.base.cross.json_write(PRIVATE/'heartbeat.json',row)
    with (PRIVATE/'events.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)


def fit(cfg,identity,verify=False):
    parent.checked_training(); parent.parent.check_frozen(identity['parent']['parent'])
    old_freeze=json.loads((parent.PUBLIC/'prediction_freeze.json').read_text())
    for ref in old_freeze['receipts']:
        assert artifact(ROOT/ref['path'])==ref
        for a in json.loads((ROOT/ref['path']).read_text())['artifacts'].values(): assert artifact(ROOT/a['path'])==a
    refs=[]; seconds=[]
    for v in parent.views(identity['parent']):
        if shutil.disk_usage(PRIVATE).free < 10*2**30: raise OSError('Preserve10GiB; per-view resume available')
        out=PRIVATE/'probes'/v['tag']; receipt=out/'complete.json'
        if receipt.exists() and not verify:
            row=json.loads(receipt.read_text()); assert row['registration']==artifact(PUBLIC/'registration_lock.json')
            for a in row['artifacts'].values(): assert artifact(ROOT/a['path'])==a
            refs.append(artifact(receipt)); seconds.append(row['fit_seconds']); continue
        start=time.monotonic(); bank={}
        for inner in parent.method.fitting_sites(v['sites'],v['outer']):
            home=parent.PRIVATE/'heads'/v['tag']/inner; row=json.loads((home/'complete.json').read_text())
            with np.load(home/'scores.npz',allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'],v['ids']); pred=z['scores'].copy()
            bank[inner]=dict(prediction=pred,training_sites=row['input']['training_sites'],cut=row['input']['cut'])
        with np.load(parent.parent.frozen_directory(v['tag'],'original')/'scores.npz',allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],v['held_ids']); frozen=z['scores'].copy()
        old=json.loads((parent.PRIVATE/'probes'/v['tag']/'models.json').read_text())
        common=method.common_event_target(v['raw'],v['cv'],v['sites'],v['outer'],v['pr']['positive_easy_cut'])
        np.testing.assert_array_equal(common,v['outer_y'])
        w=np.where(v['env']>0,v['pr']['weights'],0)
        models,alignment,accounting,scores={},{},{},{}
        for variant in cfg['residual_variants']:
            p,y,cuts,producers=parent.method.assemble(bank,v['raw'],v['cv'],v['sites'],v['outer'],variant)
            assert array_hash(p)==old['alignment'][variant]['prediction_sha256']
            models[variant]=parent.fit_residual_bank(v['context'],p,common,w,v['sites'],v['outer'],ridge=cfg['ridge'],variant=variant)
            alignment[variant]=dict(prediction_sha256=array_hash(p),inner_target_sha256=array_hash(y),
                common_target_sha256=array_hash(common),row_producer_sha256=array_hash(producers))
            accounting[variant]={}
            for arm in cfg['probe_arms']:
                model=models[variant][arm]
                model['response_event']='outer_three_locality_fitting_event'
                model['teacher_event']='unchanged_inner_two_locality_event'
                accounting[variant][arm]=method.transport_accounting(old['models'][variant][arm],model,
                    v['context'],y,common,w,v['held_context'],frozen)
                scores[variant+'__'+arm]=residual.predict(model,v['held_context'],frozen)
        doc=dict(models=models,alignment=alignment,accounting=accounting,
                 ids_sha256=array_hash(v['ids']),held_context_sha256=array_hash(v['held_context']),
                 outer_cut=v['pr']['positive_easy_cut'],outer_held_labels_used=False)
        if verify:
            assert json.loads((out/'models.json').read_text())==doc
            with np.load(out/'scores.npz',allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'],v['held_ids'])
                for k,p in scores.items(): np.testing.assert_array_equal(p,z[k])
            seconds.append(json.loads(receipt.read_text())['fit_seconds'])
        else:
            immutable_json(out/'models.json',doc)
            base.previous.parent.atomic_npz(out/'scores.npz',ids=v['held_ids'],**scores)
            elapsed=time.monotonic()-start; seconds.append(elapsed)
            immutable_json(receipt,dict(registration=artifact(PUBLIC/'registration_lock.json'),fit_seconds=elapsed,
                artifacts=dict(models=artifact(out/'models.json'),scores=artifact(out/'scores.npz'))))
        refs.append(artifact(receipt)); beat('fits_replayed' if verify else 'fits_frozen',views=len(refs),tag=v['tag'])
    assert len(refs)==144
    immutable_json(PUBLIC/('fit_replay.json' if verify else 'prediction_freeze.json'),
        dict(receipts=refs,closed_form_fits=864,new_neural_updates=0,outer_held_labels_used=False,
             summed_fit_seconds=sum(seconds)))


def check_freeze():
    base.previous.require_committed(PUBLIC/'prediction_freeze.json')
    for ref in json.loads((PUBLIC/'prediction_freeze.json').read_text())['receipts']:
        assert artifact(ROOT/ref['path'])==ref
        for a in json.loads((ROOT/ref['path']).read_text())['artifacts'].values(): assert artifact(ROOT/a['path'])==a


def evaluate(cfg,identity,verify=False):
    check_freeze(); groups={}; by_cache={}; refs=[]; direct=0
    for v in parent.views(identity['parent']):
        name=v['g']['group']+'_'+v['pair']; path=PUBLIC/'groups'/(name+'.json')
        if name not in by_cache: by_cache={name:base.pair_inputs(v['g'],v['data'],v['pairs'],v['pair'])[2]}
        cv=v['data']['baseline_ade'][v['held_ids'],1]
        y=source.tail.diagnostic.event_targets(by_cache[name][v['te']],cv,v['pr']['positive_easy_cut'])
        old=json.loads((parent.PUBLIC/'groups'/(name+'.json')).read_text())
        prior=next(f for f in old['folds'] if f['held']==v['outer'])
        assert prior['target_sha256']==array_hash(y) and prior['held_ids_sha256']==array_hash(v['held_ids'])
        metrics=dict(prior['metrics'])
        orig=json.loads((source.PUBLIC/'groups'/(name+'.json')).read_text())
        edges=next(f for f in orig['folds'] if f['held']==v['outer'])['training']['cost_only']['edges']
        env=v['pairs']['B'][v['pair']][1][v['te']]
        with np.load(parent.parent.frozen_directory(v['tag'],'original')/'scores.npz',allow_pickle=False) as z: frozen=z['scores'].copy()
        with np.load(PRIVATE/'probes'/v['tag']/'scores.npz',allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],v['held_ids'])
            for variant in cfg['residual_variants']:
                for arm in cfg['probe_arms']:
                    k=variant+'__'+arm; p=z[k]
                    np.testing.assert_array_equal(p[:,:3],frozen[:,:3])
                    assert np.isfinite(p).all() and (p[:,3]>=0).all() and (p[:,3]<=p[:,1]).all()
                    metric=source.parent.parent.measure(p,y,env,np.repeat(v['outer'],len(y)),edges)
                    for subset,use in (('all',np.ones(len(y),bool)),('envelope_positive',env>0)):
                        use &= np.isfinite(y).all(1)
                        np.testing.assert_allclose(np.mean((p[use,3]-y[use,3])**2),metric[subset]['harm_MSE'],rtol=1e-12,atol=1e-12)
                        direct+=1
                    metrics['common__'+k]=metric
        groups.setdefault(name,[]).append(dict(held=v['outer'],metrics=metrics,
            target_sha256=array_hash(y),held_ids_sha256=array_hash(v['held_ids'])))
        if len(groups[name])==4:
            row=dict(group=v['g']['group'],producer=v['g']['producer'],controller=v['g']['controller'],
                seed=int(v['g']['group'].split('_seed')[1].split('_')[0]),pair=v['pair'],folds=groups.pop(name),
                result_source='fresh_run_common_event_readout_cached_verified_controls_source_development')
            if verify: assert json.loads(path.read_text())==row
            else: immutable_json(path,row)
            refs.append(artifact(path)); beat('readout_replayed' if verify else 'readout',groups=len(refs),group=name)
    assert len(refs)==36 and direct==1728 and not groups
    immutable_json(PUBLIC/('eval_replay.json' if verify else 'completion_checks.json'),
        dict(all_passed=True,groups=refs,direct_MSE_checks=direct))


def report(cfg,identity):
    rows=[]
    for a in json.loads((PUBLIC/'completion_checks.json').read_text())['groups']:
        assert artifact(ROOT/a['path'])==a; rows.append(json.loads((ROOT/a['path']).read_text()))
    controls=dict(old_oof='oof__context_bias',original='original',outer_context='outer_in_sample__context_bias',
        common_global='common__oof__global_bias',common_next='common__in_sample_next__context_bias',
        common_prev='common__in_sample_prev__context_bias')
    comparisons={'common_oof_vs_'+k:('common__oof__context_bias',a) for k,a in controls.items()}
    for v in cfg['residual_variants']:
        for a in cfg['probe_arms']:
            comparisons[v+'_'+a+'_common_vs_inner']=('common__'+v+'__'+a,v+'__'+a)
    contrasts={}
    for key,(a,b) in comparisons.items():
        proxy=[dict(r,folds=[dict(f,metrics={'fractional':f['metrics'][a],'mean':f['metrics'][b]}) for f in r['folds']]) for r in rows]
        contrasts[key]=parent.parent.paired_contrasts(proxy,cfg)
    summary=parent.parent.summarize_contrasts(contrasts)
    primary=[summary['common_oof_vs_'+k]['full'] for k in controls]
    mechanism=primary[0]['envelope_positive__harm_MSE_gain_percent']['positive']==6
    positive=all(v['envelope_positive__harm_MSE_gain_percent']['positive']==6 for v in primary)
    guards=all(v['envelope_positive__'+k]['negative']==v['envelope_positive__'+k]['not_estimable']==0
        for v in primary for k in ('top10_gain_pp','coverage_log_error_reduction'))
    gates=dict(event_alignment_signal=mechanism,primary_six_positive_all_controls=positive,
        tail_coverage_guards=guards,common_event_repair_signal=positive and guards,
        policy_evaluated=False,deployment_changed=False,independent_confirmation=False,
        stage5c_executed=False,smc_enabled=False)
    immutable_json(PUBLIC/'aggregate_metrics.json',dict(contrasts=contrasts,summary=summary,gates=gates))
    accounting=[]
    for a in json.loads((PUBLIC/'prediction_freeze.json').read_text())['receipts']:
        row=json.loads((ROOT/a['path']).read_text()); doc=json.loads((ROOT/row['artifacts']['models']['path']).read_text())
        for variant,arms in doc['accounting'].items():
            for arm,q in arms.items(): accounting.append(dict(tag=Path(a['path']).parent.name,variant=variant,arm=arm,**q))
    assert len(accounting)==864
    immutable_json(PUBLIC/'event_accounting.json',dict(rows=accounting,identity_checks=864,causal_effect_proven=False))
    lines=['# Common-Event Residual Results','','## Material Passport',
        'fresh_run:864 fixed ridge probes; cached_verified:432 inner risk heads and original outer estimators.',
        'No new neural or trajectory training. Source-development only, not independent confirmation.', '',
        '| Comparison / inputs | Positive / negative / overlapping / missing MSE intervals | Point range (%) |','|---|---|---|']
    for k,ps in summary.items():
        for p,v in ps.items():
            q=v['envelope_positive__harm_MSE_gain_percent']
            lines.append(f"| {k} / {p} | {q['positive']} / {q['negative']} / {q['overlap']} / {q['not_estimable']} | {q['point_range']} |")
    lines+=['','Range denotes six point estimates, not one CI. Three seeds averaged per locality,3000 paired locality resamples.',
        'Six assignments overlap; four localities per assignment, no multiplicity adjustment. No negative interval is not a safety proof.',
        'Common event is meta-fitting-only; inner teachers still predict their original events. No teacher retraining or independent calibration.',
        'The864 exact preclip event projections isolate this fixed label change algebraically, not a causal explanation of all errors.',
        '', '```json',json.dumps(gates,indent=2),'```','',
        'Eight observed/twelve predicted annotation steps, detector pixels. No metric/seconds, human-gold, physical-safety, true3D or foundation claim.']
    (PUBLIC/'results.md').write_text('\n'.join(lines)+'\n'); print(json.dumps(gates,indent=2))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--phase',required=True,choices=['register','fit','evaluate','report','verify_fit','verify_eval'])
    args=ap.parse_args(); PUBLIC.mkdir(parents=True,exist_ok=True); PRIVATE.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    with (PRIVATE/'lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); cfg,identity=registration(args.phase=='register')
        if args.phase in ('fit','verify_fit'): fit(cfg,identity,args.phase=='verify_fit')
        elif args.phase in ('evaluate','verify_eval'): evaluate(cfg,identity,args.phase=='verify_eval')
        elif args.phase=='report': report(cfg,identity)


if __name__=='__main__': main()
