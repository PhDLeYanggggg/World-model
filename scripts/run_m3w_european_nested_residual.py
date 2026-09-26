"""Nested fitting-locality residual experiment; reserved roles remain unopened."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Use native arm64 Python before importing Torch')
from scripts import run_m3w_european_context_residual as parent
from src.world_model import m3w_nested_residual as method
from src.world_model import m3w_membership_auxiliary as neural
import numpy as np
import torch
source, residual = parent.source, parent.method
base = source.base
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_nested_residual_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_nested_residual_v1'
CONFIG = 'configs/m3w_european_nested_residual_v1.json'
FILES = [CONFIG, 'src/world_model/m3w_nested_residual.py', 'tests/test_m3w_nested_residual.py',
         'scripts/run_m3w_european_nested_residual.py', str(PUBLIC.relative_to(ROOT)/'protocol.md')]
artifact, digest, immutable_json, array_hash = parent.artifact, parent.digest, parent.immutable_json, parent.array_hash


def registration(create=False):
    cfg = json.loads((ROOT/CONFIG).read_text()); _, pid = parent.registration()
    v = json.loads((parent.PUBLIC/'verification.json').read_text()); assert v['all_passed']
    for p,h in v['artifacts'].items(): assert digest(parent.PUBLIC/p) == h
    for p,h in v['source_bindings'].items(): assert digest(ROOT/p) == h
    assert cfg['head_training'] == json.loads((ROOT/'configs/m3w_european_membership_auxiliary_v1.json').read_text())['head_training']
    assert (cfg['outer_views'], cfg['new_inner_heads'], cfg['updates'], cfg['closed_form_fits']) == (144,432,864000,864)
    assert cfg['residual_variants'] == list(method.VARIANTS) and cfg['probe_arms'] == list(residual.ARMS)
    assert cfg['ridge'] == .1
    assert not any(cfg[k] for k in ('threshold_refit','selection_access','reserved_calibration_access',
        'confirmation_access','new_forecaster_training','deployment_changed','stage5c_executed','smc_enabled'))
    identity = dict(parent=pid, source=pid['parent']['parent']['source'],
        parent_verification=artifact(parent.PUBLIC/'verification.json'), bindings={p:digest(ROOT/p) for p in FILES})
    path = PUBLIC/'registration_lock.json'
    if create: immutable_json(path, identity)
    else:
        assert json.loads(path.read_text()) == identity; base.previous.require_committed(path)
    return cfg, identity


def beat(state, **kw):
    row = dict(pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),state=state,**kw)
    base.base.cross.json_write(PRIVATE/'heartbeat.json',row)
    with (PRIVATE/'events.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)


def views(identity):
    for g,data,pairs in base.contexts(identity['source']):
        bi = pairs['B']['ids']; sites = data['sites'][bi]
        for pair in ('full','motion_only'):
            bx,be,r,p = pairs['B'][pair]
            context = residual.features(data['geometry'][bi],data['width'][bi],r,p)
            for outer in sorted(set(sites)):
                x,env,y,e,ss,ids,pr = source.diagnosis.fitting_inputs(g,data,pairs,pair,outer)
                tr,te = sites != outer,sites == outer; tag = g['group']+'_'+pair+'_'+outer
                np.testing.assert_array_equal(ids,bi[tr])
                old = json.loads((parent.frozen_directory(tag,'original')/'complete.json').read_text())
                for k,a in (('train_x_sha256',x),('train_y_sha256',y),('train_ids_sha256',ids)):
                    assert old['input'][k] == array_hash(a)
                assert outer not in g['producer_roster'] and not set(ss) & set(g['producer_roster'])
                yield dict(g=g,data=data,pairs=pairs,bi=bi,tr=tr,te=te,pair=pair,outer=outer,tag=tag,
                    x=x,env=env,raw=y[:,:2],outer_y=y,cv=data['baseline_ade'][ids,1],sites=ss,
                    ids=ids,pr=pr,context=context[tr],held_context=context[te],held_ids=bi[te])


def support(cfg,identity):
    rows = []
    for v in views(identity):
        for inner in method.fitting_sites(v['sites'],v['outer']):
            take,pr,y,e = method.inner_inputs(v['x'],v['raw'],v['cv'],v['sites'],v['outer'],inner)
            known = pr['known']; positive = int((y[known,3] > 0).sum())
            rows.append(dict(tag=v['tag'],pair=v['pair'],inner=inner,training_sites=pr['training_sites'],
                known=int(known.sum()),easy=int(np.nansum(e)),easy_harm_rows=positive,
                cut=pr['positive_easy_cut'],train_ids_sha256=array_hash(v['ids'][take]),
                target_sha256=array_hash(y),numerically_supported=positive>0 and 0<np.nansum(e)<known.sum()))
        beat('support',inner_views=len(rows),tag=v['tag'])
    assert len(rows) == 432
    doc = dict(registration=artifact(PUBLIC/'registration_lock.json'),rows=rows,
        training_allowed=all(r['numerically_supported'] for r in rows),outer_held_labels_used=False,
        statistical_power_established=False)
    immutable_json(PUBLIC/'support_report.json',doc)
    text = ['# Inner Fitting Support','','Numerical support, not independent sample size or power.',
            'Training allowed: '+str(doc['training_allowed']), '', '| Inputs | Inner fits | Minimum known rows | Minimum easy-harm rows |',
            '|---|---:|---:|---:|']
    for pair in cfg['pairs']:
        group = [r for r in rows if r['pair']==pair]
        text.append(f"| {pair} | {len(group)} | {min(r['known'] for r in group)} | {min(r['easy_harm_rows'] for r in group)} |")
    (PUBLIC/'support_report.md').write_text('\n'.join(text)+'\n')


def input_record(v,inner,take,pr,y):
    return dict(tag=v['tag'],inner=inner,outer=v['outer'],training_sites=pr['training_sites'],
        train_ids_sha256=array_hash(v['ids'][take]),train_x_sha256=array_hash(v['x'][take]),
        train_y_sha256=array_hash(y),score_ids_sha256=array_hash(v['ids']),cut=pr['positive_easy_cut'],
        outer_held_labels_used=False,inner_held_labels_used_for_fit=False)


def train(cfg,identity,pilot=False,resume=False,verify=False):
    support_doc = json.loads((PUBLIC/'support_report.json').read_text())
    assert support_doc['training_allowed'] and support_doc['registration']==artifact(PUBLIC/'registration_lock.json')
    base.previous.require_committed(PUBLIC/'support_report.json'); parent.check_frozen(identity['parent'])
    receipts = []
    for v in views(identity):
        seed = int(v['g']['group'].split('_seed')[1].split('_')[0])
        for inner in method.fitting_sites(v['sites'],v['outer']):
            if shutil.disk_usage(PRIVATE).free < 10*2**30: raise OSError('Preserve10GiB; resume retained checkpoints')
            out = PRIVATE/'heads'/v['tag']/inner; receipt = out/'complete.json'
            take,pr,y,e = method.inner_inputs(v['x'],v['raw'],v['cv'],v['sites'],v['outer'],inner)
            inp = input_record(v,inner,take,pr,y)
            ipath = PRIVATE/'inputs'/v['tag']/(inner+'.json'); immutable_json(ipath,inp)
            hid = dict(registration=artifact(PUBLIC/'registration_lock.json'),input=artifact(ipath),seed=seed)
            if receipt.exists():
                row = json.loads(receipt.read_text()); assert row['identity'] == hid and row['input']==inp
                for a in row['artifacts'].values(): assert artifact(ROOT/a['path']) == a
                if verify:
                    model,state = neural.restore(out)
                    for k in ('mean','std','known','weights'): np.testing.assert_array_equal(state['preprocess'][k],pr[k])
                    for k in ('cost_scale','positive_easy_cut','hard_cut'): assert state['preprocess'][k] == pr[k]
                    assert state['step'] == cfg['head_training']['steps'] and state['identity'] == hid
                    assert state['seed']==seed and state['settings']==cfg['head_training'] and state['arm']=='cost_only'
                    target=np.where(pr['known'][:,None],y/pr['cost_scale'],0).astype(np.float32)
                    rms=np.sqrt(np.sum(pr['weights'][:,None]*target.astype(float)**2,axis=0)).clip(1e-4)
                    np.testing.assert_array_equal(state['loss_scales'],rms)
                    assert state['draws'][~pr['known']].sum() == 0
                    pred,_ = neural.predict(model,v['x'],v['env'],pr)
                    with np.load(out/'scores.npz',allow_pickle=False) as z:
                        np.testing.assert_array_equal(z['ids'],v['ids']); np.testing.assert_array_equal(z['scores'],pred)
            else:
                if verify: raise ValueError('Missing completed inner head')
                model,fit = neural.fit(v['x'][take],y,e,v['sites'][take],v['env'][take],pr,
                    arm='cost_only',seed=seed,settings=cfg['head_training'],identity=hid,directory=out,
                    resume=resume,stop_at=100 if pilot else None,
                    heartbeat=lambda **kw:beat(tag=v['tag'],inner=inner,**kw))
                if pilot:
                    immutable_json(PRIVATE/'pilot.json',dict(fit=fit,checkpoint=artifact(out/'checkpoint.pt'),
                        projected_fit_seconds=fit['seconds']/100*cfg['updates'],excludes_loading_inference_verification=True))
                    return
                pred,_ = neural.predict(model,v['x'],v['env'],pr)
                base.previous.parent.atomic_npz(out/'scores.npz',ids=v['ids'],scores=pred)
                immutable_json(receipt,dict(identity=hid,input=inp,fit=fit,
                    artifacts=dict(checkpoint=artifact(out/'checkpoint.pt'),scores=artifact(out/'scores.npz'))))
            receipts.append(artifact(receipt)); beat('inner_replayed' if verify else 'inner_frozen',completed=len(receipts),tag=v['tag'],inner=inner)
    assert len(receipts)==432; parent.check_frozen(identity['parent'])
    manifest = dict(registration=artifact(PUBLIC/'registration_lock.json'),heads=receipts,
                    updates=864000,all_passed=True,outer_held_labels_used=False)
    immutable_json(PRIVATE/('inner_replay.json' if verify else 'training_complete.json'),manifest)
    immutable_json(PUBLIC/('inner_replay.json' if verify else 'inner_prediction_freeze.json'),
        dict(manifest=artifact(PRIVATE/('inner_replay.json' if verify else 'training_complete.json')),
             heads=432,updates=864000,outer_readout_run=False))


def checked_training():
    doc = json.loads((PRIVATE/'training_complete.json').read_text()); assert doc['all_passed']
    assert doc['registration']==artifact(PUBLIC/'registration_lock.json') and len(doc['heads'])==432 and doc['updates']==864000
    for a in doc['heads']:
        assert artifact(ROOT/a['path'])==a
        for ref in json.loads((ROOT/a['path']).read_text())['artifacts'].values(): assert artifact(ROOT/ref['path'])==ref
    return doc


def probes(cfg,identity,verify=False):
    checked_training(); parent.check_frozen(identity['parent'])
    base.previous.require_committed(PUBLIC/'inner_prediction_freeze.json'); refs=[]
    for v in views(identity):
        out = PRIVATE/'probes'/v['tag']; receipt = out/'complete.json'
        if receipt.exists() and not verify:
            row = json.loads(receipt.read_text()); assert row['registration']==artifact(PUBLIC/'registration_lock.json')
            for a in row['artifacts'].values(): assert artifact(ROOT/a['path'])==a
            refs.append(artifact(receipt)); continue
        bank = {}
        for inner in method.fitting_sites(v['sites'],v['outer']):
            home = PRIVATE/'heads'/v['tag']/inner; row = json.loads((home/'complete.json').read_text())
            with np.load(home/'scores.npz',allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'],v['ids']); pred = z['scores'].copy()
            bank[inner] = dict(prediction=pred,training_sites=row['input']['training_sites'],cut=row['input']['cut'])
        with np.load(parent.frozen_directory(v['tag'],'original')/'scores.npz',allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],v['held_ids']); frozen = z['scores'].copy()
        models,drift,alignment,scores = {},{},{},{}
        w = np.where(v['env'] > 0,v['pr']['weights'],0)
        for variant in cfg['residual_variants']:
            p,y,cuts,producers = method.assemble(bank,v['raw'],v['cv'],v['sites'],v['outer'],variant)
            models[variant] = residual.fit(v['context'],p,y,w,v['sites'],v['outer'],ridge=cfg['ridge'])
            drift[variant] = method.cut_drift(v['cv'],cuts,v['pr']['positive_easy_cut'])
            alignment[variant] = dict(prediction_sha256=array_hash(p),target_sha256=array_hash(y),
                row_producer_sha256=array_hash(producers),cut_sha256=array_hash(cuts))
            for arm in cfg['probe_arms']:
                scores[variant+'__'+arm] = residual.predict(models[variant][arm],v['held_context'],frozen)
        doc = dict(models=models,cut_drift=drift,alignment=alignment,ids_sha256=array_hash(v['ids']),
                   held_context_sha256=array_hash(v['held_context']),outer_cut=v['pr']['positive_easy_cut'])
        if verify:
            assert json.loads((out/'models.json').read_text())==doc
            with np.load(out/'scores.npz',allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'],v['held_ids'])
                for k,p in scores.items(): np.testing.assert_array_equal(z[k],p)
        else:
            immutable_json(out/'models.json',doc)
            base.previous.parent.atomic_npz(out/'scores.npz',ids=v['held_ids'],**scores)
            immutable_json(receipt,dict(registration=artifact(PUBLIC/'registration_lock.json'),
                artifacts=dict(models=artifact(out/'models.json'),scores=artifact(out/'scores.npz'))))
        refs.append(artifact(receipt)); beat('probes_replayed' if verify else 'probes_frozen',views=len(refs),tag=v['tag'])
    assert len(refs)==144
    immutable_json(PUBLIC/('probe_replay.json' if verify else 'prediction_freeze.json'),dict(receipts=refs,closed_form_fits=864,outer_held_labels_used=False))


def evaluate(cfg,identity,verify=False):
    checked_training(); parent.check_frozen(identity['parent']); base.previous.require_committed(PUBLIC/'prediction_freeze.json')
    for ref in json.loads((PUBLIC/'prediction_freeze.json').read_text())['receipts']:
        assert artifact(ROOT/ref['path'])==ref
        for a in json.loads((ROOT/ref['path']).read_text())['artifacts'].values(): assert artifact(ROOT/a['path'])==a
    groups = {}; by_cache = {}; refs=[]
    for v in views(identity):
        name = v['g']['group']+'_'+v['pair']; path = PUBLIC/'groups'/(name+'.json')
        if name not in by_cache:
            by_cache = {name:base.pair_inputs(v['g'],v['data'],v['pairs'],v['pair'])[2]}
        cv = v['data']['baseline_ade'][v['held_ids'],1]
        y = source.tail.diagnostic.event_targets(by_cache[name][v['te']],cv,v['pr']['positive_easy_cut'])
        old = json.loads((parent.PUBLIC/'groups'/(name+'.json')).read_text())
        prior = next(f for f in old['folds'] if f['held']==v['outer'])
        assert prior['target_sha256']==array_hash(y) and prior['held_ids_sha256']==array_hash(v['held_ids'])
        metrics = {k:prior['metrics'][j] for k,j in [('original','original'),
            ('outer_in_sample__global_bias','original__global_bias'),('outer_in_sample__context_bias','original__context_bias')]}
        original = json.loads((source.PUBLIC/'groups'/(name+'.json')).read_text())
        edges = next(f for f in original['folds'] if f['held']==v['outer'])['training']['cost_only']['edges']
        env = v['pairs']['B'][v['pair']][1][v['te']]
        with np.load(parent.frozen_directory(v['tag'],'original')/'scores.npz',allow_pickle=False) as z: frozen=z['scores'].copy()
        with np.load(PRIVATE/'probes'/v['tag']/'scores.npz',allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],v['held_ids'])
            for variant in cfg['residual_variants']:
                for arm in cfg['probe_arms']:
                    k=variant+'__'+arm; p=z[k]; np.testing.assert_array_equal(p[:,:3],frozen[:,:3])
                    assert np.isfinite(p).all() and (p[:,3]>=0).all() and (p[:,3]<=p[:,1]).all()
                    metrics[k]=source.parent.parent.measure(p,y,env,np.repeat(v['outer'],len(y)),edges)
        doc=json.loads((PRIVATE/'probes'/v['tag']/'models.json').read_text())
        groups.setdefault(name,[]).append(dict(held=v['outer'],metrics=metrics,cut_drift=doc['cut_drift'],
            target_sha256=array_hash(y),held_ids_sha256=array_hash(v['held_ids'])))
        if len(groups[name])==4:
            row=dict(group=v['g']['group'],producer=v['g']['producer'],controller=v['g']['controller'],
                seed=int(v['g']['group'].split('_seed')[1].split('_')[0]),pair=v['pair'],folds=groups.pop(name),
                result_source='fresh_run_nested_residual_readout_cached_verified_outer_estimators_source_development')
            if verify: assert json.loads(path.read_text())==row
            else: immutable_json(path,row)
            refs.append(artifact(path)); beat('readout_replayed' if verify else 'readout',groups=len(refs),group=name)
    assert len(refs)==36 and not groups
    immutable_json(PUBLIC/('eval_replay.json' if verify else 'completion_checks.json'),dict(all_passed=True,groups=refs))


def report(cfg,identity):
    rows=[]
    for a in json.loads((PUBLIC/'completion_checks.json').read_text())['groups']:
        assert artifact(ROOT/a['path'])==a; rows.append(json.loads((ROOT/a['path']).read_text()))
    controls=dict(original='original',outer_context='outer_in_sample__context_bias',
        oof_global='oof__global_bias',next='in_sample_next__context_bias',prev='in_sample_prev__context_bias')
    comparisons={'oof_context_vs_'+k:('oof__context_bias',a) for k,a in controls.items()}
    comparisons.update({v+'_'+a+'_vs_original':(v+'__'+a,'original') for v in method.VARIANTS for a in residual.ARMS
                        if (v,a)!=('oof','context_bias')})
    contrasts={}
    for key,(a,b) in comparisons.items():
        proxy=[dict(r,folds=[dict(f,metrics={'fractional':f['metrics'][a],'mean':f['metrics'][b]}) for f in r['folds']]) for r in rows]
        contrasts[key]=parent.paired_contrasts(proxy,cfg)
    summary=parent.summarize_contrasts(contrasts)
    primary=[summary['oof_context_vs_'+k]['full'] for k in controls]
    positive=all(v['envelope_positive__harm_MSE_gain_percent']['positive']==6 for v in primary)
    guards=all(v['envelope_positive__'+k]['negative']==v['envelope_positive__'+k]['not_estimable']==0
        for v in primary for k in ('top10_gain_pp','coverage_log_error_reduction'))
    drift={pair:{v:{'label_disagreement_range':[min(f['cut_drift'][v]['easy_label_disagreement_fraction'] for r in rows if r['pair']==pair for f in r['folds']),
        max(f['cut_drift'][v]['easy_label_disagreement_fraction'] for r in rows if r['pair']==pair for f in r['folds'])],
        'median_label_disagreement':float(np.median([f['cut_drift'][v]['easy_label_disagreement_fraction'] for r in rows if r['pair']==pair for f in r['folds']]))}
        for v in method.VARIANTS} for pair in cfg['pairs']}
    gates=dict(primary_six_positive_all_controls=positive,tail_coverage_guards=guards,
        nested_residual_diagnostic_signal=positive and guards,new_forecaster_training=False,
        independent_calibration=False,independent_confirmation=False,policy_evaluated=False,
        deployment_changed=False,stage5c_executed=False,smc_enabled=False)
    immutable_json(PUBLIC/'aggregate_metrics.json',dict(contrasts=contrasts,summary=summary,cut_drift=drift,gates=gates))
    heads=[json.loads((ROOT[a['path']]).read_text()) for a in checked_training()['heads']]
    compute=dict(heads=len(heads),updates=sum(h['fit']['step'] for h in heads),
        summed_fit_seconds=sum(h['fit']['seconds'] for h in heads),unknown_draws=sum(h['fit']['unknown_rows_sampled'] for h in heads),
        median_initial_cost_loss=float(np.median([h['fit']['trace'][0]['cost_loss'] for h in heads])),
        median_final_cost_loss=float(np.median([h['fit']['trace'][-1]['cost_loss'] for h in heads])),
        threads=4,interop=1,workers=0,real_torch=True,new_trajectory_training=False,
        private_training_manifest=artifact(PRIVATE/'training_complete.json'))
    assert compute['updates']==864000 and compute['unknown_draws']==0
    immutable_json(PUBLIC/'compute_receipt.json',compute)
    lines=['# Nested Residual Results','','## Material Passport',
        'Fresh432 native Torch cost heads /864000 updates and864 closed-form probes. Frozen trajectory producers and outer estimators are cached_verified.',
        'Source-development readout; not independent calibration or confirmation. No new trajectory training or deployment.', '',
        '| Comparison / inputs | Positive / negative / overlap / missing MSE intervals | Point range (%) |','|---|---|---|']
    for k,ps in summary.items():
        for p,v in ps.items():
            q=v['envelope_positive__harm_MSE_gain_percent']
            lines.append(f"| {k} / {p} | {q['positive']} / {q['negative']} / {q['overlap']} / {q['not_estimable']} | {q['point_range']} |")
    lines+=['','Three seeds averaged within locality,3000 resamples of four localities per assignment. Six dependent assignments retained; no multiplicity adjustment.',
        'All inner cuts/scales are fitting-only. Different inner/outer cuts and two-vs-three locality training remain transport limitations, not hidden endpoint changes.',
        'No negative tail interval is not a safety or noninferiority proof.', '', '```json',json.dumps(gates,indent=2),'```', '',
        'Eight observed/twelve predicted annotation steps; detector pixels only. No metric/seconds, human-gold, physical-safety, true3D or foundation claim.']
    (PUBLIC/'results.md').write_text('\n'.join(lines)+'\n'); print(json.dumps(dict(gates=gates,compute=compute),indent=2))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--phase',required=True,choices=['register','support','pilot','train','probes','evaluate','report','verify_inner','verify_probes','verify_eval'])
    ap.add_argument('--resume',action='store_true'); args=ap.parse_args()
    PUBLIC.mkdir(parents=True,exist_ok=True); PRIVATE.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    with (PRIVATE/'lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); cfg,identity=registration(args.phase=='register')
        if args.phase=='support': support(cfg,identity)
        elif args.phase in ('pilot','train','verify_inner'): train(cfg,identity,pilot=args.phase=='pilot',resume=args.resume,verify=args.phase=='verify_inner')
        elif args.phase in ('probes','verify_probes'): probes(cfg,identity,verify=args.phase=='verify_probes')
        elif args.phase in ('evaluate','verify_eval'): evaluate(cfg,identity,verify=args.phase=='verify_eval')
        elif args.phase=='report': report(cfg,identity)


if __name__=='__main__': main()
