"""Fixed credit/denominator mechanism controls; decisions precede outcome readout."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 required before numerical imports')
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(k, '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import scipy
import torch
from scripts.build_m3w_native_scene_context import load_past_queries
from scripts.run_m3w_native_forecast import file_digest, immutable_json, json_write, assert_current
from scripts.run_m3w_native_nested import write_arrays
from src.world_model.m3w_risk_subsidy_controls import select, direct_risk
from src.evaluation.m3w_native_metrics import paired_scene_metrics
from src.evaluation.m3w_native_matched_coverage import paired_scene_contrast

CONFIG = 'configs/m3w_risk_subsidy_v1.json'
NEW = ('net_population', 'net_clipped_population', 'net_selected', 'net_clipped_selected')
MATCHED = ('matched_net_population', 'matched_net_clipped_population', 'matched_net_selected')


def require_replay(root, cfg, phase, expected_identity):
    ip, mp = root/'identity.json', root/'decisions_complete.json'
    if not ip.is_file() or not mp.is_file(): raise ValueError('Replay requires original identity and complete manifest')
    if json.loads(ip.read_text()) != expected_identity: raise ValueError('Replay identity changed')
    dm=json.loads(mp.read_text())
    if dm['experiment_sha256'] != file_digest(ip) or dm['queries'] != 188388:
        raise ValueError('Replay manifest incomplete or foreign')
    if not dm['receipts']: raise ValueError('Replay receipts absent')
    for ref in dm['receipts']:
        rp=ROOT/ref['path']
        if not rp.is_file() or file_digest(rp)!=ref['sha256']: raise ValueError('Replay receipt missing or changed')
        r=json.loads(rp.read_text()); path=ROOT/r['path']
        if (r['experiment_sha256']!=dm['experiment_sha256'] or not path.is_file()
                or file_digest(path)!=r['sha256']): raise ValueError('Replay decision archive missing or changed')
    if phase=='evaluate':
        path=ROOT/cfg['reports']/'analysis.json'
        if not path.is_file(): raise ValueError('Replay first readout missing')
        a=json.loads(path.read_text())
        if a['experiment_sha256']!=dm['experiment_sha256'] or a['decision_manifest_sha256']!=file_digest(mp):
            raise ValueError('Replay first readout identity mismatch')
        expected={str((root/'outcomes'/f'{x}_seed{s}.npz').relative_to(ROOT)) for x in cfg['actions'] for s in cfg['seeds']}
        if {v['path'] for v in a['outcome_archives']}!=expected: raise ValueError('Replay outcome inventory incomplete')
        for ref in a['outcome_archives']:
            path=ROOT/ref['path']
            if not path.is_file() or file_digest(path)!=ref['sha256']: raise ValueError('Replay outcome archive missing or changed')


def load():
    cfg = json.loads((ROOT/CONFIG).read_text()); pr = ROOT/cfg['parent_output']; pp = ROOT/cfg['parent_reports']
    assert file_digest(pr/'identity.json') == cfg['parent_identity_sha256']
    assert file_digest(pp/'analysis.json') == cfg['parent_analysis_sha256']
    original = json.loads((pr/'identity.json').read_text()); assert_current(original)
    parent = json.loads((pp/'analysis.json').read_text())
    assert parent['experiment_sha256'] == cfg['parent_identity_sha256']
    assert cfg['rho'] == .02 and cfg['bootstrap_resamples'] == 3000 and cfg['solver_seconds'] == 5.
    assert cfg['sites'] == parent['sites'] and cfg['seeds'] == parent['seeds']
    assert cfg['actions'] == original['config']['actions']
    assert cfg['policies'] == ['old_strict', 'positive_population', 'net_point', 'parent_net_population', *NEW, *MATCHED]
    assert not any(cfg[k] for k in ('new_training','threshold_search','independent_calibration',
        'independent_confirmation','external_readout','deployment','stage5c_executed','smc_enabled'))
    bindings = dict(original['source_bindings'])

    def bind(path, expected=None):
        path = Path(path); path = path if path.is_absolute() else ROOT/path
        key, sha = str(path.relative_to(ROOT)), file_digest(path)
        if expected is not None and sha != expected: raise ValueError('Changed source: '+key)
        if key in bindings and bindings[key] != sha: raise ValueError('Conflicting dependency: '+key)
        bindings[key] = sha

    bind(pr/'identity.json', cfg['parent_identity_sha256']); bind(pp/'analysis.json', cfg['parent_analysis_sha256'])
    bind(pr/'decisions_complete.json', parent['decision_manifest_sha256'])
    dm = json.loads((pr/'decisions_complete.json').read_text())
    assert dm['experiment_sha256'] == cfg['parent_identity_sha256'] and dm['query_action_seed_instances'] == 188388
    for name in ('decision_replay.json', 'aggregate_replay.json', 'independent_arithmetic.json'):
        path = pp/name; bind(path); record = json.loads(path.read_text()); assert record['all_checks_passed']
        if 'analysis_sha256' in record: assert record['analysis_sha256'] == cfg['parent_analysis_sha256']
        if 'decision_manifest_sha256' in record: assert record['decision_manifest_sha256'] == parent['decision_manifest_sha256']
    for f in parent['fits']:
        assert f['fit']['complete'] and f['fit']['trees'] == 128
        assert f['view'].rsplit('_seed', 1)[0] not in f['identity']['training_sites']
        bind(f['checkpoint'], f['checkpoint_sha256'])
    assert len(parent['fits']) == 36
    old_path = ROOT/'outputs/publication_readiness_2026_09/easy_moment_v1/analysis.json'
    old = json.loads(old_path.read_text()); bind(old_path)
    for r in old['outcome_archives']: bind(r['path'], r['sha256'])
    for ref in dm['receipts']:
        bind(ref['path'], ref['sha256']); r = json.loads((ROOT/ref['path']).read_text())
        assert r['experiment_sha256'] == cfg['parent_identity_sha256']
        bind(r['path'], r['sha256'])
    for path in (CONFIG, cfg['registration'], 'scripts/run_m3w_risk_subsidy.py',
            'src/world_model/m3w_risk_subsidy_controls.py', 'tests/test_m3w_risk_subsidy_controls.py',
            'tests/test_m3w_risk_subsidy_replay.py',
            'outputs/publication_readiness_2026_09/zero_reference_support_v2/analysis.json'):
        bind(path)
    data = load_past_queries(bindings)
    data['sites'] = np.array([r.split('/')[0] for r in data['recordings']])
    assert len(data['sites']) == 175756 and set(data['sites']) == set(cfg['sites'])
    identity = dict(config=cfg, source_bindings=bindings, numpy=np.__version__, scipy=scipy.__version__,
        torch=torch.__version__, architecture=platform.machine(), data_role='design_exposed_development',
        runtime=dict(torch_threads=4, interop_threads=1, num_workers=0))
    return cfg, data, parent, old, dm, identity


def decide(pack, args, beat):
    cfg, data, parent, old, dm, identity = pack; root = ROOT/cfg['output']; arms = cfg['policies']
    ish = file_digest(root/'identity.json'); manifest = []; queries = 0; new_chunks = 0
    for ref in dm['receipts']:
        r = json.loads((ROOT/ref['path']).read_text()); source_path = ROOT/r['path']
        out = root/'decisions'/source_path.relative_to(ROOT/cfg['parent_output']/'decisions')
        rp = out.with_suffix('.json')
        if rp.exists() and not args.verify:
            if not args.resume: raise ValueError('Existing decisions require explicit --resume or --verify')
            record = json.loads(rp.read_text())
            assert record['experiment_sha256'] == ish and record['source_receipt_sha256'] == ref['sha256']
            assert file_digest(out) == record['sha256']
            manifest.append(dict(path=str(rp.relative_to(ROOT)), sha256=file_digest(rp)))
            queries += len(r['queries']); continue
        if args.verify and not rp.exists(): raise ValueError('Replay cannot create missing decisions')
        with np.load(source_path, allow_pickle=False) as z: z = {k:z[k].copy() for k in z.files}
        ids = z['ids']; bits = np.zeros((len(ids), len(arms)), bool)
        for j, name in enumerate(arms[:4]): bits[:, j] = z['choices'][:, parent['policies'].index(name.removeprefix('parent_'))]
        records = []; seen = np.zeros(len(ids), bool)
        for q in r['queries']:
            rows = np.flatnonzero((data['recordings'][ids] == q['recording']) & (data['frames'][ids] == q['frame']))
            assert len(rows) and not seen[rows].any(); seen[rows] = True
            g, risk, den, ok = [z[k][rows] for k in ('gain','net_risk','denominator','support')]
            report = {}
            for mode in NEW:
                b, info = select(g, risk, den, ok, mode, seconds=cfg['solver_seconds'])
                bits[rows, arms.index(mode)] = b; report[mode] = info
            reference = report['net_clipped_selected']; count = reference['selected']
            for mode in MATCHED:
                b, info = select(g, risk, den, ok, mode.removeprefix('matched_'), count=count,
                    seconds=cfg['solver_seconds'])
                bits[rows, arms.index(mode)] = b
                report[mode] = dict(info, requested_count=count, reference_optimal=reference['optimal'],
                    reference_failed_closed=reference['failed_closed'])
            records.append(dict(recording=q['recording'], frame=q['frame'], arms=report))
        assert seen.all()
        write_arrays(out, dict(ids=ids, choices=bits))
        result = dict(experiment_sha256=ish, source_receipt_sha256=ref['sha256'],
            view=r['view'], action=r['action'], path=str(out.relative_to(ROOT)), sha256=file_digest(out), queries=records)
        immutable_json(rp, result); manifest.append(dict(path=str(rp.relative_to(ROOT)), sha256=file_digest(rp)))
        queries += len(records); new_chunks += 1
        beat(state='decision_replay' if args.verify else 'past_only_decisions', queries=queries,
            view=r['view'], action=r['action'], new_chunks=new_chunks)
        if args.limit_chunks and new_chunks >= args.limit_chunks:
            beat(state='pilot_complete_not_full_matrix', queries=queries); return
    assert queries == 188388; assert_current(identity)
    immutable_json(root/'decisions_complete.json', dict(experiment_sha256=ish,
        queries=queries, receipts=manifest, future_arrays_opened_for_decisions=False))
    if args.verify:
        immutable_json(ROOT/cfg['reports']/'decision_replay.json', dict(all_checks_passed=True,
            queries=queries, manifest_sha256=file_digest(root/'decisions_complete.json')))
    beat(state='decisions_complete', queries=queries)


def evaluate(pack, args, beat):
    cfg, data, parent, old, _, identity = pack; root = ROOT/cfg['output']; public = ROOT/cfg['reports']
    if args.verify and not (public/'analysis.json').exists(): raise ValueError('Cannot replay absent first readout')
    if args.verify and not all((root/'outcomes'/f'{a}_seed{s}.npz').exists() for a in cfg['actions'] for s in cfg['seeds']):
        raise ValueError('Cannot replay missing outcome choice archives')
    dm = json.loads((root/'decisions_complete.json').read_text())
    assert dm['experiment_sha256'] == file_digest(root/'identity.json') and dm['queries'] == 188388
    n = len(data['sites']); arms = cfg['policies']; bits = {(s,a):np.zeros((n,len(arms)),bool) for s in cfg['seeds'] for a in cfg['actions']}
    seen = {k:np.zeros(n,bool) for k in bits}; solver = {a:{m:dict(queries=0, canonical_optimum_unverified=0,
        failed_closed=0, count_failures=0, reference_optimum_unverified=0,
        reference_failed_closed=0) for m in (*NEW,*MATCHED)} for a in cfg['actions']}
    for ref in dm['receipts']:
        assert file_digest(ROOT/ref['path']) == ref['sha256']; r = json.loads((ROOT/ref['path']).read_text())
        assert r['experiment_sha256'] == dm['experiment_sha256']
        assert file_digest(ROOT/r['path']) == r['sha256']
        key = int(r['view'].rsplit('_seed',1)[1]), r['action']
        with np.load(ROOT/r['path'],allow_pickle=False) as z:
            ids=z['ids']; assert not seen[key][ids].any(); seen[key][ids]=True; bits[key][ids]=z['choices']
        for q in r['queries']:
            for mode, info in q['arms'].items():
                assert info['constraint_pass'] and info['direct_constraint_pass']
                v=solver[r['action']][mode]; v['queries']+=1
                v['canonical_optimum_unverified']+=not info['canonical_optimal_verified']
                v['failed_closed']+=info['failed_closed'];v['count_failures']+=not info['exact_count_pass']
                v['reference_optimum_unverified']+=not info.get('reference_optimal',True)
                v['reference_failed_closed']+=info.get('reference_failed_closed',False)
    assert all(v.all() for v in seen.values())

    def metric(e, cv, mask=None, ci=False):
        use=np.ones(n,bool) if mask is None else mask
        return paired_scene_metrics(e[use],cv[use],data['sites'][use],expected_scenes=cfg['sites'],
            dataset='sdd',coordinate_unit='annotation_pixel',bootstrap_resamples=3000 if ci else 0)

    summary={}; contrasts={}; outcomes=[]
    pairs=[('net_population','parent_net_population'),('net_clipped_population','net_population'),('net_selected','net_population'),
        ('net_clipped_selected','net_clipped_population'),('net_clipped_selected','net_selected'),
        ('net_clipped_selected','net_population'),('net_clipped_selected','net_point')]
    pairs += [(m,'net_clipped_selected') for m in MATCHED]
    pairs += [(m,'old_strict') for m in NEW]
    for action in cfg['actions']:
        errors={p:[] for p in arms}; ends={p:[] for p in arms}; detail={p:{} for p in arms}
        for seed in cfg['seeds']:
            ref=next(r for r in old['outcome_archives'] if r['path'].endswith(f'{action}_seed{seed}.npz'))
            assert file_digest(ROOT/ref['path'])==ref['sha256']
            with np.load(ROOT/ref['path'],allow_pickle=False) as z: o={k:z[k].copy() for k in z.files}
            cv,cf=o['cv'],o['cf']; masks={k:o[k] for k in ('complete','zero_CV','positive_easy','hard')}
            for j,p in enumerate(arms):
                b=bits[seed,action][:,j]; e=np.where(b,o['candidate_ade'],cv); f=np.where(b,o['candidate_fde'],cf)
                errors[p].append(e); ends[p].append(f)
                detail[p][str(seed)]=dict(ADE=metric(e,cv),FDE=metric(f,cf),
                    subsets={k:metric(e,cv,m) for k,m in masks.items()},selected=int(b.sum()),
                    selected_unknown=int((b&np.isnan(cv)).sum()),selected_incomplete=int((b&~masks['complete']).sum()),
                    zero_CV_harmed=int((b&masks['zero_CV']&(e>0)).sum()),
                    full_grid_gain_bounds={s:[float(np.where(b,o[k],0)[data['sites']==s].mean()) for k in ('lower','upper')] for s in cfg['sites']})
            op=root/'outcomes'/f'{action}_seed{seed}.npz'; write_arrays(op,dict(choices=bits[seed,action]))
            outcomes.append(dict(path=str(op.relative_to(ROOT)),sha256=file_digest(op)))
        mean={p:np.mean(errors[p],axis=0) for p in arms}
        for p in arms:
            summary[action+'__'+p]=dict(ADE=metric(mean[p],cv,ci=True),FDE=metric(np.mean(ends[p],axis=0),cf,ci=True),
                subsets={k:metric(mean[p],cv,m,True) for k,m in masks.items()},seeds=detail[p])
        contrasts[action]={}
        for left,right in pairs:
            value={}
            for subset in ('all','hard','positive_easy'):
                mask=None if subset=='all' else masks[subset]
                l,r=metric(mean[left],cv,mask)['by_scene'],metric(mean[right],cv,mask)['by_scene']
                value[subset]=paired_scene_contrast([l[s]['gain_percent'] for s in cfg['sites']],
                    [r[s]['gain_percent'] for s in cfg['sites']],resamples=3000)
            contrasts[action][left+'_minus_'+right]=value
    assert_current(identity)
    result=dict(result_source='fresh_causal_allocation_cached_verified_moments_forecasts_labels_no_new_training',
        experiment_sha256=file_digest(root/'identity.json'),decision_manifest_sha256=file_digest(root/'decisions_complete.json'),
        rows=n,sites=cfg['sites'],seeds=cfg['seeds'],policies=arms,summary=summary,contrasts=contrasts,
        solver=solver,outcome_archives=outcomes,new_training=False,independent_confirmation=False,
        independent_calibration=False,deployment=False,stage5c_executed=False,smc_enabled=False)
    immutable_json(public/'analysis.json',result)
    if args.verify: immutable_json(public/'aggregate_replay.json',dict(all_checks_passed=True,analysis_sha256=file_digest(public/'analysis.json')))
    beat(state='evaluation_complete',rows=n)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['preflight','decide','evaluate'],default='preflight')
    parser.add_argument('--resume',action='store_true'); parser.add_argument('--verify',action='store_true')
    parser.add_argument('--limit-chunks',type=int)
    args=parser.parse_args(); torch.set_num_threads(4); torch.set_num_interop_threads(1)
    if args.limit_chunks is not None and args.limit_chunks <= 0: parser.error('--limit-chunks must be positive')
    pack=load(); cfg=pack[0]; root=ROOT/cfg['output']; root.mkdir(parents=True,exist_ok=True)
    def beat(**v):
        record=dict(pid=os.getpid(),updated_unix=time.time(),**v); json_write(root/'heartbeat.json',record)
        with (root/'events.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
    with (root/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.verify:
            require_replay(root, cfg, args.phase, pack[-1])
        immutable_json(root/'identity.json',pack[-1])
        if args.phase=='preflight': beat(state='preflight_pass',rows=len(pack[1]['sites']),bindings=len(pack[-1]['source_bindings']))
        elif args.phase=='decide': decide(pack,args,beat)
        else: evaluate(pack,args,beat)


if __name__=='__main__': main()
