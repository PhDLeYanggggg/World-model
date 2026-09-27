"""Fitting-only moment readouts with unchanged source-held cost evaluation."""
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
    raise RuntimeError('Use native arm64 before importing Torch')
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '4')
from scripts import run_m3w_european_regime_transport as previous
from src.world_model import m3w_cost_mass as method
import numpy as np
import torch

magnitude = previous.previous
strong = magnitude.strong
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_cost_mass_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_cost_mass_v1'
CONFIG = 'configs/m3w_european_cost_mass_v1.json'
FILES = [CONFIG, 'src/world_model/m3w_cost_mass.py', 'tests/test_m3w_cost_mass.py',
    'scripts/run_m3w_european_cost_mass.py', 'scripts/report_m3w_european_cost_mass.py',
    'tests/test_m3w_cost_mass_report.py', str(PUBLIC.relative_to(ROOT)/'protocol.md')]
artifact, digest, immutable_json, array_hash = previous.artifact, previous.digest, previous.immutable_json, previous.array_hash
require_committed = previous.require_committed


def beat(state, **kw):
    row = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), state=state, **kw)
    strong.risk.base.base.cross.json_write(PRIVATE/'heartbeat.json', row)
    with (PRIVATE/'events.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
    print(json.dumps(row), flush=True)


def registration(create=False):
    cfg = json.loads((ROOT/CONFIG).read_text()); pcfg, pid = previous.registration()
    seal = magnitude.previous.checked_seal(previous.PUBLIC/'verification.json')
    for ref in json.loads((previous.PUBLIC/'verification.json').read_text())['local_detailed_metrics']:
        assert artifact(ROOT/ref['path']) == ref
    for key, value in pcfg.items():
        if value is False: assert cfg[key] is False
    assert (cfg['views'], cfg['fresh_readouts'], cfg['max_slope'], cfg['root_iterations']) == (144,432,8.,80)
    assert cfg['arms'] == ['cost_only','cap_aux','shuffled_aux'] and not cfg['new_neural_training']
    identity = dict(parent=pid, parent_verification=seal, bindings={p:digest(ROOT/p) for p in FILES})
    path = PUBLIC/'registration_lock.json'
    if create: immutable_json(path, identity)
    else:
        assert json.loads(path.read_text()) == identity; require_committed(path)
    return cfg, identity


def views(identity):
    return previous.views(identity['parent'])


def fitting(cfg, identity, *, pilot=False, verify=False):
    magnitude.checked_inner(); magnitude.check_freeze(); refs = []; start = time.monotonic()
    if not pilot: assert json.loads((PRIVATE/'pilot.json').read_text())['complete']
    for v in views(identity):
        if shutil.disk_usage(PRIVATE).free < 10*2**30: raise OSError('Preserve10GiB;completed fits are resumable')
        assert not (set(v['sites']) & set(v['g']['producer_roster'])) and v['outer'] not in v['sites']
        old_home = magnitude.PRIVATE/'readouts'/v['tag']; magnitude.checked_receipt(old_home)
        old = json.loads((old_home/'models.json').read_text())
        assert old['training_ids_sha256'] == array_hash(v['ids'])
        assert old['held_ids_sha256'] == array_hash(v['held_ids'])
        _, all_env, _, _ = v['pairs']['B'][v['pair']]; held_env = all_env[v['te']]
        models, details, scores = {}, {}, {}
        with np.load(old_home/'scores.npz', allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved['ids'], v['held_ids'])
            for arm in cfg['arms']:
                p, y, cuts, lineage = magnitude.oof_bank(v, arm)
                prior = old['details'][arm]
                assert array_hash(p) == prior['oof_score_sha256'] and array_hash(y) == prior['target_sha256']
                assert lineage == prior['references']
                l2 = magnitude.method.fit_magnitude(p,y,v['env'],v['sites'],v['outer'],max_slope=cfg['max_slope'])
                assert l2 == old['models'][arm]
                mass = method.fit(p,y,v['env'],v['sites'],v['outer'],max_slope=cfg['max_slope'],iterations=cfg['root_iterations'])
                raw_model = dict(slopes=[1.,1.],max_slope=cfg['max_slope'])
                np.testing.assert_array_equal(method.predict(raw_model,p,v['env']),p)
                unprojected = p.astype(float).copy(); unprojected[:,[1,3]] *= np.asarray(l2['slopes'])
                fitted = dict(raw=p, unprojected_L2=unprojected,
                    scaled=method.predict(l2,p,v['env']),mass=method.predict(mass,p,v['env']))
                definitions = dict(raw=raw_model,unprojected_L2=l2,scaled=l2,mass=mass)
                w = method.fitting_weights(p,y,v['env'],v['sites'],v['outer'])
                equal = {key:method.decompose(p,y,score,w,v['sites'],definitions[key]) for key,score in fitted.items()}
                row_w = (w>0).astype(float); row_w /= row_w.sum()
                row = {key:method.decompose(p,y,score,row_w,v['sites'],definitions[key]) for key,score in fitted.items()}
                for key in fitted:
                    for component in ('H_all','H_easy'):
                        for d in (equal[key][component],row[key][component]):
                            np.testing.assert_allclose(d['MSE'],d['zero_target_SSE']+d['positive_target_SSE'],rtol=1e-13)
                            np.testing.assert_allclose(d['MSE'],sum(s['MSE_contribution'] for s in d['localities'].values()),rtol=1e-13)
                edges = strong.parent.edges_for(fitted['mass'],v['env'],v['pr'],cfg)
                models[arm] = mass
                details[arm] = dict(oof_sha256=array_hash(p),target_sha256=array_hash(y),
                    cut_sha256=array_hash(cuts),producer_lineage=lineage,weights_sha256=array_hash(w),
                    equal_locality=equal,row_weighted=row,edges=edges)
                scores[arm+'_mass'] = method.predict(mass,saved[arm+'_raw'],held_env)
        record = dict(registration=artifact(PUBLIC/'registration_lock.json'),
            parent=artifact(old_home/'complete.json'),tag=v['tag'],pair=v['pair'],outer=v['outer'],
            fitting_sites=sorted(set(v['sites'])),forecast_producer_sites=v['g']['producer_roster'],
            held_ids_sha256=array_hash(v['held_ids']),training_ids_sha256=array_hash(v['ids']),
            models=models,details=details,outer_labels_used=False,new_neural_training=False)
        out = PRIVATE/'readouts'/v['tag']
        if verify or (out/'complete.json').exists():
            magnitude.checked_receipt(out); assert json.loads((out/'models.json').read_text()) == record
            with np.load(out/'scores.npz',allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'],v['held_ids'])
                for key,value in scores.items(): np.testing.assert_array_equal(z[key],value)
        else:
            out.mkdir(parents=True,exist_ok=True); immutable_json(out/'models.json',record)
            magnitude.atomic_npz(out/'scores.npz',ids=v['held_ids'],**scores)
            immutable_json(out/'complete.json',dict(registration=artifact(PUBLIC/'registration_lock.json'),
                artifacts=dict(models=artifact(out/'models.json'),scores=artifact(out/'scores.npz'))))
        refs.append(artifact(out/'complete.json'))
        beat('fit_replayed' if verify else 'fit_frozen',views=len(refs),readouts=len(refs)*3,tag=v['tag'])
        if pilot:
            immutable_json(PRIVATE/'pilot.json',dict(complete=True,receipt=refs[0],
                seconds_excluding_ancestry=time.monotonic()-start,views=1,readouts=3,
                free_GiB=shutil.disk_usage(PRIVATE).free/2**30,
                held_labels_used=False,new_neural_training=False))
            return
    assert len(refs) == 144
    freeze = dict(registration=artifact(PUBLIC/'registration_lock.json'),readouts=refs,
        fitted_readouts=432,parameters_per_readout=2,outer_labels_used=False,independent_roles_read=False)
    if verify:
        assert json.loads((PUBLIC/'prediction_freeze.json').read_text()) == freeze
        immutable_json(PUBLIC/'fitting_replay.json',dict(exact=True,views=144,readouts=432,old_L2_matches=432))
    else: immutable_json(PUBLIC/'prediction_freeze.json',freeze)


def check_freeze():
    path = PUBLIC/'prediction_freeze.json'; require_committed(path)
    doc = json.loads(path.read_text()); assert doc['fitted_readouts'] == 432
    for ref in doc['readouts']:
        assert artifact(ROOT/ref['path']) == ref; magnitude.checked_receipt((ROOT/ref['path']).parent)
    return doc


def evaluate(cfg,identity,verify=False):
    check_freeze(); groups = {}; cached = {}; checks = 0
    old = {r['group']+'_'+r['pair']:r for r in json.loads((magnitude.PUBLIC/'readout.json').read_text())['rows']}
    for v in views(identity):
        name = v['g']['group']+'_'+v['pair']
        if name not in cached:
            cached = {name:strong.risk.base.pair_inputs(v['g'],v['data'],v['pairs'],v['pair'])[2]}
        _,env,_,_ = v['pairs']['B'][v['pair']]; he = env[v['te']]
        y = strong.tail.event_targets(cached[name][v['te']],v['data']['baseline_ade'][v['held_ids'],1],v['pr']['positive_easy_cut'])
        prior = next(r for r in old[name]['folds'] if r['held'] == v['outer'])
        assert array_hash(y) == prior['target_sha256'] and array_hash(v['held_ids']) == prior['held_ids_sha256']
        out = PRIVATE/'readouts'/v['tag']; record = json.loads((out/'models.json').read_text())
        old_home = magnitude.PRIVATE/'readouts'/v['tag']; old_record = json.loads((old_home/'models.json').read_text())
        metrics = {}
        with np.load(out/'scores.npz',allow_pickle=False) as new, np.load(old_home/'scores.npz',allow_pickle=False) as old_scores:
            np.testing.assert_array_equal(new['ids'],v['held_ids']); np.testing.assert_array_equal(old_scores['ids'],v['held_ids'])
            for arm in cfg['arms']:
                for mode in ('raw','scaled','mass'):
                    key = arm+'_'+mode; score = new[key] if mode == 'mass' else old_scores[key]
                    edges = record['details'][arm]['edges'] if mode == 'mass' else old_record['details'][arm]['edges'][mode]
                    m = strong.measure(score,y,he,v['data']['sites'][v['held_ids']],edges)
                    for subset,mask in [('all',np.ones(len(y),bool)),('envelope_positive',he>0)]:
                        known = np.isfinite(y).all(1)&mask
                        np.testing.assert_allclose(m[subset]['component_MSE'],((score[known]-y[known])**2).mean(0),rtol=1e-13,atol=1e-13)
                        checks += 1
                    if mode != 'mass': assert m == prior['metrics'][key]
                    metrics[key] = m
        group = groups.setdefault(name,dict(group=v['g']['group'],producer=v['g']['producer'],
            controller=v['g']['controller'],seed=magnitude.seed_of(v),pair=v['pair'],folds=[]))
        group['folds'].append(dict(held=v['outer'],metrics=metrics,
            target_sha256=array_hash(y),held_ids_sha256=array_hash(v['held_ids'])))
        beat('held_scored',views=sum(len(g['folds']) for g in groups.values()),tag=v['tag'])
    assert len(groups) == 36 and checks == 2592
    doc = dict(rows=list(groups.values()),direct_MSE_checks=checks,
        result_source='fresh_run_source_held_cost_readout_no_new_neural_training',
        independent_roles_read=False,new_policy_evaluated=False)
    if verify:
        assert json.loads((PUBLIC/'readout.json').read_text()) == doc
        immutable_json(PUBLIC/'evaluation_replay.json',dict(exact=True,views=144,direct_MSE_checks=checks))
    else: immutable_json(PUBLIC/'readout.json',doc)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--phase',required=True,
        choices=['register','pilot','fit','evaluate','verify_fit','verify_eval'])
    args = parser.parse_args(); PUBLIC.mkdir(parents=True,exist_ok=True); PRIVATE.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    with (PRIVATE/'lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); cfg,identity = registration(args.phase == 'register')
        if args.phase in ('pilot','fit','verify_fit'):
            fitting(cfg,identity,pilot=args.phase == 'pilot',verify=args.phase == 'verify_fit')
        elif args.phase in ('evaluate','verify_eval'):
            evaluate(cfg,identity,verify=args.phase == 'verify_eval')


if __name__ == '__main__': main()
