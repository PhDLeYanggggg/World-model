"""Fixed source-only crossed fitting-regime/easy-cut experiment."""
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
    raise RuntimeError('Use native arm64 before Torch import')
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '4')
from scripts import run_m3w_european_oof_magnitude as previous
from scripts.recover_m3w_oof_magnitude_checkpoint import recover
from src.world_model import m3w_regime_transport as method
import numpy as np
import torch

neural, archive = previous.neural, previous.method
strong, nested = previous.strong, previous.nested
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_regime_transport_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_regime_transport_v1'
CONFIG = 'configs/m3w_european_regime_transport_v1.json'
FILES = [CONFIG, 'src/world_model/m3w_regime_transport.py', 'tests/test_m3w_regime_transport.py',
    'scripts/run_m3w_european_regime_transport.py', 'scripts/report_m3w_european_regime_transport.py',
    'tests/test_m3w_regime_transport_report.py', str(PUBLIC.relative_to(ROOT)/'protocol.md')]
artifact, digest, immutable_json, array_hash = previous.artifact, previous.digest, previous.immutable_json, previous.array_hash
require_committed = previous.require_committed


def beat(state, **kw):
    row = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), state=state, **kw)
    strong.risk.base.base.cross.json_write(PRIVATE/'heartbeat.json', row)
    with (PRIVATE/'events.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
    print(json.dumps(row), flush=True)


def registration(create=False):
    cfg = json.loads((ROOT/CONFIG).read_text()); pcfg, pid = previous.registration()
    seal = previous.previous.checked_seal(previous.PUBLIC/'verification.json')
    for ref in json.loads((previous.PUBLIC/'verification.json').read_text())['local_detailed_metrics']:
        assert artifact(ROOT/ref['path']) == ref
    assert cfg['head_training'] == pcfg['head_training'] and cfg['cells'] == list(method.CELLS)
    assert (cfg['views'], cfg['new_heads'], cfg['new_updates']) == (144, 864, 1728000)
    for key, value in pcfg.items():
        if value is False: assert cfg[key] is False
    identity = dict(parent=pid, parent_verification=seal, bindings={p:digest(ROOT/p) for p in FILES})
    path = PUBLIC/'registration_lock.json'
    if create: immutable_json(path, identity)
    else:
        assert json.loads(path.read_text()) == identity; require_committed(path)
    return cfg, identity


def views(identity):
    return previous.views(identity['parent'])


def cell_input(v, omitted, cell):
    take, pr, y, easy, lineage = method.inputs(v['x'], v['raw'], v['cv'], v['sites'], v['outer'], omitted, cell)
    assert not (set().union(*(set(lineage[k]) for k in ('row_sites', 'preprocessing_sites', 'cut_sites')))
                & set(v['g']['producer_roster']))
    record = dict(tag=v['tag'], lineage=lineage, cut=pr['positive_easy_cut'],
        row_ids_sha256=array_hash(v['ids'][take]), x_sha256=array_hash(v['x'][take]),
        target_sha256=array_hash(y), easy_sha256=array_hash(easy),
        envelope_sha256=array_hash(v['env'][take]), held_ids_sha256=array_hash(v['held_ids']),
        forecast_producer_sites=v['g']['producer_roster'],
        held_outcomes_used_for_fit=False, not_inner_OOF=cell != 'two_cut2')
    return take, pr, y, easy, record


def native(v, omitted, regime):
    home = (nested.PRIVATE/'heads'/v['tag']/omitted if regime == 'two'
            else strong.PRIVATE/'heads'/v['tag']/'cost_only')
    previous.checked_receipt(home)
    model, state = neural.restore(home)
    return model, state, artifact(home/'complete.json')


def support(cfg, identity):
    rows = []
    for v in views(identity):
        for omitted in sorted(set(v['sites'])):
            for cell in method.CELLS:
                take, pr, y, easy, record = cell_input(v, omitted, cell)
                known = pr['known']; e = float(pr['weights']@np.nan_to_num(easy, nan=0))
                rows.append(dict(input=record, known=int(known.sum()),
                    easy_rows=int(np.nansum(easy)), easy_harm_rows=int((y[known, 3] > 0).sum()),
                    supported=bool(0 < e < 1 and (v['env'][take][known] > 0).any()),
                    fresh_training=cell in method.NEW_CELLS))
        beat('support', views=len(rows)//12, tag=v['tag'])
    assert len(rows) == 1728 and sum(r['fresh_training'] for r in rows) == 864
    immutable_json(PUBLIC/'support_report.json', dict(registration=artifact(PUBLIC/'registration_lock.json'),
        rows=rows, training_allowed=all(r['supported'] for r in rows), statistical_power=False,
        outer_labels_used=False, crossed_heads_are_not_inner_OOF=True))
    (PUBLIC/'support_report.md').write_text('# Crossed Fitting Support\n\n'
        f"All1728 cells supported: {all(r['supported'] for r in rows)}. "
        f"Minimum known rows:{min(r['known'] for r in rows)};easy-harm rows:{min(r['easy_harm_rows'] for r in rows)}.\n"
        '864 fresh crossed heads;native controls are reused. Numerical support,not statistical power.\n'
        'All label-definition sites are declared;two_cut3 is not inner-OOF for its omitted row site.\n')


def head(v, omitted, cell, cfg, resume, verify=False, stop_at=None, home=None):
    take, pr, y, easy, record = cell_input(v, omitted, cell)
    home = PRIVATE/'heads'/v['tag']/omitted/cell if home is None else home
    hid = dict(registration=artifact(PUBLIC/'registration_lock.json'), input=record, seed=previous.seed_of(v))
    if (home/'complete.json').exists():
        receipt = previous.checked_receipt(home); assert receipt['identity'] == hid
        model, state = archive.restore(home); fit = receipt['fit']
    else:
        if verify: raise ValueError('Missing completed crossed head')
        if shutil.disk_usage(PRIVATE).free < 10*2**30: raise OSError('Keep10GiB;resume retained checkpoints')
        if resume and (home/'checkpoint.pt.gz').exists():
            recovered = recover(home, PRIVATE, restore=True)
            if recovered: beat('checkpoint_recovered', recovery=recovered)
        model, fit = neural.fit(v['x'][take], y, easy, v['sites'][take], v['env'][take], pr,
            arm='cost_only', seed=previous.seed_of(v), settings=cfg['head_training'], identity=hid,
            directory=home, resume=resume, stop_at=stop_at,
            heartbeat=lambda **kw:beat(tag=v['tag'], omitted=omitted, cell=cell, **kw))
        if not fit['complete']: return model, None, fit, None
        checkpoint = archive.compress_checkpoint(home); model, state = archive.restore(home)
        immutable_json(home/'complete.json', dict(identity=hid, fit=fit,
            artifacts=dict(checkpoint=artifact(checkpoint)), result_source='fresh_run_native_torch'))
    assert state['identity'] == hid and state['step'] == 2000 and state['arm'] == 'cost_only'
    for key in ('mean', 'std', 'known', 'weights'):
        np.testing.assert_array_equal(state['preprocess'][key], pr[key])
    assert state['preprocess']['positive_easy_cut'] == pr['positive_easy_cut']
    assert not state['draws'][~pr['known']].any()
    return model, state, fit, artifact(home/'complete.json')


def train(cfg, identity, pilot=False, resume=False, verify=False):
    support_doc = json.loads((PUBLIC/'support_report.json').read_text())
    assert support_doc['training_allowed'] and support_doc['registration'] == artifact(PUBLIC/'registration_lock.json')
    require_committed(PUBLIC/'support_report.json'); previous.check_freeze()
    registered = {(r['input']['tag'],r['input']['lineage']['omitted'],r['input']['lineage']['cell']):r['input']
        for r in support_doc['rows']}
    if not pilot:
        assert json.loads((PRIVATE/'pilot.json').read_text())['native_exact']
    refs, prediction_refs, matches, bridge_checked = [], [], 0, False
    for v in views(identity):
        bx, env, _, _ = v['pairs']['B'][v['pair']]; hx, he = bx[v['te']], env[v['te']]
        _, native3, native3ref = native(v, None, 'three')
        old_readout = previous.PRIVATE/'readouts'/v['tag']
        previous.checked_receipt(old_readout)
        magnitude = json.loads((old_readout/'models.json').read_text())
        fixed_readout = magnitude['models']['cost_only']
        for omitted in sorted(set(v['sites'])):
            _, native2, native2ref = native(v, omitted, 'two')
            if verify and not bridge_checked:
                model, bridge, _, _ = head(v, omitted, 'three_cut3', cfg, resume, verify=True,
                    home=PRIVATE/'pilot_native_bridge')
                method.check_cut_match(bridge, native3)
                for key in model.state_dict(): assert torch.equal(model.state_dict()[key], native3['model'][key])
                bridge_checked = True
            if pilot:
                model, bridge, fit, ref = head(v, omitted, 'three_cut3', cfg, resume,
                    home=PRIVATE/'pilot_native_bridge')
                method.check_cut_match(bridge, native3)
                for key in model.state_dict(): assert torch.equal(model.state_dict()[key], native3['model'][key])
                _, _, partial, _ = head(v, omitted, 'two_cut3', cfg, resume, stop_at=200)
                immutable_json(PRIVATE/'pilot.json', dict(native_bridge=ref, native_exact=True,
                    native_updates=fit['step'], crossed_updates=partial['step'], fit=partial,
                    projected_crossed_fit_seconds=partial['seconds']/partial['step']*cfg['new_updates'],
                    excludes_preflight_IO_and_diagnostics=True, free_GiB=shutil.disk_usage(PRIVATE).free/2**30))
                beat('pilot_complete'); return
            score, provenance, cuts = {}, {}, {}
            for cell in method.CELLS:
                take, pr, _, _, record = cell_input(v, omitted, cell)
                assert record == registered[(v['tag'], omitted, cell)]
                if cell in method.NEW_CELLS:
                    model, state, _, ref = head(v, omitted, cell, cfg, resume, verify)
                    method.check_cut_match(state, native2 if cell.startswith('two_') else native3)
                    refs.append(ref); matches += 1
                else:
                    model, state, ref = native(v, omitted, 'two' if cell.startswith('two_') else 'three')
                    for key in ('mean', 'std', 'known', 'weights'):
                        np.testing.assert_array_equal(pr[key], state['preprocess'][key])
                    assert pr['positive_easy_cut'] == state['preprocess']['positive_easy_cut']
                method.check_prediction_sites([v['outer']], record['lineage'])
                raw, _ = neural.predict(model, hx, he, pr)
                score[cell+'_raw'] = raw
                score[cell+'_scaled'] = archive.predict_magnitude(fixed_readout, raw, he)
                provenance[cell] = dict(head=ref, input=record)
                cuts[cell] = pr['positive_easy_cut']
            with np.load(old_readout/'scores.npz', allow_pickle=False) as old:
                np.testing.assert_array_equal(old['ids'], v['held_ids'])
                np.testing.assert_array_equal(score['three_cut3_raw'], old['cost_only_raw'])
                np.testing.assert_array_equal(score['three_cut3_scaled'], old['cost_only_scaled'])
            out = PRIVATE/'predictions'/v['tag']/omitted
            doc = dict(provenance=provenance, cuts=cuts, omitted=omitted,
                parent_magnitude=artifact(old_readout/'complete.json'),
                edges=magnitude['details']['cost_only']['edges'], held_features_sha256=array_hash(hx),
                held_ids_sha256=array_hash(v['held_ids']), outer=v['outer'], outer_labels_used=False)
            if verify or (out/'complete.json').exists():
                previous.checked_receipt(out); assert json.loads((out/'models.json').read_text()) == doc
                with np.load(out/'scores.npz', allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved['ids'], v['held_ids'])
                    for key,value in score.items(): np.testing.assert_array_equal(saved[key], value)
            else:
                out.mkdir(parents=True, exist_ok=True); immutable_json(out/'models.json', doc)
                previous.atomic_npz(out/'scores.npz', ids=v['held_ids'], **score)
                immutable_json(out/'complete.json', dict(registration=artifact(PUBLIC/'registration_lock.json'),
                    artifacts=dict(models=artifact(out/'models.json'), scores=artifact(out/'scores.npz'))))
            prediction_refs.append(artifact(out/'complete.json'))
            beat('replica_replayed' if verify else 'replica_frozen', replicas=len(prediction_refs), heads=len(refs), tag=v['tag'])
    assert len(refs) == matches == 864 and len(prediction_refs) == 432
    frozen = dict(registration=artifact(PUBLIC/'registration_lock.json'), heads=refs, predictions=prediction_refs,
        matched_cut_checks=matches, new_updates=1728000, pilot_bridge=artifact(PRIVATE/'pilot_native_bridge'/'complete.json'),
        prediction_lineage_excludes_outer=True, independent_roles_read=False)
    if verify:
        assert bridge_checked
        assert json.loads((PUBLIC/'prediction_freeze.json').read_text()) == frozen
        immutable_json(PUBLIC/'training_replay.json', dict(exact=True, heads=864, replicas=432, native_bridge=True))
    else: immutable_json(PUBLIC/'prediction_freeze.json', frozen)


def check_freeze():
    require_committed(PUBLIC/'prediction_freeze.json')
    doc = json.loads((PUBLIC/'prediction_freeze.json').read_text())
    for ref in doc['heads']+doc['predictions']+[doc['pilot_bridge']]:
        assert artifact(ROOT/ref['path']) == ref; previous.checked_receipt((ROOT/ref['path']).parent)
    return doc


def evaluate(cfg, identity, verify=False):
    check_freeze(); rows = []; old = {r['group']+'_'+r['pair']:r
        for r in json.loads((previous.PUBLIC/'readout.json').read_text())['rows']}
    checks = 0; cached = {}
    for v in views(identity):
        name = v['g']['group']+'_'+v['pair']
        if name not in cached:
            cached = {name:strong.risk.base.pair_inputs(v['g'],v['data'],v['pairs'],v['pair'])[2]}
        full_labels = cached[name]
        _, env, _, _ = v['pairs']['B'][v['pair']]
        cv = v['data']['baseline_ade'][v['held_ids'],1]; he = env[v['te']]
        outer_y = strong.tail.event_targets(full_labels[v['te']], cv, v['pr']['positive_easy_cut'])
        old_fold = next(r for r in old[v['g']['group']+'_'+v['pair']]['folds'] if r['held'] == v['outer'])
        assert array_hash(outer_y) == old_fold['target_sha256']
        replicas = []
        for omitted in sorted(set(v['sites'])):
            out = PRIVATE/'predictions'/v['tag']/omitted; record = json.loads((out/'models.json').read_text())
            targets = dict(outer_cut3=outer_y, inner_cut2_diagnostic=strong.tail.event_targets(
                full_labels[v['te']], cv, record['cuts']['two_cut2']))
            metrics = {}
            with np.load(out/'scores.npz', allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved['ids'], v['held_ids'])
                for label,y in targets.items():
                    metrics[label] = {}
                    for cell in cfg['cells']:
                        for mode in ('raw','scaled'):
                            key = cell+'_'+mode; score = saved[key]
                            m = strong.measure(score, y, he, v['data']['sites'][v['held_ids']], record['edges'][mode])
                            for subset,mask in [('all',np.ones(len(y),bool)),('envelope_positive',he>0)]:
                                use = mask & np.isfinite(y).all(1)
                                np.testing.assert_allclose(m[subset]['component_MSE'], ((score[use]-y[use])**2).mean(0), rtol=1e-13, atol=1e-13)
                                checks += 1
                            metrics[label][key] = m
                    if label == 'outer_cut3':
                        for mode in ('raw','scaled'):
                            for subset in ('all','envelope_positive'):
                                assert metrics[label]['three_cut3_'+mode][subset] == old_fold['metrics']['cost_only_'+mode][subset]
            replicas.append(dict(omitted=omitted,metrics=metrics,cuts=record['cuts']))
        rows.append(dict(group=v['g']['group'],producer=v['g']['producer'],controller=v['g']['controller'],
            seed=previous.seed_of(v),pair=v['pair'],held=v['outer'],replicas=replicas,
            target_sha256=array_hash(outer_y),held_ids_sha256=array_hash(v['held_ids'])))
        beat('held_scored', views=len(rows), tag=v['tag'])
    assert len(rows) == 144 and checks == 13824
    doc = dict(rows=rows,direct_MSE_checks=checks,result_source='fresh_run_exposed_source_development',
        independent_roles_read=False,new_policy_evaluated=False)
    if verify:
        assert json.loads((PUBLIC/'readout.json').read_text()) == doc
        immutable_json(PUBLIC/'evaluation_replay.json',dict(exact=True,views=144,replicas=432,direct_MSE_checks=checks))
    else: immutable_json(PUBLIC/'readout.json',doc)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--phase',required=True,
        choices=['register','support','pilot','train','evaluate','verify_training','verify_eval'])
    parser.add_argument('--resume',action='store_true'); args = parser.parse_args()
    PUBLIC.mkdir(parents=True,exist_ok=True); PRIVATE.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    with (PRIVATE/'lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); cfg,identity = registration(args.phase == 'register')
        if args.phase == 'support': support(cfg,identity)
        elif args.phase in ('pilot','train','verify_training'):
            train(cfg,identity,args.phase == 'pilot',args.resume,args.phase == 'verify_training')
        elif args.phase in ('evaluate','verify_eval'): evaluate(cfg,identity,args.phase == 'verify_eval')


if __name__ == '__main__': main()
