"""Nested, producer-excluded magnitude fitting; no independent-role access."""
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
    raise RuntimeError('Native arm64 required before Torch import')
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '4')
from scripts import run_m3w_european_aux_prior as previous
from src.world_model import m3w_oof_magnitude as method
from src.world_model import m3w_membership_auxiliary as neural
import numpy as np
import torch

strong = previous.parent; nested = strong.risk.nested
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_oof_magnitude_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_oof_magnitude_v1'
CONFIG = 'configs/m3w_european_oof_magnitude_v1.json'
FILES = [CONFIG, 'src/world_model/m3w_oof_magnitude.py', 'tests/test_m3w_oof_magnitude.py',
    'scripts/run_m3w_european_oof_magnitude.py', 'scripts/report_m3w_european_oof_magnitude.py',
    str(PUBLIC.relative_to(ROOT)/'protocol.md')]
artifact, digest, immutable_json, array_hash = previous.artifact, previous.digest, previous.immutable_json, previous.array_hash
atomic_npz = strong.risk.base.previous.parent.atomic_npz
require_committed = strong.risk.base.previous.require_committed


def beat(state, **kw):
    row = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), state=state, **kw)
    strong.risk.base.base.cross.json_write(PRIVATE/'heartbeat.json', row)
    with (PRIVATE/'events.jsonl').open('a') as handle: handle.write(json.dumps(row)+'\n')
    print(json.dumps(row), flush=True)


def registration(create=False):
    cfg = json.loads((ROOT/CONFIG).read_text()); pcfg, pid = previous.registration()
    seal = previous.checked_seal(previous.PUBLIC/'verification.json')
    doc = json.loads((previous.PUBLIC/'verification.json').read_text())
    for ref in doc['local_detailed_metrics']: assert artifact(ROOT/ref['path']) == ref
    assert cfg['head_training'] == pcfg['head_training'] and cfg['max_slope'] == 8
    assert cfg['arms'] == ['cost_only', 'cap_aux', 'shuffled_aux']
    assert cfg['new_updates'] == (cfg['reference_heads']+cfg['inner_auxiliary_heads'])*2000 == 2016000
    for key, value in pcfg.items():
        if value is False: assert cfg[key] is False
    identity = dict(parent=pid, parent_verification=seal, bindings={p:digest(ROOT/p) for p in FILES})
    path = PUBLIC/'registration_lock.json'
    if create: immutable_json(path, identity)
    else:
        assert json.loads(path.read_text()) == identity; require_committed(path)
    return cfg, identity


def views(identity):
    return strong.views(identity['parent']['parent'])


def seed_of(v):
    return int(v['g']['group'].split('_seed')[1].split('_')[0])


def ref_input(v, site):
    take = v['sites'] == site
    pr = method.reference_preprocess(v['x'][take], v['raw'][take], v['cv'][take],
        v['sites'][take], set(v['sites'])-{site} | {v['outer']})
    y = nested.method.labels(v['raw'][take], v['cv'][take], pr['positive_easy_cut'])
    easy = strong.labels(v['cv'][take], pr['positive_easy_cut'])
    record = dict(group=v['g']['group'], pair=v['pair'], training_site=site,
        forecast_producer_sites=v['g']['producer_roster'], seed=seed_of(v),
        ids_sha256=array_hash(v['ids'][take]), x_sha256=array_hash(v['x'][take]),
        target_sha256=array_hash(y), cv_sha256=array_hash(v['cv'][take]),
        envelope_sha256=array_hash(v['env'][take]), cut=pr['positive_easy_cut'])
    return take, pr, y, easy, record


def reference_home(v, site):
    return PRIVATE/'references'/(v['g']['group']+'_'+v['pair'])/site


def support(cfg, identity):
    refs = {}; inners = []
    for v in views(identity):
        for site in sorted(set(v['sites'])):
            take, pr, y, easy, record = ref_input(v, site)
            key = str(reference_home(v, site).relative_to(PRIVATE))
            known = pr['known']; row = dict(input=record, known=int(known.sum()),
                easy=int(np.nansum(easy)), easy_harm=int(np.sum(y[known, 3] > 0)),
                supported=bool(0 < np.nansum(easy) < known.sum() and (v['env'][take][known] > 0).any()))
            if key in refs: assert refs[key] == row
            refs[key] = row
        for inner in sorted(set(v['sites'])):
            take, pr, y, easy = nested.method.inner_inputs(v['x'], v['raw'], v['cv'], v['sites'], v['outer'], inner)
            method.check_lineage([inner], pr['training_sites'], v['outer'], v['g']['producer_roster'])
            inp = nested.input_record(v, inner, take, pr, y)
            old = json.loads((nested.PRIVATE/'heads'/v['tag']/inner/'complete.json').read_text())
            assert old['input'] == inp
            inners.append(dict(tag=v['tag'], inner=inner, input=inp, cached_control=artifact(
                nested.PRIVATE/'heads'/v['tag']/inner/'complete.json'), known=int(pr['known'].sum()),
                supported=bool(0 < np.nansum(easy) < pr['known'].sum())))
        beat('support', views=len(inners)//3, unique_references=len(refs))
    assert len(refs) == 144 and len(inners) == 432
    doc = dict(registration=artifact(PUBLIC/'registration_lock.json'), references=refs, inners=inners,
        training_allowed=all(r['supported'] for r in list(refs.values())+inners),
        outer_labels_used=False, held_inner_labels_used=False, statistical_power_established=False)
    immutable_json(PUBLIC/'support_report.json', doc)
    (PUBLIC/'support_report.md').write_text('# Nested Fitting Support\n\n'
        f"144 exact-deduplicated single-site references;432 two-site controller fits. Allowed: {doc['training_allowed']}.\n\n"
        f"Minimum single-site supported rows {min(r['known'] for r in refs.values())}; "
        f"easy-harm rows {min(r['easy_harm'] for r in refs.values())}. "
        'Numerical support only, not independent sample size or statistical power.\n'
        'No new training or outer source-held readout yet. Independent roles remain closed.\n')


def checked_receipt(home):
    doc = json.loads((home/'complete.json').read_text())
    for ref in doc['artifacts'].values(): assert artifact(ROOT/ref['path']) == ref
    return doc


def room():
    if shutil.disk_usage(PRIVATE).free < 10*2**30:
        raise OSError('Preserve 10GiB. Existing checkpoints retained for resume; no deletion of old assets.')


def reference(v, site, cfg, resume, verify):
    take, pr, y, easy, record = ref_input(v, site); home = reference_home(v, site)
    hid = dict(registration=artifact(PUBLIC/'registration_lock.json'), input=record)
    if (home/'complete.json').exists():
        receipt = checked_receipt(home); assert receipt['identity'] == hid
        model, state = method.restore(home)
    else:
        if verify: raise ValueError('Reference incomplete')
        room()
        model, fit = neural.fit(v['x'][take], y, easy, v['sites'][take], v['env'][take], pr,
            arm='cost_only', seed=seed_of(v), settings=cfg['head_training'], identity=hid,
            directory=home, resume=resume, heartbeat=lambda **kw:beat(reference=str(home.relative_to(PRIVATE)), **kw))
        checkpoint = method.compress_checkpoint(home); model, state = method.restore(home)
        immutable_json(home/'complete.json', dict(identity=hid, fit=fit, result_source='fresh_run_native_torch',
            artifacts=dict(checkpoint=artifact(checkpoint))))
    assert state['identity'] == hid and state['step'] == 2000 and state['arm'] == 'cost_only'
    for key in ('mean', 'std', 'known', 'weights'): np.testing.assert_array_equal(state['preprocess'][key], pr[key])
    assert state['preprocess']['training_sites'] == [site]
    assert state['preprocess']['positive_easy_cut'] == pr['positive_easy_cut']
    assert not state['draws'][~pr['known']].any()
    return model, pr, artifact(home/'complete.json')


def inner_event(v, inner, take, y, refs):
    fit_sites = v['sites'][take]; prediction = np.zeros((take.sum(), 4)); lineage = []
    for destination in sorted(set(fit_sites)):
        source = next(s for s in set(fit_sites) if s != destination)
        model, pr, receipt = refs[source]
        method.check_lineage([destination, inner], pr['training_sites'], v['outer'], v['g']['producer_roster'])
        loc = fit_sites == destination
        prediction[loc], _ = neural.predict(model, v['x'][take][loc], v['env'][take][loc], pr)
        lineage.append(dict(prediction_site=destination, reference_training_sites=pr['training_sites'], reference=receipt))
    event = strong.cap.method.event_target(y, prediction, v['env'][take])
    return event, lineage, array_hash(prediction)


def train(cfg, identity, pilot=False, resume=False, verify=False):
    doc = json.loads((PUBLIC/'support_report.json').read_text()); assert doc['training_allowed']
    assert doc['registration'] == artifact(PUBLIC/'registration_lock.json'); require_committed(PUBLIC/'support_report.json')
    nested.checked_training(); strong.check_sources(identity['parent']['parent'])
    receipts = []; reference_receipts = {}; started = time.monotonic()
    for v in views(identity):
        refs = {s:reference(v, s, cfg, resume, verify) for s in sorted(set(v['sites']))}
        for _, _, ref in refs.values(): reference_receipts[ref['path']] = ref
        for inner in sorted(set(v['sites'])):
            take, pr, y, easy = nested.method.inner_inputs(v['x'], v['raw'], v['cv'], v['sites'], v['outer'], inner)
            event, lineage, refhash = inner_event(v, inner, take, y, refs)
            held = ~take
            inp = dict(original=nested.input_record(v, inner, take, pr, y), lineage=lineage,
                reference_scores_sha256=refhash, event_sha256=array_hash(event),
                score_ids_sha256=array_hash(v['ids'][held]), event_positive=int(np.nansum(event)),
                event_known=int(np.isfinite(event).sum()), entire_lineage_excludes=[v['outer'], inner])
            control_home = nested.PRIVATE/'heads'/v['tag']/inner
            control, cs = neural.restore(control_home); checked_receipt(control_home)
            assert cs['step'] == 2000 and cs['settings'] == cfg['head_training'] and cs['seed'] == seed_of(v)
            assert cs['arm'] == 'cost_only'
            for arm in ('cap_aux', 'shuffled_aux'):
                home = PRIVATE/'inner'/v['tag']/inner/arm
                hid = dict(registration=artifact(PUBLIC/'registration_lock.json'), input=inp, arm=arm, seed=seed_of(v))
                if (home/'complete.json').exists():
                    receipt = checked_receipt(home); assert receipt['identity'] == hid
                    model, state = method.restore(home); fit = receipt['fit']
                else:
                    if verify: raise ValueError('Inner auxiliary incomplete')
                    room()
                    model, fit = previous.method.fit(v['x'][take], y, easy, event, v['sites'][take],
                        inner, v['env'][take], pr, arm=arm, seed=seed_of(v), settings=cfg['head_training'],
                        identity=hid, directory=home, resume=resume, stop_at=200 if pilot else None,
                        heartbeat=lambda **kw:beat(tag=v['tag'], inner=inner, arm=arm, **kw))
                    if pilot:
                        immutable_json(PRIVATE/'pilot.json', dict(fit=fit, checkpoint=artifact(home/'checkpoint.pt'),
                            reference_heads_completed=len(reference_receipts), pilot_auxiliary_steps=200,
                            projected_auxiliary_fit_seconds=fit['seconds']/200*cfg['inner_auxiliary_heads']*2000,
                            available_GiB=shutil.disk_usage(PRIVATE).free/2**30,
                            wall_seconds_excluding_ancestry_preflight=time.monotonic()-started))
                        beat('pilot_complete'); return
                    method.compress_checkpoint(home); model, state = method.restore(home)
                assert state['step'] == 2000 and state['identity'] == hid
                strong.match_state(state, cs)
                np.testing.assert_array_equal(state['auxiliary_target'], previous.method.auxiliary_target(event, v['sites'][take], arm, seed_of(v)))
                assert not state['draws'][~pr['known']].any()
                score, prob = neural.predict(model, v['x'][held], v['env'][held], pr)
                if (home/'complete.json').exists():
                    with np.load(home/'scores.npz', allow_pickle=False) as z:
                        np.testing.assert_array_equal(z['ids'], v['ids'][held]); np.testing.assert_array_equal(z['scores'], score)
                        np.testing.assert_array_equal(z['probability'], prob)
                else:
                    atomic_npz(home/'scores.npz', ids=v['ids'][held], scores=score, probability=prob)
                    immutable_json(home/'complete.json', dict(identity=hid, fit=fit,
                        result_source='fresh_run_native_torch', matched_cost_control=artifact(control_home/'complete.json'),
                        artifacts=dict(checkpoint=artifact(home/'checkpoint.pt.gz'), scores=artifact(home/'scores.npz'))))
                receipts.append(artifact(home/'complete.json'))
                beat('inner_replayed' if verify else 'inner_frozen', completed=len(receipts), tag=v['tag'], inner=inner, arm=arm)
    assert len(receipts) == 864 and len(reference_receipts) == 144
    result = dict(registration=artifact(PUBLIC/'registration_lock.json'), references=list(reference_receipts.values()),
        inner_heads=receipts, new_heads=1008, new_updates=2016000, cached_controls=432,
        unknown_draws=0, inner_and_outer_labels_excluded_from_fit=True)
    if verify:
        assert json.loads((PUBLIC/'inner_prediction_freeze.json').read_text()) == result
        immutable_json(PUBLIC/'training_replay.json', dict(exact=True, references=144, inner_heads=864))
    else: immutable_json(PUBLIC/'inner_prediction_freeze.json', result)
    beat('training_complete', phase_seconds_excluding_ancestry_preflight=time.monotonic()-started)


def checked_inner():
    path = PUBLIC/'inner_prediction_freeze.json'; require_committed(path)
    doc = json.loads(path.read_text()); assert doc['new_heads'] == 1008
    for ref in doc['references']+doc['inner_heads']:
        assert artifact(ROOT/ref['path']) == ref; checked_receipt((ROOT/ref['path']).parent)


def oof_bank(v, arm):
    prediction = np.zeros((len(v['ids']), 4)); target = prediction.copy(); cuts = np.zeros(len(target)); refs = []
    for inner in sorted(set(v['sites'])):
        take, pr, _, _ = nested.method.inner_inputs(v['x'], v['raw'], v['cv'], v['sites'], v['outer'], inner)
        held = ~take
        home = (nested.PRIVATE/'heads'/v['tag']/inner if arm == 'cost_only'
                else PRIVATE/'inner'/v['tag']/inner/arm)
        receipt = checked_receipt(home); refs.append(artifact(home/'complete.json'))
        with np.load(home/'scores.npz', allow_pickle=False) as z:
            if arm == 'cost_only':
                np.testing.assert_array_equal(z['ids'], v['ids']); prediction[held] = z['scores'][held]
            else:
                np.testing.assert_array_equal(z['ids'], v['ids'][held]); prediction[held] = z['scores']
        target[held] = nested.method.labels(v['raw'][held], v['cv'][held], pr['positive_easy_cut'])
        cuts[held] = pr['positive_easy_cut']
    return prediction, target, cuts, refs


def readouts(cfg, identity, verify=False):
    checked_inner(); previous.check_freeze(); receipts = []
    for v in views(identity):
        out = PRIVATE/'readouts'/v['tag']; models = {}; scores = {}; details = {}
        _, _, _, _, _, frozen, env, _, _, _ = strong.inputs(v)
        target_hash = None
        for arm in cfg['arms']:
            p, y, cuts, refs = oof_bank(v, arm)
            if target_hash is None: target_hash = array_hash(y)
            assert target_hash == array_hash(y)
            model = method.fit_magnitude(p, y, v['env'], v['sites'], v['outer'], max_slope=cfg['max_slope'])
            home = (strong.PRIVATE if arm == 'cost_only' else previous.PRIVATE)/'heads'/v['tag']/arm
            checked_receipt(home)
            with np.load(home/'scores.npz', allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'], v['held_ids']); raw = z['scores'].copy()
            np.testing.assert_array_equal(raw[:, [0, 2]], frozen[:, [0, 2]])
            scores[arm+'_raw'] = raw; scores[arm+'_scaled'] = method.predict_magnitude(model, raw, env)
            fitted = method.predict_magnitude(model, p, v['env'])
            edges = strong.parent.edges_for(fitted, v['env'], v['pr'], cfg)
            raw_edges = strong.parent.edges_for(p, v['env'], v['pr'], cfg)
            models[arm] = model
            details[arm] = dict(oof_score_sha256=array_hash(p), target_sha256=target_hash, references=refs,
                cut_drift=nested.method.cut_drift(v['cv'], cuts, v['pr']['positive_easy_cut']),
                edges=dict(raw=raw_edges, scaled=edges),
                fitting_raw=strong.measure(p, y, v['env'], v['sites'], raw_edges),
                fitting_scaled=strong.measure(fitted, y, v['env'], v['sites'], edges),
                outer_original=artifact(home/'complete.json'))
        record = dict(models=models, details=details, training_ids_sha256=array_hash(v['ids']),
            held_ids_sha256=array_hash(v['held_ids']), fitting_sites=sorted(set(v['sites'])), outer=v['outer'],
            outer_labels_used=False, thresholds_searched=False)
        if verify:
            assert json.loads((out/'models.json').read_text()) == record
            with np.load(out/'scores.npz', allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'], v['held_ids'])
                for key, value in scores.items(): np.testing.assert_array_equal(z[key], value)
        else:
            immutable_json(out/'models.json', record); atomic_npz(out/'scores.npz', ids=v['held_ids'], **scores)
            immutable_json(out/'complete.json', dict(registration=artifact(PUBLIC/'registration_lock.json'),
                artifacts=dict(models=artifact(out/'models.json'), scores=artifact(out/'scores.npz'))))
        receipts.append(artifact(out/'complete.json')); beat('readout_frozen', views=len(receipts), tag=v['tag'])
    assert len(receipts) == 144
    doc = dict(registration=artifact(PUBLIC/'registration_lock.json'), receipts=receipts, fitted_magnitude_heads=432,
        parameters_per_head=2, outer_labels_used=False, independent_roles_read=False)
    if verify:
        assert json.loads((PUBLIC/'prediction_freeze.json').read_text()) == doc
        immutable_json(PUBLIC/'magnitude_replay.json', dict(exact=True, views=144, heads=432))
    else: immutable_json(PUBLIC/'prediction_freeze.json', doc)


def check_freeze():
    path = PUBLIC/'prediction_freeze.json'; require_committed(path)
    doc = json.loads(path.read_text()); assert doc['fitted_magnitude_heads'] == 432
    for ref in doc['receipts']:
        assert artifact(ROOT/ref['path']) == ref; checked_receipt((ROOT/ref['path']).parent)


def evaluate(cfg, identity, verify=False):
    check_freeze(); groups = {}; cached = {}; direct = 0
    old = {r['group']+'_'+r['pair']:r for r in json.loads((previous.PUBLIC/'readout.json').read_text())['rows']}
    for v in views(identity):
        _, _, _, _, _, _, env, _, _, _ = strong.inputs(v)
        name = v['g']['group']+'_'+v['pair']
        if name not in cached: cached = {name:strong.risk.base.pair_inputs(v['g'], v['data'], v['pairs'], v['pair'])[2]}
        y = strong.tail.event_targets(cached[name][v['te']], v['data']['baseline_ade'][v['held_ids'], 1], v['pr']['positive_easy_cut'])
        old_fold = next(f for f in old[name]['folds'] if f['held'] == v['outer'])
        assert old_fold['target_sha256'] == array_hash(y) and old_fold['held_ids_sha256'] == array_hash(v['held_ids'])
        home = PRIVATE/'readouts'/v['tag']; record = json.loads((home/'models.json').read_text()); metrics = {}
        with np.load(home/'scores.npz', allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], v['held_ids'])
            for arm in cfg['arms']:
                for mode in ('raw', 'scaled'):
                    key = arm+'_'+mode; score = z[key]
                    metrics[key] = strong.measure(score, y, env, v['data']['sites'][v['held_ids']], record['details'][arm]['edges'][mode])
                    for subset, mask in [('all', np.ones(len(y), bool)), ('envelope_positive', env > 0)]:
                        known = np.isfinite(y).all(1) & mask
                        np.testing.assert_allclose(metrics[key][subset]['component_MSE'], ((score[known]-y[known])**2).mean(0), rtol=1e-13, atol=1e-13)
                        direct += 1
        group = groups.setdefault(name, dict(group=v['g']['group'], producer=v['g']['producer'],
            controller=v['g']['controller'], seed=seed_of(v), pair=v['pair'], folds=[]))
        group['folds'].append(dict(held=v['outer'], metrics=metrics, models=record['models'],
            cut_drift=record['details']['cost_only']['cut_drift'], target_sha256=array_hash(y), held_ids_sha256=array_hash(v['held_ids'])))
        beat('source_held_readout', views=sum(len(g['folds']) for g in groups.values()), tag=v['tag'])
    assert direct == 1728 and len(groups) == 36
    doc = dict(rows=list(groups.values()), direct_MSE_checks=direct,
        result_source='fresh_run_exposed_source_development', independent_roles_read=False, policy_evaluated=False)
    if verify:
        assert json.loads((PUBLIC/'readout.json').read_text()) == doc
        immutable_json(PUBLIC/'evaluation_replay.json', dict(exact=True, views=144, direct_MSE_checks=direct))
    else: immutable_json(PUBLIC/'readout.json', doc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', required=True, choices=['register', 'support', 'pilot', 'train', 'readouts',
        'evaluate', 'verify_training', 'verify_readouts', 'verify_eval'])
    parser.add_argument('--resume', action='store_true'); args = parser.parse_args()
    PUBLIC.mkdir(parents=True, exist_ok=True); PRIVATE.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    with (PRIVATE/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB); cfg, identity = registration(args.phase == 'register')
        if args.phase == 'support': support(cfg, identity)
        elif args.phase in ('pilot', 'train', 'verify_training'):
            train(cfg, identity, args.phase == 'pilot', args.resume, args.phase == 'verify_training')
        elif args.phase in ('readouts', 'verify_readouts'): readouts(cfg, identity, args.phase == 'verify_readouts')
        elif args.phase in ('evaluate', 'verify_eval'): evaluate(cfg, identity, args.phase == 'verify_eval')


if __name__ == '__main__': main()
