"""Hash-frozen readout after all matched heads finish; missing controls stop it."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Use native arm64 before numerical imports')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
from scripts import run_m3w_temporal_auxiliary as training
from scripts import run_m3w_leaf_quality_extension as controls
from scripts import verify_m3w_temporal_auxiliary_readout as scalar
from src.world_model import m3w_temporal_auxiliary_readout as api

PUBLIC, PRIVATE = training.PUBLIC/'readout', training.PRIVATE/'readout'
sha, once, parent = training.sha, training.once, training.parent


def registration():
    cfg, reg = training.registration()
    assert reg == json.loads((training.PUBLIC/'registration.json').read_text())
    assert controls.registration() == json.loads((controls.PUBLIC/'registration.json').read_text())
    rows = controls.read_rows(controls.diagnostic.fitted_run.PUBLIC)
    expected = [{k: r[k] for k in ('group', 'source', 'head_seed')} for _, r in sorted(rows.items())]
    paths = parent.forest.closure(ROOT, ['scripts.run_m3w_temporal_auxiliary_readout'])
    paths += [PUBLIC/'protocol.md', ROOT/'tests/test_m3w_temporal_auxiliary_readout.py',
              ROOT/'tests/test_m3w_temporal_auxiliary_readout_runner.py']
    return cfg, dict(bindings={str(p.relative_to(ROOT)): sha(p) for p in paths},
        training_registration_sha256=sha(training.PUBLIC/'registration.json'),
        strong_control_registration_sha256=sha(controls.PUBLIC/'registration.json'),
        expected_source_heads=expected, controls=list(api.ARMS), independent_roles_read=False,
        checkpoint_selection=False, threshold_search=False, risk_budget=cfg['risk_budget'])


def training_documents(root, home, expected, cfg):
    """Validate the entire final manifest before any development predictions."""
    freeze = json.loads((home/'training_freeze.json').read_text())
    if (freeze['neural_fits'] != cfg['neural_fits'] or freeze['validation_labels_scored'] is not False
            or freeze['independent_roles_read'] is not False or freeze['deployment_changed'] is not False):
        raise ValueError('Complete unscored fixed-final training freeze required')
    wanted = {(r['group'], r['source'], r['head_seed'], arm) for r in expected for arm in training.api.ARMS}
    found = {}
    for ref in freeze['fits']:
        path = (root/ref['path']).resolve()
        if not path.is_relative_to(home.resolve()/'fits') or sha(path) != ref['sha256']:
            raise ValueError('Changed or misplaced training receipt')
        row = json.loads(path.read_text()); identity = row['identity']; cp = row['checkpoint']
        key = (identity['group'], identity['source'], identity['seed'], row['arm'])
        if key not in wanted or key in found or row['step'] != cfg['head_training']['steps']:
            raise ValueError('Missing, duplicate or partial fixed-final head')
        if row['validation_labels_scored'] is not False:
            raise ValueError('Training receipt already scored validation')
        cp_path = (root/cp['path']).resolve()
        private = root/'data/stage_cvpr2027_experiments'/training.NAME/'heads'
        if not cp_path.is_relative_to(private.resolve()) or cp_path.stat().st_size != cp['bytes'] or sha(cp_path) != cp['sha256']:
            raise ValueError('Changed or misplaced neural checkpoint')
        found[key] = row
    if set(found) != wanted or len(found) != cfg['neural_fits']:
        raise ValueError('All 216 matched heads must finish before readout')
    return found


def admission(cfg, reg):
    result = dict(training_status='not_run_missing_training_freeze', neural_fits_verified=0,
        new_validation_predictions_read=False, readout_allowed=False,
        controls_status='metadata_hash_verified_remote_payloads_not_probed',
        local_training_storage=training.storage(cfg))
    if (training.PUBLIC/'training_freeze.json').exists():
        documents = training_documents(ROOT, training.PUBLIC, reg['expected_source_heads'], cfg)
        result.update(training_status='cached_verified_complete_manifest', neural_fits_verified=len(documents),
            readout_allowed=True, controls_status='payload_verification_required_before_each_group')
    return result


def frozen_controls(state, source, anchor, po, ad, readers, site, seed, partition, tid, vid, x, env, y, q):
    cost_reader, poisson_reader, add_reader = readers
    h = parent.base.inter.array_hash
    assert source['partition'] == partition == anchor['partition'] == po['partition']
    assert h(tid) == anchor['training_ids_hash'] and h(vid) == anchor['validation_ids_hash']
    assert h(y) == anchor['targets_hash'] and h(q[vid]) == anchor['past_quality_hash']
    fitted, meta = cost_reader.fetch(anchor['checkpoint'])
    fp, mp = poisson_reader.fetch(po['checkpoint'])
    for m, a, directory in ((meta, anchor, controls.diagnostic.fitted_run.PUBLIC), (mp, po, controls.prior.PUBLIC)):
        assert m['fit'] == a['training']
        assert m['identity'] == dict(group=anchor['group'], source=site, head_seed=seed, partition=partition,
            checkpoint=source['checkpoint'], training_ids_hash=h(tid), registration_sha256=sha(directory/'registration.json'))
        assert api.forest.fingerprint(q[tid]) == m['fit']['past_quality_hash']
    op, add, support, _, _, _ = controls.prior.additive_control(state, add_reader, ad, source, partition, tid, vid, x, env, q[vid])
    positive = controls.api.diagnostic.model.positive
    _, pp, sp, _ = positive.predict(state, fp, x, env, q[vid])
    _, cp, sc, _ = positive.predict(state, fitted, x, env, q[vid])
    np.testing.assert_array_equal(sp, support); np.testing.assert_array_equal(sc, support)
    pred = dict(original=op, additive=add, poisson=pp, cost=cp)
    for name in api.CONTROLS:
        assert h(pred[name]) == anchor['prediction_hashes'][name]
    return pred, support


def beat(**kw):
    row = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **kw)
    (PRIVATE/'heartbeat.json').write_text(json.dumps(row)+'\n')
    with (PRIVATE/'events.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
    print(json.dumps(row), flush=True)


def run(cfg, reg, documents, *, resume, verify):
    start = time.monotonic(); refs, rows = [], []; count = fetched = 0
    source_docs = parent.parent.docs()
    cost_docs = controls.read_rows(controls.diagnostic.fitted_run.PUBLIC)
    poisson_docs = controls.read_rows(controls.prior.PUBLIC)
    additive_docs = controls.prior.prior_rows()
    # Check connectivity/owned streams before loading the scientific arrays.
    with controls.FrozenReader(controls.diagnostic.SERVER) as cr, controls.FrozenReader(controls.reader.SERVER) as pr, controls.FrozenReader(controls.prior.previous.SERVER) as ar:
        beat(state='owned_read_only_strong_controls_connected')
        _, _, data, jobs, oid, _, _, _ = parent.inner.old.load()
        q, provenance = controls.prior.prior.past_quality(data)
        assert provenance == json.loads((controls.prior.prior.PUBLIC/'feature_provenance.json').read_text())
        for c in parent.parent.contexts(data, jobs, oid):
            for site in parent.inner.sources(c):
                group = c['name']+'_fit_'+site
                at, ids, x, env, y, _, upstream = parent.inner.training_arrays(c, data, site)
                tr, val, partition = parent.forest.parent.api.source_partition(data['recordings'][ids], data['frames'][ids], site)
                tid, vid = ids[tr], ids[val]
                assert not set(data['recordings'][tid]) & set(data['recordings'][vid])
                temporal, _ = training.previous.api.targets(c['floor'][at][tr].astype(float)+data['origin'][tid, None],
                    c['prediction'][at][tr].astype(float)+data['origin'][tid, None], data['target_eval'][tid], data['valid'][tid])
                input_hashes = {k: api.forest.fingerprint(np.asarray(a)) for k, a in dict(x=x[tr], envelope=env[tr],
                    primary=y[tr], temporal=temporal, sites=data['sites'][tid], recordings=data['recordings'][tid], frames=data['frames'][tid]).items()}
                preprocessing = api.forest.core.preprocess(x[tr], env[tr], y[tr], data['sites'][tid], data['recordings'][tid], data['frames'][tid], training_site=site)
                for seed in cfg['head_seeds']:
                    if time.monotonic()-start > cfg['hard_runtime_limit_seconds']:
                        raise TimeoutError('12-hour bound; verified group outputs are resumable')
                    beat(state='group_readout', group=group, head_seed=seed)
                    source, anchor = source_docs[group, seed], cost_docs[group, seed]
                    assert sha(ROOT/source['checkpoint']['path']) == source['checkpoint']['sha256']
                    state = joblib.load(ROOT/source['checkpoint']['path'])
                    assert state['identity']['upstream'] == upstream
                    api.forest.core.exact(preprocessing, state['preprocess'])
                    trained = []
                    for arm in training.api.ARMS:
                        doc = documents[group, site, seed, arm]
                        model = training.api.core.read_checkpoint(ROOT/doc['checkpoint']['path'])
                        expected_identity = dict(group=group, source=site, seed=seed, upstream=upstream,
                            partition=partition, training_ids_sha256=api.forest.fingerprint(tid),
                            source_checkpoint=source['checkpoint'], registration_sha256=sha(training.PUBLIC/'registration.json'))
                        for k, v in dict(identity=expected_identity, step=cfg['head_training']['steps'],
                            settings=cfg['head_training'], preprocess=preprocessing, arm=arm, seed=seed, input_hashes=input_hashes).items():
                            training.api.core.exact(model[k], v)
                        assert doc['input_hashes'] == input_hashes and doc['draw_hash'] == model['draw_hash']
                        trained.append(model)
                    training.api.assert_matched(trained)
                    pred, support = frozen_controls(state, source, anchor, poisson_docs[group, seed], additive_docs[group, seed],
                        (cr, pr, ar), site, seed, partition, tid, vid, x[val], env[val], y[val], q)
                    fetched += sum(anchor[k]['bytes'] for k in ('checkpoint', 'poisson_checkpoint', 'additive_checkpoint'))
                    moving = c['moving'][at][val]
                    old, old_actions = controls.api.diagnostic.model.evaluate(state=state, predictions=pred,
                        targets=y[val], envelope=env[val], moving=moving, support=support,
                        sites=data['sites'][vid], recordings=data['recordings'][vid], frames=data['frames'][vid], ids=vid)
                    assert old == anchor['result']
                    for k, a in old_actions.items(): assert parent.base.inter.array_hash(a) == anchor['action_hashes'][k]
                    for arm, model in zip(training.api.ARMS, trained):
                        pred[arm], _, sp = training.api.predict(model, x[val], env[val])
                        np.testing.assert_array_equal(support, sp)
                        replay, _, sr = training.api.predict(model, x[val], env[val])
                        np.testing.assert_array_equal(pred[arm], replay); np.testing.assert_array_equal(sr, support)
                    args = (preprocessing, pred, y[val], env[val], moving, support, data['sites'][vid], data['recordings'][vid], data['frames'][vid], vid)
                    result, actions = api.evaluate(*args)
                    checks = scalar.verify(*args, result, actions); count += checks
                    h = parent.base.inter.array_hash
                    row = dict(group=group, source=site, head_seed=seed, partition=partition, result=result,
                        independent_scalar_checks=checks, exact_inference_replay=True, strong_control_replay_exact=True,
                        training_ids_hash=h(tid), validation_ids_hash=h(vid), targets_hash=h(y[val]),
                        prediction_hashes={k: h(v) for k, v in pred.items()}, action_hashes={k: h(v) for k, v in actions.items()},
                        input_hashes=input_hashes, training_freeze_sha256=sha(training.PUBLIC/'training_freeze.json'),
                        readout_registration_sha256=sha(PUBLIC/'registration.json'),
                        result_source='fresh_run_fixed_final_development_readout', model_source='cached_verified',
                        new_training=False, independent_roles_read=False)
                    dest = PUBLIC/'groups'/(group+'_head'+str(seed)+'.json')
                    if dest.exists() and not (resume or verify): raise RuntimeError('Explicit resume required')
                    if verify: assert dest.exists()
                    once(dest, row); rows.append(row); refs.append(dict(path=str(dest.relative_to(ROOT)), sha256=sha(dest)))
                    beat(state='group_verified', groups=len(rows), seconds=time.monotonic()-start)
    summary = api.summarize(rows, reg['expected_source_heads'], cfg)
    count += scalar.verify_summary(rows, cfg, summary)
    if verify: assert (PUBLIC/'summary.json').exists()
    once(PUBLIC/'summary.json', summary)
    receipt = dict(groups=refs, summary_sha256=sha(PUBLIC/'summary.json'),
        training_freeze_sha256=sha(training.PUBLIC/'training_freeze.json'), independent_scalar_checks=count,
        exact_inference_replay=True, strong_control_replay_exact=True, local_numeric_cache=False,
        seconds=time.monotonic()-start, fetched_checkpoint_bytes=fetched,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024),
        new_training=False, independent_roles_read=False, deployment_changed=False)
    once(PUBLIC/('verification.json' if verify else 'complete.json'), receipt)
    beat(state='complete', verify=verify, seconds=time.monotonic()-start)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['register', 'preflight', 'run', 'verify'])
    p.add_argument('--resume', action='store_true'); args = p.parse_args()
    cfg, reg = registration()
    if args.phase == 'register':
        once(PUBLIC/'registration.json', reg); print('Registered fixed seven-arm readout; no new validation predictions'); return
    assert reg == json.loads((PUBLIC/'registration.json').read_text())
    parent.base.inter.committed(PUBLIC/'registration.json')
    if args.phase == 'preflight':
        value = admission(cfg, reg)
        stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
        once(PUBLIC/('preflight_'+stamp+'.json'), value); print(json.dumps(value, indent=2)); return
    documents = training_documents(ROOT, training.PUBLIC, reg['expected_source_heads'], cfg)
    parent.base.inter.committed(training.PUBLIC/'training_freeze.json')
    if args.phase == 'run': assert not (PUBLIC/'complete.json').exists()
    else: assert (PUBLIC/'complete.json').exists()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    training.api.torch.set_num_threads(cfg['cpu_threads']); training.api.torch.set_num_interop_threads(1)
    with (PRIVATE/'process.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(cfg, reg, documents, resume=args.resume, verify=args.phase == 'verify')


if __name__ == '__main__':
    main()
