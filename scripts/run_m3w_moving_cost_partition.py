"""Fixed source-held risk partitions with original-mean readout and no new test."""
import argparse
from pathlib import Path
import platform
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 .venv-pytorch required')
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_peer_path_probe as previous
from src.evaluation import m3w_cost_partition_probe as method

source, rel, np = previous.source, previous.rel, previous.np
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/moving_cost_partition_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/moving_cost_partition_v1'
CONFIG = ROOT/'configs/m3w_moving_cost_partition_v1.json'; REG = PUBLIC/'registration_v2.json'
FILES = (Path(__file__), CONFIG, PUBLIC/'protocol.md', ROOT/'src/evaluation/m3w_cost_partition_probe.py',
    ROOT/'tests/test_m3w_cost_partition_probe.py', PUBLIC/'preflight_amendment.md')


def register(beat):
    cfg = source.read(CONFIG); _, values = previous.register(beat)
    if (cfg['arms'] != list(method.ARMS) or cfg['features'] != 'summary120' or cfg['tree'] != method.parent.TREE
            or cfg['primary'] != 'cost_partition_minus_mse_partition' or cfg['minimum_passing_contexts'] != 2
            or cfg['internal_screen_minimum_cost_reduction'] != .01
            or cfg['risk_budget'] != .02 or cfg['easy_net_budget'] != .02
            or any(cfg[k] for k in ('threshold_search', 'new_H_readout', 'identity_certification',
                'deployment_changed', 'submission_ready', 'stage5c_executed', 'smc_enabled'))):
        raise ValueError('Fixed model comparison and unchanged scientific rules required')
    verification = source.read(previous.PUBLIC/'verification.json')
    rel.base.exact(verification['analysis_sha256'], source.sha(previous.PUBLIC/'analysis.json'))
    source.once(REG, source.stable(dict(config=cfg, sklearn_version=previous.sklearn.__version__,
        files={str(p.relative_to(ROOT)):source.sha(p) for p in FILES},
        parent_registration=source.artifact(previous.REG), parent_analysis=source.artifact(previous.PUBLIC/'analysis.json'),
        parent_verification=source.artifact(previous.PUBLIC/'verification.json'), jobs=values[3],
        partitions={j['key']:values[5][j['key']] for j in values[3]},
        fresh_fits=16, cached_controls=8, primary='cost_partition_minus_mse_partition',
        original_rules_unchanged=True, identity_certification=False, new_V_or_H=False)))
    return cfg, values


def cached_control(d, job, site, *, refit=False):
    name = job['key']+'_'+str(site)+'_extra_trees_summary120.json'
    ref = source.read(previous.PUBLIC/'fits'/name); saved = rel.base.read_checkpoint(source.checked(ref['checkpoint']))
    rel.base.exact(saved['identity']['feature_hash'], rel.parent.kin.digest(d['xs']['summary120']))
    rel.base.exact(saved['identity']['target_hash'], rel.parent.kin.digest(d['target']))
    train, held = d['sites'] != site, d['sites'] == site
    rel.base.exact(saved['identity']['train_ids_hash'], rel.parent.kin.digest(d['ids'][train]))
    rel.base.exact(saved['identity']['held_ids_hash'], rel.parent.kin.digest(d['ids'][held]))
    pred = method.parent.predict(saved['state'], d['xs']['summary120'][held]); rel.base.exact(pred, saved['pred'])
    if refit:
        p, model = method.parent.fit_predict(d['xs']['summary120'], d['target'], train, held, d['weights'], 'extra_trees')
        rel.base.exact(p, pred); rel.base.exact(method.parent.fingerprint(model), method.parent.fingerprint(saved['state']))
    return dict(saved['state'], arm='mse_partition'), ref['checkpoint']


def cell(d, job, site, arm, guard, *, resume=False, verify=False, kind='fits'):
    train, held = d['sites'] != site, d['sites'] == site; x, y, w = d['xs']['summary120'], d['target'], d['weights']
    guard()
    binding = dict(registration_sha256=source.sha(REG), job=job['key'], held_source=str(site), arm=arm,
        input_hash=rel.parent.kin.digest(x), target_hash=rel.parent.kin.digest(y),
        train_ids_hash=rel.parent.kin.digest(d['ids'][train]), held_ids_hash=rel.parent.kin.digest(d['ids'][held]))
    if arm == 'mse_partition':
        state, ref = cached_control(d, job, site, refit=verify); seconds = None
    else:
        path = PRIVATE/kind/(job['key']+'_'+str(site)+'_'+arm+'.pt.gz')
        saved = None
        if path.exists():
            if not resume and not verify:
                raise ValueError('Existing model requires explicit resume')
            saved = rel.base.read_checkpoint(path); rel.base.exact(saved['binding'], binding)
        elif verify:
            raise ValueError('Cannot verify absent model')
        if saved is None or verify:
            start = time.monotonic(); state = method.fit(x, y, train, w, arm)
            fitted = dict(binding=binding, state=state, prediction=method.predict(state, x[held]), seconds=time.monotonic()-start)
            if saved is None:
                path.parent.mkdir(parents=True, exist_ok=True); rel.base.save_checkpoint(path, fitted); saved = fitted
            else:
                rel.base.exact(method.fingerprint(state), method.fingerprint(saved['state']))
                rel.base.exact(fitted['prediction'], saved['prediction'])
        state, seconds, ref = saved['state'], saved['seconds'], source.artifact(path)
    prediction = method.predict(state, x[held])
    if arm != 'mse_partition':
        rel.base.exact(prediction, saved['prediction'])
    record = dict(**binding, checkpoint=ref, seconds=seconds, rows=int(held.sum()),
        result_source='cached_verified' if arm == 'mse_partition' else 'fresh_run',
        prediction_hash=rel.parent.kin.digest(prediction),
        model_fingerprint_sha256=source.hashlib.sha256(source.json.dumps(
            {k:rel.parent.kin.digest(v) for k,v in method.fingerprint(state).items()}, sort_keys=True).encode()).hexdigest(),
        metrics=method.parent.previous.metrics(prediction, y[held], w[held], state['target_scale']),
        fitting_metrics=method.parent.previous.metrics(method.predict(state, x[train]), y[train], w[train], state['target_scale']),
        tree_nodes=sum(t.tree_.node_count for t in state['model'].estimators_),
        decomposition=method.decomposition(state, x, y, train, w))
    return source.stable(record)


def target_diagnosis(d):
    y, w = d['target'], d['weights']; w = w/w.sum(); joint = (y <= 0).all(-1)
    return dict(known_rows=len(y), known_joint_safe_rows=int(joint.sum()),
        weighted_joint_safe_fraction=float(w@joint),
        channel_positive_mass=np.sum(w[:, None]*np.maximum(y, 0), axis=0).tolist(),
        channel_negative_mass=np.sum(w[:, None]*np.maximum(-y, 0), axis=0).tolist(),
        channel_weighted_means=np.sum(w[:, None]*y, axis=0).tolist(),
        zero_joint_rows=int((y == 0).all(-1).sum()),
        descriptive_known_fitting_labels_only=True, not_a_future_input=True)


def pilot(cfg, values, guard, beat, resume):
    job = values[3][-1]; d = previous.inputs(values, job, guard); site = sorted(set(d['sites']))[0]
    a = cell(d, job, site, 'cost_partition', guard, resume=resume, kind='pilot')
    b = cell(d, job, site, 'cost_partition', guard, verify=True, kind='pilot'); rel.base.exact(a, b)
    source.once(PUBLIC/'pilot.json', dict(model=a, full64_tree_repeat_exact=True, local_training_reasonable=True,
        seconds_per_fit=a['seconds'], fresh_fits=2, new_V_or_H=False))
    beat(state='pilot_complete', seconds_per_fit=a['seconds'])


def execute(cfg, values, guard, beat, *, resume=False, verify=False):
    if not source.read(PUBLIC/'pilot.json')['full64_tree_repeat_exact']:
        raise ValueError('Real complete pilot required')
    contexts = []; complete = 0
    for job in values[3]:
        d = previous.inputs(values, job, guard); folds = []
        for site in sorted(set(d['sites'])):
            fold = dict(held_source=str(site), queries=int(len(set(d['query'][d['sites'] == site]))), arms={})
            for arm in method.ARMS:
                r = cell(d, job, site, arm, guard, resume=resume, verify=verify)
                source.once(PUBLIC/'cells'/(job['key']+'_'+str(site)+'_'+arm+'.json'), r)
                fold['arms'][arm] = r; complete += 1
                beat(state='cell_verified' if verify else 'cell_complete', completed=complete, total=24, arm=arm, job=job['key'])
            folds.append(fold)
        contexts.append(dict(**d['audit'], target_diagnosis=target_diagnosis(d), folds=folds, summary=method.summary(folds)))
        del d; source.gc.collect(); guard()
    passes = sum(c['summary']['passes_internal_screen'] for c in contexts)
    result = source.stable(dict(contexts=contexts, cells=complete, fresh_primary_fits=16, cached_controls=8,
        primary_passing_contexts=passes, neural_mechanism_followup_supported=passes >= 2,
        original_rules_unchanged=True, identity_certification=False, new_V_or_H=False, independent_confirmation=False,
        neural_training=False, deployment_changed=False, submission_ready=False, stage5c_executed=False, smc_enabled=False))
    source.once(PUBLIC/'analysis.json', result)
    if verify:
        source.once(PUBLIC/'verification.json', dict(all16_new_full_refits_exact=True, all8_legacy_full_refits_exact=True,
            all24_metric_replays_exact=True, analysis_sha256=source.sha(PUBLIC/'analysis.json')))


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('phase', choices=('register', 'pilot', 'run', 'verify'))
    p.add_argument('--resume', action='store_true'); args = p.parse_args(); PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        source.fcntl.flock(lock, source.fcntl.LOCK_EX | source.fcntl.LOCK_NB)
        runtime = previous.old.run.old.old.previous.teacher.lineage; runtime.PRIVATE = PRIVATE
        start, guard, beat = runtime.runtime({}, source.read(CONFIG), args.phase)
        cfg, values = register(beat)
        if args.phase == 'register':
            return
        for path in (*FILES, REG):
            blob = source.subprocess.run(['git', 'show', 'HEAD:'+str(path.relative_to(ROOT))], cwd=ROOT, capture_output=True, check=True).stdout
            if source.hashlib.sha256(blob).hexdigest() != source.sha(path):
                raise ValueError('Commit code and protocol before fitting')
        receipt = PUBLIC/(args.phase+'_execution.json')
        if receipt.exists():
            raise ValueError('Completed phase is immutable')
        if args.phase == 'pilot':
            pilot(cfg, values, guard, beat, args.resume)
        else:
            execute(cfg, values, guard, beat, resume=args.resume, verify=args.phase == 'verify')
        source.once(receipt, dict(pid=source.os.getpid(), seconds=time.monotonic()-start,
            peak_RSS_bytes=source.resource.getrusage(source.resource.RUSAGE_SELF).ru_maxrss, new_V_or_H=False, neural_training=False))
        beat(state='complete')


if __name__ == '__main__':
    main()
