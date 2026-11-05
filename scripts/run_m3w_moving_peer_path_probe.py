"""Directional peer-path information probes on exposed own-B sources only."""
import argparse
from pathlib import Path
import platform
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Use native arm64 .venv-pytorch before numerical imports')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import sklearn
from scripts import run_m3w_moving_peer_participation as parent
from scripts import diagnose_m3w_candidate_neighbors as old
from src.world_model import m3w_candidate_peer_paths as features
from src.evaluation import m3w_peer_path_probe as method

source, rel, np = parent.source, old.api, old.np
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/moving_peer_path_probe_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/moving_peer_path_probe_v1'
CONFIG = ROOT/'configs/m3w_moving_peer_path_probe_v1.json'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), CONFIG, PUBLIC/'protocol.md', ROOT/'src/world_model/m3w_candidate_peer_paths.py',
    ROOT/'src/evaluation/m3w_peer_path_probe.py', ROOT/'tests/test_m3w_peer_path_probe.py')


def register(beat):
    cfg = source.read(CONFIG); _, values = parent.register(beat)
    if (cfg['arms'] != list(method.ARMS) or cfg['families'] != list(method.FAMILIES)
            or cfg['primary_family'] != 'extra_trees' or cfg['tree'] != method.TREE
            or cfg['ridge_penalty'] != .01 or cfg['risk_budget'] != .02 or cfg['easy_net_budget'] != .02
            or any(cfg[k] for k in ('threshold_search', 'new_H_readout', 'identity_certification',
                'deployment_changed', 'submission_ready', 'stage5c_executed', 'smc_enabled'))):
        raise ValueError('Fixed source probe and unchanged scientific rules required')
    for public in (parent.PUBLIC, old.PUBLIC):
        rel.base.exact(source.read(public/'verification.json')['analysis_sha256'], source.sha(public/'analysis.json'))
    oldreg = source.read(old.PUBLIC/'registration_v3.json')
    for path, sha in oldreg['files'].items():
        rel.base.exact(source.sha(ROOT/path), sha)
    source.once(REG, source.stable(dict(config=cfg, sklearn_version=sklearn.__version__,
        files={str(f.relative_to(ROOT)):source.sha(f) for f in FILES}, feature_names=list(features.NAMES),
        parent_registration=source.artifact(parent.REG), previous_factorial=source.artifact(parent.PUBLIC/'analysis.json'),
        previous_neighbor_probe=source.artifact(old.PUBLIC/'analysis.json'),
        jobs=values[3], partitions={j['key']:values[5][j['key']] for j in values[3]},
        scientific_role='own_B_source_held_information_diagnostic', new_V_or_H=False,
        primary='extra_trees_aligned180_minus_summary120', identity_certification=False,
        original_rules_unchanged=True, original24group_primary='not_estimable_not_passed',
        deployment_changed=False, submission_ready=False, stage5c_executed=False, smc_enabled=False)))
    return cfg, values


def inputs(values, job, guard):
    data, warm = parent.inputs(values, job); c = data['causal']
    previous = source.read(old.PUBLIC/(job['key']+'.json'))
    rel.base.exact(previous['input_hash'], warm['input_hash'])
    guard(); added = features.chunked(c['x']); shifted, donors = features.misaligned(added, c['groups'], c['ids'])
    summary = old.features.chunked(c['x'])
    common = np.concatenate((rel.parent.kin.motion_basis(c['x']), rel.basis(c['x']), summary), -1)
    if (c['allowed'].sum(1) > 1).any():
        raise ValueError('Only frozen one-proposal supervision allowed')
    row, action = np.nonzero(c['allowed'] & data['observed_future'][:, None])
    target = rel.parent.kin.parent.signed(data['truth_by_arm']['bound_aware']).numpy()[row, action].astype(float)
    query = np.full(len(c['ids']), -1, int)
    for i, ids in enumerate(c['groups']):
        query[ids] = i
    if (query < 0).any():
        raise ValueError('Every causal row must belong to one complete query')
    with np.load(source.checked(previous['saved_probes']), allow_pickle=False) as saved:
        cached = {k:saved[k].copy() for k in saved.files if k != 'all_neighbor_basis'}
        rel.base.exact(saved['all_neighbor_basis'], summary)
        rel.base.exact(saved['ids'], c['ids'][row]); rel.base.exact(saved['actions'], action)
        rel.base.exact(saved['original_proposal'], c['allowed'])
    xs = dict(summary120=common[row, action].astype(float),
        aligned180=np.concatenate((common[row, action], added[row, action]), -1).astype(float),
        misaligned180=np.concatenate((common[row, action], shifted[row, action]), -1).astype(float))
    record = dict(job=job['key'], own_B_rows=len(c['ids']), known_proposed_rows=len(row),
        unknown_proposed_rows=int((c['allowed'] & ~data['observed_future'][:, None]).sum()),
        complete_queries=len(c['groups']), sources=sorted(set(c['sites'])),
        aligned_neighbor_row_fraction=float((summary[:, 0, 0] > 0).mean()),
        shifted_rows=int((donors != np.arange(len(donors))).sum()),
        feature_constructed_before_outcome_filter=True,
        causal_basis_hash=rel.parent.kin.digest(added), donor_hash=rel.parent.kin.digest(donors),
        ids_hash=rel.parent.kin.digest(c['ids'][row]), target_hash=rel.parent.kin.digest(target),
        feature_hashes={k:rel.parent.kin.digest(v) for k,v in xs.items()},
        parent_input_hash=warm['input_hash'], old_probe_ref=source.artifact(old.PUBLIC/(job['key']+'.json')))
    return dict(xs=xs, target=target, sites=c['sites'][row], query=query[row], ids=c['ids'][row],
        weights=method.previous.balanced_weights(c['sites'][row], query[row]), audit=record, cached=cached)


def cell(d, job, site, family, arm, guard, *, resume=False, verify=False, kind='fits'):
    train, held = d['sites'] != site, d['sites'] == site; x = d['xs'][arm]
    identity = dict(job=job['key'], held_source=str(site), family=family, arm=arm,
        registration_sha256=source.sha(REG), feature_hash=rel.parent.kin.digest(x),
        target_hash=rel.parent.kin.digest(d['target']), train_ids_hash=rel.parent.kin.digest(d['ids'][train]),
        held_ids_hash=rel.parent.kin.digest(d['ids'][held]))
    path = PRIVATE/kind/(job['key']+'_'+str(site)+'_'+family+'_'+arm+'.pt.gz')
    guard()
    if path.exists():
        if not resume and not verify:
            raise ValueError('Existing fit requires explicit resume or verification')
        saved = rel.base.read_checkpoint(path); rel.base.exact(saved['identity'], identity)
    elif verify:
        raise ValueError('Cannot verify an absent fit')
    else:
        saved = None
    if saved is None or verify:
        started = time.monotonic()
        pred, state = method.fit_predict(x, d['target'], train, held, d['weights'], family)
        fitted = dict(identity=identity, state=state, pred=pred, seconds=time.monotonic()-started)
        if saved is None:
            path.parent.mkdir(parents=True, exist_ok=True); rel.base.save_checkpoint(path, fitted); saved = fitted
        else:
            rel.base.exact(method.fingerprint(state), method.fingerprint(saved['state']))
            rel.base.exact(pred, saved['pred'])
    prediction = method.predict(saved['state'], x[held]); rel.base.exact(prediction, saved['pred'])
    legacy = family == 'ridge' and arm == 'summary120'
    if legacy:
        prefix = str(site)+'_plus_candidate_neighbors_'
        rel.base.exact(prediction, d['cached'][prefix+'predictions'])
        for k, value in method.fingerprint(saved['state']).items():
            rel.base.exact(value, d['cached'][prefix+k])
    return dict(identity=identity, rows=int(held.sum()), queries=int(len(set(d['query'][held]))),
        train_sources=sorted(set(d['sites'][train])), checkpoint=source.artifact(path),
        seconds=saved['seconds'], prediction_hash=rel.parent.kin.digest(prediction),
        model_fingerprint_sha256=source.hashlib.sha256(source.json.dumps(
            {k:rel.parent.kin.digest(v) for k,v in method.fingerprint(saved['state']).items()}, sort_keys=True).encode()).hexdigest(),
        old_ridge_control_exact=legacy, result_source='fresh_run_probe_fit',
        metrics=method.previous.metrics(prediction, d['target'][held], d['weights'][held], saved['state']['target_scale']))


def pilot(cfg, values, guard, beat, resume):
    job = values[3][-1]; d = inputs(values, job, guard); site = sorted(set(d['sites']))[0]
    row = cell(d, job, site, 'extra_trees', 'aligned180', guard, resume=resume, kind='pilot')
    repeated = cell(d, job, site, 'extra_trees', 'aligned180', guard, verify=True, kind='pilot')
    rel.base.exact(row, repeated)
    source.once(PUBLIC/'pilot.json', dict(pid=source.os.getpid(), full_forest_seconds=row['seconds'],
        full_forest_trees=64, direct_repeat_exact=True, complete_probes=2,
        local_training_reasonable=True, model=row, no_new_V_or_H=True))
    beat(state='pilot_complete', forest_seconds=row['seconds'])


def execute(cfg, values, guard, beat, *, resume=False, verify=False):
    if not source.read(PUBLIC/'pilot.json')['direct_repeat_exact']:
        raise ValueError('Real deterministic pilot required')
    contexts = []; completed = 0
    for job in values[3]:
        d = inputs(values, job, guard); rows = []
        source.once(PUBLIC/'inputs'/(job['key']+'.json'), source.stable(d['audit']))
        for site in sorted(set(d['sites'])):
            record = dict(held_source=str(site), families={})
            for family in method.FAMILIES:
                record['families'][family] = {}
                for arm in method.ARMS:
                    r = cell(d, job, site, family, arm, guard, resume=resume, verify=verify)
                    name = job['key']+'_'+str(site)+'_'+family+'_'+arm+'.json'
                    source.once(PUBLIC/'fits'/name, source.stable(r)); completed += 1
                    record['families'][family][arm] = r['metrics']
                    record.update(rows=r['rows'], queries=r['queries'], train_sources=r['train_sources'])
                    beat(state='probe_verified' if verify else 'probe_fitted', complete=completed, total=48,
                        job=job['key'], family=family, arm=arm, seconds=r['seconds'])
            rows.append(record)
        contexts.append(dict(**d['audit'], folds=rows, families=method.summarize(rows)))
        del d; source.gc.collect(); guard()
    passing = sum(r['families']['extra_trees']['passes_internal_screen'] for r in contexts)
    result = source.stable(dict(status='fresh_run_own_B_source_held_information_diagnostic', contexts=contexts,
        primary_family='extra_trees', passing_contexts=passing, followup_neural_supported=passing >= 2,
        fits=completed, repeated_legacy_controls=8, neural_training=False, new_V_or_H=False,
        original_rules_unchanged=True, identity_certification=False, independent_confirmation=False,
        original24group_primary='not_estimable_not_passed', deployment_changed=False, submission_ready=False,
        stage5c_executed=False, smc_enabled=False))
    source.once(PUBLIC/'analysis.json', result)
    if verify:
        source.once(PUBLIC/'verification.json', dict(all48_refits_and_predictions_exact=True,
            all8_old_ridge_controls_exact=True, analysis_sha256=source.sha(PUBLIC/'analysis.json')))


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('phase', choices=('register', 'pilot', 'run', 'verify'))
    p.add_argument('--resume', action='store_true'); args = p.parse_args(); PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        source.fcntl.flock(lock, source.fcntl.LOCK_EX | source.fcntl.LOCK_NB)
        runtime = old.run.old.old.previous.teacher.lineage; runtime.PRIVATE = PRIVATE
        started, guard, beat = runtime.runtime({}, source.read(CONFIG), args.phase)
        cfg, values = register(beat)
        if args.phase == 'register':
            return
        for f in (*FILES, REG):
            content = source.subprocess.run(['git', 'show', 'HEAD:'+str(f.relative_to(ROOT))], cwd=ROOT, capture_output=True, check=True).stdout
            if source.hashlib.sha256(content).hexdigest() != source.sha(f):
                raise ValueError('Commit fixed protocol and code before fitting')
        receipt = PUBLIC/(args.phase+'_execution.json')
        if receipt.exists():
            raise ValueError('Completed phase immutable')
        if args.phase == 'pilot':
            pilot(cfg, values, guard, beat, args.resume)
        else:
            execute(cfg, values, guard, beat, resume=args.resume, verify=args.phase == 'verify')
        source.once(receipt, dict(pid=source.os.getpid(), seconds=time.monotonic()-started,
            peak_RSS_bytes=source.resource.getrusage(source.resource.RUSAGE_SELF).ru_maxrss,
            neural_training=False, new_V_or_H=False))
        beat(state='complete')


if __name__ == '__main__':
    main()
