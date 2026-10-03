"""Replay frozen cost heads; measure TRAIN support and surrogate/ensemble gaps."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import resource
import shlex
import subprocess
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 required')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
from scripts import run_m3w_cost_harm_newton as fitted_run
from scripts.m3w_bounded_packet_write import with_keepalive
from src.world_model import m3w_cost_support_diagnostic as api

reader = fitted_run.runner.reader
prior, parent, sha, once = reader.prior, reader.parent, reader.sha, reader.once
NAME = 'european_cost_support_diagnostic_v1'
PUBLIC, PRIVATE = fitted_run.PUBLIC.parent/NAME, fitted_run.PRIVATE.parent/NAME
CONFIG = ROOT/'configs'/('m3w_'+NAME+'.json')
SERVER = reader.SERVER.replace('european_positive_harm_v1', 'european_cost_harm_newton_v1').replace('_positive', '_cost')


class CostReader(reader.PositiveReader):
    def __enter__(self):
        path = ROOT/'data/stage_cvpr2027_experiments/create_handoff_20260923'
        assert sha(path/'compute_handoff_20260927.json') == 'f78d85acb9d51270747e67b7f3533b1bae023d66874d0fa4ef03fdb6f266d1b9'
        ssh = json.loads((path/'observations.json').read_text())['ssh_arguments']
        command = with_keepalive(ssh)+[shlex.join(['/usr/bin/python3', '-c', SERVER])]
        self.p = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, bufsize=0)
        try:
            assert self.line() == {'ready': True}
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise
        return self


def registration():
    assert fitted_run.registration() == json.loads((fitted_run.PUBLIC/'registration.json').read_text())
    receipt = json.loads((fitted_run.PUBLIC/'verification.json').read_text())
    for name in ('summary', 'complete', 'checkpoint_manifest'):
        assert sha(fitted_run.PUBLIC/(name+'.json')) == receipt[name+'_sha256']
    paths = parent.forest.closure(ROOT, ['scripts.run_m3w_cost_support_diagnostic'])
    paths += [CONFIG, PUBLIC/'protocol.md', ROOT/'tests/test_m3w_cost_support_diagnostic.py']
    return dict(bindings={str(p.relative_to(ROOT)): sha(p) for p in paths},
        parent_verification_sha256=sha(fitted_run.PUBLIC/'verification.json'),
        source_heads=72, new_training=False, independent_roles_read=False, policy_selection=False)


def beat(**kw):
    row = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **kw)
    (PRIVATE/'heartbeat.json').write_text(json.dumps(row)+'\n')
    with (PRIVATE/'events.jsonl').open('a') as f:
        f.write(json.dumps(row)+'\n')
    print(json.dumps(row), flush=True)


def summarize(rows, cfg):
    def interval(values):
        return api.model.positive.base.interval([(r['source'], v) for r, v in zip(rows, values)],
                                                cfg['bootstrap_resamples'], cfg['bootstrap_seed'])
    def reduce(get):
        d = dict(rows=sum(get(r)['rows'] for r in rows), unknown_rows=sum(get(r)['unknown_rows'] for r in rows))
        for k in ('known_query_weight_mass', 'global_weighted_MSE_change', 'conditional_MSE_change'):
            d[k] = interval([get(r)[k] for r in rows])
        for field, names in (('moment_contributions', api.prior.MOMENTS), ('score_contributions', api.prior.SCORES)):
            d[field] = {name: interval([get(r)[field][j] for r in rows]) for j, name in enumerate(names)}
        return d
    out = dict(groups=len(rows), localities=len({r['source'] for r in rows}), new_training=False,
               independent_confirmation=False, policy_selection=False, deployment_changed=False)
    for section in ('cohorts', 'strata', 'projection_effects'):
        out[section] = {name: reduce(lambda r: r['diagnosis'][section][name]) for name in rows[0]['diagnosis'][section]}
    out['raw_error_change'] = reduce(lambda r: r['diagnosis']['raw_error_change'])
    out['training_projected_error_change'] = reduce(lambda r: r['training_projected_error_change'])
    out['tree_scores'] = {role: {name: interval([r['tree_scores'][role][name] for r in rows])
        for name in ('mean_tree_MSE_change', 'raw_ensemble_MSE_change', 'dispersion_change', 'fixed_penalty')}
        for role in ('train', 'validation')}
    out['support_cohorts'] = {name: dict(rows=sum(r['diagnosis']['support_cohorts'][name]['rows'] for r in rows),
        descriptor_mean={key: interval([r['diagnosis']['support_cohorts'][name]['descriptor_mean'][key] for r in rows])
                         for key in api.DESCRIPTORS}) for name in rows[0]['diagnosis']['support_cohorts']}
    out['maximum_surrogate_reconstruction_residual'] = max(abs(r['surrogate_reconstruction_residual']) for r in rows)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['register', 'pilot', 'run', 'verify'])
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    cfg, reg = json.loads(CONFIG.read_text()), registration()
    if args.phase == 'register':
        once(PUBLIC/'registration.json', reg)
        print('Registered frozen support and objective diagnostic')
        return
    assert reg == json.loads((PUBLIC/'registration.json').read_text())
    parent.base.inter.committed(PUBLIC/'registration.json')
    if args.phase != 'verify':
        assert not (PUBLIC/'complete.json').exists()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    parent.core.torch.set_num_threads(cfg['cpu_threads'])
    parent.core.torch.set_num_interop_threads(1)
    start = time.monotonic()
    with (PRIVATE/'process.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        trained = {}
        for ref in json.loads((fitted_run.PUBLIC/'complete.json').read_text())['groups']:
            assert sha(ROOT/ref['path']) == ref['sha256']
            r = json.loads((ROOT/ref['path']).read_text())
            trained[r['group'], r['head_seed']] = r
        docs = parent.parent.docs()
        _, _, data, jobs, oid, _, _, _ = parent.inner.old.load()
        q, provenance = prior.prior.past_quality(data)
        assert provenance == json.loads((prior.prior.PUBLIC/'feature_provenance.json').read_text())
        rows, refs = [], []
        size = fetched = scalar_checks = 0
        with CostReader() as stream:
            beat(state='owned_read_only_cost_stream_connected', new_HPC_jobs=0)
            for c in parent.parent.contexts(data, jobs, oid):
                for site in parent.inner.sources(c):
                    group = c['name']+'_fit_'+site
                    at, ids, x, env, y, _, upstream = parent.inner.training_arrays(c, data, site)
                    tr, val, partition = parent.forest.parent.api.source_partition(data['recordings'][ids], data['frames'][ids], site)
                    tid, vid = ids[tr], ids[val]
                    h = parent.base.inter.array_hash
                    for seed in cfg['head_seeds']:
                        assert time.monotonic()-start < cfg['hard_runtime_limit_seconds']
                        beat(state='frozen_support_and_tree_ensemble_diagnosis', group=group, head_seed=seed)
                        source, anchor = docs[group, seed], trained[group, seed]
                        assert sha(ROOT/source['checkpoint']['path']) == source['checkpoint']['sha256']
                        state = joblib.load(ROOT/source['checkpoint']['path'])
                        assert state['identity']['upstream'] == upstream
                        assert source['partition'] == partition == anchor['partition']
                        assert h(tid) == anchor['training_ids_hash'] and h(vid) == anchor['validation_ids_hash']
                        assert h(y[val]) == anchor['targets_hash'] and h(q[vid]) == anchor['past_quality_hash']
                        fitted, meta = stream.fetch(anchor['checkpoint'])
                        fetched += anchor['checkpoint']['bytes']
                        assert meta['fit'] == anchor['training']
                        assert meta['identity'] == dict(group=group, source=site, head_seed=seed, partition=partition,
                            checkpoint=source['checkpoint'], training_ids_hash=h(tid),
                            registration_sha256=sha(fitted_run.PUBLIC/'registration.json'))
                        assert api.forest.fingerprint(q[tid]) == meta['fit']['past_quality_hash']
                        tables = api.training_support(state, fitted, x[tr], env[tr], y[tr], q[tid],
                            data['sites'][tid], data['recordings'][tid], data['frames'][tid])
                        raw_values = api.raw_predictions(state, fitted, x[val], env[val], q[vid], tables)
                        repeated = api.raw_predictions(state, fitted, x[val], env[val], q[vid], tables)
                        for a, b in zip(raw_values, repeated):
                            np.testing.assert_array_equal(a, b)
                        ro, rc, support, descriptor, extra = raw_values
                        raw = dict(original=ro, cost=rc)
                        pred = {k: api.forest.project_moments(v, env[val]) for k, v in raw.items()}
                        for k, v in pred.items():
                            assert h(v) == anchor['prediction_hashes'][k]
                        base = api.model.positive.base
                        actions = {k: base.eligible(v, c['moving'][at][val], support) for k, v in pred.items()}
                        a, b = base.matched(actions['original'], actions['cost'],
                            api.prior.signed(pred['original'])[:, 0], api.prior.signed(pred['cost'])[:, 0],
                            data['recordings'][vid], data['frames'][vid], vid)
                        actions.update(original_matched_cost=a, cost_matched_original=b)
                        for k, act in actions.items():
                            assert h(act) == anchor['action_hashes'][k]
                            result = base.completion_bounds(y[val], act, env[val])
                            assert result == anchor['result']['policies'][k]
                            scalar_checks += prior.prior.leaf.previous.check_scalars(result,
                                prior.prior.leaf.previous.independent.scalar_bounds(y[val], act, env[val]))
                        diagnosis = api.diagnose(state, raw, pred, y[val], actions, data['sites'][vid],
                            data['recordings'][vid], data['frames'][vid], descriptor, extra)
                        np.testing.assert_allclose(diagnosis['cohorts']['all']['global_weighted_MSE_change'],
                            anchor['result']['contrasts']['cost_minus_original_signed_MSE'], atol=1e-9, rtol=1e-9)
                        scores = {}
                        for role, mask, index in (('train', tr, tid), ('validation', val, vid)):
                            scores[role] = api.tree_score_diagnostic(state, fitted, x[mask], env[mask], q[index], y[mask],
                                data['sites'][index], data['recordings'][index], data['frames'][index])
                        np.testing.assert_allclose(scores['validation']['raw_ensemble_MSE_change'],
                            diagnosis['raw_error_change']['global_weighted_MSE_change'], atol=1e-9, rtol=1e-9)
                        train_raw = api.raw_predictions(state, fitted, x[tr], env[tr], q[tid], tables)
                        np.testing.assert_array_equal(train_raw[4][np.isfinite(y[tr]).all(1), 2], 0)
                        train_pred = [api.forest.project_moments(v, env[tr]) for v in train_raw[:2]]
                        w, _ = api.forest.core.weights(data['sites'][tid], data['recordings'][tid], data['frames'][tid], np.isfinite(y[tr]).all(1))
                        train_error = api.prior.error_slice(*train_pred, y[tr], w, state['preprocess'], np.ones(len(tid), bool))
                        loss = anchor['training']['training_loss']
                        residual = scores['train']['mean_tree_MSE_change']+scores['train']['fixed_penalty']-(sum(loss['after'])-sum(loss['before']))
                        np.testing.assert_allclose(residual, 0, atol=1e-6, rtol=0)
                        row = dict(group=group, source=site, head_seed=seed, diagnosis=diagnosis, tree_scores=scores,
                            training_projected_error_change=train_error, surrogate_reconstruction_residual=float(residual),
                            result_source='fresh_run_frozen_diagnostic', models_source='cached_verified',
                            checkpoint=anchor['checkpoint'], parent_checkpoint=source['checkpoint'],
                            training_ids_hash=h(tid), validation_ids_hash=h(vid), targets_hash=h(y[val]),
                            prediction_hashes={k: h(v) for k, v in pred.items()}, action_hashes={k: h(v) for k, v in actions.items()},
                            registration_sha256=sha(PUBLIC/'registration.json'))
                        rows.append(row)
                        if args.phase != 'pilot':
                            path = PUBLIC/'groups'/(group+'_head'+str(seed)+'.json')
                            if path.exists() and not (args.resume or args.phase == 'verify'):
                                raise RuntimeError('Explicit resume required')
                            once(path, row)
                            size += path.stat().st_size
                            assert size < cfg['aggregate_output_cap_bytes']
                            refs.append(dict(path=str(path.relative_to(ROOT)), sha256=sha(path)))
                        beat(state='group_complete', groups=len(rows), seconds=time.monotonic()-start, fetched_bytes=fetched)
                        if args.phase == 'pilot': break
                    if args.phase == 'pilot': break
                if args.phase == 'pilot': break
        runtime = dict(pid=os.getpid(), seconds=time.monotonic()-start, groups=refs,
            fetched_checkpoint_bytes=fetched, independent_scalar_checks=scalar_checks,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            native_architecture=platform.machine(), cpu_threads=cfg['cpu_threads'], num_workers=0,
            exact_raw_inference_replay=True, exact_readout_replay=True, new_training=False,
            local_numeric_cache=False, new_HPC_jobs=0)
        if args.phase == 'pilot':
            once(PUBLIC/'pilot.json', runtime)
        else:
            assert len(rows) == 72
            summary = summarize(rows, cfg)
            once(PUBLIC/'summary.json', summary)
            runtime['summary_sha256'] = sha(PUBLIC/'summary.json')
            dest = PRIVATE/'additional_replays'/(str(time.time_ns())+'.json') if args.phase == 'verify' else PUBLIC/'complete.json'
            once(dest, runtime)
        beat(state='complete', phase=args.phase, seconds=time.monotonic()-start)


if __name__ == '__main__':
    main()
