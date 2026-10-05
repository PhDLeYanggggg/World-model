"""Frozen-action temporal target audit and TRAIN-only analytic probe fitting."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 required before Torch imports')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
from scripts import run_m3w_label_support as labels
from src.world_model import m3w_temporal_target_audit as api

parent, sha, once = labels.parent, labels.sha, labels.once
NAME = 'european_temporal_target_audit_v1'
PUBLIC, PRIVATE = labels.PUBLIC.parent/NAME, labels.PRIVATE.parent/NAME
CONFIG = ROOT/'configs'/('m3w_'+NAME+'.json')
PREVIOUS = labels.PUBLIC.parent/'european_leaf_quality_extension_v1'
REGISTRATION = PUBLIC/'registration_amended.json'


def registration():
    assert labels.registration() == json.loads((labels.PUBLIC/'registration.json').read_text())
    for home in (labels.PUBLIC, PREVIOUS):
        receipt = json.loads((home/'verification.json').read_text())
        for key in ('summary', 'complete'):
            assert sha(home/(key+'.json')) == receipt[key+'_sha256']
    files = parent.forest.closure(ROOT, ['scripts.run_m3w_temporal_target_audit'])
    files += [CONFIG, PUBLIC/'protocol.md', PUBLIC/'amendment.md',
              ROOT/'tests/test_m3w_temporal_target_audit.py', ROOT/'tests/test_m3w_temporal_target_reporting.py']
    return dict(bindings={str(p.relative_to(ROOT)): sha(p) for p in files},
                original_registration_sha256=sha(PUBLIC/'registration.json'),
                source_label_verification_sha256=sha(labels.PUBLIC/'verification.json'),
                parent_extension_verification_sha256=sha(PREVIOUS/'verification.json'),
                source_heads=72, independent_roles_read=False, deployment_changed=False)


def beat(**kw):
    row = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **kw)
    (PRIVATE/'heartbeat.json').write_text(json.dumps(row)+'\n')
    with (PRIVATE/'events.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
    print(json.dumps(row), flush=True)


def decomposition(d, y, chosen):
    known = d['valid_steps'] > 0; selected = chosen & known
    # Positive easy harm can exist when easy reference error is exactly zero.
    easy = selected & ((y[:, 3] > 0) | (y[:, 4] > 0))
    den = float(y[easy, 3].sum())
    return dict(rows=len(y), unknown_rows=int((~known).sum()),
        full_label_rows=int((d['valid_steps'] == 12).sum()),
        step_counts=[int((d['valid_steps'] == k).sum()) for k in range(13)],
        selected=int(chosen.sum()), unknown_selected=int((chosen & ~known).sum()),
        selected_opposite_sign=int((selected & d['opposite_sign']).sum()),
        selected_netharm_rows=int((selected & (d['harm'] > 0)).sum()),
        selected_netharm_with_cancellation=int((selected & (d['harm'] > 0) & d['opposite_sign']).sum()),
        selected_nonharmful_but_step_harmful=int((selected & (d['harm'] == 0) & (d['gross_harm'] > 0)).sum()),
        selected_harm=float(d['harm'][selected].sum()),
        selected_step_harm=float(d['gross_harm'][selected].sum()),
        selected_cancellation=float(d['cancellation'][selected].sum()),
        known_selected_easy_reference=den,
        known_selected_easy_harm=float(y[easy, 4].sum()),
        known_selected_easy_step_harm=float(d['gross_harm'][easy].sum()),
        original_known_easy_ratio=float(y[easy, 4].sum()/den) if den > 0 else None,
        changed_target_known_easy_ratio=float(d['gross_harm'][easy].sum()/den) if den > 0 else None)


def summary(rows, cfg):
    ci = lambda get: labels.api.locality_interval(rows, get, cfg['bootstrap_resamples'], cfg['bootstrap_seed'])
    contrasts = {}
    for cohort in ('validation', 'complete_validation', 'selected_validation'):
        for other in ('rowmean_leaf', 'global_temporal'):
            for channel, index in (('signed_error', 0), ('reference_error', 1), ('mean_channels', None)):
                key = cohort+'_temporal_minus_'+other+'_'+channel
                def get(r, cohort=cohort, other=other, index=index):
                    m = r['probes'][cohort]
                    if m['temporal_leaf'] is None: return None
                    diff = np.asarray(m['temporal_leaf'])-m[other]
                    return float(diff.mean() if index is None else diff[index])
                contrasts[key] = ci(get)
    failures = []
    for cohort in ('validation', 'complete_validation'):
        for other in ('rowmean_leaf', 'global_temporal'):
            for channel in ('signed_error', 'mean_channels'):
                key = cohort+'_temporal_minus_'+other+'_'+channel
                v = contrasts[key]
                if v['CI95'] is None or v['CI95'][1] >= 0: failures.append(key)
    return dict(groups=len(rows), localities=len({r['source'] for r in rows}), contrasts=contrasts,
        pooled_validation={k: sum(r['decomposition']['validation'][k] for r in rows)
            for k in ('rows', 'unknown_rows', 'full_label_rows', 'selected', 'unknown_selected',
                      'selected_opposite_sign', 'selected_netharm_rows', 'selected_netharm_with_cancellation',
                      'selected_nonharmful_but_step_harmful', 'selected_harm', 'selected_step_harm', 'selected_cancellation')},
        advance_to_auxiliary_design=not failures, failure_reasons=failures,
        policy_changed=False, new_neural_training=False, fresh_analytic_probe_fits=len(rows),
        independent_confirmation=False, deployment_changed=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['register', 'pilot', 'run', 'verify'])
    p.add_argument('--resume', action='store_true'); args = p.parse_args()
    cfg, reg = json.loads(CONFIG.read_text()), registration()
    if args.phase == 'register':
        once(REGISTRATION, reg); print('Registered amended temporal target audit'); return
    assert reg == json.loads(REGISTRATION.read_text())
    parent.base.inter.committed(REGISTRATION)
    if args.phase != 'verify': assert not (PUBLIC/'complete.json').exists()
    if args.phase == 'run':
        pilot = json.loads((PUBLIC/'pilot_amended.json').read_text())
        assert pilot['exact_refit'] and pilot['exact_inference']
        assert pilot['peak_RSS_bytes'] < 40*2**30 and pilot['seconds']*72 < cfg['hard_runtime_limit_seconds']
    PRIVATE.mkdir(parents=True, exist_ok=True)
    api.forest.core.torch.set_num_threads(cfg['cpu_threads']); api.forest.core.torch.set_num_interop_threads(1)
    start = time.monotonic()
    with (PRIVATE/'process.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        beat(state='checking_source_assets_loading')
        _, _, data, jobs, oid, _, _, _ = parent.inner.old.load()
        docs = parent.parent.docs(); rows, refs = [], []; scalar_checks = 0
        for c in parent.parent.contexts(data, jobs, oid):
            for site in parent.inner.sources(c):
                group = c['name']+'_fit_'+site
                at, ids, x, env, y, _, upstream = parent.inner.training_arrays(c, data, site)
                tr, val, partition = parent.forest.parent.api.source_partition(data['recordings'][ids], data['frames'][ids], site)
                assert not set(data['recordings'][ids[tr]]) & set(data['recordings'][ids[val]])
                series, d = api.targets(c['floor'][at].astype(float)+data['origin'][ids, None],
                    c['prediction'][at].astype(float)+data['origin'][ids, None], data['target_eval'][ids], data['valid'][ids])
                known = d['valid_steps'] > 0
                np.testing.assert_array_equal(known, np.isfinite(y).all(1))
                for j, key in ((0, 'benefit'), (1, 'harm'), (2, 'reference')):
                    np.testing.assert_allclose(d[key][known], y[known, j], rtol=1e-9, atol=1e-8)
                np.testing.assert_allclose((d['gross_harm']-d['cancellation'])[known], y[known, 1], rtol=1e-9, atol=1e-8)
                for seed in cfg['head_seeds']:
                    assert time.monotonic()-start < cfg['hard_runtime_limit_seconds']
                    beat(state='fitting_temporal_probe', group=group, head_seed=seed)
                    source = docs[group, seed]
                    assert sha(ROOT/source['checkpoint']['path']) == source['checkpoint']['sha256']
                    state = joblib.load(ROOT/source['checkpoint']['path'])
                    assert state['identity']['upstream'] == upstream and source['partition'] == partition
                    tid, vid = ids[tr], ids[val]; h = parent.base.inter.array_hash
                    params = (state, x[tr], env[tr], series[tr], data['sites'][tid], data['recordings'][tid], data['frames'][tid])
                    fitted = api.fit(*params); again = api.fit(*params)
                    assert api.fingerprint(fitted) == api.fingerprint(again)
                    per_role, decomposed, hashes = {}, {}, {}
                    for role, use in (('training', tr), ('validation', val)):
                        ix = ids[use]
                        old, support = api.forest.predict(state, x[use], env[use])
                        selected = parent.api.eligible(old, c['moving'][at][use], support)
                        if role == 'validation':
                            assert h(old) == source['validation']['prediction_hashes']['forest']
                            assert h(selected) == source['validation']['action_hashes']['forest']
                            bounds = parent.api.completion_bounds(y[use], selected, env[use])
                            scalar_checks += labels.leaf.previous.check_scalars(bounds, source['validation']['completion_screen'])
                        probes, temporal_support = api.predict(state, fitted, x[use], env[use])
                        replay, rs = api.predict(state, again, x[use], env[use])
                        np.testing.assert_array_equal(temporal_support, rs)
                        for key in probes: np.testing.assert_array_equal(probes[key], replay[key])
                        normalized = series[use]/state['preprocess']['scale']
                        per_role[role] = api.score(probes, normalized, data['sites'][ix], data['recordings'][ix], data['frames'][ix])
                        rd = {k: v[use] for k, v in d.items()}
                        decomposed[role] = decomposition(rd, y[use], selected)
                        selected_known = selected & np.isfinite(y[use]).all(1)
                        np.testing.assert_allclose(decomposed[role]['known_selected_easy_harm'], y[use][selected_known, 4].sum(), atol=1e-8)
                        if role == 'validation':
                            for name, mask in (('complete_validation', rd['valid_steps'] == 12), ('selected_validation', selected)):
                                per_role[name] = api.score({k: v[mask] for k, v in probes.items()}, normalized[mask],
                                    data['sites'][ix][mask], data['recordings'][ix][mask], data['frames'][ix][mask])
                            support_stats = dict(mean_tree_step_supported_fraction=float(temporal_support.mean()),
                                observed_steps_using_any_global_fallback=int((data['valid'][ix] & (temporal_support < 1-1e-12)).sum()))
                        hashes[role] = dict(ids=h(ix), temporal_target=h(series[use]), original_prediction=h(old),
                            original_action=h(selected), probes={k: h(v) for k, v in probes.items()})
                    row = dict(group=group, source=site, head_seed=seed, partition=partition, checkpoint=source['checkpoint'],
                        probes=per_role, decomposition=decomposed, support=support_stats, hashes=hashes,
                        fitted_tables_sha256=api.fingerprint(fitted), exact_refit=True, exact_inference=True,
                        original_policy_unchanged=True, policy_changed=False, fresh_analytic_probe_fit=True,
                        result_source='fresh_run', parent_source='cached_verified', new_neural_training=False,
                        registration_sha256=sha(REGISTRATION))
                    rows.append(row)
                    if args.phase != 'pilot':
                        path = PUBLIC/'groups'/(group+'_head'+str(seed)+'.json')
                        if path.exists() and not (args.resume or args.phase == 'verify'): raise RuntimeError('Use explicit resume')
                        once(path, row); refs.append(dict(path=str(path.relative_to(ROOT)), sha256=sha(path)))
                        assert sum((ROOT/r['path']).stat().st_size for r in refs) < cfg['aggregate_output_cap_bytes']
                    beat(state='group_complete', heads=len(rows), seconds=time.monotonic()-start)
                    if args.phase == 'pilot': break
                if args.phase == 'pilot': break
            if args.phase == 'pilot': break
        runtime = dict(pid=os.getpid(), seconds=time.monotonic()-start, groups=refs,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            disk_free_bytes=shutil.disk_usage(ROOT).free, native_architecture=platform.machine(),
            cpu_threads=4, num_workers=0, scalar_checks=scalar_checks, exact_refit=True, exact_inference=True,
            new_neural_training=False, new_numerical_cache=False, remote_access=False, policy_changed=False)
        if args.phase == 'pilot':
            once(PUBLIC/'pilot_amended.json', runtime)
        else:
            assert len(rows) == cfg['source_heads']
            once(PUBLIC/'summary.json', summary(rows, cfg))
            runtime['summary_sha256'] = sha(PUBLIC/'summary.json')
            name = 'replay.json' if args.phase == 'verify' else 'complete.json'
            once(PUBLIC/name, runtime)
        beat(state='complete', phase=args.phase, seconds=time.monotonic()-start)


if __name__ == '__main__': main()
