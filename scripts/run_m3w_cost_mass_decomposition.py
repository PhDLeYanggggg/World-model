"""Fresh frozen-model SOURCE cost decomposition, never a new policy trial."""
import argparse
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 required before Torch import')
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '4')
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts import run_m3w_bound_aware_cost as parent
from src.evaluation.m3w_cost_mass_decomposition import cost_masses, risks
from src.world_model.m3w_supported_bound_targets import supported_bound_targets

NAME = 'european_bound_cost_decomposition_v1'
PUBLIC = ROOT/'outputs/publication_readiness_2026_09'/NAME
PRIVATE = ROOT/'data/stage_cvpr2027_experiments'/NAME
FILES = (Path(__file__), ROOT/'src/evaluation/m3w_cost_mass_decomposition.py',
    ROOT/'tests/test_m3w_cost_mass_decomposition.py', PUBLIC/'protocol.md')
sha, once, checked, artifact = parent.sha, parent.once, parent.checked, parent.artifact


def register():
    cfg, jobs, src, ids, refs = parent.admission()
    v = json.loads((parent.PUBLIC/'verification.json').read_text())
    if not v['all72_metric_groups_exact'] or v['summary_sha256'] != sha(parent.PUBLIC/'summary.json'):
        raise ValueError('Complete parent readout and replay required')
    definition = dict(bindings={str(p.relative_to(ROOT)): sha(p) for p in FILES},
        parent_registration=artifact(parent.REG), parent_summary=artifact(parent.PUBLIC/'summary.json'),
        parent_training=artifact(parent.PUBLIC/'training_freeze.json'),
        parent_predictions=artifact(parent.PUBLIC/'policy_freeze.json'),
        jobs=jobs, actual_models=36, roles=['critic_fit', 'double_excluded'], new_optimizer_updates=0,
        diagnosis_not_independent_confirmation=True, future_labels_target_only=True,
        thresholds_changed=False, deployment_changed=False, stage5c_executed=False, smc_enabled=False)
    once(PUBLIC/'registration.json', definition)
    return cfg, jobs, src, ids, refs


def one(job, role, src, ids, refs, fitted, beat):
    training = role == 'critic_fit'
    packet_job = dict(job, evaluation_sites=job['critic_sites'], evaluation_scope=job['fit_scope']) if training else job
    data = parent.bank(packet_job, src, ids, refs, training=False)
    c, paths = data['causal'], data['paths']; floor, proposal = paths[:, job['floor_index']], paths[:, 6]
    out = dict(job=job['key'], role=role, sites=sorted(set(data['sites'])),
        rows=len(data['ids']), ids_sha256=parent.prior.frozen.fingerprint(data['ids']), arms={})
    for arm in parent.api.ARMS:
        meta = fitted[job['key'], arm]
        state = parent.api.base.read_checkpoint(checked(meta['checkpoint']))
        pred = parent.api.base.predict(state, c)
        alpha = parent.api.amplitudes(pred, c['allowed'])
        if not training:
            old = json.loads((parent.PUBLIC/'policies'/(job['key']+'_'+arm+'.json')).read_text())
            np.testing.assert_array_equal(pred, np.load(checked(old['costs']), allow_pickle=False))
            np.testing.assert_array_equal(alpha, np.load(checked(old['amplitudes']), allow_pickle=False)[:, 0])
        # Evaluation labels are read after inference, never added to c or state preprocessing.
        y = np.load(parent.prior.SOURCE/'target_eval.npy', mmap_mode='r', allow_pickle=False)[data['ids']]-src['origin'][data['ids'], None]
        valid = np.load(parent.prior.SOURCE/'valid.npy', mmap_mode='r', allow_pickle=False)[data['ids']]
        truth = supported_bound_targets(paths, y, valid, job['floor_index'], job['easy_cut'])['moments'][:, 6]
        mixed = parent.api.prediction(floor, proposal, alpha)
        effective = np.linalg.norm(mixed-floor, axis=-1).max(1) > 0
        mixed_bank = paths.copy(); mixed_bank[:, 6] = mixed
        actual = supported_bound_targets(mixed_bank, y, valid, job['floor_index'], job['easy_cut'])['moments'][:, 6]
        p = pred[:, 6]; records = {}
        old_groups = {} if training else {site: json.loads((parent.PUBLIC/'groups'/(job['key']+'_'+site+'.json')).read_text()) for site in job['evaluation_sites']}
        for rec in sorted(set(data['recordings'])):
            at = data['recordings'] == rec; take = at & effective
            site = str(data['sites'][np.flatnonzero(at)[0]])
            full = cost_masses(p, truth, at & c['allowed'][:, 6])
            selected = cost_masses(p, truth, take)
            scaled = cost_masses(p, truth, take, alpha)
            actual_pred = p.copy(); actual_pred[:, [0, 1, 4]] *= alpha[:, None]
            realized = cost_masses(actual_pred, actual, take)
            r = risks(realized)
            if not training:
                prior = old_groups[site]['policies'][arm+'_bounded']['recordings'][str(rec)]
                for key, oldkey in (('target_all', 'completion_all_risk_upper'), ('target_easy', 'completion_easy_risk_upper')):
                    expected = prior[oldkey]
                    if expected is None:
                        if r[key] is not None: raise ValueError('Support differs from parent')
                    else: np.testing.assert_allclose(r[key], expected, rtol=1e-12, atol=1e-12)
            records[str(rec)] = dict(site=site, rows=int(at.sum()), selected=int(take.sum()),
                unknown_selected=int((take & ~valid.any(1)).sum()),
                full_legal_neural=full, selected_full_proposal=selected,
                selected_scaled_proposal=scaled, selected_actual_interpolation=realized,
                selected_actual_risks=r)
        out['arms'][arm] = dict(checkpoint=meta['checkpoint'], records=records,
            nominal_active_rows=int((alpha > 0).sum()), actual_active_rows=int(effective.sum()),
            held_predictions_exact=not training)
        del pred, state, y, valid, truth, mixed, mixed_bank, actual, p, records
        beat(job=job['key'], role=role, arm=arm, state='role_arm_complete')
    del data, c, paths; gc.collect()
    return out


def summarize(groups):
    result = {}
    for role in ('critic_fit', 'double_excluded'):
        result[role] = {}
        for arm in parent.api.ARMS:
            records = [r for g in groups if g['role'] == role for r in g['arms'][arm]['records'].values()]
            summary = dict(record_views=len(records), selected_rows=sum(r['selected'] for r in records))
            for scope in ('full_legal_neural', 'selected_full_proposal', 'selected_scaled_proposal', 'selected_actual_interpolation'):
                values = {}
                for moment in ('B', 'H', 'R', 'ER', 'EH'):
                    x = [r[scope][moment]['predicted_over_target'] for r in records]
                    a = np.array([v for v in x if v is not None])
                    values[moment] = dict(defined_record_views=len(a),
                        quantiles_0_25_50_75_100=np.quantile(a, [0, .25, .5, .75, 1]).tolist() if len(a) else None)
                summary[scope] = values
            for channel in ('all', 'easy'):
                pairs = [(r['selected_actual_risks']['predicted_'+channel], r['selected_actual_risks']['target_'+channel]) for r in records]
                summary[channel+'_risk'] = dict(defined_views=sum(b is not None for a, b in pairs),
                    predicted_safe_actual_violations=sum(a is not None and b is not None and a <= .02+1e-12 and b > .02 for a, b in pairs),
                    actual_violations=sum(b is not None and b > .02 for a, b in pairs))
            result[role][arm] = summary
    return dict(result_source='fresh_zero_update_frozen_SOURCE_inference_and_same_set_target_diagnosis',
        jobs=18, job_role_groups=len(groups), models=36, inference_role_passes=72,
        held36cost_and_amplitude_arrays_exact=True, held_completion_risks_match_parent=True,
        descriptive_record_view_quantiles_not_independent_site_inference=True,
        new_optimizer_updates=0, independent_confirmation=False, deployment_changed=False,
        stage5c_executed=False, smc_enabled=False, results=result)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('phase', choices=('register', 'run', 'verify'))
    p.add_argument('--resume', action='store_true'); args = p.parse_args()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cfg, jobs, src, ids, refs = register()
        if args.phase == 'register': return
        for f in (*FILES, PUBLIC/'registration.json'):
            blob = subprocess.run(['git', 'show', 'HEAD:'+str(f.relative_to(ROOT))], cwd=ROOT, capture_output=True, check=True).stdout
            if hashlib.sha256(blob).hexdigest() != sha(f): raise ValueError('Commit diagnostic definition first')
        end = PUBLIC/('verification.json' if args.phase == 'verify' else 'complete.json')
        if end.exists(): raise ValueError('Complete diagnostic phase immutable')
        if args.phase == 'verify' and not (PUBLIC/'complete.json').exists(): raise ValueError('Complete original pass required')
        torch.set_num_threads(4); torch.set_num_interop_threads(1); start = time.monotonic()
        def beat(**kw):
            if time.monotonic()-start > 43200: raise TimeoutError('12h phase boundary; resume unchanged groups')
            value = dict(pid=os.getpid(), elapsed_seconds=time.monotonic()-start, **kw)
            parent.write(PRIVATE/'heartbeat.json', value); print(json.dumps(value), flush=True)
        models = parent.fitted(); groups = []
        for job in jobs:
            for role in ('critic_fit', 'double_excluded'):
                path = PUBLIC/'groups'/(job['key']+'_'+role+'.json')
                if path.exists() and args.resume and args.phase == 'run':
                    value = json.loads(path.read_text())
                else:
                    value = one(job, role, src, ids, refs, models, beat)
                    if args.phase == 'verify' and not path.exists(): raise ValueError('Missing diagnostic group')
                    once(path, value)
                groups.append(value); beat(state='group_complete', groups=len(groups), total=36)
        summary = summarize(groups); once(PUBLIC/'summary.json', summary)
        once(end, dict(pid=os.getpid(), seconds=time.monotonic()-start,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            summary_sha256=sha(PUBLIC/'summary.json'), registration_sha256=sha(PUBLIC/'registration.json'),
            all36_job_role_summaries_exact=args.phase == 'verify', new_optimizer_updates=0))
        print(json.dumps(summary), flush=True)


if __name__ == '__main__': main()
