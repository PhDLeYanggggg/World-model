"""Diagnose the complete frozen feature experiment without optimizer updates."""
import argparse
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 required before Torch import')
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts import run_m3w_interaction_feature_cost as parent
from src.evaluation.m3w_cost_mass_decomposition import cost_masses, risks
from src.evaluation.m3w_feature_cost_diagnosis import ARMS, ROLES, packet_for_role, summarize
from src.world_model.m3w_supported_bound_targets import supported_bound_targets

NAME = 'european_feature_cost_diagnosis_v1'
PUBLIC = ROOT/'outputs/publication_readiness_2026_09'/NAME
PRIVATE = ROOT/'data/stage_cvpr2027_experiments'/NAME
FILES = (Path(__file__), ROOT/'src/evaluation/m3w_feature_cost_diagnosis.py',
         ROOT/'tests/test_m3w_feature_cost_diagnosis.py', PUBLIC/'protocol.md')
sha, once, checked, artifact = parent.sha, parent.once, parent.checked, parent.artifact


def register():
    args = parent.admission()
    cfg, jobs, _, _, _, _ = args
    v = json.loads((parent.PUBLIC/'verification.json').read_text())
    if not v['all72_metric_groups_exact'] or v['summary_sha256'] != sha(parent.PUBLIC/'summary.json'):
        raise ValueError('Complete verified parent required')
    definition = dict(bindings={str(p.relative_to(ROOT)): sha(p) for p in FILES},
        parent_registration=artifact(parent.REG), parent_summary=artifact(parent.PUBLIC/'summary.json'),
        parent_training=artifact(parent.PUBLIC/'training_freeze.json'),
        parent_predictions=artifact(parent.PUBLIC/'policy_freeze.json'),
        jobs=jobs, models=36, roles=list(ROLES), arms=list(ARMS),
        new_optimizer_updates=0, future_labels_diagnostic_targets_only=True,
        descriptive_not_policy_selection=True, independent_confirmation=False,
        thresholds_changed=False, deployment_changed=False, stage5c_executed=False, smc_enabled=False)
    once(PUBLIC/'registration.json', definition)
    return args


def one(job, role, src, ids, refs, caches, fitted, beat):
    packet = packet_for_role(job, role)
    # Both roles use the causal inference-only path; fit labels never refit preprocessing.
    data = parent.bank(packet, src, ids, refs, caches, training=False)
    c, paths = data['causal'], data['paths']
    floor, proposal = paths[:, job['floor_index']], paths[:, 6]
    out = dict(job=job['key'], role=role, sites=sorted(set(data['sites'])),
        rows=len(data['ids']), ids_sha256=parent.parent.prior.frozen.fingerprint(data['ids']), arms={})
    for arm in ARMS:
        meta = fitted[job['key'], arm]
        state = parent.api.base.base.read_checkpoint(checked(meta['checkpoint']))
        pred = parent.api.predict(state, c)
        alpha = parent.api.base.amplitudes(pred, c['allowed'])
        if role == 'double_excluded':
            saved = json.loads((parent.PUBLIC/'policies'/(job['key']+'_'+arm+'.json')).read_text())
            np.testing.assert_array_equal(pred, np.load(checked(saved['costs']), allow_pickle=False))
            np.testing.assert_array_equal(alpha, np.load(checked(saved['amplitudes']), allow_pickle=False)[:, 0])
        # Only after inference, load targets for cost diagnosis on the same rows.
        y = np.load(parent.parent.prior.SOURCE/'target_eval.npy', mmap_mode='r', allow_pickle=False)[data['ids']]-src['origin'][data['ids'], None]
        valid = np.load(parent.parent.prior.SOURCE/'valid.npy', mmap_mode='r', allow_pickle=False)[data['ids']]
        truth = supported_bound_targets(paths, y, valid, job['floor_index'], job['easy_cut'])['moments'][:, 6]
        mixed = parent.api.base.prediction(floor, proposal, alpha)
        effective = np.linalg.norm(mixed-floor, axis=-1).max(1) > 0
        mixed_bank = paths.copy()
        mixed_bank[:, 6] = mixed
        actual = supported_bound_targets(mixed_bank, y, valid, job['floor_index'], job['easy_cut'])['moments'][:, 6]
        p = pred[:, 6]
        old = {} if role == 'critic_fit' else {site: json.loads((parent.PUBLIC/'groups'/(job['key']+'_'+site+'.json')).read_text()) for site in job['evaluation_sites']}
        records = {}
        actual_pred = p.copy()
        actual_pred[:, [0, 1, 4]] *= alpha[:, None]
        for rec in sorted(set(data['recordings'])):
            at = data['recordings'] == rec
            take = at & effective
            site = str(data['sites'][np.flatnonzero(at)[0]])
            realized = cost_masses(actual_pred, actual, take)
            r = risks(realized)
            if role == 'double_excluded':
                expected = old[site]['policies'][arm+'_bounded']['recordings'][str(rec)]
                for key, oldkey in (('target_all', 'completion_all_risk_upper'), ('target_easy', 'completion_easy_risk_upper')):
                    value = expected[oldkey]
                    if value is None:
                        if r[key] is not None:
                            raise ValueError('Missing support differs from parent')
                    else:
                        np.testing.assert_allclose(r[key], value, rtol=1e-12, atol=1e-12)
            records[str(rec)] = dict(site=site, rows=int(at.sum()), selected=int(take.sum()),
                unknown_selected=int((take & ~valid.any(1)).sum()),
                full_legal_neural=cost_masses(p, truth, at & c['allowed'][:, 6]),
                selected_full_proposal=cost_masses(p, truth, take),
                selected_scaled_proposal=cost_masses(p, truth, take, alpha),
                selected_actual_interpolation=realized, selected_actual_risks=r)
        out['arms'][arm] = dict(checkpoint=meta['checkpoint'], records=records,
            nominal_active_rows=int((alpha > 0).sum()), actual_active_rows=int(effective.sum()),
            held_predictions_exact=role == 'double_excluded')
        del pred, state, y, valid, truth, mixed, mixed_bank, actual, p, actual_pred
        beat(job=job['key'], role=role, arm=arm, state='role_arm_complete')
    del data, c, paths
    gc.collect()
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'run', 'verify'))
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cfg, jobs, src, ids, refs, caches = register()
        if args.phase == 'register':
            return
        for path in (*FILES, PUBLIC/'registration.json'):
            blob = subprocess.run(['git', 'show', 'HEAD:'+str(path.relative_to(ROOT))], cwd=ROOT, capture_output=True, check=True).stdout
            if hashlib.sha256(blob).hexdigest() != sha(path):
                raise ValueError('Commit diagnostic definition before execution')
        end = PUBLIC/('verification.json' if args.phase == 'verify' else 'complete.json')
        if end.exists():
            raise ValueError('Completed phases immutable')
        if args.phase == 'verify' and not (PUBLIC/'complete.json').exists():
            raise ValueError('Complete first pass required')
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        start = time.monotonic()
        def beat(**kw):
            if time.monotonic()-start > 43200:
                raise TimeoutError('12h phase boundary; resume unchanged groups')
            if shutil.disk_usage(ROOT).free < cfg['disk_reserve_bytes']:
                raise OSError('Retain original disk reserve; no deletion')
            value = dict(pid=os.getpid(), elapsed_seconds=time.monotonic()-start, **kw)
            parent.write(PRIVATE/'heartbeat.json', value)
            with (PRIVATE/'events.jsonl').open('a') as stream:
                stream.write(json.dumps(value)+'\n')
            print(json.dumps(value), flush=True)
        models = parent.fitted()
        groups = []
        beat(state='runtime_ready', architecture=platform.machine(), workers=0, threads=4)
        for job in jobs:
            for role in ROLES:
                path = PUBLIC/'groups'/(job['key']+'_'+role+'.json')
                if path.exists() and args.resume and args.phase == 'run':
                    value = json.loads(path.read_text())
                else:
                    value = one(job, role, src, ids, refs, caches, models, beat)
                    if args.phase == 'verify' and not path.exists():
                        raise ValueError('Missing first-pass group')
                    once(path, value)
                groups.append(value)
                beat(state='group_complete', groups=len(groups), total=36)
        summary = summarize(groups, jobs)
        once(PUBLIC/'summary.json', summary)
        once(end, dict(pid=os.getpid(), seconds=time.monotonic()-start,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            summary_sha256=sha(PUBLIC/'summary.json'), registration_sha256=sha(PUBLIC/'registration.json'),
            all36_job_role_summaries_exact=args.phase == 'verify', new_optimizer_updates=0))
        print(json.dumps(dict(complete=str(end), summary=str(PUBLIC/'summary.json'))), flush=True)


if __name__ == '__main__':
    main()
