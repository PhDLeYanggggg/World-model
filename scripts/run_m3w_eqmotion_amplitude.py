"""Registered SOURCE-only bounded-amplitude mechanism experiment."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import signal
import subprocess
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Use the native arm64 environment before importing Torch')
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '4')
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts import run_m3w_european_eqmotion_controls as parent
from src.evaluation import m3w_eqmotion_amplitude as api
from src.evaluation import m3w_selection_forecast_readout as metrics

NAME = 'european_eqmotion_amplitude_v1'
PUBLIC = ROOT/'outputs/publication_readiness_2026_09'/NAME
PRIVATE = ROOT/'data/stage_cvpr2027_experiments'/NAME
CONFIG = ROOT/'configs/m3w_eqmotion_amplitude_v1.json'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), CONFIG, ROOT/'src/evaluation/m3w_eqmotion_amplitude.py',
         ROOT/'tests/test_m3w_eqmotion_amplitude.py', PUBLIC/'protocol.md')
run = parent.run


def admission(register=False):
    cfg = json.loads(CONFIG.read_text())
    values = parent.admission()
    _, data, jobs, _, _, modern_jobs, _, _, whitelist, _ = values
    for new, current in zip(jobs, modern_jobs):
        if new[0] != current['key'] or new[1]['baseline_index'] != current['design']['baseline_index']:
            raise ValueError('Both families must retain the same fitting-selected floor')
    for path, field in ((run.PUBLIC/'verification.json', 'all72_SOURCE_groups_exact'),
                        (parent.PUBLIC/'verification.json', 'all144_secondary_groups_exact')):
        receipt = json.loads(path.read_text())
        if not receipt[field] or receipt['summary_sha256'] != run.sha(path.parent/'summary.json'):
            raise ValueError('Complete primary and stronger-control replay required')
    if (cfg['amplitudes'] != [0, .001, .005, .01, .02, .05, .1, .25, .5, 1]
            or cfg['primary_amplitude'] != .25 or len(whitelist) != cfg['source_whitelist_rows']
            or cfg['positive_harm_budget'] != .02 or cfg['easy_net_degradation_budget'] != .02
            or any(cfg[k] for k in ('new_training', 'threshold_selection', 'development_arrays_opened',
                'calibration_arrays_opened', 'confirmation_opened', 'deployment_changed',
                'stage5c_executed', 'smc_enabled'))):
        raise ValueError('Fixed complete curve, original role boundaries and recording risk required')
    paths = (*FILES, run.REG, parent.REG, run.PUBLIC/'prediction_freeze.json',
             parent.PUBLIC/'prediction_freeze.json', run.PUBLIC/'verification.json',
             parent.PUBLIC/'verification.json',
             ROOT/'src/evaluation/m3w_selection_forecast_readout.py',
             ROOT/'src/evaluation/m3w_native_metrics.py',
             ROOT/'src/world_model/m3w_european_source_forecast.py')
    value = dict(bindings={str(p.relative_to(ROOT)):run.sha(p) for p in paths},
        source_rows=len(whitelist), source_localities=sorted(set(data['sites'][whitelist])),
        source_recordings=len(set(data['recordings'][whitelist])),
        trials=[j[0] for j in jobs], actual_seeds=[17, 29, 43], amplitudes=cfg['amplitudes'],
        primary_amplitude=cfg['primary_amplitude'], source_is_exposed_development=True,
        new_training=False, threshold_selection=False, independent_roles_opened=False,
        deployment_changed=False, stage5c_executed=False, smc_enabled=False)
    if register:
        if (PUBLIC/'complete.json').exists(): raise ValueError('Completed experiment immutable')
        run.once(REG, value)
    if json.loads(REG.read_text()) != value: raise ValueError('Amplitude contract changed')
    if not register:
        for path in (*FILES, REG):
            name = str(path.relative_to(ROOT))
            blob = subprocess.run(['git', 'show', 'HEAD:'+name], cwd=ROOT,
                                  capture_output=True, check=True).stdout
            if hashlib.sha256(blob).hexdigest() != run.sha(path):
                raise ValueError('Commit the complete contract before outcome reductions')
    return cfg, data, jobs, whitelist, value


def summarize(groups, cfg, registration):
    roster = registration['source_localities']
    expected = {(j, s) for j in registration['trials'] for s in roster}
    # Each forecaster excludes eight localities, rather than all twelve.
    if (len(groups) != 72 or len({(g['trial'], g['site']) for g in groups}) != 72
            or not {(g['trial'], g['site']) for g in groups}.issubset(expected)):
        raise ValueError('All 72 original producer-excluded groups required')
    ci = lambda f: api.locality_interval(groups, f, roster, cfg['bootstrap_resamples'], cfg['bootstrap_seed'])
    curve = {}
    for amplitude in cfg['amplitudes']:
        key = str(amplitude)
        arms = {}
        for family in ('eqmotion', 'history_PCT'):
            arms[family] = dict(
                known_gain=ci(lambda g: g['policies'][family][key]['known_ADE_gain_floor_percent']),
                completion_lower_gain=ci(lambda g: g['policies'][family][key]['completion_lower_gain_floor_percent']),
                easy_gain=ci(lambda g: g['policies'][family][key]['subsets']['easy']['gain_percent']),
                hard_gain=ci(lambda g: g['policies'][family][key]['subsets']['hard']['gain_percent']),
                switch_rate=ci(lambda g: g['policies'][family][key]['switch_rate']),
                safety=api.aggregate_risk(groups, family, amplitude))
        paired = {f:ci(lambda g: g['paired'][key][f])
                  for f in ('known_pp', 'completion_lower_pp', 'completion_upper_pp')}
        curve[key] = dict(arms=arms, EqMotion_minus_current_PCT=paired)
    focal = curve[str(cfg['primary_amplitude'])]['arms']['eqmotion']
    lower = focal['completion_lower_gain']['CI95']
    return dict(result_source='fresh_SOURCE_amplitude_reduction_of_cached_verified_forecasts',
        source_rows=registration['source_rows'], source_recordings=registration['source_recordings'],
        source_localities=roster, groups=72, policy_group_evaluations=72*2*len(cfg['amplitudes']),
        amplitudes=cfg['amplitudes'], primary_amplitude=cfg['primary_amplitude'], curve=curve,
        quarter_amplitude_full_record_risk_pass=focal['safety']['full_contract_pass'],
        quarter_amplitude_lower_gain_supported=bool(lower is not None and lower[0] > 0),
        chosen_deployment_amplitude=None, threshold_selection=False, new_training=False,
        independent_confirmation=False, private_independent_arrays_read=False,
        deployment_changed=False, submission_ready=False, stage5c_executed=False, smc_enabled=False)


def evaluate(cfg, data, jobs, whitelist, registration, *, verify=False, resume=False):
    start = time.monotonic()
    end = PUBLIC/('verification.json' if verify else 'complete.json')
    if end.exists(): raise ValueError('Completed receipts are immutable')
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('Resume completed groups unchanged')))
    def beat(**kw):
        if time.monotonic()-start > cfg['hard_runtime_limit_seconds']: raise TimeoutError('12-hour phase boundary; resume')
        if shutil.disk_usage(ROOT).free < cfg['disk_reserve_bytes']: raise OSError('Preserve10GiB disk reserve')
        row = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                   phase='verify' if verify else 'evaluate', **kw)
        run.write(PRIVATE/'heartbeat.json', row)
        with (PRIVATE/'events.jsonl').open('a') as log: log.write(json.dumps(row)+'\n')
        print(json.dumps(row), flush=True)
    beat(state='runtime_ready', architecture=platform.machine(), threads=4, workers=0, new_training=False)
    primary = json.loads((run.PUBLIC/'prediction_freeze.json').read_text())
    controls = json.loads((parent.PUBLIC/'prediction_freeze.json').read_text())
    groups = []
    for (key, design, identity), eqref, cref in zip(jobs, primary['predictions'], controls['predictions']):
        if key != cref['trial']: raise ValueError('Fixed producer order changed')
        meta = json.loads((ROOT/eqref['path']).read_text())
        if (run.sha(ROOT/eqref['path']) != eqref['sha256']
                or run.sha(ROOT/meta['prediction']) != meta['sha256']
                or run.sha(ROOT/cref['history_prediction']) != cref['history_sha256']):
            raise ValueError('Complete frozen prediction identities required')
        with np.load(ROOT/meta['prediction'], allow_pickle=False) as z:
            full_ids, p = z['ids'], z['prediction'].astype(float)
        with np.load(ROOT/cref['history_prediction'], allow_pickle=False) as z:
            ids, q = z['ids'], z['prediction'].astype(float)
        expected = whitelist[np.isin(whitelist, design['held_ids'])]
        np.testing.assert_array_equal(ids, expected)
        p = p[parent.aligned_rows(full_ids, ids)]
        floor = run.source.baseline_numpy(data['history'][ids], design['baseline_index'])-data['origin'][ids, None]
        cv = run.source.baseline_numpy(data['history'][ids], 1)-data['origin'][ids, None]
        y = data['target_eval'][ids]-data['origin'][ids, None]
        valid = np.asarray(data['valid'][ids]); sites = data['sites'][ids]; recordings = data['recordings'][ids]
        for site in sorted(set(sites)):
            path = PUBLIC/'groups'/(key+'_'+site+'.json')
            if path.exists() and resume and not verify:
                row = json.loads(path.read_text())
                if row['registration_sha256'] != run.sha(REG): raise ValueError('Partial group registration changed')
                groups.append(row); continue
            if path.exists() and not verify: raise ValueError('Partial evaluation requires --resume')
            if verify and not path.exists(): raise ValueError('Cannot verify absent group')
            use = sites == site
            b, c, target, mask, rec = floor[use], cv[use], y[use], valid[use], recordings[use]
            be, _ = metrics.errors(b, target, mask); ce, _ = metrics.errors(c, target, mask)
            easy = np.isfinite(ce) & (ce > 0) & (ce <= design['easy_cut'])
            policies = {a:{} for a in ('eqmotion', 'history_PCT')}
            diagnosis = {a:{} for a in policies}; paired = {}
            for amplitude in cfg['amplitudes']:
                akey = str(amplitude)
                ep, qp = api.attenuate(b, p[use], amplitude), api.attenuate(b, q[use], amplitude)
                for family, pred in (('eqmotion', ep), ('history_PCT', qp)):
                    policies[family][akey] = metrics.policy(pred, b, c, target, mask, rec,
                        easy_cut=design['easy_cut'], hard_cut=design['hard_cut'])
                    diagnosis[family][akey] = api.risk_diagnosis(pred, b, target, mask, rec,
                        easy=easy, baseline_cost=be)
                paired[akey] = metrics.paired(qp, ep, b, target, mask)
            row = dict(trial=key, seed=identity['seed'], site=site,
                registration_sha256=run.sha(REG), ids_sha256=run.source.array_hash(ids[use]),
                rows=int(use.sum()), baseline_index=int(design['baseline_index']),
                policies=policies, diagnosis=diagnosis, paired=paired)
            run.once(path, row); groups.append(row)
            beat(state='group_verified' if verify else 'group_complete', trial=key, site=site, groups=len(groups), total=72)
        del p, q, floor, cv, y
    summary = summarize(groups, cfg, registration)
    if verify and not (PUBLIC/'summary.json').exists(): raise ValueError('No primary summary to replay')
    run.once(PUBLIC/'summary.json', summary)
    result = dict(pid=os.getpid(), seconds=time.monotonic()-start,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        registration_sha256=run.sha(REG), summary_sha256=run.sha(PUBLIC/'summary.json'),
        groups=len(groups), policy_group_evaluations=summary['policy_group_evaluations'],
        metric_replay_exact=verify, fresh_Torch_inference=False, new_training=False,
        independent_confirmation=False)
    if not verify:
        result['groups_manifest'] = {str((PUBLIC/'groups'/(g['trial']+'_'+g['site']+'.json')).relative_to(ROOT)):
            run.sha(PUBLIC/'groups'/(g['trial']+'_'+g['site']+'.json')) for g in groups}
    run.once(end, result)
    print(json.dumps({k:v for k,v in result.items() if k != 'groups_manifest'}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('register', 'evaluate', 'verify'))
    p.add_argument('--resume', action='store_true'); args = p.parse_args()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        values = admission(register=args.phase == 'register')
        if args.phase != 'register': evaluate(*values, verify=args.phase == 'verify', resume=args.resume)


if __name__ == '__main__': main()
