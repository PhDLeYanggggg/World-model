"""Allocated-node, source-only selected-set calibration with exact replay."""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from src.world_model import m3w_selected_set_calibration as api


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def once(path, value):
    text = json.dumps(value, indent=2, allow_nan=False)+'\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_text() == text, 'Immutable output differs'
    else:
        with path.open('x') as f:
            f.write(text)


def compute(z, cfg):
    meta = json.loads(str(z['meta_json']))
    for k, h in meta['array_hashes'].items():
        assert api.action_hash(z[k]) == h
    result = api.evaluate(z['p'], z['y'], z['env'], z['moving'], z['support'],
        z['recordings'], rounds=cfg['round_cap'], quantile=cfg['empirical_quantile'])
    for key in ('parent_oof_action_hashes','parent_source','parent_final'):
        assert result[key] == meta[key], 'Parent reproduction failed'
    known = np.isfinite(z['y']).all(1)
    return dict(identity=meta['identity'], full_known_reference=float(z['y'][known,2].sum()),
        source=meta['source'], rows_in_packet=len(z['p']), result=result)


def interval(pairs, cfg):
    sites = sorted({s for s,v in pairs if v is not None})
    if not sites:
        return dict(defined_views=0, undefined_views=len(pairs), locality_count=0,
                    mean=None, CI95=None)
    v = np.array([np.mean([x for t,x in pairs if t == s and x is not None]) for s in sites])
    rng = np.random.default_rng(cfg['bootstrap_seed'])
    samples = v[rng.integers(0, len(v), (cfg['bootstrap_resamples'],len(v)))].mean(1)
    n = sum(x is not None for _,x in pairs)
    return dict(defined_views=n, undefined_views=len(pairs)-n, locality_count=len(v),
        mean=float(v.mean()), CI95=np.quantile(samples,[.025,.975]).tolist(),
        strict_all_views_mean=float(v.mean()) if n == len(pairs) else None,
        nominal_development_only=True)


def summary(groups, cfg):
    rows = [dict(r, source=g['source'], total_ref=g['full_known_reference'])
            for g in groups for r in g['result']['rows']]
    out = {}
    for mode in api.parent.MODES:
        for role in ('source_oof','source_resubstitution'):
            subset = [r for r in rows if r['mode'] == mode and r['role'] == role]
            d = dict(views=len(subset))
            for arm in ('parent','selected_set'):
                rr = [r[arm] for r in subset]
                risks = [r['easy_selected_risk_upper'] for r in rr]
                d[arm] = dict(complete_support_pass=sum(r['finite_completion_supported'] for r in rr),
                    selected_occurrences=sum(r['selected_count'] for r in rr),
                    selected_unknown_occurrences=sum(r['selected_unknown'] for r in rr),
                    nonempty_views=sum(r['selected_count'] > 0 for r in rr),
                    easy_risk_defined=sum(v is not None for v in risks),
                    easy_risk_undefined=sum(v is None for v in risks),
                    easy_risk_violations=sum(v is not None and v > .02+1e-12 for v in risks),
                    easy_risk_worst=max((v for v in risks if v is not None),default=None))
            d['utility_change_percent_full_reference'] = interval([(r['source'],
                100*(r['selected_set']['selected_net_gain_lower_mass']-r['parent']['selected_net_gain_lower_mass'])/r['total_ref']
                if r['total_ref'] > 0 else None) for r in subset],cfg)
            d['common_defined_easy_risk_change'] = interval([(r['source'],
                r['selected_set']['easy_selected_risk_upper']-r['parent']['easy_selected_risk_upper']
                if all(r[a]['easy_selected_risk_upper'] is not None for a in ('parent','selected_set')) else None)
                for r in subset],cfg)
            d['lost_complete_support'] = sum(r['parent']['finite_completion_supported'] and
                not r['selected_set']['finite_completion_supported'] for r in subset)
            d['gained_complete_support'] = sum(not r['parent']['finite_completion_supported'] and
                r['selected_set']['finite_completion_supported'] for r in subset)
            calibrators = [f['calibration'] for r in subset for f in r['folds']] if role == 'source_oof' else [r['final'] for r in subset]
            d['calibrator_status_counts'] = dict(Counter(c['status'] for c in calibrators))
            d['max_rounds_used'] = max((len(c['trace']) for c in calibrators),default=0)
            out[mode+'_'+role] = d
    return dict(comparisons=out, source_groups=len(groups), source_only=True,
        result_source='fresh_run_source_calibration_on_cached_verified_forecasts',
        neural_updates=0, transfer_evaluated=False, independent_roles_read=False,
        deployment_promoted=False, stage5c_executed=False, smc_enabled=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--home',required=True)
    p.add_argument('--resume',action='store_true')
    args = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Use allocated CREATE CPU node')
    home = Path(args.home)
    cfg = json.loads((home/'config.json').read_text())
    reg = json.loads((home/'registration.json').read_text())
    for name, h in reg['remote_code'].items():
        assert digest(ROOT/name) == h
    assert digest(home/'config.json') == reg['config_sha256']
    manifest = json.loads((home/'input_manifest.json').read_text())
    assert manifest['registration_sha256'] == digest(home/'registration.json')
    assert len(manifest['packets']) == cfg['source_heads']
    assert not (home/'complete.json').exists(), 'Completed run: inspect, not rerun'
    torch.set_num_threads(cfg['cpu_threads']); torch.set_num_interop_threads(1)
    started = time.monotonic()
    with (home/'process.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        initial = []
        for phase in ('compute','replay'):
            refs, groups, used = [], [], 0
            for i, entry in enumerate(manifest['packets']):
                path = home/'inputs'/(entry['group']+'.npz')
                assert digest(path) == entry['sha256']
                out = home/'groups'/(entry['group']+'.json')
                if phase == 'compute' and out.exists() and not args.resume:
                    raise RuntimeError('Existing group requires --resume')
                with np.load(path,allow_pickle=False) as z:
                    got = compute(z,cfg)
                text_bytes = len(json.dumps(got,indent=2,allow_nan=False).encode())+1
                used += text_bytes
                assert used < cfg['output_byte_cap'], 'Output cap reached; preserve groups'
                once(out,got)
                refs.append(dict(group=entry['group'],sha256=digest(out),bytes=out.stat().st_size))
                groups.append(got)
                beat = dict(job_id=os.environ['SLURM_JOB_ID'],pid=os.getpid(),phase=phase,
                    groups=i+1,seconds=time.monotonic()-started,
                    peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
                (home/'heartbeat.json').write_text(json.dumps(beat)+'\n')
                if i == 0 or (i+1)%12 == 0:
                    print(json.dumps(beat),flush=True)
            report = summary(groups,cfg)
            once(home/'summary.json',report)
            if phase == 'compute': initial = refs
            else: assert refs == initial
        once(home/'complete.json',dict(groups=refs,exact_replay=True,source_heads=len(refs),
            summary_sha256=digest(home/'summary.json'),job_id=os.environ['SLURM_JOB_ID'],
            seconds=time.monotonic()-started,new_neural_updates=0,transfer_evaluated=False,
            independent_roles_read=False,registration_sha256=digest(home/'registration.json')))
        print(json.dumps(dict(complete=True,source_heads=len(initial))),flush=True)


if __name__ == '__main__':
    main()
