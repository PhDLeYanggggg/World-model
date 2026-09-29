"""Allocated-node diagnostic and exact full replay of frozen risk packets."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
import fcntl

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from src.world_model.m3w_boundary_diagnostic import diagnostic


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_once(p, value):
    text = json.dumps(value, indent=2, allow_nan=False)+'\n'
    if p.exists():
        assert p.read_text() == text, 'Immutable output mismatch'
    else:
        tmp = p.with_suffix('.tmp'); tmp.write_text(text); os.replace(tmp, p)


def strict_mean(v):
    defined = [x for x in v if x is not None]
    return dict(mean=float(np.mean(defined)) if defined and len(defined) == len(v) else None,
                conditional_mean=float(np.mean(defined)) if defined else None,
                defined=len(defined), undefined=len(v)-len(defined))


def verify_result(r):
    count = 0
    assert r['rows'] == r['known_rows']+r['unknown_rows']
    for arm in r['arms'].values():
        for d in [n for e in arm['scopes'].values() for n in e.values()]+arm['bin_excess']:
            assert d['selected_rows'] == d['known_selected']+d['unknown_selected']
            expected = d['predicted_excess']+d['harm_underestimate']+d['reference_overestimate']
            np.testing.assert_allclose(expected, d['true_excess'], rtol=1e-10, atol=1e-8)
            np.testing.assert_allclose(d['true_harm']-.02*d['true_reference'], d['true_excess'], rtol=1e-10, atol=1e-8)
            if d['true_reference'] > 0:
                np.testing.assert_allclose(d['realized_risk_ratio'], d['true_harm']/d['true_reference'])
                np.testing.assert_allclose(d['true_excess_over_selected_reference'], d['realized_risk_ratio']-.02)
            else:
                assert d['realized_risk_ratio'] is None and d['true_excess_over_selected_reference'] is None
            assert not d['risk_certified']
            assert d['tail_harm'] <= d['true_harm']+1e-8
            assert d['tail_positive_underestimate'] <= d['positive_harm_underestimate']+1e-8
            count += 8
        assert sum(b['rows'] for b in arm['bins']) == arm['scopes']['eligible']['all']['selected_rows']
    return count


def aggregate(records, cfg):
    keys = ['realized_risk_ratio', 'predicted_risk_ratio', 'true_excess_over_selected_reference',
        'predicted_excess_over_selected_reference', 'harm_underestimate_over_selected_reference',
        'reference_overestimate_over_selected_reference', 'tail_harm_share',
        'tail_positive_underestimate_share', 'harmful_switch_fraction']
    result = {}
    comparison = []
    for phase in ('fitting', 'internal_transfer'):
        rows = [r for r in records if r['meta']['phase'] == phase]
        result[phase] = dict(groups=len(rows), rows=sum(r['rows'] for r in rows),
            known_rows=sum(r['known_rows'] for r in rows), unknown_rows=sum(r['unknown_rows'] for r in rows),
            query_occurrences=sum(r['query_count'] for r in rows), arms={})
        for arm in cfg['arms']:
            result[phase]['arms'][arm] = {}
            for scope in ('eligible', 'selected', 'matched', 'near_boundary'):
                result[phase]['arms'][arm][scope] = {}
                for event in ('all', 'easy'):
                    by_site = {}
                    for site in sorted({r['meta']['site'] for r in rows}):
                        subset = [r['arms'][arm]['scopes'][scope][event] for r in rows if r['meta']['site'] == site]
                        by_site[site] = {k: strict_mean([r[k] for r in subset]) for k in keys}
                    ds = [r['arms'][arm]['scopes'][scope][event] for r in rows]
                    z = dict(by_site=by_site,
                        equal_site={k: strict_mean([s[k]['mean'] for s in by_site.values()]) for k in keys},
                        defined_views=sum(d['realized_risk_ratio'] is not None for d in ds),
                        violating_views=sum(d['realized_risk_ratio'] is not None and d['realized_risk_ratio'] > .02+1e-10 for d in ds),
                        unknown_selected=sum(d['unknown_selected'] for d in ds),
                        selected_rows=sum(d['selected_rows'] for d in ds),
                        known_selected=sum(d['known_selected'] for d in ds))
                    result[phase]['arms'][arm][scope][event] = z
        if phase == 'internal_transfer':
            for r in rows:
                comparison.append(dict(view=r['meta']['view'],
                    risk={arm: {scope: {event: r['arms'][arm]['scopes'][scope][event]['realized_risk_ratio']
                        for event in ('all', 'easy')} for scope in ('selected', 'matched')} for arm in cfg['arms']}))
    return dict(phases=result, parent_readout_comparison=comparison,
                result_source='fresh_run_frozen_action_diagnostic_on_CREATE',
                inference_policy_changed=False, independent_confirmation=False)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--home', required=True)
    p.add_argument('--resume', action='store_true'); a = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Scientific analysis requires allocated compute node')
    home = Path(a.home); cfg = json.loads((home/'config.json').read_text())
    reg = json.loads((home/'registration.json').read_text()); manifest = json.loads((home/'manifest.json').read_text())
    assert cfg['groups'] == 288 and cfg['parameter_updates'] == 0 and cfg['risk_budget'] == .02
    assert not any(cfg[k] for k in ('independent_roles_read','threshold_search','deployment_changed','stage5c_executed','smc_enabled'))
    for k, h in reg['remote_code'].items():
        assert digest(ROOT/k) == h
    assert len(manifest['packets']) == cfg['groups']
    (home/'groups').mkdir(exist_ok=True)
    if (home/'receipt.json').exists():
        raise ValueError('Already completed: verify rather than duplicate')
    began = time.monotonic(); refs = []; records = []; arithmetic = 0
    def beat(phase, complete):
        out = dict(job_id=os.environ['SLURM_JOB_ID'], pid=os.getpid(), phase=phase, groups=complete,
                   utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        (home/'heartbeat.json').write_text(json.dumps(out)+'\n')
        with (home/'events.jsonl').open('a') as f:
            f.write(json.dumps(out)+'\n')
        print(json.dumps(out), flush=True)
    with (home/'process.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for entry in manifest['packets']:
            path = home/'inputs'/(entry['group']+'.npz'); assert digest(path) == entry['sha256']
            dest = home/'groups'/(entry['group']+'.json')
            if dest.exists() and not a.resume:
                raise ValueError('Use resume for completed groups')
            with np.load(path, allow_pickle=False) as z:
                meta = json.loads(str(z['meta_json'])); arrays = {k: z[k] for k in z.files if k != 'meta_json'}
                r = diagnostic(arrays, meta, cfg)
            r['packet_sha256'] = entry['sha256']; arithmetic += verify_result(r)
            write_once(dest, r); refs.append(dict(path=str(dest.relative_to(home)), sha256=digest(dest)))
            records.append(r)
            if len(refs) % 12 == 0: beat('diagnose', len(refs))
        first_seconds = time.monotonic()-began
        for i, entry in enumerate(manifest['packets']):
            path = home/'inputs'/(entry['group']+'.npz'); assert digest(path) == entry['sha256']
            with np.load(path, allow_pickle=False) as z:
                r = diagnostic({k: z[k] for k in z.files if k != 'meta_json'}, json.loads(str(z['meta_json'])), cfg)
            r['packet_sha256'] = entry['sha256']
            assert r == records[i]
            if (i+1) % 12 == 0: beat('exact_replay', i+1)
        summary = aggregate(records, cfg)
        summary['accounting_checks'] = arithmetic
        write_once(home/'summary.json', summary)
        receipt = dict(groups=288, fitting_groups=72, transfer_groups=216, first_pass_seconds=first_seconds,
            replay_seconds=time.monotonic()-began-first_seconds, artifacts=refs,
            summary_sha256=digest(home/'summary.json'), config_sha256=digest(home/'config.json'),
            manifest_sha256=digest(home/'manifest.json'), registration_sha256=digest(home/'registration.json'),
            job_id=os.environ['SLURM_JOB_ID'], pid=os.getpid(), full_replay_exact=True, accounting_checks=arithmetic,
            numpy=np.__version__, peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            parameter_updates=0, independent_roles_read=False, deployment_changed=False)
        write_once(home/'receipt.json', receipt); beat('complete', 288)


if __name__ == '__main__': main()
