"""Registered finite-completion source screen with unchanged utility ranking."""
import argparse
from collections import Counter, defaultdict
import fcntl
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 required before Torch import')
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from scripts import run_m3w_unknown_outcome_bounds as parent
from src.world_model import m3w_completion_screen_policy as api

NAME = 'european_completion_screen_policy_v1'
PUBLIC, PRIVATE = parent.PUBLIC.parent/NAME, parent.PRIVATE.parent/NAME
CONFIG = ROOT/'configs'/('m3w_'+NAME+'.json')
old, base, core = parent.parent, parent.base, parent.core
digest, immutable = parent.digest, parent.immutable


def registration():
    cfg = json.loads(CONFIG.read_text()); path = parent.PUBLIC/'verification.json'
    assert digest(path) == cfg['parent_seal_sha256']
    seal = json.loads(path.read_text())
    for p, h in seal['source_bindings'].items(): assert digest(ROOT/p) == h, p
    for p, h in seal['artifacts'].items(): assert digest(parent.PUBLIC/p) == h, p
    assert cfg['risk_budget'] == .02 and cfg['candidate_order'] == list(api.NAMES)
    assert not any(cfg[k] for k in ('parameter_updates', 'threshold_search', 'independent_roles_read',
        'deployment_changed', 'stage5c_executed', 'smc_enabled'))
    paths = old.closure(ROOT, ['scripts.run_m3w_completion_screen_policy'])
    paths += [CONFIG, PUBLIC/'protocol.md', ROOT/'tests/test_m3w_completion_screen_policy.py']
    return cfg, dict(parent_seal_sha256=digest(path),
        bindings={str(p.relative_to(ROOT)): digest(p) for p in paths},
        new_parameter_updates=0, independent_roles_read=False)


def beat(**kw):
    row = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **kw)
    base.inter.json_write(PRIVATE/'heartbeat.json', row)
    with (PRIVATE/'events.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
    print(json.dumps(row), flush=True)


def select():
    groups = defaultdict(dict)
    for row in json.loads((parent.PUBLIC/'readout.json').read_text())['rows']:
        assert row['head'] not in groups[row['group']]
        groups[row['group']][row['head']] = row
    controls = {r['group']: r for r in json.loads((old.PUBLIC/'source_choices.json').read_text())['rows']}
    assert set(groups) == set(controls) and len(groups) == 72
    rows = []
    for group, candidates in groups.items():
        choice = api.choose(candidates); head = choice['selected']
        rows.append(dict(group=group, source=controls[group]['source'], selection=choice,
            control_head=controls[group]['selection']['selected'],
            selected_checkpoint_step=candidates[head]['checkpoint_step'] if head != 'fallback' else None,
            source_validation_ids_hash=controls[group]['source_validation_ids_hash'],
            checkpoint=controls[group]['checkpoint']))
    immutable(PUBLIC/'source_choices.json', dict(rows=rows, parameter_updates=0, independent_calibration=False))


def views(data, jobs, oid):
    base.inter.committed(PUBLIC/'source_choices.json')
    choices = {r['group']: r for r in json.loads((PUBLIC/'source_choices.json').read_text())['rows']}
    controls = {r['view']: r for r in json.loads((old.PUBLIC/'decision_freeze.json').read_text())['rows']}
    for c, at, ids, p, _, pr, meta in old.parent.views(data, jobs, oid):
        choice = choices[c['name']+'_fit_'+meta['source']]
        z = (c['x'][at]-pr['mean'])/pr['std']
        support = np.sqrt(np.mean(z*z, axis=1)) <= pr['support_limit']
        available = dict(mse=p['validation'], final=p['final'], fallback=np.zeros_like(p['validation']))
        names = [choice['control_head'], choice['selection']['selected']]
        if 'initial' in names:
            assert base.artifact(ROOT/choice['checkpoint']['path']) == choice['checkpoint']
            state = core.read_checkpoint(ROOT/choice['checkpoint']['path'])
            available['initial'], ok = core.predict(state, c['x'][at], c['env'][at], initial=True)
            np.testing.assert_array_equal(support, ok)
        scores = dict(affine=available[names[0]], nonlinear=available[names[1]])
        chosen = core.decisions(scores, c['moving'][at], support, data['recordings'][ids], data['frames'][ids], ids)
        actions = {name: chosen[key] for name, key in [('control', 'affine'), ('bounded', 'nonlinear'),
            ('control_matched', 'affine_matched'), ('bounded_matched', 'nonlinear_matched')]}
        assert base.inter.array_hash(actions['control']) == controls[meta['view']]['action_hashes']['selected']
        out = dict(view=meta['view'], site=meta['site'], source=meta['source'], seed=meta['seed'],
            control_head=names[0], bounded_head=names[1], selected_checkpoint_step=choice['selected_checkpoint_step'],
            ids_hash=meta['ids_hash'], score_hashes={k: base.inter.array_hash(v) for k, v in scores.items()},
            action_hashes={k: base.inter.array_hash(v) for k, v in actions.items()})
        yield c, at, ids, actions, out


def decide(data, jobs, oid):
    rows = []
    for _, _, _, _, meta in views(data, jobs, oid):
        rows.append(meta)
        if len(rows) % 36 == 0: beat(state='causal_decisions', views=len(rows))
    assert len(rows) == 216
    immutable(PUBLIC/'decision_freeze.json', dict(rows=rows, causal_only=True, control_hashes_exact=True))


def evaluate(data, jobs, oid, replay=False):
    base.inter.committed(PUBLIC/'decision_freeze.json'); started = time.monotonic()
    expected = {r['view']: r for r in json.loads((PUBLIC/'decision_freeze.json').read_text())['rows']}
    rows = []; checks = queries = completion_checks = 0
    for c, at, ids, actions, meta in views(data, jobs, oid):
        assert meta == expected[meta['view']]
        cv, cf, (floor, ff), (neural, nf) = base.floor_api.costs(c, data, at)
        y = core.targets(cv, floor, neural, c['job']['design']['easy_cut'])
        known = np.isfinite(cv); easy = known & (cv > 0) & (cv <= c['job']['design']['easy_cut'])
        _, groups = np.unique(np.rec.fromarrays([data['recordings'][ids], data['frames'][ids]]), return_inverse=True)
        counts = {k: np.bincount(groups, weights=a.astype(int)) for k, a in actions.items()}
        matched = np.minimum(counts['control'], counts['bounded'])
        for name in ('control_matched', 'bounded_matched'):
            np.testing.assert_array_equal(counts[name], matched)
        queries += len(matched)
        for name, take in {**actions, 'floor': np.zeros(len(ids), bool)}.items():
            m = base.floor_api.metric(cv, floor, neural, cf, ff, nf,
                np.where(take, neural, floor), np.where(take, nf, ff), take, data['valid'][ids],
                c['job']['design']['easy_cut'], c['job']['design']['hard_cut'])
            selected_easy = known & easy & take; ref = float(floor[selected_easy].sum())
            m['selected_easy_positive_harm_ratio'] = float(np.maximum(neural-floor, 0)[selected_easy].sum())/ref if ref > 0 else None
            checks += old.accounting.audit_metric(m, cv, floor, neural, cf, ff, nf, take, data['valid'][ids],
                c['job']['design']['easy_cut'], c['job']['design']['hard_cut'])
            bound = parent.completion_bounds(y, take, c['env'][at])
            completion_checks += parent.independent_accounting(y, take, c['env'][at], bound)
            rows.append(dict(**meta, policy=name, metric=m, completion_diagnostic=bound))
        if len(rows) % 180 == 0: beat(state='outcome_readout', views=len(rows)//5)
    assert len(rows) == 1080
    immutable(PUBLIC/'readout.json', dict(rows=rows, independent_metric_checks=checks,
        independent_completion_checks=completion_checks, query_count_checks=queries,
        new_parameter_updates=0, independent_confirmation=False, deployment_changed=False))
    immutable(PUBLIC/('evaluation_replay.json' if replay else 'evaluation_runtime.json'),
        dict(seconds=time.monotonic()-started, all216_frozen_decisions_reverified=True, exact_readout_replay=replay))


def report(cfg, reg):
    assert json.loads((PUBLIC/'evaluation_replay.json').read_text())['exact_readout_replay']
    assert json.loads((PUBLIC/'source_choice_replay.json').read_text())['all72_exact']
    doc = json.loads((PUBLIC/'readout.json').read_text()); rows = doc['rows']
    choices = json.loads((PUBLIC/'source_choices.json').read_text())['rows']
    indexed = {(r['view'], r['policy']): r for r in rows}
    ci = lambda pairs: old.locality_interval(pairs, cfg['bootstrap_resamples'], cfg['bootstrap_seed'])
    contrasts = {}; seed_contrasts = {}
    for label, a, b in [('primary_unmatched', 'control', 'bounded'),
                         ('secondary_matched', 'control_matched', 'bounded_matched')]:
        items = []
        for row in rows:
            if row['policy'] != a: continue
            other = indexed[row['view'], b]
            x, y = row['metric']['error_sum'], other['metric']['error_sum']
            items.append(dict(site=row['site'], seed=row['seed'], gain=100*(x-y)/x if x > 0 else None))
        contrasts[label] = ci([(r['site'], r['gain']) for r in items])
        seed_contrasts[label] = {str(seed): ci([(r['site'], r['gain']) for r in items if r['seed'] == seed])
                                for seed in (17, 29, 43)}
    policies = {}
    for name in ('control', 'bounded', 'control_matched', 'bounded_matched'):
        rr = [r for r in rows if r['policy'] == name]
        policies[name] = {k: ci([(r['site'], r['metric'][k]) for r in rr]) for k in
            ('all_gain_floor', 'easy_gain_floor', 'hard_gain_floor', 'FDE_gain_floor', 'intervention_rate')}
        for label, key in [('all', 'selected_positive_harm_ratio'), ('easy', 'selected_easy_positive_harm_ratio')]:
            policies[name][label+'_risk'] = old.accounting.risk_coverage(rr, key)
        policies[name]['worst_easy_degradation_percent'] = max(-r['metric']['easy_gain_floor'] for r in rr
            if r['metric']['easy_gain_floor'] is not None)
        policies[name]['finite_completion_supported_views'] = sum(r['completion_diagnostic']['finite_completion_supported'] for r in rr)
    summary = dict(contrasts=contrasts, by_seed=seed_contrasts, policies=policies,
        source_choices=dict(Counter(r['selection']['selected'] for r in choices)),
        nonfallback_chosen_trained_steps=sum((r['selected_checkpoint_step'] or 0) > 0 for r in choices),
        independent_metric_checks=doc['independent_metric_checks'],
        independent_completion_checks=doc['independent_completion_checks'], query_count_checks=doc['query_count_checks'],
        independent_confirmation=False, deployment_changed=False, new_training=False)
    immutable(PUBLIC/'summary.json', summary)
    tests = subprocess.run([sys.executable, '-m', 'pytest', 'tests/test_m3w_completion_screen_policy.py',
        'tests/test_m3w_unknown_outcome_bounds.py', 'tests/test_m3w_source_policy_selection.py', '-q'],
        cwd=ROOT, capture_output=True, text=True)
    if tests.returncode: raise RuntimeError(tests.stdout+tests.stderr)
    (PUBLIC/'scoped_tests.txt').write_text(tests.stdout+tests.stderr)
    immutable(PUBLIC/'verification.json', dict(status='verified_development_completion_screen_contrast',
        source_bindings=reg['bindings'], artifacts={p.name: digest(p) for p in PUBLIC.iterdir()
            if p.is_file() and p.name != 'verification.json'}, exact_readout_replay=True,
        independent_metric_checks=doc['independent_metric_checks'], independent_completion_checks=doc['independent_completion_checks'],
        independent_confirmation=False, deployment_changed=False, full_legacy_suite='not_run'))
    print(json.dumps(summary, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['register', 'select', 'replay_select', 'decide', 'evaluate', 'replay_evaluate', 'report'])
    phase = p.parse_args().phase
    PUBLIC.mkdir(parents=True, exist_ok=True); PRIVATE.mkdir(parents=True, exist_ok=True)
    cfg, reg = registration()
    if phase == 'register':
        immutable(PUBLIC/'registration.json', reg); print('Registered completion-screen policy contrast'); return
    assert json.loads((PUBLIC/'registration.json').read_text()) == reg
    base.inter.committed(PUBLIC/'registration.json')
    if shutil.disk_usage(PRIVATE).free < cfg['maximum_new_artifact_bytes']+2**30:
        raise OSError('Insufficient space for aggregate-only policy contrast')
    core.torch.set_num_threads(4); core.torch.set_num_interop_threads(1)
    with (PRIVATE/'process.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); beat(state='starting', phase=phase)
        if phase in ('select', 'replay_select'):
            select()
            if phase == 'replay_select': immutable(PUBLIC/'source_choice_replay.json', dict(all72_exact=True))
        elif phase == 'report': report(cfg, reg)
        else:
            _, _, data, jobs, oid, _, _, _ = parent.inner.old.load()
            if phase == 'decide': decide(data, jobs, oid)
            else: evaluate(data, jobs, oid, phase == 'replay_evaluate')
        beat(state='complete', phase=phase)


if __name__ == '__main__': main()
