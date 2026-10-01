"""Frozen-source validation diagnostic, not a new deployment policy."""
import argparse
from collections import Counter
import fcntl
import json
import math
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
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts import run_m3w_source_policy_selection as parent
from src.world_model.m3w_unknown_outcome_bounds import completion_bounds

NAME = 'european_unknown_outcome_bounds_v1'
PUBLIC, PRIVATE = parent.PUBLIC.parent/NAME, parent.PRIVATE.parent/NAME
CONFIG = ROOT/'configs'/('m3w_'+NAME+'.json')
base, core, digest, immutable = parent.base, parent.core, parent.digest, parent.immutable
inner = parent.parent.parent.parent


def registration():
    cfg = json.loads(CONFIG.read_text())
    path = parent.PUBLIC/'verification.json'
    assert digest(path) == cfg['parent_seal_sha256']
    seal = json.loads(path.read_text())
    for p, h in seal['source_bindings'].items():
        assert digest(ROOT/p) == h, p
    for p, h in seal['artifacts'].items():
        assert digest(parent.PUBLIC/p) == h, p
    assert cfg['risk_budget'] == .02 and cfg['easy_degradation_limit'] == .02
    assert not any(cfg[k] for k in ('parameter_updates', 'new_policy_selection',
        'transfer_outcome_readout', 'independent_roles_read', 'deployment_changed',
        'stage5c_executed', 'smc_enabled'))
    paths = parent.closure(ROOT, ['scripts.run_m3w_unknown_outcome_bounds'])
    paths += [CONFIG, PUBLIC/'protocol.md', ROOT/'tests/test_m3w_unknown_outcome_bounds.py']
    return cfg, dict(parent_seal_sha256=digest(path),
        bindings={str(p.relative_to(ROOT)): digest(p) for p in paths},
        candidates=216, source_fits=72, no_new_policy_selection=True)


def beat(**kw):
    row = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **kw)
    base.inter.json_write(PRIVATE/'heartbeat.json', row)
    with (PRIVATE/'events.jsonl').open('a') as f:
        f.write(json.dumps(row)+'\n')
    print(json.dumps(row), flush=True)


def independent_accounting(y, take, envelope, result):
    """Scalar summation, separate from the production vectorized bound."""
    known = np.isfinite(y).all(1)
    values = dict(benefit=[], harm=[], ref=[], easy_ref=[], easy_harm=[], easy_benefit=[])
    unknown, full_easy_ref = [], []
    for row, selected, ok, radius in zip(y, take, known, envelope):
        if ok:
            full_easy_ref.append(float(row[3]))
        if not selected:
            continue
        if not ok:
            unknown.append(float(radius))
            continue
        for k, value in zip(values, [*row[:3], row[3], row[4], row[0] if row[3] else 0.]):
            values[k].append(float(value))
    mass = {k: math.fsum(v) for k, v in values.items()}
    d, full_ref = math.fsum(unknown), math.fsum(full_easy_ref)
    ratio = lambda n, r: n/r if r > 0 else None
    easy_n = mass['easy_harm']-mass['easy_benefit']+d
    expected = dict(
        selected_unknown_envelope_mass=d,
        selected_net_gain_lower_mass=mass['benefit']-mass['harm']-d,
        all_budget_slack_mass=.02*mass['ref']-mass['harm'],
        easy_budget_slack_mass=.02*mass['easy_ref']-mass['easy_harm'],
        all_selected_risk_upper=ratio(mass['harm']+d, mass['ref']),
        easy_selected_risk_upper=ratio(mass['easy_harm']+d, mass['easy_ref']),
        easy_degradation_upper=ratio(max(easy_n, 0) if (~known).any() else easy_n, full_ref))
    for k, value in expected.items():
        if value is None:
            assert result[k] is None
        else:
            np.testing.assert_allclose(result[k], value, rtol=1e-10, atol=1e-8)
    return len(expected)


def run(replay=False):
    started = time.monotonic()
    fits = parent.fit_docs()
    choices = {r['group']: r for r in json.loads((parent.PUBLIC/'source_choices.json').read_text())['rows']}
    _, _, data, jobs, oid, _, _, _ = inner.old.load()
    causal = {k: data[k] for k in inner.old.parent.CAUSAL_KEYS}
    rows, checks, ids_seen, partial_seen = [], 0, set(), set()
    for c in base.floor_api.contexts(causal, jobs, oid):
        for site in inner.sources(c):
            group = c['name']+'_fit_'+site
            state = core.read_checkpoint(ROOT/fits[group]['checkpoint']['path'])
            at, ids, x, env, y, _, identity = inner.training_arrays(c, data, site)
            assert identity == state['identity']['upstream'] and state['step'] == 2000
            _, val, partition = parent.parent.api.source_partition(data['recordings'][ids], data['frames'][ids], site)
            assert partition == state['partition'] == choices[group]['partition']
            vid = ids[val]
            assert base.inter.array_hash(vid) == choices[group]['source_validation_ids_hash']
            predictions, support = parent.parent.api.paired_predictions(state, x[val], env[val])
            initial, other = core.predict(state, x[val], env[val], initial=True)
            np.testing.assert_array_equal(support, other)
            predictions = dict(mse=predictions['validation'], final=predictions['final'], initial=initial)
            actions = {k: parent.causal_action(p, c['moving'][at][val], support,
                data['recordings'][vid], data['frames'][vid], vid) for k, p in predictions.items()}
            assert parent.api.choose(y[val], actions) == choices[group]['selection']
            ids_seen.update(map(int, vid))
            counts = data['valid'][vid].sum(1)
            partial_seen.update(map(int, vid[(counts > 0) & (counts < 12)]))
            floor, neural = c['floor'][at][val].astype(float), c['prediction'][at][val].astype(float)
            disagreement = np.linalg.norm(neural-floor, axis=-1).max(1)
            assert np.all(disagreement <= env[val]+1e-10)
            for name, take in actions.items():
                assert base.inter.array_hash(predictions[name]) == choices[group]['validation_score_hashes'][name]
                assert base.inter.array_hash(take) == choices[group]['validation_action_hashes'][name]
                before = base.inter.array_hash(take)
                bound = completion_bounds(y[val], take, env[val])
                assert before == base.inter.array_hash(take)
                checks += independent_accounting(y[val], take, env[val], bound)
                old = choices[group]['selection']['validation'][name]
                rows.append(dict(group=group, source=site, head=name,
                    checkpoint_step={'initial': 0, 'mse': state['best_step'], 'final': state['step']}[name],
                    ids_hash=choices[group]['source_validation_ids_hash'], action_hash=before,
                    target_hash=base.inter.array_hash(y[val]), envelope_hash=base.inter.array_hash(env[val]),
                    old=old, bound=bound,
                    rejected_only_unknown=old['reasons'] == ['selected_outcomes_unknown_on_source_validation']))
            if len(rows) % 36 == 0:
                beat(state='source_bounds', candidate_views=len(rows), source_fits=len(rows)//3)
    assert len(rows) == 216
    immutable(PUBLIC/'readout.json', dict(rows=rows, source_fits=72,
        independent_scalar_checks=checks, unique_source_validation_ids=len(ids_seen),
        unique_partial_label_ids=len(partial_seen), no_new_policy_selection=True,
        new_parameter_updates=0, transfer_outcome_readout=False))
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() != 'Darwin':
        rss *= 1024
    immutable(PUBLIC/('replay.json' if replay else 'runtime.json'), dict(
        seconds=time.monotonic()-started, peak_RSS_bytes=rss, pid=os.getpid(),
        all216_exact_replay=replay, source_selection_reconstructed=True,
        envelope_contract_checked=True, new_row_cache_bytes=0, new_checkpoint_bytes=0))


def report(reg):
    assert json.loads((PUBLIC/'replay.json').read_text())['all216_exact_replay']
    doc = json.loads((PUBLIC/'readout.json').read_text())
    rows = doc['rows']
    summarize = lambda rr: dict(
        candidate_views=len(rr), supported=sum(r['bound']['finite_completion_supported'] for r in rr),
        supported_trained=sum(r['bound']['finite_completion_supported'] and r['checkpoint_step'] > 0 for r in rr),
        reasons=dict(Counter(x for r in rr for x in r['bound']['reasons'])))
    subsets = dict(all=rows, unknown_only=[r for r in rows if r['rejected_only_unknown']],
        no_selected_unknown=[r for r in rows if not r['old']['selected_unknown']],
        prior_supported=[r for r in rows if r['old']['eligible']],
        previously_risk_failed=[r for r in rows if any('exceeds_budget' in v for v in r['old']['reasons'])])
    summary = dict(subsets={k: summarize(v) for k, v in subsets.items()},
        by_head={n: summarize([r for r in rows if r['head'] == n]) for n in parent.api.NAMES},
        supported=[{k: r[k] for k in ('group', 'source', 'head', 'checkpoint_step', 'rejected_only_unknown')}
                   for r in rows if r['bound']['finite_completion_supported']],
        independent_scalar_checks=doc['independent_scalar_checks'],
        unique_source_validation_ids=doc['unique_source_validation_ids'],
        unique_partial_label_ids=doc['unique_partial_label_ids'],
        no_new_policy_selection=True, new_training=False, transfer_evaluation='not_run',
        independent_confirmation=False, deployment_changed=False)
    immutable(PUBLIC/'summary.json', summary)
    tests = subprocess.run([sys.executable, '-m', 'pytest', 'tests/test_m3w_unknown_outcome_bounds.py',
        'tests/test_m3w_source_policy_selection.py', '-q'], cwd=ROOT, capture_output=True, text=True)
    if tests.returncode:
        raise RuntimeError(tests.stdout+tests.stderr)
    (PUBLIC/'scoped_tests.txt').write_text(tests.stdout+tests.stderr)
    immutable(PUBLIC/'verification.json', dict(status='verified_development_finite_completion_diagnostic',
        source_bindings=reg['bindings'], artifacts={p.name: digest(p) for p in PUBLIC.iterdir()
            if p.is_file() and p.name != 'verification.json'},
        exact_replay=True, independent_scalar_checks=doc['independent_scalar_checks'],
        no_new_policy_selection=True, independent_confirmation=False, full_legacy_suite='not_run'))
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['register', 'run', 'replay', 'report'])
    phase = parser.parse_args().phase
    PUBLIC.mkdir(parents=True, exist_ok=True); PRIVATE.mkdir(parents=True, exist_ok=True)
    cfg, reg = registration()
    if phase == 'register':
        immutable(PUBLIC/'registration.json', reg)
        print('Registered fixed-action completion diagnostic; no new policy selection')
        return
    assert json.loads((PUBLIC/'registration.json').read_text()) == reg
    base.inter.committed(PUBLIC/'registration.json')
    if shutil.disk_usage(PRIVATE).free < cfg['maximum_new_artifact_bytes']+2**30:
        raise OSError('Insufficient space for bounded aggregate-only diagnostic')
    core.torch.set_num_threads(4); core.torch.set_num_interop_threads(1)
    with (PRIVATE/'process.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        beat(state='starting', phase=phase)
        if phase == 'report':
            report(reg)
        else:
            run(phase == 'replay')
        beat(state='complete', phase=phase)


if __name__ == '__main__':
    main()
