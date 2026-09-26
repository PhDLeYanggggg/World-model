"""Training-only frozen gradients; two disposable optimizer steps per head."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import sys
import time
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Use native .venv-pytorch arm64')
from scripts import run_m3w_european_membership_auxiliary as parent
from src.evaluation import m3w_task_gradients as method
import numpy as np
import torch

PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_task_gradients_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_task_gradients_v1'
CONFIG = 'configs/m3w_european_task_gradients_v1.json'
artifact, digest, immutable_json = parent.artifact, parent.digest, parent.immutable_json
FILES = [CONFIG, 'src/evaluation/m3w_task_gradients.py', 'src/evaluation/m3w_source_gradient_diagnostic.py',
         'tests/test_m3w_task_gradients.py', 'scripts/run_m3w_european_task_gradients.py',
         str(PUBLIC.relative_to(ROOT)/'protocol.md')]


def registration(create=False):
    cfg = json.loads((ROOT/CONFIG).read_text()); _, pid = parent.registration()
    v = json.loads((parent.PUBLIC/'verification.json').read_text()); assert v['all_passed']
    for p, h in v['artifacts'].items(): assert digest(parent.PUBLIC/p) == h
    for p, h in v['source_bindings'].items(): assert digest(ROOT/p) == h
    assert cfg['frozen_heads'] == 288 and cfg['new_persisted_training_updates'] == 0
    assert not any(cfg[k] for k in ('held_outcome_readout', 'selection_access', 'reserved_calibration_access',
        'confirmation_access', 'deployment_changed', 'stage5c_executed', 'smc_enabled'))
    identity = dict(parent=pid, parent_verification=artifact(parent.PUBLIC/'verification.json'),
                    bindings={p: digest(ROOT/p) for p in FILES})
    path = PUBLIC/'registration_lock.json'
    if create: immutable_json(path, identity)
    else:
        assert json.loads(path.read_text()) == identity
        parent.base.previous.require_committed(path)
    return cfg, identity


def beat(**kwargs):
    row = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **kwargs)
    parent.base.base.cross.json_write(PRIVATE/'heartbeat.json', row)
    with (PRIVATE/'events.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
    print(json.dumps(row), flush=True)


def fitting_inputs(g, data, pairs, pair, held):
    base = parent.base; bi = pairs['B']['ids']; sites = data['sites'][bi]; tr = sites != held
    ids = bi[tr]; x, env, reference, candidate = pairs['B'][pair]
    # Slice before accessing future labels. Held/C labels are not diagnostic inputs.
    cv = data['baseline_ade'][ids, 1]
    def error(pred):
        return base.native_errors(pred[tr].astype(float)+data['origin'][ids, None],
            data['target_eval'][ids], data['valid'][ids], np.ones(len(ids)))[0]
    by = base.method.event_targets(cv, error(reference), error(candidate), g['easy_cut'])
    pr = base.base.ordinary.preprocess(x[tr], by[:, :2], cv, sites[tr], held)
    y = parent.tail.diagnostic.event_targets(by, cv, pr['positive_easy_cut'])
    easy = parent.parent.parent.method.labels(cv, pr['positive_easy_cut'])
    assert held not in pr['training_sites'] and held not in g['producer_roster']
    return x[tr], env[tr], y, easy, sites[tr], ids, pr


def compute(cfg, identity, verify=False):
    parent.checked_training(identity['parent']); refs = []; started = time.monotonic()
    for g, data, pairs in parent.base.contexts(identity['parent']['source']):
        name = g['group']; sites = data['sites'][pairs['B']['ids']]
        for pair in cfg['pairs']:
            path = PUBLIC/'groups'/(name+'_'+pair+'.json'); receipt = PRIVATE/'readout'/path.name
            if path.exists() and not verify:
                assert artifact(path) == json.loads(receipt.read_text()); refs.append(artifact(path)); continue
            records = []
            for held in sorted(set(sites)):
                x, env, y, easy, fitting_sites, ids, pr = fitting_inputs(g, data, pairs, pair, held)
                tag = name+'_'+pair+'_'+held
                for arm in cfg['arms']:
                    directory = parent.PRIVATE/'heads'/tag/arm; cp = artifact(directory/'checkpoint.pt')
                    r = json.loads((directory/'complete.json').read_text()); inp = r['input']
                    for k, a in (('train_x_sha256', x), ('train_y_sha256', y), ('train_easy_sha256', easy), ('train_ids_sha256', ids)):
                        assert inp[k] == parent.array_hash(a), k
                    model, state = parent.method.restore(directory)
                    assert state['step'] == 2000 and state['arm'] == arm
                    for k in ('mean', 'std', 'known', 'weights'): np.testing.assert_array_equal(state['preprocess'][k], pr[k])
                    assert state['preprocess']['cost_scale'] == pr['cost_scale']
                    batch = method.fitting_batch(x, env, y, easy, fitting_sites, held, state)
                    diagnostic = method.diagnose(model, state, batch)
                    assert artifact(directory/'checkpoint.pt') == cp
                    if arm == 'cost_only': assert diagnostic['shared_BCE_relation']['cost']['norm'] == 0
                    records.append(dict(held=held, arm=arm, checkpoint=cp, fitting_ids_sha256=parent.array_hash(ids),
                        fixed_batch_ids_sha256=parent.array_hash(ids[state['fixed_ids']]), diagnostic=diagnostic))
            row = dict(group=name, pair=pair, producer=g['producer'], controller=g['controller'],
                seed=int(name.split('_seed')[1].split('_')[0]), records=records,
                result_source='fresh_run_gradients_and_virtual_steps_cached_verified_models_and_fitting_labels',
                held_outcomes_used=False, new_persisted_updates=0, policy_changed=False)
            if verify: assert json.loads(path.read_text()) == row
            else: immutable_json(path, row); immutable_json(receipt, artifact(path))
            refs.append(artifact(path)); beat(state='replayed' if verify else 'computed', groups=len(refs), group=name, pair=pair)
    assert len(refs) == 36
    parent.checked_training(identity['parent'])
    out = dict(identity=identity, groups=refs, all_passed=True, frozen_heads=288, disposable_optimizer_steps=576,
        new_persisted_updates=0, held_outcome_readout=False, checkpoint_hashes_unchanged=True)
    immutable_json(PUBLIC/('replay_receipt.json' if verify else 'completion_checks.json'), out)
    beat(state='replay_complete' if verify else 'complete', elapsed_seconds=time.monotonic()-started)


def report(cfg, identity):
    done = json.loads((PUBLIC/'completion_checks.json').read_text()); assert done['identity'] == identity
    rows = []
    for ref in done['groups']:
        assert artifact(ROOT/ref['path']) == ref
        g = json.loads((ROOT/ref['path']).read_text())
        for r in g['records']: rows.append(dict(group=g['group'], pair=g['pair'], producer=g['producer'],
            controller=g['controller'], arm=r['arm'], diagnostic=r['diagnostic']))
    def select(pair, arm, producer=None, controller=None):
        return method.population([r['diagnostic'] for r in rows if r['pair'] == pair and r['arm'] == arm
            and (producer is None or r['producer'] == producer) and (controller is None or r['controller'] == controller)],
            cfg['effect_relative_tolerance'])
    populations = {p: {a: select(p, a) for a in cfg['arms']} for p in cfg['pairs']}
    roles = sorted(set((r['producer'], r['controller']) for r in rows))
    assignments = {p: {a: {f'{u}->{v}': select(p, a, u, v) for u, v in roles} for a in cfg['arms']} for p in cfg['pairs']}
    full = populations['full']['membership_aux']; n = full['dependent_fitting_views']
    gradient = full['gradients']['cost']['negative_cosines'] > n/2
    step = full['optimizer']['cost']['auxiliary_worse'] > n/2
    decision = dict(majority_negative_cost_cosine=gradient, majority_virtual_cost_worse=step,
        registered_gradient_repair_motivated=gradient and step, method_improvement_proven=False,
        deployment_changed=False, independent_confirmation=False, stage5c_executed=False, smc_enabled=False)
    aggregate = dict(populations=populations, assignments=assignments, decision=decision)
    immutable_json(PUBLIC/'aggregate_metrics.json', aggregate)
    lines = ['# Frozen Task-Gradient Results', '',
        'Fresh fitting-batch calculation; cached_verified checkpoints and source data. No new held-out readout.',
        '288 frozen heads; 576 disposable optimizer steps, zero persisted updates. All parent checkpoint hashes unchanged.', '',
        '| Features / frozen arm | Views | Negative cost cosine | Median cosine | Auxiliary virtual cost worse / better / tied |',
        '|---|---:|---:|---:|---|']
    for p, arms in populations.items():
        for a, v in arms.items():
            grad = v['gradients']['cost']; opt = v['optimizer']['cost']
            lines.append(f"| {p} / {a} | {v['dependent_fitting_views']} | {grad['negative_cosines']} | {grad['cosine']['median']} | {opt['auxiliary_worse']} / {opt['auxiliary_better']} / {opt['tied']} |")
    lines += ['', '## Components', '', '| Features / arm / component | Negative cosine | Median cosine | Worse / better / tied |', '|---|---:|---:|---|']
    for p, arms in populations.items():
        for a, v in arms.items():
            for k in ('H', 'H_E'):
                grad, opt = v['gradients'][k], v['optimizer'][k]
                lines.append(f"| {p} / {a} / {k} | {grad['negative_cosines']} | {grad['cosine']['median']} | {opt['auxiliary_worse']} / {opt['auxiliary_better']} / {opt['tied']} |")
    lines += ['', '## Decision', '', '```json', json.dumps(decision, indent=2), '```', '',
        'Dependent fitting views, not independent samples or causal explanations of the full training path.',
        'Raw gradient conflict alone is not AdamW interference. Full assignment detail and changes in every component are retained in aggregate_metrics.json.',
        'The cost-only control has zero membership shared gradient by construction; an undefined cosine is not evidence of alignment.',
        'One virtual update may worsen even cost-only loss because stored momentum, curvature and the fixed batch differ from ongoing stochastic training.',
        'No metric/seconds, true3D, foundation, physical safety, new deployment or submission-ready claim. Stage5C and SMC remain off.']
    (PUBLIC/'results.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(aggregate, indent=2))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--phase', required=True, choices=['register', 'run', 'verify', 'report']); args = ap.parse_args()
    PUBLIC.mkdir(parents=True, exist_ok=True); PRIVATE.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    with (PRIVATE/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cfg, identity = registration(args.phase == 'register')
        if args.phase in ('run', 'verify'): compute(cfg, identity, args.phase == 'verify')
        elif args.phase == 'report': report(cfg, identity)


if __name__ == '__main__': main()
