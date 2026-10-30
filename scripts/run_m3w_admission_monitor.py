"""Replay19 complete frozen fits with non-intervening loss/gradient telemetry."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import signal
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 required before Torch')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts import run_m3w_admission_cost as parent
from src.world_model import m3w_admission_monitor as api

NAME = 'european_admission_monitor_v1'
PUBLIC = ROOT / 'outputs/publication_readiness_2026_09' / NAME
PRIVATE = ROOT / 'data/stage_cvpr2027_experiments' / NAME
REG = PUBLIC / 'registration.json'
FILES = (Path(__file__), ROOT / 'src/world_model/m3w_admission_monitor.py',
    ROOT / 'tests/test_m3w_admission_monitor.py', PUBLIC / 'protocol.md')


def admission(register=False):
    cfg, values, store, banks, _ = parent.admission()
    check = json.loads((parent.PUBLIC / 'verification.json').read_text())
    if not check['all72groups_exact'] or check['summary_sha256'] != parent.sha(parent.PUBLIC / 'summary.json'):
        raise ValueError('Verified completed parent required')
    parent.once(REG, dict(bindings={str(p.relative_to(ROOT)): parent.sha(p) for p in FILES},
        parent_freeze_sha256=parent.sha(parent.PUBLIC / 'training_freeze.json'),
        parent_verification_sha256=parent.sha(parent.PUBLIC / 'verification.json'),
        contexts=json.loads(parent.REG.read_text())['contexts'],
        full_replays=19, diagnostic_optimizer_updates=38000,
        original_scientific_fields_exact_required=True, no_new_policy=True,
        held_scored=False, independent_roles_opened=False, deployment_changed=False))
    if not register:
        for rel, digest in json.loads(REG.read_text())['bindings'].items():
            if parent.sha(ROOT / rel) != digest:
                raise ValueError('Registered instrumentation changed')
            parent.parent.frozen.engine.committed(ROOT / rel)
        parent.parent.frozen.engine.committed(REG)
    return cfg, values, store, banks, parent.models()


def summarize(receipts):
    selected = [r for r in receipts if r['objective_arm'] == 'admitted_score']
    if len(receipts) != 19 or len(selected) != 18 or any(not r['scientific_state_exact'] for r in receipts):
        raise ValueError('Complete19exact replays required')
    monitors = [r['telemetry'] for r in selected]
    snapshots = [v for m in monitors for v in m['rows']]
    totals = {k: sum(m[k] for m in monitors) for k in ('step', 'row_draws', 'unknown_draws',
        'unknown_supervised_draws', 'selected_rows', 'false_all_rows', 'false_easy_rows', 'steps_aux_positive')}
    sums = {k: sum(m['summed_losses'][k] for m in monitors) for k in monitors[0]['summed_losses']}
    def distribution(values):
        x = np.array([v for v in values if v is not None], float)
        return dict(defined=len(x), quantiles=np.quantile(x, [0, .1, .5, .9, 1]).tolist() if len(x) else None)
    ratios = {k: distribution([r[k] for r in snapshots]) for k in ('auxiliary_over_anchor_gradient',
        'easy_auxiliary_over_anchor_gradient', 'auxiliary_anchor_cosine', 'easy_auxiliary_anchor_cosine')}
    return dict(replays=19, actual_diagnostic_optimizer_updates=38000,
        admitted_training_totals=totals, admitted_summed_losses=sums,
        auxiliary_over_anchor_loss=sums['aux']/(sums['base']+sums['upper']),
        first_aux_positive_steps=[m['first_aux_positive_step'] for m in monitors],
        snapshot_gradient_distributions=ratios, snapshots=len(snapshots),
        snapshot_false_easy_gate_under_001=sum(r['false_easy_gate_fraction_under_001'] for r in snapshots),
        snapshot_false_easy_gate_defined=sum(r['false_easy_gate_fraction_defined'] for r in snapshots),
        all19_original_scientific_states_exact=True,
        result_source='fresh_instrumented_full_Torch_training_replay_of_cached_verified_SOURCE_fits',
        no_new_candidate_or_policy=True, effectiveness_recomputed=False,
        independent_confirmation=False, held_scored=False, independent_roles_opened=False,
        deployment_changed=False, submission_ready=False, stage5c_executed=False, smc_enabled=False)


def execute(resume=False):
    cfg, values, store, banks, models = admission()
    if (PUBLIC / 'complete.json').exists():
        raise ValueError('Completed diagnosis immutable')
    parent.parent.torch.set_num_threads(4)
    parent.parent.torch.set_num_interop_threads(1)
    start = time.monotonic()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('Resume same telemetry checkpoints')))
    def guard():
        if time.monotonic()-start > 43200 or shutil.disk_usage(ROOT).free < 10*2**30:
            raise RuntimeError('12h/10GiB guard; retain fixed roster and resume')
        if sum(p.stat().st_size for p in PRIVATE.rglob('*.pt.gz')) > 128*2**20:
            raise OSError('128MiB diagnostic checkpoint cap')
    def beat(**kw):
        guard()
        v = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **kw)
        parent.write(PRIVATE / 'heartbeat.json', v)
        print(json.dumps(v), flush=True)
    beat(state='runtime_ready', threads=4, workers=0, full_replays=19)
    rows, refs = [], []
    first = sorted(values[2]['contexts'])[0]
    for context, items in sorted(values[2]['contexts'].items()):
        op, npackets, original, data = parent.parent.frozen.prepared_context(context, items, values, store, banks, beat)
        roster = [(seed, 'admitted_score') for seed in cfg['seeds']]
        if context == first:
            roster.insert(0, (17, 'upper_score'))
        for seed, arm in roster:
            path = PRIVATE / f'{context}_seed{seed}_{arm}.pt.gz'
            receipt = PUBLIC / 'replays' / f'{context}_seed{seed}_{arm}.json'
            reference = models[context, seed, arm]
            if receipt.exists():
                if not resume:
                    raise ValueError('Explicit partial resume required')
                row = json.loads(receipt.read_text())
                if parent.sha(path) != row['checkpoint']['sha256'] or row['registration_sha256'] != parent.sha(REG):
                    raise ValueError('Completed replay receipt changed')
            else:
                state = api.fit(data, objective_arm=arm, variant='new_neural', settings=cfg['training'], seed=seed,
                    identity=dict(context=context, seed=seed, arm=arm, registration_sha256=parent.sha(REG)),
                    path=path, heartbeat=lambda **kw: beat(context=context, seed=seed, objective_arm=arm, **kw),
                    guard=guard, resume=resume and path.exists())
                observed = dict(state)
                telemetry = observed.pop('admission_monitor')
                old = parent.parent.frozen.api.completion.base.read_checkpoint(ROOT / reference['checkpoint']['path'])
                parent.parent.frozen.api.assert_scientific_equal(old, observed)
                if (telemetry['step'] != 2000 or telemetry['draw_hash'] != state['draw_hash']
                        or telemetry['row_draws'] != state['row_draws'] or telemetry['unknown_supervised_draws']):
                    raise ValueError('Exact complete known-only diagnostic replay required')
                row = dict(context=context, seed=seed, objective_arm=arm, scientific_state_exact=True,
                    actual_optimizer_updates=2000, seconds=state['seconds'], telemetry=telemetry,
                    checkpoint=dict(path=str(path.relative_to(ROOT)), sha256=parent.sha(path), bytes=path.stat().st_size),
                    reference_checkpoint_sha256=reference['checkpoint']['sha256'],
                    registration_sha256=parent.sha(REG), held_scored=False, independent_roles_opened=False)
                parent.once(receipt, row)
            rows.append(row)
            refs.append(dict(path=str(receipt.relative_to(ROOT)), sha256=parent.sha(receipt)))
            beat(state='exact_replay_complete', context=context, seed=seed, arm=arm, completed=len(rows), total=19)
            if len(rows) == 1:
                parent.once(PUBLIC / 'runtime_pilot.json', dict(actual_full_updates=2000,
                    seconds=row['seconds'], peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    scientific_state_exact=True, projected_19fit_optimizer_seconds=row['seconds']*19,
                    projection_excludes_packet_preparation=True))
        del op, npackets, original, data
    summary = summarize(rows)
    parent.once(PUBLIC / 'summary.json', summary)
    parent.once(PUBLIC / 'complete.json', dict(pid=os.getpid(), seconds=time.monotonic()-start,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, replays=refs,
        summary_sha256=parent.sha(PUBLIC / 'summary.json'), actual_optimizer_updates=38000,
        no_new_policy=True, held_scored=False, independent_roles_opened=False))
    beat(state='all19_replays_exact')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'run'))
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE / 'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        admission(register=True) if args.phase == 'register' else execute(args.resume)
