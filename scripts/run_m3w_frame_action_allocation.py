"""Registered same-bank current-query allocation, checkpointed and replayable."""
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
    raise RuntimeError('Native arm64 .venv-pytorch required before Torch')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts import run_m3w_gradient_balanced_admission as source
from scripts.report_m3w_feature_cost_evidence import aggregate_groups
from src.world_model import m3w_frame_action_allocation as api
from src.evaluation import m3w_frame_action_readout as evaluation
from src.evaluation.m3w_multiaction_support import solve_action_bank

NAME = 'european_frame_action_allocation_v1'
PUBLIC = ROOT / 'outputs/publication_readiness_2026_09' / NAME
PRIVATE = ROOT / 'data/stage_cvpr2027_experiments' / NAME
CONFIG = ROOT / 'configs/m3w_frame_action_allocation_v1.json'
REG = PUBLIC / 'registration.json'
BASE = source.parent.parent.frozen
sha, once, write = source.sha, source.once, source.write
FILES = (Path(__file__), CONFIG, ROOT / 'src/world_model/m3w_frame_action_allocation.py',
    ROOT / 'src/evaluation/m3w_frame_action_readout.py', ROOT / 'tests/test_m3w_frame_action_allocation.py',
    ROOT / 'tests/test_m3w_frame_action_readout.py', PUBLIC / 'protocol.md')


def admission(register=False):
    cfg = json.loads(CONFIG.read_text())
    _, values, store, banks, _ = source.admission()
    check = json.loads((source.PUBLIC / 'verification.json').read_text())
    if (cfg['anchor_objective'] != 'admitted_score' or cfg['seeds'] != [17, 29, 43]
            or cfg['risk_budget'] != .02 or cfg['actions'] != 7 or cfg['groups'] != 72
            or cfg['new_training_updates'] or not check['all72groups_exact']
            or check['summary_sha256'] != sha(source.PUBLIC / 'summary.json')
            or any(cfg[k] for k in ('held_scored', 'independent_roles_opened',
                'deployment_changed', 'submission_ready', 'stage5c_executed', 'smc_enabled'))):
        raise ValueError('Verified prior diagnosis and unchanged closed fitting roles required')
    models = source.parent.models()
    readout_reg = source.PUBLIC / 'readout_registration.json'
    for rel, digest in json.loads(readout_reg.read_text())['bindings'].items():
        if sha(ROOT / rel) != digest:
            raise ValueError('Frozen parent readout dependency changed: '+rel)
    refs = [dict(context=c, seed=s, checkpoint=models[c, s, 'admitted_score']['checkpoint'])
            for c in sorted(values[2]['contexts']) for s in cfg['seeds']]
    once(REG, dict(bindings={str(p.relative_to(ROOT)): sha(p) for p in FILES},
        source_complete_sha256=sha(source.PUBLIC / 'complete.json'),
        source_verification_sha256=sha(source.PUBLIC / 'verification.json'),
        source_readout_registration_sha256=sha(readout_reg),
        anchor_training_freeze_sha256=sha(source.parent.PUBLIC / 'training_freeze.json'),
        contexts=json.loads(source.REG.read_text())['contexts'], cached_verified_models=refs,
        primary='joint_matched_minus_independent_full_policy', query_key='source_recording_current_raw_frame',
        same_intervention_count_per_query=True, new_training_updates=0,
        inference_labels_used=False, pair_interactions_optimized=False,
        held_scored=False, independent_roles_opened=False, deployment_changed=False,
        submission_ready=False, stage5c_executed=False, smc_enabled=False))
    if not register:
        for rel, digest in json.loads(REG.read_text())['bindings'].items():
            if sha(ROOT / rel) != digest:
                raise ValueError('Registered frame-allocation implementation changed: '+rel)
            BASE.engine.committed(ROOT / rel)
        BASE.engine.committed(REG)
    return cfg, values, store, banks, models


def runtime(cfg):
    BASE.torch.set_num_threads(4); BASE.torch.set_num_interop_threads(1)
    start = time.monotonic()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('Resume fixed query checkpoints')))
    def guard():
        if time.monotonic()-start > cfg['hard_runtime_limit_seconds']:
            raise TimeoutError('12h boundary; resume registered query allocation')
        if shutil.disk_usage(ROOT).free < cfg['disk_reserve_bytes']:
            raise OSError('Keep10GiB free reserve')
        if sum(p.stat().st_size for p in PRIVATE.rglob('*.pt.gz')) > cfg['private_progress_cap_bytes']:
            raise OSError('Private action-progress cap exceeded')
    def beat(**event):
        guard(); row = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **event)
        write(PRIVATE / 'heartbeat.json', row)
        with (PRIVATE / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')
        print(json.dumps(row), flush=True)
    beat(state='runtime_ready', architecture=platform.machine(), threads=4, workers=0, device='cpu')
    return start, beat, guard


def infer(packet, state, store, reference):
    extra = store.read(packet['ids'], 'history')
    p = BASE.api.detector.predict(state, packet['x'], packet['scene'], extra, packet['envelope'], packet['scales'])
    c, rank = BASE.api.completion.base.select_actions(p, packet['allowed'], arm='physical_neural')
    for value, expected in ((p, reference['prediction_sha256']['admitted_score']),
            (c, reference['action_sha256']['admitted_score']), (packet['ids'], reference['ids_sha256'])):
        if BASE.engine.fingerprint(value) != expected:
            raise ValueError('Frozen causal inputs/predictions/independent choices changed')
    return p, c, rank


def pilot():
    cfg, values, store, banks, models = admission()
    if (PUBLIC / 'pilot.json').exists():
        raise ValueError('Completed real pilot immutable')
    start, beat, guard = runtime(cfg)
    context, items = sorted(values[2]['contexts'].items())[0]
    _, packets, _, _ = BASE.prepared_context(context, items, values, store, banks, beat)
    name = sorted(packets)[0]; packet = packets[name]
    reference = aggregate_groups(source.parent.NAME)[f'{context}_{name}_seed17_fit.json']
    state = BASE.api.completion.base.read_checkpoint(ROOT / models[context, 17, 'admitted_score']['checkpoint']['path'])
    p, _, _ = infer(packet, state, store, reference)
    groups = api.query_groups(packet['recordings'], packet['frames'], packet['ids'])
    selected = sorted(set(range(min(cfg['pilot_prefix_queries'], len(groups)))) |
        set(sorted(range(len(groups)), key=lambda i: (-len(groups[i]), i))[:cfg['pilot_largest_queries']]))
    at = np.sort(np.concatenate([groups[i] for i in selected]))
    args = (p[at], packet['allowed'][at], packet['recordings'][at], packet['frames'][at], packet['ids'][at])
    cp = PRIVATE / 'pilot_progress.pt.gz'
    class PlannedInterruption(Exception):
        pass
    def stop(value):
        BASE.api.completion.base.save_checkpoint(cp, value)
        raise PlannedInterruption()
    try:
        api.allocate(*args, seconds=cfg['solver_seconds_per_query'], checkpoint=stop)
    except PlannedInterruption:
        pass
    resumed = api.allocate(*args, seconds=cfg['solver_seconds_per_query'],
        resume_state=BASE.api.completion.base.read_checkpoint(cp), heartbeat=lambda **kw: beat(state='pilot_resumed', **kw))
    began = time.monotonic()
    direct = api.allocate(*args, seconds=cfg['solver_seconds_per_query'], heartbeat=lambda **kw: beat(state='pilot_direct', **kw))
    direct_seconds = time.monotonic()-began
    BASE.api.completion.base.exact(resumed, direct)
    once(PUBLIC / 'pilot.json', dict(pid=os.getpid(), seconds=time.monotonic()-start,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        context=context, source=name, seed=17, causal_queries=len(selected), rows=len(at),
        selected_query_index_sha256=BASE.engine.fingerprint(np.array(selected)),
        direct_seconds=direct_seconds, exact_real_interrupted_recovery=True,
        source_queries=len(groups), rough72group_projection_seconds=direct_seconds/len(selected)*len(groups)*72,
        projection_based_on_one_source_not_full_roster=True, diagnostics=direct[2],
        prediction_control_exact=True, outcome_scoring=False, training_updates=0,
        registration_sha256=sha(REG), engineering_pass=True))
    beat(state='real_frame_pilot_complete_no_outcome_scoring')


def execute(*, verify=False, resume=False):
    cfg, values, store, banks, models = admission()
    if (PUBLIC / 'complete.json').exists() and not verify:
        raise ValueError('Completed fixed roster immutable; use verify')
    if not json.loads((PUBLIC / 'pilot.json').read_text())['engineering_pass']:
        raise ValueError('Real recovery/runtime pilot required')
    start, beat, guard = runtime(cfg)
    old = aggregate_groups(source.parent.NAME); oracle_control = aggregate_groups(BASE.NAME)
    prior = aggregate_groups(NAME) if verify else {}
    prior_refs = {Path(r['path']).name: r for r in json.loads((PUBLIC / 'complete.json').read_text())['groups']} if verify else {}
    refs, rows = [], []
    for context, items in sorted(values[2]['contexts'].items()):
        _, packets, _, _ = BASE.prepared_context(context, items, values, store, banks, beat)
        job = next(j for j in values[4] if j['key'] == context.split('_controller')[0])
        for seed in cfg['seeds']:
            model_ref = models[context, seed, 'admitted_score']['checkpoint']
            if sha(ROOT / model_ref['path']) != model_ref['sha256']:
                raise ValueError('Frozen unit-admission head changed')
            state = BASE.api.completion.base.read_checkpoint(ROOT / model_ref['path'])
            for name, packet in sorted(packets.items()):
                key = f'{context}_{name}_seed{seed}_fit.json'; path = PUBLIC / 'groups' / key
                if path.exists() and not verify:
                    if not resume:
                        raise ValueError('Explicit partial-roster resume required')
                    row = json.loads(path.read_text())
                    if row['registration_sha256'] != sha(REG):
                        raise ValueError('Partial readout registration differs')
                else:
                    p, independent, rank = infer(packet, state, store, old[key])
                    if evaluation.bank.policy_metrics(packet, independent) != old[key]['result']['policies']['admitted_score']:
                        raise ValueError('Original complete-policy control metrics differ')
                    groups = api.query_groups(packet['recordings'], packet['frames'], packet['ids'])
                    agents = values[3]['agents'][packet['ids']]
                    if any(len(set(agents[v].astype(str))) != len(v) for v in groups):
                        raise ValueError('A current query cannot contain duplicate target agent IDs')
                    progress = PRIVATE / 'progress' / (key+'.pt.gz'); progress.parent.mkdir(parents=True, exist_ok=True)
                    if progress.exists() and not (resume or verify):
                        raise ValueError('Explicit action-checkpoint resume required')
                    choices, ranks, info = api.allocate(p, packet['allowed'], packet['recordings'],
                        packet['frames'], packet['ids'], seconds=cfg['solver_seconds_per_query'],
                        heartbeat=lambda **kw: beat(state='allocating_queries', context=context, source=name, seed=seed, **kw),
                        checkpoint=(lambda value: None) if verify else lambda value: BASE.api.completion.base.save_checkpoint(progress, value),
                        resume_state=BASE.api.completion.base.read_checkpoint(progress) if resume and progress.exists() and not verify else None)
                    np.testing.assert_array_equal(choices['independent'], independent)
                    oracle = np.full(len(p), -1, np.int64)
                    for rec in sorted(set(packet['recordings'])):
                        at = packet['recordings'] == rec
                        diagnostic, selected = solve_action_bank(packet['y'][at], packet['envelope'][at], packet['allowed'][at])
                        if diagnostic != oracle_control[key]['result']['bank_opportunity_diagnostic']['new_neural'][str(rec)]:
                            raise ValueError('Original whole-record label-aware oracle differs')
                        oracle[at] = selected
                    result = evaluation.readout(packet, choices, ranks, info, job['design']['easy_cut'],
                        values[3]['origin'][packet['ids']], oracle)
                    row = dict(context=context, source=name, seed=seed, result=result,
                        prediction_sha256=BASE.engine.fingerprint(p),
                        action_sha256={a: BASE.engine.fingerprint(c) for a, c in choices.items()},
                        ids_sha256=BASE.engine.fingerprint(packet['ids']), control_exact=True, record_oracle_exact=True,
                        registration_sha256=sha(REG), model_checkpoint_sha256=model_ref['sha256'],
                        result_source='fresh_registered_query_allocation_on_cached_verified_SOURCE_fitting_assets',
                        held_scored=False, independent_roles_opened=False)
                    if verify:
                        if row != prior[key]:
                            raise ValueError('Separate full allocation/inference/metric replay differs')
                    else:
                        once(path, row)
                rows.append(row)
                refs.append(prior_refs[key] if verify else dict(path=str(path.relative_to(ROOT)), sha256=sha(path)))
                beat(state='full_group_complete', context=context, source=name, seed=seed, groups=len(rows), total=72)
        del packets, state
    summary = evaluation.summarize(rows, json.loads(REG.read_text())['contexts'], cfg)
    if verify:
        if (summary != json.loads((PUBLIC / 'summary.json').read_text())
                or refs != json.loads((PUBLIC / 'complete.json').read_text())['groups']):
            raise ValueError('Complete fixed-roster summary differs')
        once(PUBLIC / 'verification.json', dict(all72groups_exact=True,
            all72prediction_action_full_metric_controls_exact=True, all444record_oracles_exact=True,
            summary_sha256=sha(PUBLIC / 'summary.json'), complete_sha256=sha(PUBLIC / 'complete.json')))
    else:
        once(PUBLIC / 'summary.json', summary)
        once(PUBLIC / 'complete.json', dict(pid=os.getpid(), seconds=time.monotonic()-start,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            groups=refs, summary_sha256=sha(PUBLIC / 'summary.json'), training_updates=0,
            held_scored=False, independent_roles_opened=False))
    write(PRIVATE / ('verification_runtime.json' if verify else 'readout_runtime.json'),
        dict(pid=os.getpid(), seconds=time.monotonic()-start, peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    beat(state='all72groups_verified' if verify else 'all72groups_complete', fitting_advancement_gate=summary['fitting_advancement_gate'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'pilot', 'run', 'verify'))
    parser.add_argument('--resume', action='store_true'); args = parser.parse_args()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE / 'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.phase == 'register':
            admission(register=True)
        elif args.phase == 'pilot':
            pilot()
        else:
            execute(verify=args.phase == 'verify', resume=args.resume)
