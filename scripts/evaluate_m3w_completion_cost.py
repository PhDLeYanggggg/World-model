"""Read committed final completion-cost heads once, then replay exactly."""
import argparse
import fcntl
import json
from pathlib import Path
import platform
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 required before Torch import')
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from scripts import run_m3w_completion_cost as run
from src.evaluation import m3w_completion_cost_readout as api
from src.evaluation import m3w_multiaction_cost_readout as base
from scripts.report_m3w_feature_cost_evidence import aggregate_groups

REG = run.PUBLIC/'readout_registration.json'
FILES = (Path(__file__), ROOT/'src/evaluation/m3w_completion_cost_readout.py',
         ROOT/'tests/test_m3w_completion_cost_readout.py', run.PUBLIC/'protocol.md')


def admission(register=False):
    cfg, values = run.admission(); learned = run.models()
    run.once(REG, dict(bindings={str(p.relative_to(ROOT)): run.sha(p) for p in FILES},
        training_freeze_sha256=run.sha(run.PUBLIC/'training_freeze.json'),
        first_full_replay_sha256=run.sha(run.PUBLIC/'first_full_replay.json'),
        training_registration_sha256=run.sha(run.REG), required_fitting_groups=144,
        primary='completion_bounds_minus_known_only_physical_neural', scope='original_fitting_only',
        cached_neural_only_reference_not_newly_trained=True, original_unknown_labels_preserved=True,
        proxy_target_MSE_is_not_actual_future_error_calibration=True,
        held_scored=False, independent_roles_opened=False, deployment_changed=False))
    if not register:
        for rel, digest in json.loads(REG.read_text())['bindings'].items():
            if run.sha(ROOT/rel) != digest:
                raise ValueError('Registered paired readout changed: '+rel)
            run.parent.parent.committed(ROOT/rel)
        run.parent.parent.committed(REG)
    return cfg, values, learned


def execute(verify=False, resume=False):
    cfg, values, learned = admission(); finish = run.PUBLIC/'complete.json'
    if finish.exists() and not verify:
        raise ValueError('Completed paired readout immutable; use verify')
    started, beat, guard = run.runtime(cfg)
    old_groups = aggregate_groups(run.parent.NAME)
    cached_refs = run.parent.models()
    _, origin_refs = run.parent.parent.old.direct.ORIGINS.models(values[2])
    rows, refs = [], []
    for context, items in sorted(values[2]['contexts'].items()):
        packets = run.packets(context, items, values, beat)
        data = run.api.training_data(packets, run.parent)
        input_hashes, preprocess = data['input_hashes'], data['preprocess']; del data
        job = next(j for j in values[4] if j['key'] == context.split('_controller')[0])
        for seed in cfg['seeds']:
            states = {(bank, mode): run.api.base.read_checkpoint(ROOT/learned[context, seed, bank, mode]['checkpoint']['path'])
                      for bank, mode in run.cells(cfg)}
            cached = {bank: run.api.base.read_checkpoint(ROOT/cached_refs[context, seed, bank]['checkpoint']['path'])
                      for bank in run.api.base.ARMS}
            for state in states.values():
                run.api.base.exact(input_hashes, state['input_hashes'])
                run.api.base.exact(preprocess, state['preprocess'])
            for state in cached.values():
                run.api.base.exact(preprocess, state['preprocess'])
            origin = run.parent.parent.old.base.core.read_checkpoint(ROOT/origin_refs[context, seed, 'self']['checkpoint']['path'])
            originals = run.parent.parent.old.direct.origin_predictions(origin, {s: p['args'] for s, p in packets.items()})
            for source, packet in sorted(packets.items()):
                guard(); kw = packet['x'], packet['scene'], packet['envelope'], packet['scales']
                predictions = {key: run.api.predict(state, *kw) for key, state in states.items()}
                old_predictions = {bank: run.parent.api.predict(state, *kw) for bank, state in cached.items()}
                choices, ranks = {}, {}
                for (bank, mode), p in predictions.items():
                    choices[bank, mode], ranks[bank, mode] = run.api.base.select_actions(p, packet['allowed'], arm=bank)
                old_choices, old_ranks = {}, {}
                for bank, p in old_predictions.items():
                    old_choices[bank], old_ranks[bank] = run.api.base.select_actions(p, packet['allowed'], arm=bank)
                op = originals[source]
                take = run.parent.parent.old.base.actions(op, run.parent.parent.old.base.shared.eligible(packet['args'])) & packet['allowed'][:, 6]
                for mode in cfg['modes']:
                    path = run.PUBLIC/'groups'/f'{context}_{source}_seed{seed}_{mode}_fit.json'
                    if path.exists() and not (verify or resume):
                        raise ValueError('Explicit partial readout resume required')
                    current = {bank: choices[bank, mode] for bank in cfg['banks']}
                    order = {bank: ranks[bank, mode] for bank in cfg['banks']}
                    pred = {bank: predictions[bank, mode] for bank in cfg['banks']}
                    current['neural_only'], order['neural_only'] = old_choices['neural_only'], old_ranks['neural_only']
                    pred['neural_only'] = old_predictions['neural_only']
                    for bank in cfg['banks']:
                        current[bank+'_static_parent'], order[bank+'_static_parent'] = old_choices[bank], old_ranks[bank]
                    current['frozen_origin_binary'] = np.where(take, 6, -1)
                    order['frozen_origin_binary'] = op[:, 0]-op[:, 1]
                    current['floor'], order['floor'] = np.full(len(take), -1, np.int64), np.zeros(len(take))
                    result = base.group_readout(packet, current, order, pred)
                    old = old_groups[f'{context}_{source}_seed{seed}_fit.json']
                    for bank in run.api.base.ARMS:
                        name = bank if bank == 'neural_only' else bank+'_static_parent'
                        if (result['policies'][name] != old['result']['policies'][bank]
                                or run.parent.fingerprint(current[name]) != old['action_sha256'][bank]):
                            raise ValueError('Frozen static parent fitting readout changed')
                    result['supervision_contrasts'] = {bank: api.paired(packet,
                        choices[bank, 'known_only'], choices[bank, 'completion_bounds'],
                        ranks[bank, 'known_only'], ranks[bank, 'completion_bounds']) for bank in cfg['banks']}
                    result['parent_contrasts'] = {bank: api.paired(packet,
                        old_choices[bank], choices[bank, mode], old_ranks[bank], ranks[bank, mode]) for bank in cfg['banks']}
                    result['proxy_diagnostics'] = {bank: api.proxy_diagnostics(packet, pred[bank], bank) for bank in cfg['banks']}
                    result['legacy_calibration_role'] = 'descriptive_known_outcomes_all_seven_actions_not_a_risk_certificate'
                    result['neural_only_reference'] = 'cached_verified_static_head_not_trained_in_this_experiment'
                    result['subsets'] = api.subsets(packet, current, job['design']['easy_cut'])
                    value = dict(context=context, source=source, seed=seed, mode=mode, scope='fit', result=result,
                        action_sha256={name: run.parent.fingerprint(c) for name, c in current.items()},
                        prediction_sha256={name: run.parent.fingerprint(p) for name, p in pred.items()},
                        registration_sha256=run.sha(REG), result_source='fresh_fitting_inference_of_new_paired_Torch_heads',
                        original_unknown_labels_preserved=True, future_mask_inference_input=False, held_scored=False)
                    if verify:
                        if value != json.loads(path.read_text()):
                            raise ValueError('Exact paired completion readout differs')
                    else:
                        run.once(path, value)
                    rows.append(value); refs.append(dict(path=str(path.relative_to(ROOT)), sha256=run.sha(path)))
            beat(state='completion_readout_progress', context=context, seed=seed, groups=len(rows), total=144)
        del packets
    summary = api.summarize(rows, values[2]['contexts'], values[1])
    if verify:
        if summary != json.loads((run.PUBLIC/'summary.json').read_text()) or refs != json.loads(finish.read_text())['groups']:
            raise ValueError('Exact full paired summary differs')
        run.once(run.PUBLIC/'verification.json', dict(exact_replay=True, groups=144, summary_sha256=run.sha(run.PUBLIC/'summary.json')))
    else:
        run.once(run.PUBLIC/'summary.json', summary)
        run.once(finish, dict(pid=run.os.getpid(), seconds=time.monotonic()-started, groups=refs,
            peak_RSS_bytes=run.rss_bytes(), summary_sha256=run.sha(run.PUBLIC/'summary.json'),
            held_scored=False, independent_roles_opened=False, new_optimizer_updates=0))
    run.json_write(run.PRIVATE/('verification_runtime.json' if verify else 'readout_runtime.json'),
        dict(pid=run.os.getpid(), seconds=time.monotonic()-started, peak_RSS_bytes=run.rss_bytes()))
    beat(state='completion_readout_verified' if verify else 'completion_readout_complete', groups=144)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'run', 'verify')); parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(); run.PRIVATE.mkdir(parents=True, exist_ok=True)
    with (run.PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.phase == 'register':
            admission(register=True)
        else:
            execute(args.phase == 'verify', args.resume)
