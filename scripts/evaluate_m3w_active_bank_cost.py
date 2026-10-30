"""Fixed matched-bank fitting readout and separate exact policy inference replay."""
import argparse
import fcntl
import json
from pathlib import Path
import platform
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 required before Torch')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts import run_m3w_active_bank_cost as run
from scripts.report_m3w_feature_cost_evidence import aggregate_groups
from src.evaluation import m3w_active_bank_cost_readout as score

REG = run.PUBLIC / 'readout_registration.json'
FILES = (Path(__file__), ROOT / 'src/evaluation/m3w_active_bank_cost_readout.py',
         ROOT / 'tests/test_m3w_active_bank_readout.py', ROOT / 'tests/test_m3w_active_bank_inference.py',
         ROOT / 'src/evaluation/m3w_multiaction_cost_readout.py',
         ROOT / 'src/evaluation/m3w_completion_cost_readout.py',
         ROOT / 'src/evaluation/m3w_multiaction_forest_readout.py',
         ROOT / 'src/evaluation/m3w_multiaction_support.py',
         ROOT / 'src/evaluation/m3w_easy_constrained_policy.py',
         ROOT / 'src/world_model/m3w_scaled_risk_readout.py',
         ROOT / 'src/world_model/m3w_multiaction_cost.py', ROOT / 'src/world_model/m3w_detector_cost.py',
         run.PUBLIC / 'readout_protocol.md')


def admission(register=False):
    cfg, values, legacy, store, _ = run.admission()
    learned, banks = run.models(), run.cache_refs()
    run.once(REG, dict(bindings={str(p.relative_to(ROOT)): run.sha(p) for p in FILES},
        training_freeze_sha256=run.sha(run.PUBLIC / 'training_freeze.json'),
        bank_freeze_sha256=run.sha(run.PUBLIC / 'bank_complete.json'),
        old_readout_complete_sha256=run.sha(run.old.PUBLIC / 'complete.json'),
        groups=72, primary='new_neural_minus_old_neural', policy_thresholds_unchanged=True,
        original_fitting_only=True, empty_count_comparison_not_estimable=True,
        same_head_physical_restriction_diagnostic_only=True, held_scored=False,
        independent_roles_opened=False, deployment_changed=False, submission_ready=False))
    if not register:
        for rel, digest in json.loads(REG.read_text())['bindings'].items():
            if run.sha(ROOT / rel) != digest:
                raise ValueError('Registered matched-bank readout changed')
            run.engine.committed(ROOT / rel)
        run.engine.committed(REG)
    return cfg, values, legacy, store, learned, banks


def execute(verify=False, resume=False):
    cfg, values, legacy, store, learned, banks = admission()
    finish = run.PUBLIC / 'complete.json'
    if finish.exists() and not verify:
        raise ValueError('Completed matched-bank readout immutable; use verify')
    start, beat, guard = run.runtime(cfg)
    prior = aggregate_groups(run.old.NAME)
    rows, refs = [], []
    for context, items in sorted(values[2]['contexts'].items()):
        old_packets, new_packets, original, changed = run.prepared_context(context, items, values, store, banks, beat)
        job = next(j for j in values[4] if j['key'] == context.split('_controller')[0])
        for seed in cfg['seeds']:
            states = {}
            for variant in cfg['variants']:
                state = run.api.completion.base.read_checkpoint(ROOT / learned[context, seed, variant]['checkpoint']['path'])
                data = changed if variant == 'new_neural' else original
                run.api.completion.base.exact(state['preprocess'], data['preprocess'])
                run.api.completion.base.exact(state['input_hashes'], data['input_hashes'])
                states[variant] = state
            for source, old_packet in sorted(old_packets.items()):
                guard()
                packets = dict(physical_only=old_packet, old_neural=old_packet, new_neural=new_packets[source])
                extra = store.read(old_packet['ids'], 'history')
                predictions, choices, ranks = {}, {}, {}
                for variant, packet in packets.items():
                    prediction = run.api.detector.predict(states[variant], packet['x'], packet['scene'], extra,
                        packet['envelope'], packet['scales'])
                    predictions[variant] = prediction
                    choices[variant], ranks[variant] = run.api.completion.base.select_actions(
                        prediction, packet['allowed'], arm=run.api.BANKS[variant])
                original_row = prior[f'{context}_{source}_seed{seed}_history_fit.json']
                result = score.group_readout(packets, choices, ranks, predictions, job['design']['easy_cut'])
                for variant in ('physical_only', 'old_neural'):
                    key = run.api.BANKS[variant]
                    if (run.engine.fingerprint(predictions[variant]) != original_row['prediction_sha256'][key]
                            or run.engine.fingerprint(choices[variant]) != original_row['action_sha256'][key]
                            or result['policies'][variant] != original_row['result']['policies'][key]):
                        raise ValueError('Fresh original cost predictions/actions/scores differ')
                    for record, metrics in result['recordings'].items():
                        if metrics[variant] != original_row['result']['recordings'][record][key]:
                            raise ValueError('Original required-recording risk replay differs')
                path = run.PUBLIC / 'groups' / f'{context}_{source}_seed{seed}_fit.json'
                if path.exists() and not (verify or resume):
                    raise ValueError('Explicit partial readout resume required')
                value = dict(context=context, source=source, seed=seed, scope='fit', result=result,
                    action_sha256={v: run.engine.fingerprint(c) for v, c in choices.items()},
                    prediction_sha256={v: run.engine.fingerprint(p) for v, p in predictions.items()},
                    ids_sha256=run.engine.fingerprint(old_packet['ids']), legacy_readout_exact=True,
                    old_group_scientific_hash=run.engine.fingerprint(np.frombuffer(json.dumps(original_row, sort_keys=True).encode(), dtype=np.uint8)),
                    registration_sha256=run.sha(REG), training_freeze_sha256=run.sha(run.PUBLIC / 'training_freeze.json'),
                    bank_freeze_sha256=run.sha(run.PUBLIC / 'bank_complete.json'),
                    result_source='fresh_Torch_cost_head_inference_on_cached_verified_frozen_active_banks',
                    held_scored=False, independent_roles_opened=False)
                if verify:
                    if value != json.loads(path.read_text()):
                        raise ValueError('Exact matched-bank inference/metric replay differs')
                else:
                    run.once(path, value)
                rows.append(value)
                refs.append(dict(path=str(path.relative_to(ROOT)), sha256=run.sha(path)))
            beat(state='matched_bank_readout_progress', context=context, seed=seed, groups=len(rows), total=72)
        del old_packets, new_packets, original, changed, states
    summary = score.summarize(rows, json.loads(run.REG.read_text())['contexts'], cfg,
        json.loads((run.PUBLIC / 'training_freeze.json').read_text()))
    if verify:
        if summary != json.loads((run.PUBLIC / 'summary.json').read_text()) or refs != json.loads(finish.read_text())['groups']:
            raise ValueError('Complete72group matched-bank summary replay differs')
        run.once(run.PUBLIC / 'verification.json', dict(all72groups_exact=True, all_old_predictions_actions_record_scores_exact=True,
            summary_sha256=run.sha(run.PUBLIC / 'summary.json'), complete_sha256=run.sha(finish)))
    else:
        run.once(run.PUBLIC / 'summary.json', summary)
        run.once(finish, dict(pid=run.os.getpid(), seconds=time.monotonic() - start,
            peak_RSS_bytes=run.resource.getrusage(run.resource.RUSAGE_SELF).ru_maxrss,
            groups=refs, summary_sha256=run.sha(run.PUBLIC / 'summary.json'), new_optimizer_updates=0,
            held_scored=False, independent_roles_opened=False))
    run.write(run.PRIVATE / ('verification_runtime.json' if verify else 'readout_runtime.json'),
        dict(pid=run.os.getpid(), seconds=time.monotonic() - start,
            peak_RSS_bytes=run.resource.getrusage(run.resource.RUSAGE_SELF).ru_maxrss))
    beat(state='all72bank_readouts_verified' if verify else 'all72bank_readouts_complete',
         fitting_advancement_gate=summary['fitting_advancement_gate'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'run', 'verify'))
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run.PRIVATE.mkdir(parents=True, exist_ok=True)
    with (run.PRIVATE / 'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.phase == 'register':
            admission(register=True)
        else:
            execute(args.phase == 'verify', args.resume)
