"""Fixed216group budget-score readout and separate exact inference replay."""
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
from scripts import run_m3w_budget_score_cost as run
from scripts.report_m3w_feature_cost_evidence import aggregate_groups
from src.evaluation import m3w_budget_score_readout as score

REG = run.PUBLIC / 'readout_registration.json'
FILES = (Path(__file__), ROOT / 'src/evaluation/m3w_budget_score_readout.py',
         ROOT / 'tests/test_m3w_budget_score_readout.py',
         ROOT / 'src/evaluation/m3w_active_bank_cost_readout.py',
         ROOT / 'src/evaluation/m3w_multiaction_cost_readout.py',
         ROOT / 'src/evaluation/m3w_completion_cost_readout.py',
         ROOT / 'src/evaluation/m3w_multiaction_forest_readout.py',
         ROOT / 'src/evaluation/m3w_easy_constrained_policy.py',
         run.PUBLIC / 'readout_protocol.md')


def admission(register=False):
    cfg, values, _, store, banks = run.admission()
    models = run.models()
    run.once(REG, dict(bindings={str(p.relative_to(ROOT)): run.sha(p) for p in FILES},
        training_freeze_sha256=run.sha(run.PUBLIC / 'training_freeze.json'),
        parent_complete_sha256=run.sha(run.frozen.PUBLIC / 'complete.json'),
        groups=216, primary='upper_score_new_neural_minus_original_new_neural',
        original_prediction_action_and_full_score_replay_required=True,
        prediction_decoder_and_risk_budget_unchanged=True, count_matching_utility_only=True,
        empty_comparison_not_estimable=True, fitting_roles_only=True,
        held_scored=False, independent_roles_opened=False, deployment_changed=False, submission_ready=False))
    if not register:
        for rel, digest in json.loads(REG.read_text())['bindings'].items():
            if run.sha(ROOT / rel) != digest:
                raise ValueError('Registered budget-score readout changed')
            run.frozen.engine.committed(ROOT / rel)
        run.frozen.engine.committed(REG)
    return cfg, values, store, banks, models


def execute(verify=False, resume=False):
    cfg, values, store, banks, models = admission()
    finish = run.PUBLIC / 'complete.json'
    if finish.exists() and not verify:
        raise ValueError('Completed216group readout immutable; use verify')
    start, beat, guard = run.runtime(cfg)
    previous = aggregate_groups(run.frozen.NAME)
    rows, refs = [], []
    for context, items in sorted(values[2]['contexts'].items()):
        old_packets, new_packets, original, changed = run.frozen.prepared_context(context, items, values, store, banks, beat)
        job = next(j for j in values[4] if j['key'] == context.split('_controller')[0])
        for seed in cfg['seeds']:
            states = {}
            for objective in cfg['objectives']:
                states[objective] = {}
                for variant in cfg['variants']:
                    row = models[context, seed, variant, objective]
                    state = run.frozen.api.completion.base.read_checkpoint(ROOT / row['checkpoint']['path'])
                    data = changed if variant == 'new_neural' else original
                    run.frozen.api.completion.base.exact(state['preprocess'], data['preprocess'])
                    run.frozen.api.completion.base.exact(state['input_hashes'], data['input_hashes'])
                    states[objective][variant] = state
            for source, old_packet in sorted(old_packets.items()):
                guard()
                packets = dict(physical_only=old_packet, old_neural=old_packet, new_neural=new_packets[source])
                extra = store.read(old_packet['ids'], 'history')
                predictions, choices, ranks = {}, {}, {}
                for objective in cfg['objectives']:
                    predictions[objective], choices[objective], ranks[objective] = {}, {}, {}
                    for variant, packet in packets.items():
                        prediction = run.frozen.api.detector.predict(states[objective][variant], packet['x'],
                            packet['scene'], extra, packet['envelope'], packet['scales'])
                        choice, rank = run.frozen.api.completion.base.select_actions(prediction,
                            packet['allowed'], arm=run.frozen.api.BANKS[variant])
                        predictions[objective][variant] = prediction
                        choices[objective][variant], ranks[objective][variant] = choice, rank
                computed = score.objective_readouts(packets, predictions, choices, ranks, job['design']['easy_cut'],
                    states['original']['new_neural']['preprocess']['rms'])
                reference = previous[f'{context}_{source}_seed{seed}_fit.json']
                if (computed['original']['result'] != reference['result']
                        or any(run.frozen.engine.fingerprint(predictions['original'][v]) != reference['prediction_sha256'][v]
                            or run.frozen.engine.fingerprint(choices['original'][v]) != reference['action_sha256'][v]
                            for v in cfg['variants'])):
                    raise ValueError('Original full policy prediction/action/metric control differs')
                for objective in cfg['objectives']:
                    path = run.PUBLIC / 'groups' / f'{context}_{source}_seed{seed}_{objective}_fit.json'
                    if path.exists() and not (verify or resume):
                        raise ValueError('Explicit partial readout resume required')
                    value = dict(context=context, source=source, seed=seed, objective_arm=objective,
                        **computed[objective], ids_sha256=run.frozen.engine.fingerprint(old_packet['ids']),
                        prediction_sha256={v: run.frozen.engine.fingerprint(p) for v, p in predictions[objective].items()},
                        action_sha256={v: run.frozen.engine.fingerprint(c) for v, c in choices[objective].items()},
                        original_readout_exact=objective == 'original',
                        registration_sha256=run.sha(REG), training_freeze_sha256=run.sha(run.PUBLIC / 'training_freeze.json'),
                        result_source='fresh_Torch_budget_score_head_inference_on_cached_verified_frozen_banks',
                        held_scored=False, independent_roles_opened=False)
                    if verify:
                        if value != json.loads(path.read_text()):
                            raise ValueError('Budget-score exact inference replay differs')
                    else:
                        run.once(path, value)
                    rows.append(value)
                    refs.append(dict(path=str(path.relative_to(ROOT)), sha256=run.sha(path)))
                del computed, predictions, choices, ranks
            beat(state='budget_score_readout_progress', context=context, seed=seed, groups=len(rows), total=216)
        del states, old_packets, new_packets, original, changed
    summary = score.summarize(rows, json.loads(run.REG.read_text())['contexts'], cfg,
        json.loads((run.PUBLIC / 'training_freeze.json').read_text()))
    if verify:
        if summary != json.loads((run.PUBLIC / 'summary.json').read_text()) or refs != json.loads(finish.read_text())['groups']:
            raise ValueError('Complete216group summary replay differs')
        run.once(run.PUBLIC / 'verification.json', dict(all216groups_exact=True,
            all72original_predictions_actions_full_scores_exact=True,
            summary_sha256=run.sha(run.PUBLIC / 'summary.json'), complete_sha256=run.sha(finish)))
    else:
        run.once(run.PUBLIC / 'summary.json', summary)
        run.once(finish, dict(pid=run.os.getpid(), seconds=time.monotonic() - start,
            peak_RSS_bytes=run.resource.getrusage(run.resource.RUSAGE_SELF).ru_maxrss,
            groups=refs, summary_sha256=run.sha(run.PUBLIC / 'summary.json'),
            new_optimizer_updates=0, held_scored=False, independent_roles_opened=False))
    run.write(run.PRIVATE / ('verification_runtime.json' if verify else 'readout_runtime.json'),
        dict(pid=run.os.getpid(), seconds=time.monotonic() - start,
            peak_RSS_bytes=run.resource.getrusage(run.resource.RUSAGE_SELF).ru_maxrss))
    beat(state='all216groups_verified' if verify else 'all216groups_complete',
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
