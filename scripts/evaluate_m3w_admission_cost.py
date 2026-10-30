"""Fixed72group complete-policy readout and separate exact inference replay."""
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
from scripts import run_m3w_admission_cost as run
from scripts.report_m3w_feature_cost_evidence import aggregate_groups
from src.evaluation import m3w_admission_readout as api

REG = run.PUBLIC / 'readout_registration.json'
FILES = (Path(__file__), ROOT / 'src/evaluation/m3w_admission_readout.py',
    ROOT / 'tests/test_m3w_admission_readout.py',
    ROOT / 'src/evaluation/m3w_selected_action_attribution.py',
    ROOT / 'src/evaluation/m3w_active_bank_cost_readout.py',
    ROOT / 'src/evaluation/m3w_multiaction_cost_readout.py',
    ROOT / 'src/evaluation/m3w_completion_cost_readout.py',
    ROOT / 'src/evaluation/m3w_multiaction_forest_readout.py',
    run.PUBLIC / 'readout_protocol.md')


def admission(register=False):
    cfg, values, store, banks, _ = run.admission()
    models = run.models()
    run.once(REG, dict(bindings={str(p.relative_to(ROOT)): run.sha(p) for p in FILES},
        training_freeze_sha256=run.sha(run.PUBLIC / 'training_freeze.json'),
        parent_complete_sha256=run.sha(run.parent.PUBLIC / 'complete.json'),
        groups=72, primary='admitted_score_minus_upper_score',
        both_controls_prediction_action_full_metric_exact_required=True,
        risk_rule_and_decoder_unchanged=True, matched_controls_utility_only=True,
        empty_not_estimable=True, held_scored=False, independent_roles_opened=False,
        deployment_changed=False, submission_ready=False))
    if not register:
        for rel, digest in json.loads(REG.read_text())['bindings'].items():
            if run.sha(ROOT / rel) != digest:
                raise ValueError('Registered admission readout changed')
            run.parent.frozen.engine.committed(ROOT / rel)
        run.parent.frozen.engine.committed(REG)
        run.parent.frozen.engine.committed(run.PUBLIC / 'training_freeze.json')
    return cfg, values, store, banks, models


def execute(verify=False, resume=False):
    cfg, values, store, banks, models = admission()
    if (run.PUBLIC / 'complete.json').exists() and not verify:
        raise ValueError('Completed72group readout immutable; use verify')
    start, beat, guard = run.runtime(cfg)
    previous = aggregate_groups(run.parent.NAME)
    rows, refs = [], []
    for context, items in sorted(values[2]['contexts'].items()):
        op, packets, original, changed = run.parent.frozen.prepared_context(context, items, values, store, banks, beat)
        job = next(j for j in values[4] if j['key'] == context.split('_controller')[0])
        for seed in cfg['seeds']:
            states = {o: run.parent.frozen.api.completion.base.read_checkpoint(ROOT / models[context, seed, o]['checkpoint']['path'])
                for o in cfg['objectives']}
            for source, packet in sorted(packets.items()):
                guard()
                extra = store.read(packet['ids'], 'history')
                predictions, choices, ranks = {}, {}, {}
                for obj, state in states.items():
                    p = run.parent.frozen.api.detector.predict(state, packet['x'], packet['scene'], extra,
                        packet['envelope'], packet['scales'])
                    c, rank = run.parent.frozen.api.completion.base.select_actions(p, packet['allowed'], arm='physical_neural')
                    predictions[obj], choices[obj], ranks[obj] = p, c, rank
                    if obj != 'admitted_score':
                        ref = previous[f'{context}_{source}_seed{seed}_{obj}_fit.json']
                        if (run.parent.frozen.engine.fingerprint(p) != ref['prediction_sha256']['new_neural']
                                or run.parent.frozen.engine.fingerprint(c) != ref['action_sha256']['new_neural']
                                or api.bank.policy_metrics(packet, c) != ref['result']['policies']['new_neural']):
                            raise ValueError('Original/upper full readout control changed')
                result = api.readout(packet, predictions, choices, ranks, job['design']['easy_cut'],
                    states['original']['preprocess']['rms'])
                value = dict(context=context, source=source, seed=seed, result=result, controls_exact=True,
                    prediction_sha256={o: run.parent.frozen.engine.fingerprint(p) for o, p in predictions.items()},
                    action_sha256={o: run.parent.frozen.engine.fingerprint(c) for o, c in choices.items()},
                    ids_sha256=run.parent.frozen.engine.fingerprint(packet['ids']),
                    registration_sha256=run.sha(REG), training_freeze_sha256=run.sha(run.PUBLIC / 'training_freeze.json'),
                    result_source='fresh_Torch_admission_cost_inference_on_cached_verified_frozen_banks',
                    held_scored=False, independent_roles_opened=False)
                path = run.PUBLIC / 'groups' / f'{context}_{source}_seed{seed}_fit.json'
                if path.exists() and not (verify or resume):
                    raise ValueError('Explicit partial resume required')
                if verify:
                    if value != json.loads(path.read_text()):
                        raise ValueError('Exact72group replay differs')
                else:
                    run.once(path, value)
                rows.append(value)
                refs.append(dict(path=str(path.relative_to(ROOT)), sha256=run.sha(path)))
            beat(state='readout_progress', context=context, seed=seed, groups=len(rows), total=72)
        del op, packets, original, changed, states
    summary = api.summarize(rows, json.loads(run.REG.read_text())['contexts'], cfg,
        json.loads((run.PUBLIC / 'training_freeze.json').read_text()))
    if verify:
        if (summary != json.loads((run.PUBLIC / 'summary.json').read_text())
                or refs != json.loads((run.PUBLIC / 'complete.json').read_text())['groups']):
            raise ValueError('Complete summary or group identities changed')
        run.once(run.PUBLIC / 'verification.json', dict(all72groups_exact=True,
            all144control_predictions_actions_full_metrics_exact=True,
            summary_sha256=run.sha(run.PUBLIC / 'summary.json'), complete_sha256=run.sha(run.PUBLIC / 'complete.json')))
    else:
        run.once(run.PUBLIC / 'summary.json', summary)
        run.once(run.PUBLIC / 'complete.json', dict(pid=run.os.getpid(), seconds=time.monotonic() - start,
            peak_RSS_bytes=run.resource.getrusage(run.resource.RUSAGE_SELF).ru_maxrss,
            groups=refs, summary_sha256=run.sha(run.PUBLIC / 'summary.json'), new_optimizer_updates=0,
            held_scored=False, independent_roles_opened=False))
    run.write(run.PRIVATE / ('verification_runtime.json' if verify else 'readout_runtime.json'),
        dict(pid=run.os.getpid(), seconds=time.monotonic() - start,
             peak_RSS_bytes=run.resource.getrusage(run.resource.RUSAGE_SELF).ru_maxrss))
    beat(state='all72groups_verified' if verify else 'all72groups_complete',
        fitting_advancement_gate=summary['fitting_advancement_gate'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'run', 'verify'))
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run.PRIVATE.mkdir(parents=True, exist_ok=True)
    with (run.PRIVATE / 'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        admission(register=True) if args.phase == 'register' else execute(args.phase == 'verify', args.resume)
