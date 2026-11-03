"""Post-readout fitting/held cost diagnosis; no refitting or threshold selection."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_shared_motion_cost as runner
from src.evaluation.m3w_selected_cost_audit import selected_costs
from src.world_model.m3w_multiaction_cost import select_actions
from src.world_model.m3w_supported_bound_targets import supported_bound_targets
import numpy as np

OUT = runner.PUBLIC/'selected_cost_audit'


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('evaluate', 'verify'))
    args = parser.parse_args()
    verification = runner.read(runner.PUBLIC/'verification.json')
    if (not verification['all36cost_predictions_and_policy_scores_exact']
            or verification['summary_sha256'] != runner.sha(runner.PUBLIC/'summary.json')):
        raise ValueError('Complete original policy inference and replay before post-readout diagnosis')
    cfg = runner.read(runner.CONFIG)
    start, guard, beat = runner.runtime(cfg, 'post_readout_cost_audit')
    data, jobs, refs = runner.admission(cfg, beat)
    models = {}
    for item in runner.read(runner.PUBLIC/'training_freeze.json')['fits']:
        receipt = runner.read(runner.checked(item))
        models[receipt['job']['key'], receipt['arm']] = runner.checked(receipt['checkpoint'])
    results = []
    for job in jobs:
        for role, fitting in (('critic_fitting', True), ('held', False)):
            packet = runner.view(data, job, refs, training=fitting)
            packet.pop('learning', None)
            ids, causal = packet['ids'], packet['causal']
            target = supported_bound_targets(packet['paths'],
                data['target_eval'][ids]-data['origin'][ids, None], data['valid'][ids],
                job['floor_index'], job['easy_cut'])
            for arm in cfg['arms']:
                state = runner.api.base.base.read_checkpoint(models[job['key'], arm])
                predicted = runner.api.predict(state, causal)
                independent, _ = select_actions(predicted, causal['allowed'], arm='physical_neural')
                choices = dict(all_neural=np.full(len(ids), 6, np.int64), independent=independent)
                if not fitting:
                    saved, _ = runner.allocated(predicted, causal, packet, job, arm, cfg, beat, verify=True)
                    np.testing.assert_array_equal(saved['independent'], independent)
                    choices['joint_free'] = saved['joint_free']
                results.append(dict(job=job['key'], role=role, arm=arm,
                    policies={name:selected_costs(predicted, target['moments'],
                        data['valid'][ids].any(1), choice, causal['groups']) for name,choice in choices.items()}))
                del state, predicted
            del packet, target, causal
        guard()
        beat(state='post_readout_job_complete', job=job['key'], views=len(results))
    value = dict(result_source='fresh_post_readout_diagnostic_not_independent_confirmation',
        parent_summary_sha256=runner.sha(runner.PUBLIC/'summary.json'),
        parent_verification_sha256=runner.sha(runner.PUBLIC/'verification.json'),
        bindings={str(p.relative_to(ROOT)):runner.sha(p) for p in
            (Path(__file__), ROOT/'src/evaluation/m3w_selected_cost_audit.py', ROOT/'tests/test_m3w_selected_cost_audit.py')},
        rows=results, new_training=False, new_policy_or_threshold_selected=False,
        source_roles_changed=False, stage5c_executed=False, smc_enabled=False)
    runner.once(OUT/'analysis.json', runner.stable(value))
    runner.once(OUT/(args.phase+'_execution.json'), dict(pid=runner.os.getpid(),
        seconds=time.monotonic()-start, every_scalar_replayed_exact=args.phase == 'verify',
        new_training=False, new_independent_evidence=False))
    beat(state='post_readout_complete', verified=args.phase == 'verify')


def main():
    with (runner.PRIVATE/'process.lock').open('a') as lock:
        runner.fcntl.flock(lock, runner.fcntl.LOCK_EX | runner.fcntl.LOCK_NB)
        run()


if __name__ == '__main__':
    main()
