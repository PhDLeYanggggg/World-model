"""Complete fitting-region cost check before new held-policy outcome readout."""
import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_decoder_link_repair as experiment
from src.evaluation.m3w_selected_cost_audit import selected_costs
from src.world_model.m3w_multiaction_cost import select_actions
from src.world_model.m3w_supported_bound_targets import supported_bound_targets

source, np = experiment.source, experiment.np
OUT = experiment.PUBLIC/'fitting_check'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('evaluate', 'verify'))
    args = parser.parse_args()
    if args.phase == 'evaluate' and (experiment.PUBLIC/'summary.json').exists():
        raise ValueError('Original fitting check must precede new held-policy readout')
    source.PRIVATE = experiment.PRIVATE
    with (experiment.PRIVATE/'process.lock').open('a') as lock:
        source.fcntl.flock(lock, source.fcntl.LOCK_EX | source.fcntl.LOCK_NB)
        cfg = source.read(experiment.CONFIG)
        start, guard, beat = source.runtime(cfg, 'preheld_fitting_check')
        data, jobs, refs, warm = experiment.register(cfg, beat)
        freeze = source.read(experiment.PUBLIC/'training_freeze.json')
        if freeze['complete_continuations'] != 36 or not freeze['all18pairs_exact']:
            raise ValueError('Complete all matched continuations first')
        models = {}
        for ref in freeze['fits']:
            row = source.read(source.checked(ref))
            models[row['job']['key'], row['arm']] = row['checkpoint']
        records = []
        for job in jobs:
            fitting = source.view(data, job, refs, training=True)
            fitting.pop('learning', None)
            ids, causal = fitting['ids'], fitting['causal']
            predicted, choices = {}, {}
            for arm in ('frozen_parent', *experiment.api.ARMS):
                ref = warm[job['key']] if arm == 'frozen_parent' else models[job['key'], arm]
                state = experiment.api.base.base.read_checkpoint(source.checked(ref))
                predicted[arm] = source.api.predict(state, causal)
                choices[arm+'_independent'], _ = select_actions(predicted[arm], causal['allowed'], arm='physical_neural')
                del state
            choices['all_neural'] = np.full(len(ids), 6, np.int64)
            target = supported_bound_targets(fitting['paths'], data['target_eval'][ids]-data['origin'][ids, None],
                data['valid'][ids], job['floor_index'], job['easy_cut'])
            for arm, prediction in predicted.items():
                for region, choice in choices.items():
                    records.append(dict(job=job['key'], arm=arm, region=region,
                        scores=selected_costs(prediction, target['moments'], data['valid'][ids].any(1), choice, causal['groups'])))
            beat(state='fitting_regions_checked', job=job['key'], views=len(records))
            del fitting, causal, predicted, choices, target
            guard()
        if len(records) != 216:
            raise ValueError('All18jobs times3scoring arms times4fixed causal regions required')
        value = source.stable(dict(rows=records, source_role='critic_fitting_only',
            result_source='fresh_pre_new_held_readout_fitting_diagnosis_not_independent_test',
            registration_sha256=source.sha(experiment.REG),
            training_sha256=source.sha(experiment.PUBLIC/'training_freeze.json'),
            bindings={str(p.relative_to(ROOT)):source.sha(p) for p in
                (Path(__file__), ROOT/'src/evaluation/m3w_selected_cost_audit.py')},
            original_evaluation_before_new_held_policy_readout=True,
            checkpoint_or_threshold_or_policy_selected=False, new_training=False,
            stage5c_executed=False, smc_enabled=False))
        source.once(OUT/'analysis.json', value)
        source.once(OUT/(args.phase+'_execution.json'), dict(pid=source.os.getpid(), seconds=time.monotonic()-start,
            peak_RSS_bytes=source.resource.getrusage(source.resource.RUSAGE_SELF).ru_maxrss,
            exact_scalar_replay=args.phase == 'verify', new_training=False))
        beat(state='fitting_check_complete', verify=args.phase == 'verify')


if __name__ == '__main__':
    main()
