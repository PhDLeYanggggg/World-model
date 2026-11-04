"""Fitting-only selection transport check, frozen before any new H readout."""
import argparse
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_record_excess_cost as run
from src.world_model.m3w_harm_refinement import select_actions
from src.evaluation.m3w_selected_cost_audit import selected_costs

source = run.source
PUBLIC = run.PUBLIC / 'fitting_regions'
PRIVATE = run.PRIVATE / 'fitting_regions'
REG = PUBLIC / 'registration.json'
FILES = (Path(__file__), PUBLIC / 'protocol.md')


def aggregate(cells, model, region):
    rows = [c['scores'] for c in cells if c['model'] == model and c['region'] == region]
    actual = source.np.sum([r['realized_selected_known_moments'] for r in rows], 0)
    predicted = source.np.sum([r['predicted_on_selected_known_moments'] for r in rows], 0)
    return dict(actual=actual.tolist(), predicted=predicted.tolist(), selected=sum(r['selected'] for r in rows),
        actual_over_predicted={name: float(actual[i]/predicted[i]) if predicted[i] > 0 else None
            for i, name in enumerate(('B', 'H', 'R', 'ER', 'EH'))})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'run', 'verify'))
    args = parser.parse_args()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    source.once(REG, source.stable(dict(bindings={str(p.relative_to(ROOT)): source.sha(p) for p in FILES},
        neural_runtime_registration=source.artifact(run.RUNTIME_REG),
        target_role='fitting_C_only_not_OOF_validation', new_training=False, new_H_prediction=False,
        new_decisions_for_deployment=False, threshold_search=False)))
    if args.phase == 'register':
        return
    for p in (*FILES, REG):
        body = source.subprocess.run(['git', 'show', 'HEAD:' + str(p.relative_to(ROOT))], cwd=ROOT, capture_output=True, check=True).stdout
        if source.hashlib.sha256(body).hexdigest() != source.sha(p):
            raise ValueError('Commit the C-only trigger before examining fitting regions')
    receipt = PUBLIC / (args.phase + '_execution.json')
    if receipt.exists():
        raise ValueError('Completed fitting audit is immutable')
    if args.phase == 'run' and ((run.PUBLIC / 'evaluate_execution.json').exists() or any((run.PUBLIC / 'held').glob('*.json'))):
        raise ValueError('Freeze C-only optimization trigger before new H predictions')
    if args.phase == 'verify' and not (PUBLIC / 'analysis.json').exists():
        raise ValueError('Complete the C-only diagnostic before replay')
    freeze = source.read(run.PUBLIC / 'training_freeze.json')
    if freeze['complete_new_neural_cost_heads'] != 18:
        raise ValueError('Complete all18fits first')
    with (run.PRIVATE / 'process.lock').open('a') as lock:
        source.fcntl.flock(lock, source.fcntl.LOCK_EX | source.fcntl.LOCK_NB)
        run.lineage.PRIVATE = PRIVATE
        source.PRIVATE = PRIVATE
        spec = source.read(run.CONFIG)
        cfg = source.read(run.PARENT.CONFIG)
        start, guard, beat = run.lineage.runtime(cfg, spec, 'fitting_regions_' + args.phase)
        spec, cfg, data, jobs, refs, controls = run.register(beat)
        fits = {source.read(source.checked(ref))['job']['key']: source.read(source.checked(ref)) for ref in freeze['fits']}
        cells, receipts = [], []
        for job in jobs:
            p, _, point = run.fitting(data, job, refs, controls[job['key']])
            state = run.api.base.base.read_checkpoint(source.checked(fits[job['key']]['checkpoint']))
            prediction = run.api.parent.predict(state, p['causal'])
            if source.array_hash(prediction) != fits[job['key']]['fitting_predictions_sha256']:
                raise ValueError('Frozen C prediction changed')
            masks = dict(point=p['learning']['teacher_actions'],
                record_excess=select_actions(prediction, p['causal']['allowed'], arm='physical_neural')[0])
            hashes = {k: source.array_hash(v) for k, v in masks.items()}
            ids = p['ids']
            truth = run.api.parent.targets(p, data['target_eval'][ids]-data['origin'][ids, None], data['valid'][ids], job['easy_cut'])
            part = [dict(job=job['key'], model=model, region=region,
                scores=selected_costs(value, truth['moments'], truth['observed_future'], choice, p['causal']['groups']))
                for model, value in (('point', point), ('record_excess', prediction)) for region, choice in masks.items()]
            if hashes != {k: source.array_hash(v) for k, v in masks.items()}:
                raise ValueError('Supervision changed a fitting-region mask')
            path = PUBLIC / 'groups' / (job['key'] + '.json')
            source.once(path, source.stable(dict(job=job, cells=part, choices_sha256=hashes,
                point_cost_sha256=source.array_hash(point), new_cost_sha256=source.array_hash(prediction))))
            receipts.append(source.artifact(path))
            cells.extend(part)
            beat(state='fitting_regions_complete', job=job['key'], complete=len(receipts), total=18)
            del p, state, point, prediction, truth, masks, part
            source.gc.collect()
            guard()
        values = {model: {region: aggregate(cells, model, region) for region in ('point', 'record_excess')}
            for model in ('point', 'record_excess')}
        teacher, own = values['record_excess']['point'], values['record_excess']['record_excess']
        trigger_components = [i for i in (1, 4) if own['actual'][i] > own['predicted'][i]
            and own['actual'][i]*teacher['predicted'][i] > teacher['actual'][i]*own['predicted'][i]]
        source.once(PUBLIC / 'analysis.json', source.stable(dict(values=values, receipts=receipts,
            all72fitting_region_cells_complete=True, C_only_adaptive_followup_trigger=bool(trigger_components),
            trigger_moments=[('B', 'H', 'R', 'ER', 'EH')[i] for i in trigger_components],
            trigger_frozen_before_new_Hprediction=True, fitting_only_not_validation_or_calibration=True,
            new_training=False, new_H_prediction=False, new_deployment=False, submission_ready=False,
            independent_confirmation=False, stage5c_executed=False, smc_enabled=False)))
        source.once(receipt, dict(pid=source.os.getpid(), seconds=time.monotonic()-start,
            peak_RSS_bytes=source.resource.getrusage(source.resource.RUSAGE_SELF).ru_maxrss,
            fresh_C_cost_inference=True, new_H_prediction=False))
        if args.phase == 'verify':
            source.once(PUBLIC / 'verification.json', dict(all72cells_and36C_cost_arrays_exact=True,
                analysis_sha256=source.sha(PUBLIC / 'analysis.json'), new_H_prediction=False))
        beat(state='complete')


if __name__ == '__main__':
    main()
