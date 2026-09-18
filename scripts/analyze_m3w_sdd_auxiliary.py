"""Analyze every registered auxiliary fit without selecting a model."""
import argparse
import csv
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_offline_visual_data import json_write
from src.world_model.m3w_sdd_auxiliary import load_registration


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registration',type=Path,required=True)
    args = p.parse_args()
    reg = load_registration(ROOT,args.registration); reports = ROOT/reg['reports']
    r = json.loads((reports/'report.json').read_text())
    if not r['complete'] or len(r['trials']) != 54:
        raise ValueError('Full preregistered matrix required')
    trials, summary = r['trials'], r['summary']
    for trial in trials:
        for kind in ('checkpoint','prediction'):
            if file_digest(ROOT/trial[kind+'_path']) != trial[kind+'_sha256']:
                raise ValueError('Changed fit artifact')
    data = json.loads((reports/'data_receipt.json').read_text())
    groups = {}
    for key,s in summary.items():
        rows = [t for t in trials if t['schedule']+'_'+t['modality'] == key]
        native = {}
        for record in ('eth_eth','eth_hotel','ucy_zara01','ucy_zara02','ucy_zara03'):
            slices = [t['slices'][record] for t in rows if record in t['slices']]
            native[record] = {field:float(np.mean([v[field] for v in slices])) for field in
                             ('primary_ADE','FDE','native_ADE_diagnostic','native_FDE_diagnostic')}
        events = {}
        for event in ('static_stays','static_moves','moving_stops','moving_turns','other_motion','hard_train_q75'):
            slices = [t['slices'][event] for t in rows if t['slices'].get(event,{}).get('rows',0)]
            if slices:
                events[event] = dict(rows_per_seed=sum(v['rows'] for v in slices)//3,
                    scene_count=len({t['fold'] for t in rows if t['slices'].get(event,{}).get('rows',0)}),
                    gain_percent=100*(1-np.mean([v['primary_ADE'] for v in slices])/
                                         np.mean([v['reference_ADE'] for v in slices])))
        groups[key] = dict(per_recording_native_diagnostic=native,events=events,
            training_gain_range_percent=[min(t['train_equal_scene_gain_percent'] for t in rows),
                                         max(t['train_equal_scene_gain_percent'] for t in rows)],
            held_gain_range_percent=[min(t['vs_CV']['improvement_percent'] for t in rows),
                                     max(t['vs_CV']['improvement_percent'] for t in rows)],
            easy_absolute_harm_range=[min(t['vs_CV']['easy_absolute_harm'] for t in rows),
                                      max(t['vs_CV']['easy_absolute_harm'] for t in rows)],
            easy_relative_degradation_range_percent=[min(s['easy_degradation_percent']),max(s['easy_degradation_percent'])],
            tail_ADE_p95_range=[min(t['vs_CV']['tail_ADE_p95'] for t in rows),
                               max(t['vs_CV']['tail_ADE_p95'] for t in rows)],
            sampled_auxiliary_unique_rows=[t['fit']['sampled_auxiliary_rows'] for t in rows])
    result = dict(result_source='fresh_run_analysis_of_registered_frozen_predictions',
        report_sha256=file_digest(reports/'report.json'),groups=groups,
        all_fits=54,full_population_available=229333,
        every_auxiliary_row_sampled_claim=False,
        unsupported_batches=sum(t['fit']['unsupported_batches'] for t in trials),
        optimizer_updates_verified=sum(t['fit']['step'] for t in trials),
        source_feature_clipping_fraction_by_fold={str(f):next(t['auxiliary_clipped_feature_fraction'] for t in trials if t['fold']==f) for f in range(3)},
        total_fit_seconds=sum(t['fit']['fit_seconds'] for t in trials),
        no_new_model_selected=True,independent_confirmation=False)
    json_write(reports/'analysis.json',result)
    stream = io.StringIO(); columns = ['trial','schedule','modality','seed','fold','gain_vs_CV',
        'ADE','FDE','easy_degradation','easy_absolute_harm','train_gain','fit_seconds','sampled_auxiliary_rows']
    writer = csv.DictWriter(stream,fieldnames=columns); writer.writeheader()
    for t in trials:
        m = t['vs_CV']
        writer.writerow(dict(trial=t['trial'],schedule=t['schedule'],modality=t['modality'],seed=t['seed'],fold=t['fold'],
            gain_vs_CV=m['improvement_percent'],ADE=m['primary_ADE'],FDE=m['FDE'],
            easy_degradation=m['easy_degradation_percent'],easy_absolute_harm=m['easy_absolute_harm'],
            train_gain=t['train_equal_scene_gain_percent'],fit_seconds=t['fit']['fit_seconds'],
            sampled_auxiliary_rows=t['fit']['sampled_auxiliary_rows']))
    (reports/'fit_metrics.csv').write_text(stream.getvalue())
    lines = ['# SDD Auxiliary Transfer: Complete Matched Results','',
        'Result source: fresh real Torch training; source and main caches hash-verified.',
        'All 54 preregistered fits and 324,000 updates completed. No held-fit checkpoint selection.',
        'This is exploration on three previously exposed physical sites, not independent confirmation.','',
        '| Schedule / input | Gain vs CV (%) | Descriptive site CI (%) | Gain vs same-input no-SDD (%) | Pixel gain vs mask (%) | Safe positive fits |',
        '| --- | ---: | --- | ---: | ---: | ---: |']
    for key,s in summary.items():
        ci=s['vs_CV']['exploratory_scene_ci95_percent']
        lines.append(f"| {key} | {s['vs_CV']['gain_percent']:.5f} | [{ci[0]:.5f}, {ci[1]:.5f}] | {s['vs_no_aux_same_modality']['gain_percent']:.5f} | {s['vs_same_source_mask']['gain_percent']:.5f} | {s['safe_positive_fits']}/9 |")
    lines += ['','## Interpretation Boundaries','',
        'Positive source transfer versus a neural control is not necessarily improvement over causal CV.',
        'RGB contribution is judged against the same-source spatial mask arm, not only geometry.',
        'A bootstrap over three reused sites cannot create independent scene evidence or certify safety.',
        'Easy relative degradation is retained alongside absolute normalized harm and all seed/site failures.',
        'No model is promoted by this fit-only comparison. Stage5C and SMC remain disabled.','',
        '## Source and Sampling','',
        f"Original train-only videos:40; full eligible auxiliary population:{data['rows']:,}.",
        f"Complete/partial/absent future labels:{data['complete_labels']:,}/{data['partial_labels']:,}/{data['absent_labels']:,}.",
        'All windows are indexed without future-support selection. Uniform sampling with replacement',
        'uses128,000auxiliary draws per auxiliary fit, not one full epoch. Unique row counts are inCSV/JSON.',
        'Zero-label rows stay in the population but are excluded from supported-loss averaging.',
        'SDD uses stride12rawframes,8past/12future points, endpoint+144rawframes; no seconds equivalence.',
        'Main task keeps its original annotation steps, complete labels and physical-scene-equal primary.',
        'Source histories include retrospective annotation interpolation. No strict sensor-as-of claim.',
        'No metric, true3D, foundation, generalization-success or independent-test claim.','',
        '## Remaining Uncertainty','',
        'The source contrast jointly changes dataset, viewing geometry, motion distribution and label support.',
        'It does not identify any one of these as the cause of a gain or failure.',
        'Main-train-only normalization is fixed in both arms; source clipping is reported, not tuned away.',
        'Low-resolution crops and domain mismatch remain hypotheses, not causal findings.',
        'For a publishable gain, a future protocol needs a stable method effect and untouched scenes.','',
        '## Reproduction','',
        'Run with native arm64 .venv-pytorch, four Torch threads, one interop thread, no workers.',
        'Use the registered configuration with prepare_m3w_sdd_auxiliary.py,',
        'verify_m3w_sdd_auxiliary_inputs.py, run_m3w_sdd_auxiliary.py, then its --replay mode.',
        'Rerunning the completed trainer verifies artifacts and adds zero optimizer updates.',
        'Private arrays/checkpoints are not in Git; hashes, registration, code and aggregate results are.',
        'See fit_metrics.csv for all54fits and analysis.json for event/native-coordinate diagnostics.','']
    (reports/'conclusions.md').write_text('\n'.join(lines))
    print(json.dumps(dict(total_fits=54,total_fit_seconds=result['total_fit_seconds'],
                          groups={k:v['vs_CV'] for k,v in summary.items()}),indent=2))


if __name__ == '__main__':
    main()

