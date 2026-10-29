"""New-revision export; preserve the frozen four-study evidence and manuscript."""
import argparse
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import report_m3w_cost_decision_evidence as previous
from src.evaluation.m3w_multiaction_cost_readout import interval

STUDIES = (*previous.STUDIES, 'european_shared_reference_cost_v1', 'european_static_scene_cost_v1')
DEST = previous.PUBLIC/'feature_cost_evidence_20261029'
MOMENTS = ('benefit', 'harm', 'reference', 'easy_reference', 'easy_harm')


def calibration_differences(old, new):
    if (old['scope'], old['context'], old['source'], old['seed']) != (new['scope'], new['context'], new['source'], new['seed']):
        raise ValueError('Exact paired group identity required')
    if old['result']['rows'] != new['result']['rows'] or old['result']['known_rows'] != new['result']['known_rows']:
        raise ValueError('Identical known/unknown evaluation support required')
    output = {}
    for arm in ('physical_only', 'physical_neural', 'neural_only'):
        a = old['result']['calibration'][arm]['normalized_moment_MSE']
        b = new['result']['calibration'][arm]['normalized_moment_MSE']
        if len(a) != 5 or len(b) != 5:
            raise ValueError('All five continuous costs required')
        output[arm] = dict(zip(MOMENTS, (float(y)-float(x) for x,y in zip(a,b))))
    return dict(source=new['source'], seed=new['seed'], scope=new['scope'], result=output)


def aggregate_groups(study):
    folder = previous.PUBLIC/study
    if (folder/'metric_bundle.json').exists():
        from src.evaluation.m3w_metric_bundle import verify
        verify(ROOT, folder)
        with zipfile.ZipFile(folder/'metric_groups.zip') as z:
            return {Path(n).name:json.loads(z.read(n)) for n in z.namelist()}
    result = {}
    for ref in json.loads((folder/'complete.json').read_text())['groups']:
        path = ROOT/ref['path']
        if path.parent != folder/'groups' or previous.sha(path) != ref['sha256'] or path.name in result:
            raise ValueError('Unique unchanged legacy aggregate group required')
        result[path.name] = json.loads(path.read_text())
    return result


def build():
    original = previous.STUDIES
    try:
        previous.STUDIES = STUDIES
        value = previous.build()
    finally:
        previous.STUDIES = original
    parent = aggregate_groups(STUDIES[0]); cfg = json.loads((ROOT/'configs/m3w_multiaction_cost_v1.json').read_text())
    refs = json.loads((previous.PUBLIC/STUDIES[0]/'complete.json').read_text())['groups']
    expected_hashes = {Path(r['path']).name:r['sha256'] for r in refs}
    calibration = {}
    for study in STUDIES[-2:]:
        groups = aggregate_groups(study); differences = []
        if groups.keys() != parent.keys():
            raise ValueError('Same144group identities required')
        for name, new in groups.items():
            # The complete receipt binds original exact bytes, not reserialized JSON.
            if new['parent_group_sha256'] != expected_hashes[name]:
                raise ValueError('New readout does not bind exact parent group')
            differences.append(calibration_differences(parent[name], new))
        calibration[study] = {scope:{arm:{moment:interval([r for r in differences if r['scope']==scope],
            lambda r,a=arm,m=moment:r[a][m], cfg) for moment in MOMENTS}
            for arm in ('physical_only', 'physical_neural', 'neural_only')} for scope in ('fit', 'inner_held')}
    value.update(result_source='cached_verified_training_and_utility_CIs_fresh_derived_moment_bootstrap',
        calibration_MSE_new_minus_original=calibration,
        moment_bootstrap_is_new_arithmetic_not_retraining=True,
        moment_comparisons_exploratory_no_multiple_testing_claim=True,
        export_revision='feature_cost_evidence_20261029_six_studies',
        source_adapter='EuropeanSquares_oblique_webcam_raw_box_midpoint_obs8_pred12_stride12',
        held_out_independent_data_opened=False)
    value['policies'] = [r for r in value['policies'] if not r['policy'].endswith(('_unshared_parent', '_no_scene_parent'))]
    return value


def render(value):
    lines = ['# Cost Features and Decision Failures: Six Controlled Studies', '',
        'This new revision preserves the earlier four-study export and manuscript.',
        'Training, frozen endpoints and exact readouts are source evidence; this',
        'export reuses their utility intervals and computes descriptive paired moment',
        'MSE intervals from verified aggregates. It does not train or select a model.', '',
        '## Primary Inner-Held Results', '',
        '| Study | Bank | Known ADE gain % | Conditional95%CI | Required risk views supported | Undefined easy |',
        '|---|---|---:|---:|---:|---:|']
    for row in value['policies']:
        if row['scope'] != 'inner_held' or row['policy'] not in ('physical_only', 'physical_neural', 'neural_only'):
            continue
        point, ci = row['known_ADE_gain_floor_percent'], row['conditional_CI95_percent']
        cell = 'not_estimable' if point is None else f'{point:.6f}'
        interval_text = 'not_estimable' if ci is None else f'[{ci[0]:.6f},{ci[1]:.6f}]'
        lines.append(f"| {row['study']} | {row['policy']} | {cell} | {interval_text} | "
            f"{row['supported_record_head_views']}/{row['required_record_head_views']} | {row['undefined_easy_record_head_views']} |")
    lines += ['', '## Paired Neural Contribution', '',
        'Mixed minus physical-only,percentage points. Known-label intervals cannot',
        'substitute for completion bounds or the every-record all/easy2%contract.', '',
        '| Study | Known difference [95%CI] | Completion-lower difference [95%CI] |', '|---|---:|---:|']
    for study, contrast in value['inner_held_mixed_minus_physical'].items():
        cells = []
        for key in ('full_known_pp', 'full_lower_pp'):
            row = contrast[key]; ci = row['CI95']
            cells.append('not_estimable' if row['point'] is None else f"{row['point']:.6f} [{ci[0]:.6f},{ci[1]:.6f}]")
        lines.append('| '+study+' | '+' | '.join(cells)+' |')
    lines += ['', '## Does Better Moment Prediction Explain Better Decisions?', '',
        'New minus original normalized MSE;negative is lower squared error.',
        'These are fresh3,000-locality bootstrap calculations on known-label',
        'aggregates,not independent confirmation or a familywise claim.', '',
        '| Ablation | Scope | Mixed cost | MSE change [conditional95%CI] |', '|---|---|---|---:|']
    for study, scopes in value['calibration_MSE_new_minus_original'].items():
        for scope, arms in scopes.items():
            for moment, row in arms['physical_neural'].items():
                ci = row['CI95']
                lines.append(f"| {study} | {scope} | {moment} | {row['point']:.6f} [{ci[0]:.6f},{ci[1]:.6f}] |")
    lines += ['', '## Actual Training, Not a Fresh Forecaster', '',
        '| Study | New heads | Primary updates | Seconds |', '|---|---:|---:|---:|']
    for study, row in value['training'].items():
        lines.append(f"| {study} | {row['fits']} | {row['updates']} | {row['seconds']:.3f} |")
    lines += ['', '## Claim Boundary', '',
        'The trajectory producer is fixed at seed43;three seeds refer to controller',
        'heads,not three newly trained full forecasting pipelines. Repeated context/',
        'head views are not independent recordings. SOURCE inner-held was exposed.',
        'No whole-pipeline OOF or independent selection/calibration/confirmation is claimed.',
        'Empty risk is not_estimable;count-matched controls measure utility only.',
        'MSE reduction alone is neither neural utility nor a safety certificate.',
        'Raw pixel/frame webcam detector tracks are not metric/seconds or human gold.',
        'No deployment promotion,Stage5C execution,SMC or submission readiness.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--check', action='store_true')
    args = parser.parse_args(); value = build()
    for path, text in {DEST.with_suffix('.json'):json.dumps(value, indent=2)+'\n', DEST.with_suffix('.md'):render(value)}.items():
        if args.check or path.exists():
            if path.read_text() != text:
                raise ValueError('Preserve existing export; new revision required')
        else:
            path.write_text(text)
    print(json.dumps(dict(studies=6, check=args.check, fresh_training=False,
        primary_fits=sum(v['fits'] for v in value['training'].values()),
        primary_updates=sum(v['updates'] for v in value['training'].values()))))


if __name__ == '__main__':
    main()
