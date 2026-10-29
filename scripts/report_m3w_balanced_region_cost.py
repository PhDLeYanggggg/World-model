"""Read-only exact evidence export; preserve every prior experiment revision."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import report_m3w_cost_decision_evidence as evidence
from scripts.report_m3w_policy_region_cost import interval_text

NAME='european_balanced_region_cost_v1';PUBLIC=evidence.PUBLIC/NAME


def build():
    previous=evidence.STUDIES
    try:
        evidence.STUDIES=(NAME,);value=evidence.build()
    finally:evidence.STUDIES=previous
    summary=json.loads((PUBLIC/'summary.json').read_text());freeze=json.loads((PUBLIC/'training_freeze.json').read_text())
    value.update(exact_minus_stochastic={scope:summary['scopes'][scope]['exact_minus_stochastic'] for scope in ('fit','inner_held')},
        auxiliary_teacher_rows=freeze['auxiliary_teacher_rows'],auxiliary_forward_rows=freeze['auxiliary_forward_rows'],
        same_base_draws_and_steps_not_same_FLOPs=True,
        full_roster_supported=summary['primary_full_roster_budget_supported'],
        neural_increment_supported=summary['neural_increment_conditional_CI_positive'])
    return value


def render(value):
    text=['# Complete-Region Cost Learning: Fixed Evidence','',
        'Verified completed training and exact SOURCE-development readout. This export',
        'does not train,select a threshold,create a new bootstrap or open held roles.','',
        '| Bank | Known ADE gain % [conditional95%CI] | Supported record/head views | Undefined easy | Selected (unknown) |',
        '|---|---:|---:|---:|---:|']
    for row in value['policies']:
        if row['scope']!='inner_held' or row['policy'] not in ('physical_only','physical_neural','neural_only'):continue
        cell=interval_text(dict(point=row['known_ADE_gain_floor_percent'],CI95=row['conditional_CI95_percent']))
        text.append(f"| {row['policy']} | {cell} | {row['supported_record_head_views']}/{row['required_record_head_views']} | "
            f"{row['undefined_easy_record_head_views']} | {row['selected']} ({row['selected_unknown']}) |")
    text+=['','## Paired Mixed-Bank Contrasts','','Percentage points;unknown outcomes remain in completion bounds.',
        '| Contrast | Known | Completion lower | Completion upper |','|---|---:|---:|---:|']
    for label,rows in (('complete minus stochastic',value['exact_minus_stochastic']['inner_held']),
                       ('mixed minus physical',value['inner_held_mixed_minus_physical'][NAME])):
        for mode in ('full','matched'):
            text.append('| '+label+' '+mode+' | '+' | '.join(interval_text(rows[f'{mode}_{kind}_pp']) for kind in ('known','lower','upper'))+' |')
    fit=value['training'][NAME]
    text+=['','## Actual Compute, Not Equal FLOPs','',
        f"{fit['fits']}fresh Torch cost heads,{fit['updates']:,}primary optimizer updates,",
        f"{fit['seconds']:.6f}s,peak RSS {fit['peak_RSS_bytes']:,}bytes.",
        f"Additional known teacher-row visits:{value['auxiliary_teacher_rows']:,}.",
        f"Additional two-pass query-row forwards:{value['auxiliary_forward_rows']:,}.",
        'Candidate multiplication depends on bank size. Identical base draws and steps',
        'do not mean identical training FLOPs. The trajectory producer stays frozen.',
        'Three seeds are controller seeds,not three new full forecasting pipelines.','',
        'Full required-recording all/easy2% gate: '+str(value['full_roster_supported']),
        'Positive completion-lower neural-increment CI: '+str(value['neural_increment_supported']),'',
        'Cyclic training is exact within one fixed teacher region,not an exact',
        'full-all-region update or independent calibration. Empty selected risk is',
        'not_estimable,not safe zero. Matched counts measure utility only. No harm',
        'multiplier,seed or subgroup is chosen from readout. SOURCE inner-held is',
        'exposed development;3,000-locality CIs are conditional,not confirmation.',
        'Obs8/pred12/rawstride12 webcam detector-track pixels are not metric/seconds,',
        'human gold,true3D or foundation evidence. No deployment promotion,',
        'Stage5C execution,SMC or submission readiness.','']
    return '\n'.join(text)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');a=p.parse_args();value=build()
    for path,content in {PUBLIC/'evidence.json':json.dumps(value,indent=2)+'\n',PUBLIC/'evidence.md':render(value)}.items():
        if a.check or path.exists():
            if path.read_text()!=content:raise ValueError('Preserve completed evidence;new revision required')
        else:path.write_text(content)
    print(json.dumps(dict(exact_check=a.check,new_training=False,full_roster_supported=value['full_roster_supported'],
        neural_increment_supported=value['neural_increment_supported'])))
