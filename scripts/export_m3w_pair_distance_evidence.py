"""Verify and package72 scalar diagnostic groups without raw trajectories."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts import run_m3w_pair_distance_diagnostic as diagnostic
from scripts.report_m3w_feature_cost_evidence import aggregate_groups
from src.evaluation.m3w_metric_bundle import pack,verify


def export():
    run=diagnostic.run; public=diagnostic.PUBLIC; api=diagnostic.api
    reg=json.loads(diagnostic.REG.read_text())
    for rel,digest in reg['bindings'].items():
        if run.sha(ROOT/rel)!=digest: raise ValueError('Frozen diagnostic implementation changed')
    if (public/'metric_bundle.json').exists():
        verify(ROOT,public,restore=True)
    main=aggregate_groups(run.NAME); files={p.name:p for p in (public/'groups').glob('*.json')}
    if files.keys()!=main.keys() or len(files)!=72: raise ValueError('Complete unchanged72-group cohort required')
    total=api.empty(); sources={}; refs=[]
    for name,path in sorted(files.items()):
        row=json.loads(path.read_text()); parent=main[name]
        if (row['registration_sha256']!=run.sha(diagnostic.REG) or row['group']!=name
                or row['frozen_action_sha256']!=parent['action_sha256']
                or any(row[k]!=parent[k] for k in ('context','source','seed'))
                or row['result']['label_role']!='evaluation_only_after_frozen_actions'
                or row['result']['inference_or_policy_change']):
            raise ValueError('Original action/role/group identity changed')
        v=row['result']['total']; source=sources.setdefault(row['source'],api.empty())
        for dest in (total,source):
            for k in ('requested_pair_steps','known_pair_steps','unknown_pair_steps','unknown_disagreement_mass'): dest[k]+=v[k]
            for a in dest['known_error_mass']: dest['known_error_mass'][a]+=v['known_error_mass'][a]
        refs.append(dict(path=str(path.relative_to(ROOT)),sha256=run.sha(path)))
    summary=json.loads((public/'summary.json').read_text())
    # Preserve the original accumulation order for exact byte-level summary identity;
    # sorted aggregation above is independently checked to floating precision.
    import numpy as np
    if sources.keys()!=summary['by_source'].keys(): raise ValueError('Complete original source roster required')
    for value,expected in [(total,summary['total']),*[(v,summary['by_source'][k]) for k,v in sources.items()]]:
        for k in ('requested_pair_steps','known_pair_steps','unknown_pair_steps'):
            if value[k]!=expected[k]: raise ValueError('Full support count changed')
        if value['known_pair_steps']+value['unknown_pair_steps']!=value['requested_pair_steps']:
            raise ValueError('No missing pair-step may be dropped')
        np.testing.assert_allclose(value['unknown_disagreement_mass'],expected['unknown_disagreement_mass'],rtol=1e-14,atol=1e-9)
        for k in value['known_error_mass']:
            np.testing.assert_allclose(value['known_error_mass'][k],expected['known_error_mass'][k],rtol=1e-14,atol=1e-9)
        rebuilt=api.finish(value)
        for field in ('known_only_mean_error','full_requested_mean_gain'):
            for k,v in rebuilt[field].items():
                if v is None:
                    if expected[field][k] is not None: raise ValueError('Missing support imputed')
                else: np.testing.assert_allclose(v,expected[field][k],rtol=1e-13,atol=1e-10)
    frozen=json.loads((run.PUBLIC/'frozen_action_verification.json').read_text())
    if (not frozen['all72_frozen_actions_exact'] or not frozen['all72_causal_graphs_and_full_scores_exact']
            or frozen['complete_sha256']!=run.sha(run.PUBLIC/'complete.json')):
        raise ValueError('Separate full original-action verification required')
    run.once(public/'complete.json',dict(groups=refs,summary_sha256=run.sha(public/'summary.json'),
        registration_sha256=run.sha(diagnostic.REG),frozen_action_verification_sha256=run.sha(run.PUBLIC/'frozen_action_verification.json'),
        main_complete_sha256=run.sha(run.PUBLIC/'complete.json'),scalar_aggregate_only=True,
        independent_confirmation=False,original_gates_changed=False))
    bundle=pack(ROOT,public); v=summary['total']; error=v['known_only_mean_error']; gain=v['full_requested_mean_gain']
    change=100*(error['pair_joint']/error['unary_joint']-1)
    lines=['# Annotated Pair-Distance Accuracy','',
        'Secondary post-hoc diagnostic on frozen actions. Future positions/masks are evaluation labels only;no model,threshold or original gate changes.','',
        '## Full Retained Support','',
        f'{summary["groups"]} original groups;{v["requested_pair_steps"]:,} requested pair-step views,',
        f'{v["known_pair_steps"]:,} observed and{v["unknown_pair_steps"]:,} unknown.',
        'These counts repeat contexts,critic seeds and overlapping windows. They are not independent pairs or additional training examples.',
        'Zero-pair sources remain not_estimable and are not removed from the original ADE/risk population.','',
        '## Observed Labels','',
        '|Frozen policy|Mean normalized absolute pair-distance error on observed pair-steps|','|---|---:|']
    for a in ('floor','unary_joint','pair_joint'): lines.append(f'|{a}|{error[a]:.9f}|')
    lines += ['',f'Pair-aware versus unary observed error change:{change:+.6f}%. This is a pooled point estimate,not a confidence interval.',
        'Normalization uses the geometric mean of frozen past scales. Values are not meters,seconds or a physical collision rate.','',
        '## Missing Labels','',
        f'Known contribution to mean gain over all requested pair-steps:{gain["known_contribution"]:+.9f}.',
        f'Deterministic completion interval:[{gain["completion_lower"]:+.9f},{gain["completion_upper"]:+.9f}].',
        'The interval bounds missing outcomes using predicted-distance disagreement(reverse triangle inequality). It is not a bootstrap interval.',
        'It crosses zero,so the unobserved full pair-distance outcome is not identified. Missing labels are not zero errors.','',
        '|Source|Known pair-step views|Unknown pair-step views|Known gain contribution|Completion lower|Completion upper|',
        '|---|---:|---:|---:|---:|---:|']
    for name,s in sorted(summary['by_source'].items()):
        g=s['full_requested_mean_gain']; cell=lambda x:'not_estimable' if x is None else f'{x:+.9f}'
        lines.append(f'|{name}|{s["known_pair_steps"]}|{s["unknown_pair_steps"]}|{cell(g["known_contribution"])}|{cell(g["completion_lower"])}|{cell(g["completion_upper"])}|')
    lines += ['', '## Interpretation','',
        'The fixed contraction penalty reduces its own predicted geometric proxy,but observed pair-distance error and the main ADE utility both worsen.',
        'This rejects the proxy as a sufficient repair in this experiment. It does not prove every interaction model fails or resolve missing outcomes.',
        'The original all/easy risk remains failed. These are detector-silver image-coordinate SOURCE fits,not independent confirmation,human gold or physical safety.',
        f'All72 scalar groups are byte-verified in metric_groups.zip({bundle["archive_bytes"]:,}bytes);no trajectories,images,weights or future coordinates are included.','']
    (public/'readout.md').write_text('\n'.join(lines))
    run.write(public/'evidence.json',dict(groups=72,summary_sha256=run.sha(public/'summary.json'),
        complete_sha256=run.sha(public/'complete.json'),archive_sha256=bundle['archive_sha256'],
        observed_pair_error_change_percent=change,CI_not_claimed=True,frozen_actions_verified=True,
        original_gates_changed=False,independent_confirmation=False))
    print(json.dumps(dict(groups=72,observed_pair_error_change_percent=change,bundle=bundle)))


if __name__=='__main__':export()
