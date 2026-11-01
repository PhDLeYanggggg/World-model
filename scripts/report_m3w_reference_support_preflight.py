"""Public-only evidence for expanded causal-reference action support."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_m3w_reference_support_preflight as run
from scripts.report_m3w_feature_cost_evidence import aggregate_groups
from src.evaluation.m3w_metric_bundle import verify


def build():
    p=run.PUBLIC;reg=json.loads(run.REG.read_text())
    for rel,digest in reg['bindings'].items():
        if run.sha(ROOT/rel)!=digest:raise ValueError('Frozen preflight changed')
    for name in ('summary','complete','verification'):
        if reg['previous_'+name+'_sha256']!=run.sha(run.previous.PUBLIC/(name+'.json')):raise ValueError('Original diagnosis changed')
    complete=json.loads((p/'complete.json').read_text());check=json.loads((p/'verification.json').read_text())
    if (complete['registration_sha256']!=run.sha(run.REG) or complete['summary_sha256']!=run.sha(p/'summary.json')
            or check['summary_sha256']!=run.sha(p/'summary.json') or check['complete_sha256']!=run.sha(p/'complete.json')
            or not check['exact72_numeric_replay'] or not check['all_original_controls_exact']):raise ValueError('Complete independent recalculation required')
    bundle=verify(ROOT,p);groups=aggregate_groups(run.NAME);old=aggregate_groups(run.parent.NAME)
    rows=[groups[Path(r['path']).name] for r in complete['groups']];records=set();unsupported={a:[] for a in run.api.NAMES}
    for row in rows:
        key=f'{row["context"]}_{row["source"]}_seed{row["seed"]}_fit.json';control=old[key]
        if (row['registration_sha256']!=run.sha(run.REG) or row['cost_prediction_sha256']!=control['cost_prediction_sha256']
                or row['ids_sha256']!=control['ids_sha256']):raise ValueError('All original cost predictions and target identities required')
        for a in run.api.CONTROLS:
            if row['result']['controls'][a]!=control['result']['policies'][a]['policy']:raise ValueError('Frozen complete policy scores differ')
        for rec,detail in row['result']['recordings'].items():
            records.add((row['source'],rec))
            for a in run.api.NAMES:
                if not detail[a]['budget_feasible']:unsupported[a].append(dict(context=row['context'],source=row['source'],seed=row['seed'],recording=rec,reason=detail[a]['infeasibility_reason']))
    s=json.loads((p/'summary.json').read_text());rebuilt=run.api.summarize(rows,reg['contexts'],json.loads(run.parent.CONFIG.read_text()))
    rebuilt.update(unique_target_rows=149510,unique_recordings=len(records))
    if rebuilt!=s or len(records)!=74:raise ValueError('Complete canonical preflight differs')
    return dict(summary=s,complete=complete,verification=check,bundle=bundle,unsupported=unsupported,
        previous=json.loads((run.previous.PUBLIC/'summary.json').read_text()))


def interval(v):return 'not_estimable' if v['CI95'] is None else f'{v["point"]:.6f} [{v["CI95"][0]:.6f},{v["CI95"][1]:.6f}]'


def render(v):
    p=run.PUBLIC;s=v['summary']
    lines=['# Removing The Frozen Gain-Sign Gate: Complete Preflight','','## Material Passport','',
        'fresh_run:complete72-group label-aware preflight and exact independent numerical replay.',
        'cached_verified:all forecasts,cost critics,original inputs,archived controls and previous support diagnosis.',
        'not_run:new policy training,independent selection/calibration/confirmation and deployment.',
        'This is an exposed SOURCE feasibility calculation. Oracle witnesses are not model results or inference inputs.','',
        '## One Changed Restriction','',
        'Causal action availability and positive predicted all/easy references remain. Only the old predicted benefit>harm sign restriction is removed.',
        'Both original predicted2%query budgets and original actual/bounded recording evaluation remain unchanged.',
        'No future label,oracle action or test statistic enters the new production support predicate.','',
        '|Support|Label-aware safe witness gain pp [conditional95%CI]|Supported recording views|Positive recording views|','|---|---:|---:|---:|']
    for name in ('predicted_positive','pointwise_cost_guard'):
        r=v['previous']['supports'][name];lines.append(f'|Previous:{name}|{interval(r["safe_witness_gain_pp"])}|{r["supported_record_views"]}/{r["required_record_views"]}|{r["positive_record_views"]}|')
    for name,r in s['supports'].items():lines.append(f'|Expanded:{name}|{interval(r["witness_gain_pp"])}|{r["supported_record_views"]}/{r["required_record_views"]}|{r["positive_record_views"]}|')
    lines+=['','## Preflight Decision','',f'Conservative pointwise preflight pass:{s["conservative_pointwise_preflight_pass"]}.',
        'A pass permits a controlled causal learning experiment;it is not a trained-policy safety or generalization certificate.',
        'A failure does not prove joint infeasibility:pointwise support is narrower than query budget pooling.',
        'If some views remain unsupported,exact query-pooling feasibility must be checked before investing in another actor fit.','',
        'All witness counts are free,not the trained actor per-query counts. Unknown labels never create safe easy-reference anchors.',
        'Empty risk remains not_estimable. No records are removed.','',
        '## Remaining Unsupported Views','']
    for name,r in s['supports'].items():lines.append(f'- {name}: {r["infeasible_reasons"]}')
    lines+=['','Full list:`unsupported_record_views.json`.','',
        '## Reproduction','',
        f'Complete PID{v["complete"]["pid"]},{v["complete"]["seconds"]:.6f}s,peakRSS{v["complete"]["peak_RSS_bytes"]}B;phase timing starts at runtime_ready.',
        'All72fresh calculations and original complete control scores replay exactly. Native arm64 CPU4/interop1/workers0.',
        'Registration binds the scientific code and protocol. Public report regeneration uses only scalar aggregates,not private inputs.','',
        '```bash',
        '.venv-pytorch/bin/python scripts/run_m3w_reference_support_preflight.py run',
        '.venv-pytorch/bin/python scripts/run_m3w_reference_support_preflight.py verify',
        '.venv-pytorch/bin/python scripts/report_m3w_reference_support_preflight.py','```','',
        'Completed outputs are immutable;only an interrupted unfinished run accepts `run --resume`.',
        'The first two commands need the original private assets;the report command does not claim from-scratch reproduction.','',
        '## Scope','',
        '149,510unique targets,74recordings,12exposed localities and72repeated context/critic-seed views. Forecast producer43 is shared.',
        '3,000conditional locality bootstrap draws average repeated views before resampling;not independent confirmation.',
        'EuropeanSquares oblique detector-silver/local-image,obs8/pred12/rawstride12. Not SDDt50,meters,seconds,human gold,true3D or foundation.',
        'Wholly-unknown bounds do not solve missing interior labels. No deployment,Stage5C,SMC or submission-ready claim.','']
    (p/'readout.md').write_text('\n'.join(lines));run.write(p/'unsupported_record_views.json',v['unsupported'])
    run.write(p/'evidence.json',dict(summary_sha256=run.sha(p/'summary.json'),verification_sha256=run.sha(p/'verification.json'),
        bundle_sha256=v['bundle']['archive_sha256'],new_training=False,oracle_not_model_success=True,
        conservative_pointwise_preflight_pass=s['conservative_pointwise_preflight_pass'],
        independent_confirmation=False,deployment_changed=False,submission_ready=False))


if __name__=='__main__':
    value=build();render(value);print(json.dumps(dict(groups=value['summary']['groups'],preflight_pass=value['summary']['conservative_pointwise_preflight_pass'],new_training=False)))
