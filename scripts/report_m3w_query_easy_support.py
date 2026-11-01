"""Rebuild necessary query support evidence from public scalar aggregates only."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_m3w_query_easy_support as run
from scripts.report_m3w_reference_support_preflight import build as previous_build
from scripts.report_m3w_feature_cost_evidence import aggregate_groups
from src.evaluation.m3w_metric_bundle import verify


def build():
    p=run.PUBLIC;reg=json.loads(run.REG.read_text())
    for rel,digest in reg['bindings'].items():
        if run.sha(ROOT/rel)!=digest:raise ValueError('Frozen query diagnosis changed')
    parent=previous_build()
    for name in ('complete','summary','verification'):
        if reg['previous_'+name+'_sha256']!=run.sha(run.previous.PUBLIC/(name+'.json')):raise ValueError('Verified preflight changed')
    complete=json.loads((p/'complete.json').read_text());checked=json.loads((p/'verification.json').read_text())
    if (complete['registration_sha256']!=run.sha(run.REG) or complete['summary_sha256']!=run.sha(p/'summary.json')
            or checked['summary_sha256']!=run.sha(p/'summary.json') or checked['complete_sha256']!=run.sha(p/'complete.json')
            or not checked['exact72_groups_and_remaining_query_results']):raise ValueError('Full independent query recalculation required')
    bundle=verify(ROOT,p);own=aggregate_groups(run.NAME);old=aggregate_groups(run.previous.NAME)
    rows=[own[Path(r['path']).name] for r in complete['groups']];records=set()
    expected={(c,s,k) for c,ss in reg['contexts'].items() for s in ss for k in reg['seeds']}
    if {(r['context'],r['source'],r['seed']) for r in rows}!=expected:raise ValueError('All original72identities required')
    for r in rows:
        key=f'{r["context"]}_{r["source"]}_seed{r["seed"]}_fit.json';prior=old[key]
        if (r['registration_sha256']!=run.sha(run.REG) or r['cost_prediction_sha256']!=prior['cost_prediction_sha256']
                or r['ids_sha256']!=prior['ids_sha256'] or set(r['recordings'])!=set(prior['result']['recordings'])):
            raise ValueError('All original input and recording identities required')
        for rec,value in r['recordings'].items():
            records.add((r['source'],rec));was_safe=prior['result']['recordings'][rec]['reference_pointwise']['budget_feasible']
            if (value['result_source']=='cached_verified_pointwise_witness')!=was_safe:raise ValueError('Cached and fresh provenance differ')
            if not was_safe and value['result_source']!='fresh_query_feasibility':raise ValueError('Every unresolved record must be checked')
    s=json.loads((p/'summary.json').read_text());rebuilt=run.summarize(rows);rebuilt.update(unique_target_rows=149510,unique_recordings=len(records))
    if s!=rebuilt or len(records)!=74:raise ValueError('Canonical complete query summary differs')
    return dict(summary=s,complete=complete,verification=checked,bundle=bundle,parent=parent)


def render(v):
    p=run.PUBLIC;s=v['summary']
    lines=['# Necessary Easy Support After Query Budget Pooling','','## Material Passport','',
        'fresh_run:query-support diagnosis for every unresolved record view and independent numeric replay.',
        'cached_verified:positive pointwise witnesses,original forecasts,costs,rows and complete control scores.',
        'All72groups,149,510unique targets and74recordings remain;future easy labels are diagnostic-only,never inference inputs.',
        'No new training,threshold search,independent-role opening or deployment.','',
        '## Whole Roster','',
        f'- Required recording views:{s["required_record_views"]}.',
        f'- Cached verified pointwise witnesses:{s["cached_verified_pointwise_record_views"]}.',
        f'- Fresh unresolved record checks:{s["fresh_query_record_views"]}.',
        f'- Necessary query easy support available:{s["necessary_query_easy_supported_record_views"]}.',
        f'- Obstructed recording views:{len(s["blocked_record_views"])}.',
        f'- Inconclusive recording views:{len(s["inconclusive_record_views"])}.',
        f'- Obstructed with only algebraic/singleton certificates:{s["algebraic_or_singleton_blocked_record_views"]}.','',
        'A positive easy anchor is necessary,not sufficient for positive actual safe gain. A feasible query witness does not pass a model gate.',
        'The432pointwise witnesses are not freshly recalculated in this phase;they are verified prior evidence.','',
        '## Remaining Views','','|Context|Source|Seed|Recording|Queries with known easy rows|Feasible anchor queries|Inconclusive|Reason counts|',
        '|---|---|---:|---|---:|---:|---:|---|']
    for r in s['blocked_record_views']+s['inconclusive_record_views']:
        d=r['diagnosis'];lines.append(f'|{r["context"]}|{r["source"]}|{r["seed"]}|{r["recording"]}|{d["queries_with_known_easy_rows"]}|{d["feasible_known_easy_anchor_queries"]}|{d["inconclusive_queries"]}|{d["reason_counts"]}|')
    lines+=['','## Interpretation','',
        'Singleton and nonnegative-cost sign cases are finite algebraic obstructions under the frozen supplied moments.',
        'MILP status2 is a numerical solver verdict,not an algebraic proof. Nonoptimal or failed exact-budget postchecks remain inconclusive.',
        'The binary easy-action count requires support without an arbitrary epsilon on tiny coordinate values.',
        'Every accepted witness is checked against both unchanged2%predicted budgets;the old benefit>harm sign is not reinstated.',
        'The analysis concerns original current query targets,not a claim to contain every visible agent.','',
        f'Frozen cost family has a necessary-support obstruction:{s["frozen_cost_family_has_necessary_support_obstruction"]}.',
        'If true,training another score actor while keeping these costs and budgets cannot restore support in those obstructed views.',
        'Repair or refit causal cost estimation and verify its calibration/support first;do not relax2%,insert oracle labels or discard recordings.',
        'If false,positive actual safe gain and a learned causal policy are still unproven;the necessary condition alone is not training or deployment success.','',
        '## Execution And Reproduction','',
        f'Complete PID{v["complete"]["pid"]},{v["complete"]["seconds"]:.6f}s,peakRSS{v["complete"]["peak_RSS_bytes"]}B.',
        'Phase timing starts at runtime_ready. Native arm64 CPU4/interop1/workers0,10s/query MILP cap,heartbeats and per-group resume.',
        'Independent replay verifies every original group and remaining query result;public report rebuilding reads scalar aggregates only.','',
        '```bash','.venv-pytorch/bin/python scripts/run_m3w_query_easy_support.py run',
        '.venv-pytorch/bin/python scripts/run_m3w_query_easy_support.py verify',
        '.venv-pytorch/bin/python scripts/report_m3w_query_easy_support.py','```','',
        'Completed outputs cannot be overwritten;an interrupted unfinished run accepts `run --resume`.',
        'The first two commands require private assets. Public rebuilding is not anonymous from-scratch reproduction.','',
        '## Scope','',
        'SOURCE fitting data only,12exposed localities and repeated context/critic seeds,not independent confirmation or multiple forecaster seeds.',
        'EuropeanSquares oblique detector-silver/local-image,obs8/pred12/rawstride12. Not SDDt50,meters,seconds,human gold,true3D or foundation.',
        'Unknown labels never create known reference support. No Stage5C,SMC,deployment or submission-ready claim.','']
    (p/'readout.md').write_text('\n'.join(lines))
    run.write(p/'evidence.json',dict(summary_sha256=run.sha(p/'summary.json'),verification_sha256=run.sha(p/'verification.json'),
        archive_sha256=v['bundle']['archive_sha256'],necessary_support_obstruction=s['frozen_cost_family_has_necessary_support_obstruction'],
        new_training=False,oracle_not_model_success=True,independent_confirmation=False,deployment_changed=False,submission_ready=False))


if __name__=='__main__':
    v=build();render(v);print(json.dumps(dict(groups=v['summary']['groups'],blocked=len(v['summary']['blocked_record_views']),inconclusive=len(v['summary']['inconclusive_record_views']),new_training=False)))
