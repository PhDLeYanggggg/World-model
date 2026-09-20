"""Diagnose frozen query budgets without fitting or selecting a new policy."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONFIG = 'configs/m3w_frozen_risk_forensics_v1.json'


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, allow_nan=False, separators=(',', ':'))+'\n')
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--pilot-candidates', type=int)
    args = parser.parse_args()
    if args.pilot_candidates is not None and args.pilot_candidates < 1:
        raise ValueError('Positive pilot size required')
    if platform.system() == 'Darwin' and platform.machine() != 'arm64':
        raise RuntimeError('Use the native arm64 environment')
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
        os.environ[key] = '4'
    from src.evaluation.m3w_experiment_contract import file_digest
    from src.evaluation.m3w_development_evaluation import content_digest
    from scripts.analyze_m3w_frozen_interaction import independent_check
    from src.evaluation.m3w_frozen_interaction import CONTROLS
    from src.evaluation.m3w_frozen_risk_forensics import audit_query, summarize_queries
    import numpy as np

    cfg = json.loads((ROOT/CONFIG).read_text())
    if file_digest(ROOT/cfg['parent_analysis']) != cfg['parent_analysis_sha256']:
        raise ValueError('Parent analysis changed')
    parent = json.loads((ROOT/cfg['parent_analysis']).read_text())
    old_cfg = json.loads((ROOT/cfg['parent_config']).read_text())
    if file_digest(ROOT/old_cfg['protocol']) != old_cfg['protocol_file_sha256']:
        raise ValueError('Original scientific protocol changed')
    protocol = json.loads((ROOT/old_cfg['protocol']).read_text())
    jobs, bindings = [], {cfg['parent_analysis']:cfg['parent_analysis_sha256']}
    for family in old_cfg['families']:
        directory = ROOT/old_cfg['output']/family
        complete = json.loads((directory/'completion.json').read_text())
        identity = json.loads((directory/'identity.json').read_text())
        report_path = directory/'report.json'
        if (complete['run_sha256'] != content_digest(identity)
                or complete['report_sha256'] != file_digest(report_path)
                or complete['report_sha256'] != parent['source_bindings'][str(report_path.relative_to(ROOT))]):
            raise ValueError('Parent completion/source binding changed')
        bindings[str((directory/'completion.json').relative_to(ROOT))] = file_digest(directory/'completion.json')
        bindings[str(report_path.relative_to(ROOT))] = file_digest(report_path)
        for path,digest in identity['code_sha256'].items():
            if file_digest(ROOT/path) != digest:
                raise ValueError('Frozen execution code changed')
        for path,digest in identity['source_exports'].items():
            if file_digest(ROOT/path) != digest:
                raise ValueError('Original parent exports changed')
        grouped, seen = {}, set()
        for item in complete['receipts']:
            p = ROOT/item['path']
            if item['path'] in seen or file_digest(p) != item['sha256']:
                raise ValueError('Duplicate or changed parent receipt')
            seen.add(item['path'])
            receipt = json.loads(p.read_text())
            cache = p.with_name(p.name.replace('.receipt.json','.json'))
            if receipt['run_sha256'] != complete['run_sha256'] or receipt['cache_sha256'] != file_digest(cache):
                raise ValueError('Frozen decision batch changed')
            grouped.setdefault(receipt['key'][1], []).append(cache)
        report = json.loads(report_path.read_text())
        if set(grouped) != set(report['results']):
            raise ValueError('Parent candidate membership mismatch')
        for candidate, summary in report['results'].items():
            jobs.append((family,candidate,grouped[candidate],summary))
    if len(jobs) != 24:
        raise ValueError('Retain all 24 fixed combinations')
    code = [CONFIG, 'scripts/audit_m3w_frozen_risk.py',
            'src/evaluation/m3w_frozen_risk_forensics.py', 'scripts/analyze_m3w_frozen_interaction.py']
    run_identity = dict(source_bindings=bindings, code_sha256={p:file_digest(ROOT/p) for p in code},
                        parent_protocol_sha256=old_cfg['protocol_file_sha256'],
                        new_training=False, policy_selection=False)
    run_digest = content_digest(run_identity)
    private = ROOT/cfg['private_output']
    if args.resume:
        if json.loads((private/'identity.json').read_text()) != run_identity:
            raise ValueError('Forensics resume identity changed')
    else:
        private.mkdir(parents=True, exist_ok=False)
        atomic_json(private/'identity.json', run_identity)
    already_complete = (private/'completion.json').exists()
    began, fresh, reused = time.monotonic(), 0, 0
    results, verifications, receipts = [], [], []
    for family,candidate,paths,original in jobs:
        name = content_digest([family,candidate])
        cache, receipt_path = private/(name+'.json'), private/(name+'.receipt.json')
        policy = protocol['development_evaluation']['policies'][candidate.rsplit('_',1)[-1]]
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if receipt != dict(run_sha256=run_digest, key=[family,candidate], cache_sha256=file_digest(cache)):
                raise ValueError('Forensics cached candidate changed')
            saved = json.loads(cache.read_text())
            for arm in CONTROLS:
                if summarize_queries(saved['query_records'][arm], cfg['predicted_budget_fraction_bins']) != saved['summary'][arm]:
                    raise ValueError('Saved risk reduction differs')
            reused += 1
        else:
            if already_complete:
                raise ValueError('Completed forensics missing candidate')
            rows, queries = [], []
            for path in paths:
                block = json.loads(path.read_text())
                rows.extend(block['rows']); queries.extend(block['queries'])
            checked = independent_check(rows, queries, original, policy)
            by_query = {}
            for row in rows:
                key = tuple(row[k] for k in ('recording_id','frame_id','horizon_raw'))
                by_query.setdefault(key, []).append(row)
            detail, summary = {}, {}
            for query_control,arm in zip(('independent','unary_geometry','joint'),CONTROLS):
                detail[arm] = [audit_query(by_query[tuple(q[k] for k in ('recording_id','frame_id','horizon_raw'))],q,
                    control=arm, query_control=query_control, budget=policy['max_mean_predicted_harm'],
                    easy_threshold=protocol['development_evaluation']['easy_threshold'],
                    tolerance=cfg['diagnostic_tolerance']) for q in queries]
                summary[arm] = summarize_queries(detail[arm],cfg['predicted_budget_fraction_bins'])
                m = summary[arm]; expected = original['full']['arms'][arm]
                if m['agents'] != original['full']['agent_query_count'] or m['labeled_agents'] != original['full']['label_coverage']['ade']:
                    raise ValueError('Risk diagnostic changed query/label population')
                np.testing.assert_allclose(m['easy_degradation_percent'],100*expected['easy']['degradation_fraction'],rtol=0,atol=1e-9)
                np.testing.assert_allclose(m['observed_positive_harm_sum']/m['labeled_agents'],expected['all']['mean_positive_harm'],rtol=0,atol=1e-12)
            saved = dict(query_records=detail, summary=summary, independent_parent_check=checked)
            atomic_json(cache,saved)
            atomic_json(receipt_path,dict(run_sha256=run_digest,key=[family,candidate],cache_sha256=file_digest(cache)))
            fresh += 1
        for arm,m in saved['summary'].items():
            results.append(dict(family=family,candidate=candidate,control=arm,**m))
        verifications.append(dict(family=family,candidate=candidate,**saved['independent_parent_check']))
        receipts.append(dict(path=str(receipt_path.relative_to(ROOT)),sha256=file_digest(receipt_path)))
        if not already_complete:
            status=dict(pid=os.getpid(),family=family,candidate=candidate,completed_candidates=len(receipts),
                        new_candidates=fresh,reused_candidates=reused,elapsed_seconds=time.monotonic()-began,
                        status='running_not_complete')
            atomic_json(private/'heartbeat.json',status); print(json.dumps(status),flush=True)
            if args.pilot_candidates and fresh >= args.pilot_candidates:
                status['status']='pilot_complete_full_not_complete'
                atomic_json(private/'heartbeat.json',status)
                return
    report = dict(result_source='fresh_run_risk_reduction_cached_verified_frozen_decisions',
        run_sha256=run_digest,source_bindings=bindings,results=results,verification=verifications,
        physical_sites=1,scene_ci='not_run_insufficient_independent_scenes',
        primary_metric_changed=False,new_training=False,new_inference=False,policy_selection=False,
        actual_independent_calibration=False,stage5c_executed=False,smc_enabled=False)
    output = ROOT/cfg['output']; output.mkdir(parents=True,exist_ok=True)
    report_path=output/'analysis.json'
    if already_complete:
        complete=json.loads((private/'completion.json').read_text())
        if (complete['run_sha256']!=run_digest or complete['report_sha256']!=file_digest(report_path)
                or complete['receipts']!=receipts or json.loads(report_path.read_text())!=report):
            raise ValueError('Completed report/reduction changed')
        print(json.dumps(dict(status='complete_exact_replay',new_candidates=fresh,reused_candidates=reused)),flush=True)
        return
    report_path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    flat=[]
    for r in results:
        item={k:r[k] for k in ('family','candidate','control','queries','switched_queries','unknown_selected_agents',
            'predicted_budget_failure_queries','mean_query_predicted_harm','mean_query_harm_lower_bound',
            'mean_past_agent_harm_lower_bound','easy_degradation_percent')}
        for s,v in r['budget_status'].items():
            item[s+'_queries']=v['queries']
            item[s+'_easy_harm_fraction']=r['easy_positive_harm_fraction'][s]
        flat.append(item)
    with (output/'all_controls.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    atomic_json(private/'completion.json',dict(run_sha256=run_digest,report_sha256=file_digest(report_path),
        receipts=receipts,new_candidates=fresh,reused_candidates=reused,elapsed_seconds=time.monotonic()-began))
    atomic_json(private/'heartbeat.json',dict(pid=os.getpid(),status='complete_no_policy_change',completed_candidates=len(receipts)))
    print(json.dumps(dict(status='all_frozen_risk_diagnostics_complete',comparisons=len(results),
                         new_candidates=fresh,reused_candidates=reused)),flush=True)


if __name__ == '__main__':
    main()
