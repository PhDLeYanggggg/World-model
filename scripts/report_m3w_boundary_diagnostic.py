"""Report verified remote frozen-boundary accounting without policy selection."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts import manage_m3w_boundary_diagnostic as run


def verify_summary(s):
    checks=0
    for phase,phase_rows in s['phases'].items():
        assert phase_rows['groups']==(72 if phase=='fitting' else 216)
        assert phase_rows['rows']==phase_rows['known_rows']+phase_rows['unknown_rows']
        for arm in phase_rows['arms'].values():
            for scope in arm.values():
                for event in scope.values():
                    assert 0<=event['violating_views']<=event['defined_views']<=phase_rows['groups']
                    assert event['selected_rows']==event['known_selected']+event['unknown_selected']
                    for site,metrics in event['by_site'].items():
                        keys=('true_excess_over_selected_reference','predicted_excess_over_selected_reference',
                              'harm_underestimate_over_selected_reference','reference_overestimate_over_selected_reference')
                        vals=[metrics[k]['conditional_mean'] for k in keys]
                        if all(v is not None for v in vals):
                            np.testing.assert_allclose(vals[0],sum(vals[1:]),rtol=1e-10,atol=1e-9)
                            checks+=1
                        assert len({metrics[k]['defined'] for k in keys})==1
                    for k,overall in event['equal_site'].items():
                        vals=[v[k]['mean'] for v in event['by_site'].values()]
                        finite=[v for v in vals if v is not None]
                        assert overall['defined']==len(finite) and overall['undefined']==len(vals)-len(finite)
                        if finite:
                            np.testing.assert_allclose(overall['conditional_mean'],np.mean(finite),rtol=1e-12)
                        assert (overall['mean'] is None)==(not finite or len(finite)!=len(vals))
                        checks+=1
    return checks


def value(z,key):
    row=z['equal_site'][key]
    if row['mean'] is not None:return f"{100*row['mean']:.4f}"
    if row['conditional_mean'] is None:return 'undefined'
    return f"{100*row['conditional_mean']:.4f} ({row['defined']}/12 defined localities)"


def main():
    began=time.monotonic();cfg,reg=run.registration()
    for name in ('statistical_evidence.md','failure_analysis.md'):
        assert (run.PUBLIC/name).is_file(), f'Interpretation record missing: {name}'
    assert reg==json.loads((run.PUBLIC/'registration.json').read_text())
    received=json.loads((run.PRIVATE/'collected.json').read_text())
    r=received['receipt'];s=received['summary']
    assert received['accounting']['code']==0 and received['accounting']['stdout'].startswith('COMPLETED|0:0|')
    assert r['full_replay_exact'] and r['groups']==288 and received['local_parent_risk_checks']==1728
    assert r['manifest_sha256']==run.digest(run.PUBLIC/'packet_manifest.json')
    assert r['summary_sha256']==run.digest(run.PUBLIC/'summary.json')
    checks=verify_summary(s)
    lines=['| Phase | Arm | Event | Actual risk (%) | Predicted excess / actual reference (pp) | Harm error (pp) | Reference error (pp) | Risk violations / defined | Unknown selected |',
           '|---|---|---|---:|---:|---:|---:|---:|---:|']
    tail=['| Phase | Arm | Event | Tail share of actual harm (%) | Tail share of positive harm-underestimate (%) |',
          '|---|---|---|---:|---:|']
    for phase,phase_rows in s['phases'].items():
        for arm in cfg['arms']:
            for event in ('all','easy'):
                z=phase_rows['arms'][arm]['matched'][event]
                fields=[value(z,k) for k in ('realized_risk_ratio','predicted_excess_over_selected_reference',
                                             'harm_underestimate_over_selected_reference','reference_overestimate_over_selected_reference')]
                lines.append(f'| {phase} | {arm} | {event} | '+' | '.join(fields)+
                    f" | {z['violating_views']}/{z['defined_views']} | {z['unknown_selected']} |")
                tail.append(f'| {phase} | {arm} | {event} | '+value(z,'tail_harm_share')+' | '+value(z,'tail_positive_underestimate_share')+' |')
    report=('# Frozen Decision-Boundary Risk Diagnosis\n\n## Material Passport\n\n'
        'fresh_run diagnostic and full exact replay on an allocated CREATE CPU node; '
        'cached_verified native checkpoints, causal score/action packets and source labels. '
        'No model training, parameter updates, policy selection or independent confirmation.\n\n'
        f"Slurm job `{r['job_id']}` completed with 0:0 exit. All 288 groups: 72 source-fitting and 216 internal-transfer. "
        f"First pass {r['first_pass_seconds']:.2f}s; replay {r['replay_seconds']:.2f}s. "
        'These are dependent role/seed views of 12 already opened localities.\n\n'
        '## Matched-Count Excess Decomposition\n\n'+'\n'.join(lines)+'\n\n'
        'Every row uses the same realized selected reference for its excess components. '
        'Actual risk equals 2% plus predicted excess plus harm error plus reference error. '
        'Positive harm error means actual harm exceeded its prediction; positive reference error '
        'means the predicted reference allowed too much budget. Components are signed and can offset. '
        'All entries average within locality before across localities. Undefined quantities remain '
        'explicit; conditional summaries do not certify the missing localities. '
        'Unknown selected counts cover all unknown interventions; their easy membership is itself unknown.\n\n'
        'Predicted risk itself (predicted harm / predicted reference), near-boundary, supported '
        'eligible and unmatched selected scopes are separately preserved in summary.json. '
        'Do not mistake predicted risk for the predicted-excess column.\n\n'
        '## Training-Defined Severe Tail\n\n'+'\n'.join(tail)+'\n\n'
        'The tail threshold is the 95th percentile of positive harm on each head\'s original '
        'training source only. Tail shares use actual harm or positive underestimation as denominators, '
        'not the risk denominator. They are descriptive; no held-locality quantile or threshold is fitted.\n\n'
        '## Verification\n\n'
        f"Full exact replay, {r['accounting_checks']:,} native accounting checks, "
        f"{checks:,} additional local summary reductions, 1,728 agreements with the frozen parent "
        'transfer risk readout, and three local/CREATE packet parity checks. Cross-platform numerical '
        'parity permits only 1e-10 relative/1e-9 absolute roundoff; remote replay is exact. '
        'The three parity packets are a spot check, not an independent recomputation of all rows locally.\n\n'
        '## What This Does Not Establish\n\n'
        'Observed residual bias on an adaptively selected developmental population is not proof of '
        'population conditional miscalibration or a unique causal failure mechanism. Unknown future '
        'labels are not replaced by zero and cannot become an inference filter. Fitting bias and '
        'transfer bias are reported separately; neither represents independent calibration. '
        'No new p-values or bootstrap claim is introduced by this descriptive diagnostic.\n\n'
        '## Research Boundaries\n\n'
        'Existing 4/4/2/2 producer/controller/fitting/outer roles are retained, with single-source inner '
        'fits. Independent selection/calibration/confirmation remain closed. Obs8/pred12, stride12 '
        'raw frames, image-local detector-silver. No metric/seconds, true3D, foundation, human-gold '
        'or safe deployment claim. Stage5C and SMC stay off.\n')
    (run.PUBLIC/'results.md').write_text(report)
    (run.PUBLIC/'operation.md').write_text('# Execution and Recovery\n\n'
        'Use the native arm64 local runtime. Row packets are streamed from existing caches in memory, '
        'not duplicated to local disk. Keep the old10GiB cache/training reserve; current storage '
        'shortfall blocks new local caches, not authorized isolated CREATE computation.\n\n'
        '```bash\nPYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/manage_m3w_boundary_diagnostic.py register\n'
        '# Commit registration before export.\n'
        'PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/manage_m3w_boundary_diagnostic.py export --resume\n'
        '# Commit packet_manifest.json before submitting.\n'
        'PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/manage_m3w_boundary_diagnostic.py submit\n'
        'PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/manage_m3w_boundary_diagnostic.py inspect\n'
        'PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/manage_m3w_boundary_diagnostic.py collect\n'
        'PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/report_m3w_boundary_diagnostic.py\n```\n\n'
        'Completed receipts are immutable. Do not submit again after a timeout: inspect the remote '
        'intent, submission receipt, scheduler state and heartbeat first. The allocated runner resumes '
        'completed groups with --resume. After an actual terminal error, record it, verify hashes '
        'and register any narrow repair before a replacement submission; no automatic duplicate job.\n\n'
        'Only the independent M3W CREATE directory/runtime are used. Never borrow or change the '
        'simulation project or its jobs. Shared filesystem capacity does not establish personal quota. '
        'No row packets, raw data or checkpoints belong in Git. Full legacy tests and raw-data rebuild '
        'are not run in this diagnostic.\n')
    tests=['tests/test_m3w_boundary_diagnostic.py','tests/test_m3w_boundary_report.py',
           'tests/test_m3w_inner_separability_report.py','tests/test_m3w_inner_separability.py']
    test=subprocess.run([sys.executable,'-m','pytest',*tests,'-q'],cwd=ROOT,capture_output=True,text=True)
    if test.returncode:raise RuntimeError(test.stdout+test.stderr)
    (run.PUBLIC/'scoped_tests.txt').write_text(test.stdout+test.stderr)
    out=dict(status='verified_descriptive_development_diagnosis_not_model_improvement',
        groups=288,full_remote_exact_replay=True,parent_risk_checks=1728,local_parity_packets=3,
        summary_reduction_checks=checks,native_accounting_checks=r['accounting_checks'],
        source_bindings={**reg['remote_code'],**reg['controls'],str(Path(__file__).relative_to(ROOT)):run.digest(Path(__file__)),
            'tests/test_m3w_boundary_report.py':run.digest(ROOT/'tests/test_m3w_boundary_report.py')},
        artifacts={p.name:run.digest(p) for p in sorted(run.PUBLIC.iterdir()) if p.is_file() and p.name!='verification.json'},
        full_legacy_suite='not_run_scoped_tests_only',parameter_updates=0,deployment_changed=False,
        seconds=time.monotonic()-began)
    run.immutable(run.PUBLIC/'verification.json',out)
    print(json.dumps(dict(verified=True,job_id=r['job_id'],summary_checks=checks,seconds=out['seconds'])))


if __name__=='__main__':main()
