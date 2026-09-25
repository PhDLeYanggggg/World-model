"""Validate training, lineage, full readout and scoped checks without promotion."""
import csv
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
PUBLIC=ROOT/'outputs/publication_readiness_2026_09/european_geometric_cost_v1'
PRIVATE=ROOT/'data/stage_cvpr2027_experiments/european_geometric_cost_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    a=json.loads((PUBLIC/'analysis.json').read_text())
    v=json.loads((PUBLIC/'verification.json').read_text())
    replay=json.loads((PUBLIC/'head_replay.json').read_text())
    if not v['all_passed'] or v['analysis_sha256']!=sha(PUBLIC/'analysis.json') or v['identity']!=a['identity']:
        raise ValueError('Full verified metrics required')
    if len(a['views'])!=144 or len(a['replacements'])!=108 or len(a['neural_vs_damping'])!=72:
        raise ValueError('Incomplete registered matrix')
    if a['original_views_reproduced']!=36 or replay['identity']!=a['identity'] or len(replay['checks'])!=54:
        raise ValueError('All old policies and new heads must replay')
    for path,expected in a['identity']['bindings'].items():
        if sha(ROOT/path)!=expected:raise ValueError('Frozen code/registration changed')
    events=[json.loads(line) for line in (PRIVATE/'events.jsonl').read_text().splitlines()]
    last={r['pid']:r for r in events}
    for pid,row in last.items():
        command=subprocess.run(['ps','-p',str(pid),'-o','args='],capture_output=True,text=True)
        if 'run_m3w_european_geometric_cost.py' in command.stdout or row['state']!='phase_complete':
            raise ValueError('Required phase still active or incomplete')
    records=[]
    for check in replay['checks']:
        if not check['exact'] or not check['sampler_exact'] or check['rows']!=4096 or check['total_draws']!=512000:
            raise ValueError('Matched complete checkpoint replay required')
        cp=ROOT/check['checkpoint']['path']
        if sha(cp)!=check['checkpoint']['sha256']:raise ValueError('Checkpoint changed')
        r=json.loads((cp.parent/'complete.json').read_text())
        old=ROOT/r['identity']['original_checkpoint']['path']
        parent=json.loads((old.parent/'complete.json').read_text())
        if sha(old)!=r['identity']['original_checkpoint']['sha256']:
            raise ValueError('Original model changed')
        if (r['fit']['step']!=2000 or not r['fit']['complete'] or r['fit']['unknown_rows_sampled']!=0
                or r['fit']['parameters']!=parent['fit']['parameters']):
            raise ValueError('Training budget, label support or parameter count not matched')
        records.append((check['head'],r,parent))
    with (PUBLIC/'training_losses.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['head','step','new_batch_loss','old_same_batch_loss','new_gradient_norm','old_gradient_norm'])
        for name,r,parent in records:
            oldtrace={t['step']:t for t in parent['fit']['trace']}
            for t in r['fit']['trace']:
                old=oldtrace.get(t['step'],{})
                writer.writerow([name,t['step'],t['loss'],old.get('loss'),t['gradient_norm'],old.get('gradient_norm')])
    previous=ROOT/'outputs/publication_readiness_2026_09/european_producer_transport_v1/completion_checks.json'
    tests=json.loads(previous.read_text())['scoped_test_files']+[
        'tests/test_m3w_geometric_cost_head.py','tests/test_m3w_geometric_cost_reporting.py']
    run=subprocess.run([sys.executable,'-m','pytest','-q',*tests],cwd=ROOT,capture_output=True,text=True)
    (PUBLIC/'scoped_tests.txt').write_text(run.stdout+run.stderr)
    if run.returncode or not re.search(r'\b210 passed\b',run.stdout):
        raise ValueError('Scoped tests failed; inspect scoped_tests.txt')
    output=dict(result_source='fresh_run_completion_verified_training_and_readout',
        utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),python=sys.version,architecture=platform.machine(),
        analysis_sha256=v['analysis_sha256'],summary_sha256=sha(PUBLIC/'summary_metrics.json'),
        reporter_sha256=sha(Path(__file__)),new_heads=54,new_updates=sum(r['fit']['step'] for _,r,_ in records),
        head_fit_seconds=sum(r['fit']['seconds'] for _,r,_ in records),
        parameters_per_head=records[0][1]['fit']['parameters'],label_unknown_training_draws=0,
        cached_heads=54,head_replays=54,head_sampler_matches=54,replay_rows_per_head=4096,
        replay_sampling='first_held_index_rows_not_random',old_policy_views_reproduced=36,
        pointwise_views=144,matched_replacement_contrasts=108,neural_damping_contrasts=72,
        scoped_test_files=tests,scoped_tests_passed=210,full_legacy_suite_run=False,
        scoped_tests_sha256=sha(PUBLIC/'scoped_tests.txt'),training_trace_sha256=sha(PUBLIC/'training_losses.csv'),
        required_processes_finished=sorted(last),all_required_local_processes_finished=True,
        remote_execution='not_run_local_pilot_supported_local_fit',remote_m3w_assets='not_run_directory_unverified',
        forecasts_unchanged=True,threshold_refit=False,calibration_refit=False,reserved_roles_opened=False,
        deployment_changed=False,submission_ready=False,stage5c_executed=False,smc_enabled=False)
    (PUBLIC/'completion_checks.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(heads=54,updates=output['new_updates'],tests_passed=210,processes_finished=True)))


if __name__=='__main__':
    main()
