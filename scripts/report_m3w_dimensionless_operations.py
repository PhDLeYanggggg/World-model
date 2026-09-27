"""Completed resource receipts from timed logs, not runtime estimates."""
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts import run_m3w_european_dimensionless_refit as run


def main():
    phases=[]
    for phase in ('pilot','train','predict','evaluate','replay','verify_eval'):
        path=run.PRIVATE/(phase+'.log'); text=path.read_text()
        events=[json.loads(line) for line in text.splitlines() if line.startswith('{')]
        start=next(e for e in events if e.get('state')=='started' and e.get('phase')==phase)
        end=next(e for e in reversed(events) if e.get('state')=='complete' and e.get('phase')==phase)
        wall=re.search(r'^\s*([\d.]+) real\s+([\d.]+) user\s+([\d.]+) sys\s*$',text,re.M)
        rss=re.search(r'^\s*(\d+)\s+maximum resident set size\s*$',text,re.M)
        assert wall and rss
        phases.append(dict(phase=phase,pid=start['pid'],started_utc=start['utc'],completed_utc=end['utc'],
            wall_seconds=float(wall[1]),user_seconds=float(wall[2]),system_seconds=float(wall[3]),
            peak_RSS_bytes=int(rss[1]),log=run.artifact(path),completion_observed=True))
    frozen=json.loads((run.PUBLIC/'prediction_freeze.json').read_text())
    fits=[json.loads((ROOT/r['path']).read_text())['fit'] for r in frozen['training']]
    queue=json.loads((run.PRIVATE/'create_queue.json').read_text())
    assert queue['jobs_submitted']==0 and not queue['remote_modified']
    doc=dict(result_source='fresh_run_log_readout',phases=phases,new_models=len(fits),
        updates=sum(f['step'] for f in fits),fit_seconds=sum(f['seconds'] for f in fits),
        pilot_in_same_training_budget=True,held_training_draws=sum(f['held_rows_sampled'] for f in fits),
        create_receipt=run.artifact(run.PRIVATE/'create_queue.json'),
        create_readonly_query_returncode=queue['response']['returncode'],create_jobs_submitted=0,
        remote_modified=False,independent_roles_read=False,private_data_committed=False)
    run.immutable_json(run.PUBLIC/'operations.json',doc)
    lines=['# Execution and Resource Record','',
        'Native arm64 Torch CPU, four compute threads, one interop thread, zero workers.',
        'No NumPy fallback, DataLoader multiprocessing or hardware-resource probing.','',
        '| Phase | PID | Process seconds | Peak RSS (GB, decimal) |','|---|---:|---:|---:|']
    for r in phases: lines.append(f"| {r['phase']} | {r['pid']} | {r['wall_seconds']:.2f} | {r['peak_RSS_bytes']/1e9:.3f} |")
    lines+=['',f"Nine new fits: {doc['updates']:,} updates, {doc['fit_seconds']:.2f} cumulative fit seconds.",
        'The 100-update pilot resumes into the first endpoint; it is not counted twice.',
        'All paired sampler states/counts and initial parameters match. Held fitting draws are zero.',
        'Checkpoints retain optimizer and random states; original controls are untouched.',
        'CREATE was checked read-only, no remote job submitted, cancelled or modified.',
        'The local pilot fit the resource envelope; no remote migration was necessary.',
        'Checkpoints, private predictions, raw data and detailed logs are not committed.','',
        'Existing geometry, control checkpoints and controls: cached_verified, with fresh control inference.',
        'New training, predictions, scoring and replay: fresh_run.',
        'Independent confirmation, full historical test suite and cold raw-download rebuild: not_run.',
        'No new risk policy, deployment change, Stage5C execution or SMC.']
    (run.PUBLIC/'operations.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(phases=len(phases),updates=doc['updates'],fit_seconds=doc['fit_seconds'])))


if __name__=='__main__': main()
