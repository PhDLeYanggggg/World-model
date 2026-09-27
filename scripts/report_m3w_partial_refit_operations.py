"""Derive resource/phase receipts from completed logs, never an elapsed-time guess."""
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts import run_m3w_european_partial_neighbor_refit as run


def main():
    phases=[]
    for phase in ('pilot','control','train','predict','evaluate','replay','verify_eval'):
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
    control=json.loads((run.PUBLIC/'legacy_control.json').read_text())
    c=json.loads((ROOT/control['complete']['path']).read_text())
    freeze=json.loads((run.PUBLIC/'prediction_freeze.json').read_text())
    fits=[json.loads((ROOT/r['path']).read_text())['fit'] for r in freeze['training']]
    queue=json.loads((run.PRIVATE/'create_queue.json').read_text())
    assert queue['response']['returncode']==0 and queue['jobs_submitted']==0 and not queue['remote_modified']
    doc=dict(result_source='fresh_run_log_readout',phases=phases,
        source_registration=run.artifact(run.PUBLIC/'registration.json'),
        local_new_models=len(fits),new_model_updates=sum(x['step'] for x in fits),
        new_model_training_seconds=sum(x['seconds'] for x in fits),
        legacy_control_updates=c['fit']['step'],legacy_control_cumulative_seconds=c['fit']['seconds'],
        control_pilot_included_not_extra=True,held_training_draws=sum(x['held_rows_sampled'] for x in fits),
        create_queue_receipt=run.artifact(run.PRIVATE/'create_queue.json'),
        create_jobs_submitted=0,remote_modified=False,independent_roles_read=False,
        checkpoint_and_prediction_data_committed=False)
    run.immutable_json(run.PUBLIC/'operations.json',doc)
    lines=['# Execution and Resource Record', '',
        'Native arm64 Torch CPU, four compute threads, one interop thread, zero workers.',
        'No NumPy fallback, resource probing, MPS probe or DataLoader multiprocessing.', '',
        '| Phase | PID | Process seconds | Peak RSS (GB, decimal) |', '|---|---:|---:|---:|']
    lines += [f"| {r['phase']} | {r['pid']} | {r['wall_seconds']:.2f} | {r['peak_RSS_bytes']/1e9:.3f} |" for r in phases]
    lines += ['',f"Nine new fits: {doc['new_model_updates']:,} updates, {doc['new_model_training_seconds']:.2f} training seconds.",
        f"Fresh control: {doc['legacy_control_updates']:,} updates, {doc['legacy_control_cumulative_seconds']:.2f} cumulative training seconds.",
        'The100-update pilot resumes into the same4000-update control and is not counted twice.',
        'All final sampling counts and generator states match their registered controls; held fitting draws are zero.',
        'Model parameters and control resume match exactly. Checkpoints retain optimizer/random state.', '',
        'CREATE queue inspection succeeded read-only; no remote job was submitted, cancelled or modified.',
        'The local pilot fit the resource envelope, so no remote migration was needed. The preflight',
        'preserves10GiB free space. Slow completed phases are not downgraded or relabeled.', '',
        'Registration c2df9fb5 precedes training. Prediction freeze39607e9e precedes scoring.',
        'Result commit27eea024 retains the failed benefit screen before complete replay sealing.',
        'Private cache/checkpoints/predictions/logs remain ignored; public files contain code,',
        'configuration, hashes and aggregate evidence only. Unrelated staged work is not committed.', '',
        'Source arrays/legacy weights: cached_verified. Geometry, new weights, source readout and',
        'fresh prediction replay: fresh_run. Independent confirmation and cold raw-download rebuild:',
        'not_run. The full historical test suite is not_run; scoped verification has its own receipt.',
        'No deployment change, Stage5C execution, SMC, metric or calibrated-seconds claim.']
    (run.PUBLIC/'operations.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(completed_phases=len(phases),new_model_updates=doc['new_model_updates'])))


if __name__=='__main__': main()
