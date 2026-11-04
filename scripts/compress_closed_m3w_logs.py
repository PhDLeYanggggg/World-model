"""Preserve bytes while compressing completed, closed SOURCE operational logs."""
import hashlib
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
PRIVATE=ROOT/'data/stage_cvpr2027_experiments'
PUBLIC=ROOT/'outputs/publication_readiness_2026_09'
OUTPUT=PUBLIC/'moving_harm_refinement_v1/closed_log_compression.json'


def sha(path):
    with path.open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verified-phases',action='store_true')
    parser.add_argument('--inactive-old-logs',action='store_true');args=parser.parse_args()
    output=OUTPUT if not args.verified_phases else OUTPUT.with_name('closed_verified_log_compression.json')
    if args.inactive_old_logs:output=OUTPUT.with_name('inactive_old_log_compression.json')
    if output.exists():raise ValueError('Completed storage receipt is immutable')
    paths=[];proofs={}
    for directory in PRIVATE.glob('european_*'):
        heartbeat=directory/'heartbeat.json';readout=PUBLIC/directory.name/'readout.md'
        if not heartbeat.is_file()or(not args.inactive_old_logs and not readout.is_file()):continue
        if args.inactive_old_logs and (not (PUBLIC/directory.name).is_dir()or heartbeat.stat().st_mtime>time.time()-86400):continue
        meta=json.loads(heartbeat.read_text())
        state=meta.get('state','')
        complete=state=='complete'
        if args.verified_phases:
            receipt=PUBLIC/directory.name/'verification.json'
            complete|=(meta.get('phase')=='verify'and state.startswith('all')and state.endswith('_verified')
                and not any(word in state for word in ('not_','fail','partial','running'))and receipt.is_file())
        if args.inactive_old_logs:complete=True  # Inactive logs, not an assertion that the old experiment passed.
        if not complete:continue
        try:os.kill(int(meta['pid']),0)
        except ProcessLookupError:pass
        else:continue
        proofs[directory.name]=dict(heartbeat_sha256=sha(heartbeat),observed_inactive_pid=meta['pid'],
            readout_sha256=sha(readout)if readout.is_file()else None,last_state=state,
            old_experiment_completion_claimed=not args.inactive_old_logs)
        for path in directory.iterdir():
            if (path.is_file()and not path.is_symlink()and path.resolve().is_relative_to(PRIVATE.resolve())
                    and (path.suffix=='.log'or path.name=='events.jsonl')):
                stat=path.stat()
                if args.inactive_old_logs and stat.st_mtime>time.time()-86400:continue
                if stat.st_nlink==1 and stat.st_size>=65536 and not(stat.st_flags&32):paths.append(path)
    if paths:
        opened=subprocess.run(['/usr/sbin/lsof','-nP','-F','n','--',*[str(p)for p in paths]],
            capture_output=True,text=True)
        if opened.returncode not in (0,1):raise RuntimeError('Cannot verify whether operational logs are open')
        active={line[1:]for line in opened.stdout.splitlines()if line.startswith('n/')}
        paths=[p for p in paths if str(p)not in active]
    rows=[];start=time.monotonic()
    for path in sorted(paths,key=lambda p:p.stat().st_size,reverse=True):
        before=path.stat();content=sha(path);temporary=path.with_name(path.name+'.m3w-compress-tmp')
        if temporary.exists():raise FileExistsError('Preserve existing temporary file for inspection')
        subprocess.run(['/usr/bin/ditto','--hfsCompression','--noclone',str(path),str(temporary)],check=True)
        copied=temporary.stat()
        if (sha(temporary)!=content or sha(path)!=content or path.stat().st_mtime_ns!=before.st_mtime_ns
                or copied.st_mode!=before.st_mode or copied.st_mtime_ns!=before.st_mtime_ns):
            raise ValueError('Original preserved: closed-log byte or metadata verification failed')
        if copied.st_blocks>=before.st_blocks or not(copied.st_flags&32):temporary.unlink();continue
        os.replace(temporary,path)
        if sha(path)!=content:raise ValueError('Post-compression log hash changed')
        rows.append(dict(path=str(path.relative_to(ROOT)),sha256=content,
            saved_file_blocks_bytes=(before.st_blocks-path.stat().st_blocks)*512,
            closure_proof=proofs[path.parent.name],mode_and_mtime_preserved=True))
    for row in rows:
        if sha(ROOT/row['path'])!=row['sha256']:raise ValueError('Separate closed-log hash replay failed')
    result=dict(pid=os.getpid(),seconds=time.monotonic()-start,files=rows,
        saved_file_blocks_bytes=sum(r['saved_file_blocks_bytes']for r in rows),
        all_content_hashes_separately_rechecked=bool(rows),raw_cache_checkpoint_or_unrelated_files_touched=False,
        completed_related_operational_log_replicas_only=not args.inactive_old_logs,
        inactive_related_operational_log_replicas_only=args.inactive_old_logs,scientific_bytes_changed=False)
    with output.open('x')as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps({k:v for k,v in result.items()if k!='files'}),flush=True)


if __name__=='__main__':main()
