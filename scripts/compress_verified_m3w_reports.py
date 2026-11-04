"""APFS/HFS transparent compression of unchanged tracked public JSON reports."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time
from zipfile import ZipFile

ROOT=Path(__file__).resolve().parents[1]
PUBLIC=ROOT/'outputs/publication_readiness_2026_09'
OUTPUT=PUBLIC/'moving_harm_refinement_v1/filesystem_compression.json'
ALLOWED=('european_forecast_amplitude_v1','european_dimensionless_intervention_v1',
    'european_regime_transport_v1','european_coupled_local_cost_v1',
    'european_budget_score_cost_v1','european_detector_cost_v1',
    'european_selected_cost_audit_v1','european_multiaction_forest_v1',
    'european_balanced_region_cost_v1','european_policy_region_cost_v1')


def sha(path):
    with path.open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def candidates():
    raw=subprocess.run(['git','ls-files','-z','--',*[str((PUBLIC/name).relative_to(ROOT))for name in ALLOWED]],
        cwd=ROOT,capture_output=True,check=True).stdout
    paths=[]
    for name in raw.split(b'\0'):
        if not name:continue
        path=ROOT/os.fsdecode(name)
        if path.suffix!='.json'or path.is_symlink():continue
        stat=path.stat()
        if stat.st_size>=1024**2 and stat.st_nlink==1 and not(stat.st_flags&32):paths.append(path)
    return sorted(paths,key=lambda p:p.stat().st_size,reverse=True)


def head_hash(path,revision='HEAD'):
    rel=str(path.relative_to(ROOT))
    for extra in ([],['--cached']):
        subprocess.run(['git','diff',*extra,'--quiet','--',rel],cwd=ROOT,check=True)
    head=subprocess.run(['git','show',revision+':'+rel],cwd=ROOT,capture_output=True,check=True).stdout
    expected=hashlib.sha256(head).hexdigest()
    if sha(path)!=expected:raise ValueError('Only unchanged HEAD reports are permitted')
    return expected


def archive_candidates(all_european=False,minimum_bytes=65536):
    rows=[]
    if all_european:
        raw=subprocess.run(['git','ls-files','-z','--','outputs/publication_readiness_2026_09/european_*/metric_groups.zip'],
            cwd=ROOT,capture_output=True,check=True).stdout
        archives=[ROOT/os.fsdecode(name)for name in raw.split(b'\0')if name]
    else:archives=[PUBLIC/name/'metric_groups.zip'for name in ALLOWED]
    for archive in archives:
        if not archive.exists():continue
        archive_sha=head_hash(archive)
        with ZipFile(archive)as zipped:
            for info in zipped.infolist():
                member=Path(info.filename)
                if member.is_absolute()or '..'in member.parts or member.suffix!='.json':continue
                path=archive.parent/member
                if not path.exists()or path.is_symlink()or not path.resolve().is_relative_to(PUBLIC.resolve()):continue
                stat=path.stat()
                if stat.st_size>=minimum_bytes and stat.st_nlink==1 and not(stat.st_flags&32):
                    rows.append((path,dict(path=str(archive.relative_to(ROOT)),sha256=archive_sha,member=info.filename)))
    return sorted(rows,key=lambda item:item[0].stat().st_size,reverse=True)


def compress(path,archive=None):
    rel=str(path.relative_to(ROOT));before=path.stat();original=sha(path)
    if archive is None:
        expected=head_hash(path)
    else:
        archive_path=ROOT/archive['path']
        if sha(archive_path)!=archive['sha256']:raise ValueError('Committed metric archive changed')
        with ZipFile(archive_path)as zipped:expected=hashlib.sha256(zipped.read(archive['member'])).hexdigest()
    if original!=expected:raise ValueError('Local report differs from its immutable tracked proof')
    temporary=path.with_name(path.name+'.m3w-fscompress-tmp')
    if temporary.exists():raise FileExistsError('Preserve and inspect the existing compression temporary file')
    subprocess.run(['/usr/bin/ditto','--hfsCompression','--noclone',str(path),str(temporary)],check=True)
    copied=temporary.stat()
    if sha(temporary)!=original or sha(path)!=original:
        raise ValueError('Original preserved: copied bytes do not match the original and HEAD')
    if copied.st_blocks>=before.st_blocks or not(copied.st_flags&32):
        temporary.unlink()
        return dict(path=rel,sha256=original,status='no_physical_saving_original_unchanged',saved_bytes=0,
            archive=archive,proof='direct_HEAD_file'if archive is None else'byte_exact_member_of_unchanged_HEAD_archive')
    if (copied.st_mode,copied.st_mtime_ns)!=(before.st_mode,before.st_mtime_ns):
        raise ValueError('Original preserved: metadata not copied exactly')
    os.replace(temporary,path)
    if sha(path)!=original:raise ValueError('Post-replacement content hash changed')
    after=path.stat()
    return dict(path=rel,sha256=original,status='transparent_compression_bytes_exact_to_HEAD',
        proof='direct_HEAD_file'if archive is None else'byte_exact_member_of_unchanged_HEAD_archive',
        archive=archive,
        logical_bytes=before.st_size,physical_bytes_before=before.st_blocks*512,
        physical_bytes_after=after.st_blocks*512,saved_bytes=(before.st_blocks-after.st_blocks)*512,
        mode_and_mtime_preserved=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-files',type=int,default=100)
    parser.add_argument('--target-free-bytes',type=int,default=10737418240)
    parser.add_argument('--archive-bound',action='store_true')
    parser.add_argument('--all-european-archives',action='store_true')
    parser.add_argument('--small-archive-reports',action='store_true')
    parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    if platform.system()!='Darwin':raise RuntimeError('macOS filesystem compression only')
    output=OUTPUT if not args.archive_bound else OUTPUT.with_name('filesystem_archive_compression.json')
    if args.all_european_archives:
        args.archive_bound=True;output=OUTPUT.with_name('filesystem_all_european_archive_compression.json')
    if args.small_archive_reports:
        args.archive_bound=True;args.all_european_archives=True
        output=OUTPUT.with_name('filesystem_small_archive_compression.json')
    if args.verify:
        receipt=json.loads(output.read_text())
        archives={}
        for row in receipt['files']:
            path=ROOT/row['path']
            if sha(path)!=row['sha256']:raise ValueError('Compressed report content changed')
            proof=row.get('archive')
            if proof is None and receipt.get('archive_verified_local_metric_replicas_compressed'):
                # Earlier no-saving receipts omitted the archive field; recover it
                # from the original committed revision, never from a new outcome.
                for parent in path.parents:
                    if parent==PUBLIC.parent:break
                    archive=parent/'metric_groups.zip'
                    if archive.is_file():
                        key=str(archive.relative_to(ROOT))
                        if key not in archives:archives[key]=head_hash(archive,receipt['head_verified_against'])
                        proof=dict(path=key,sha256=archives[key],member=str(path.relative_to(parent)))
                        break
                if proof is None:raise ValueError('Cannot recover original committed archive proof')
            if proof:
                archive=ROOT/proof['path']
                if proof['path']not in archives:archives[proof['path']]=head_hash(archive)
                if archives[proof['path']]!=proof['sha256']:raise ValueError('Archive changed')
                with ZipFile(archive)as zipped:
                    if hashlib.sha256(zipped.read(proof['member'])).hexdigest()!=row['sha256']:
                        raise ValueError('Archived report mismatch')
            elif head_hash(path)!=row['sha256']:raise ValueError('HEAD report changed')
        for path,value in archives.items():
            if sha(ROOT/path)!=value:raise ValueError('Archive changed during verification')
        print(json.dumps(dict(all_content_hashes_exact=True,files=len(receipt['files']),
            saved_physical_bytes=receipt['saved_physical_bytes'])));return
    if output.exists():raise ValueError('Completed compression receipt is immutable')
    start=time.monotonic();free=shutil.disk_usage(ROOT).free;rows=[]
    head=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True,check=True).stdout.strip()
    choices=archive_candidates(args.all_european_archives,4096 if args.small_archive_reports else 65536)if args.archive_bound else[(p,None)for p in candidates()]
    for path,archive in choices[:args.max_files]:
        if shutil.disk_usage(ROOT).free>=args.target_free_bytes:break
        row=compress(path,archive);rows.append(row);print(json.dumps(row),flush=True)
    result=dict(pid=os.getpid(),seconds=time.monotonic()-start,head_verified_against=head,
        free_bytes_before=free,free_bytes_after=shutil.disk_usage(ROOT).free,
        saved_physical_bytes=sum(r['saved_bytes']for r in rows),files=rows,
        raw_cache_checkpoint_or_unrelated_untracked_files_touched=False,
        archive_verified_local_metric_replicas_compressed=args.archive_bound,scientific_bytes_changed=False)
    with output.open('x')as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps({k:v for k,v in result.items()if k!='files'}),flush=True)


if __name__=='__main__':main()
