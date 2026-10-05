"""Export and execute a standalone, identity-minimized aggregate replay artifact."""
import argparse
import hashlib
import inspect
import io
import json
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from scripts import build_m3w_evidence_manuscript_v3 as original

ROOT=original.ROOT
PUBLIC=ROOT/original.BASE/'aggregate_replay_v1'
PRIVATE=ROOT/'data/stage_cvpr2027_experiments/aggregate_replay_v1'
OUTPUTS=('evidence.json','tables.md','european_contrasts.csv')
INPUTS={key:f'inputs/{i:02d}'+Path(path).suffix
        for i,(key,(path,_)) in enumerate(original.SOURCES.items(),1)}
ALLOWED={'README.md','reproduce.py','MANIFEST.json',*INPUTS.values(),
         *('expected/'+name for name in OUTPUTS)}
FORBIDDEN=(rb'/users/',rb'/home/',rb'/cephfs/',rb'yangyue',rb'k24101830',
           rb'phdleyang',rb'github\.com/',rb'[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}')

README='''# Aggregate Evidence Replay

This package rebuilds three tables and eleven European contrast rows from eight
SHA256-pinned, already public aggregate/protocol sources. Four SDD comparisons
remain a separate population. It is a frozen evidence snapshot, not a live job
status report. The failed primary tests, risk violations, unknown outcomes and
eleven-locality selected-cohort support are deliberately retained.

Run with Python 3.10 or newer; only its standard library is used:

    python reproduce.py --check
    python reproduce.py

The first command verifies every archive member and compares regenerated results
byte for byte with expected files. The second additionally writes the same three
files into replayed/. No network, repository checkout, accelerator, cluster
account, trajectory file or model checkpoint is required.

Scope: aggregate arithmetic and table reproduction only. The intervals are
copied from the verified source studies, not recomputed here. This cannot verify
raw labels, training, checkpoint contents, split independence or calibration.
It performs no training, prediction evaluation, new bootstrap or hypothesis test.
No new empirical finding, deployment promotion or submission readiness follows.

Units remain image-pixel/raw-frame or their explicitly normalized score scales;
detector/inferred labels are not human gold. No metric, seconds-level, true-3D or
foundation-model claim is made. Latent generation and particle sampling are off.

Known personal/account identifiers, absolute home paths and repository URLs are
excluded by a mechanical scan. Scientific source text and aggregate provenance
are retained unchanged, so public-source linkage may remain possible. This is
identity-minimized preparation, NOT a guarantee of venue-compliant anonymity.
It is not a complete experimental reproduction or a ready submission artifact.
'''


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def screen(items):
    for name,raw in items.items():
        for pattern in FORBIDDEN:
            if re.search(pattern,raw,re.IGNORECASE):
                raise ValueError('Identifier in archive member: '+name)


def standalone():
    functions='\n\n'.join(inspect.getsource(getattr(original,name))
                            for name in ('pointer','require','build','table_text','artifacts'))
    header='''"""Standalone aggregate arithmetic and exact table replay; no experimental rerun."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

'''
    header+='BASE=Path('+repr(str(original.BASE))+')\n'
    for key,value in dict(SOURCES=original.SOURCES,CONTRASTS=original.CONTRASTS,
                          SDD_CONTRASTS=original.SDD_CONTRASTS,INPUTS=INPUTS,OUTPUTS=OUTPUTS).items():
        header+=key+'='+repr(value)+'\n'
    main=r'''
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true');args=parser.parse_args()
    root=Path(__file__).resolve().parent
    manifest=json.loads((root/'MANIFEST.json').read_text())
    for name,ref in manifest['members'].items():
        p=root/name
        require(not Path(name).is_absolute() and '..' not in Path(name).parts,
                'Unsafe member path')
        require(p.resolve().is_relative_to(root),'Member outside archive')
        raw=p.read_bytes()
        require(len(raw)==ref['bytes'] and hashlib.sha256(raw).hexdigest()==ref['sha256'],
                'Archive member changed: '+name)
    docs={}
    for key,name in INPUTS.items():
        raw=(root/name).read_bytes()
        require(hashlib.sha256(raw).hexdigest()==SOURCES[key][1],'Pinned source changed')
        docs[key]=json.loads(raw) if name.endswith('.json') else raw.decode()
    content=artifacts(docs,'{{SDD_TABLE}}\n{{EU_TABLE}}\n{{POLICY_TABLE}}')
    for name in OUTPUTS:
        raw=content[name].encode()
        require(raw==(root/'expected'/name).read_bytes(),'Aggregate mismatch: '+name)
        if not args.check:
            p=root/'replayed'/name;p.parent.mkdir(exist_ok=True);p.write_bytes(raw)
    print(json.dumps(dict(status='aggregate_replay_verified',pinned_sources=8,tables=3,
        european_contrasts=11,sdd_contrasts=4,exact_outputs=list(OUTPUTS),new_training=False,
        new_forecast_evaluation=False,new_bootstrap=False,independent_confirmation=False,
        submission_ready=False)))

if __name__=='__main__':main()
'''
    return (header+functions+main).encode()


def members():
    docs=original.load_sources()
    expected=original.artifacts(docs,(ROOT/original.OUTPUT/'manuscript.template.md').read_text())
    items={'README.md':README.encode(),'reproduce.py':standalone()}
    for key,(name,digest) in original.SOURCES.items():
        raw=(ROOT/original.BASE/name).read_bytes();assert sha(raw)==digest
        items[INPUTS[key]]=raw
    for name in OUTPUTS:
        raw=expected[name].encode()
        if raw!=(ROOT/original.OUTPUT/name).read_bytes():raise ValueError('Saved aggregate changed: '+name)
        items['expected/'+name]=raw
    screen(items)
    manifest=dict(schema=1,scope='aggregate_replay_only',members={name:dict(bytes=len(raw),sha256=sha(raw))
        for name,raw in sorted(items.items())},no_raw_rows=True,no_checkpoints=True,
        numerical_source='cached_verified',new_training=False,new_bootstrap=False,
        independent_confirmation=False,venue_anonymity_certified=False,submission_ready=False)
    items['MANIFEST.json']=(json.dumps(manifest,indent=2,allow_nan=False)+'\n').encode()
    return items


def archive(items):
    if set(items)!=ALLOWED:raise ValueError('Archive allowlist mismatch')
    screen(items);out=io.BytesIO()
    with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name,raw in sorted(items.items()):
            info=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0));info.create_system=3
            info.external_attr=0o100644<<16;info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,raw,compresslevel=9)
    return out.getvalue()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');args=p.parse_args()
    items=members();raw=archive(items)
    archive_path=PRIVATE/'review_bundle.zip';receipt_path=PUBLIC/'verification.json'
    with tempfile.TemporaryDirectory(prefix='aggregate-replay-') as temp:
        root=Path(temp)
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            assert set(z.namelist())==ALLOWED
            for name in z.namelist():
                dest=root/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(z.read(name))
        for check in (True,False):
            command=[sys.executable,'-I',str(root/'reproduce.py')]+(['--check'] if check else [])
            run=subprocess.run(command,cwd=root,capture_output=True,text=True,timeout=20)
            if run.returncode:raise RuntimeError(run.stderr)
            result=json.loads(run.stdout)
        assert all((root/'replayed'/n).read_bytes()==items['expected/'+n] for n in OUTPUTS)
    receipt=dict(result_source='fresh_run_isolated_aggregate_replay_of_cached_verified_statistics',
        archive=str(archive_path.relative_to(ROOT)),archive_bytes=len(raw),archive_sha256=sha(raw),
        members={n:dict(bytes=len(v),sha256=sha(v)) for n,v in sorted(items.items())},
        source_manifest_sha256=sha((ROOT/original.OUTPUT/'evidence.json').read_bytes()),
        isolated_python=True,python_version=sys.version.split()[0],dependencies='standard_library_only',
        exact_output_bytes=True,known_identifiers_scan_passed=True,
        full_experimental_reproducibility=False,venue_anonymity_certified=False,**result)
    if args.check:
        assert archive_path.read_bytes()==raw
        assert json.loads(receipt_path.read_text())==receipt
    else:
        PRIVATE.mkdir(parents=True,exist_ok=True);PUBLIC.mkdir(parents=True,exist_ok=True)
        archive_path.write_bytes(raw)
        receipt_path.write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k!='members'}))


if __name__=='__main__':main()
