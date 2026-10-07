"""Standalone revision-four aggregate replay; no training or row-level evaluation."""
import argparse
import inspect
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

from scripts import build_m3w_aggregate_replay_package as prior_package
from scripts import build_m3w_evidence_manuscript_v4 as manuscript

ROOT = manuscript.ROOT
PUBLIC = ROOT/manuscript.BASE/'aggregate_replay_v2'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/aggregate_replay_v2'
SOURCES = {**manuscript.previous.SOURCES, **manuscript.SOURCES}
INPUTS = {key: f'inputs/{i:02d}'+Path(path).suffix
          for i, (key, (path, _)) in enumerate(SOURCES.items(), 1)}
OUTPUTS = ('evidence.json', 'tables.md', 'temporal_contrasts.csv')
ALLOWED = {'README.md', 'reproduce.py', 'MANIFEST.json', *INPUTS.values(),
           *('expected/'+name for name in OUTPUTS)}
sha, screen = prior_package.sha, prior_package.screen
README = '''# Completed Development Evidence: Aggregate Replay

This frozen revision-four package rebuilds six tables and eighteen temporal
contrast/metric rows from fourteen SHA256-pinned public aggregate sources.
Earlier SDD comparisons, European development comparisons and TRAIN diagnostics
remain separate. Temporal's lower prediction error does not overcome its
negative comparisons with matched neural controls or selected-risk failures.
Undefined selected-cohort comparisons and incomplete support are preserved.

Python 3.10 or newer, standard library only:

    python reproduce.py --check
    python reproduce.py

The first command checks all included members and compares rebuilt artifacts
byte for byte. The second also writes them to replayed/. No network, repository,
cluster account, trajectories, checkpoint files or numerical libraries are used.
The inner prior_evidence field is an older frozen snapshot, not current training
status; the revision-four top-level records describe completed temporal training.
The successor easy-harm-deviance experiment is not included as a scientific result.

The export process verified linked public training/readout receipts in the source
repository. Those 216 fit and 72 readout receipts are NOT included here and this
standalone script does NOT reverify them. It verifies only included aggregate
files, arithmetic and table rendering. It does not verify raw labels, checkpoint
contents, split independence, model training or population safety. The stored
confidence intervals are not recomputed. This is not a new experiment, bootstrap,
independent confirmation or complete experimental reproduction.

The package is mechanically screened for known account identifiers, home paths,
email addresses and repository URLs. Original public source hashes remain
traceable, so this is identity-minimized preparation, not an anonymity guarantee.
The paper remains not submission-ready. Units remain pixel/raw-frame or stated
normalized score units; labels are not human gold. No metric, seconds-level,
true-3D, foundation, latent-generation or particle-sampling claim is made.
'''


def standalone():
    header = ('import argparse\nimport csv\nimport hashlib\nimport io\nimport json\n'
              'import math\nfrom pathlib import Path\nfrom types import SimpleNamespace\n')
    old = manuscript.previous
    old_code = header+'BASE=Path('+repr(str(manuscript.BASE))+')\n'
    for key in ('SOURCES', 'CONTRASTS', 'SDD_CONTRASTS'):
        old_code += key+'='+repr(getattr(old, key))+'\n'
    old_code += '\n\n'.join(inspect.getsource(getattr(old, name))
                            for name in ('pointer', 'require', 'build', 'table_text'))
    text = header+'prior_scope={}\nexec('+repr(old_code)+', prior_scope)\n'
    text += 'previous=SimpleNamespace(**prior_scope)\nrequire=previous.require\n'
    for key, value in dict(SOURCES=manuscript.SOURCES, ALL_SOURCES=SOURCES, INPUTS=INPUTS,
                           OUTPUTS=OUTPUTS, ALLOWED=sorted(ALLOWED)).items():
        text += key+'='+repr(value)+'\n'
    text += 'BASE=Path('+repr(str(manuscript.BASE))+')\n'
    text += '\n\n'.join(inspect.getsource(getattr(manuscript, name))
                         for name in ('contrast', 'build', 'interval', 'table_text', 'artifacts'))
    text += r'''
def main():
    parser=argparse.ArgumentParser(description='Aggregate replay only, no experiment rerun')
    parser.add_argument('--check',action='store_true');args=parser.parse_args()
    root=Path(__file__).resolve().parent
    manifest=json.loads((root/'MANIFEST.json').read_text())
    require(set(manifest['members'])==set(ALLOWED)-{'MANIFEST.json'},'Manifest allowlist changed')
    for name,ref in manifest['members'].items():
        p=root/name
        require(not Path(name).is_absolute() and '..' not in Path(name).parts,'Unsafe member path')
        require(p.resolve().is_relative_to(root),'Member outside archive')
        raw=p.read_bytes()
        require(len(raw)==ref['bytes'] and hashlib.sha256(raw).hexdigest()==ref['sha256'],
                'Archive member changed: '+name)
    docs={}
    for key,name in INPUTS.items():
        raw=(root/name).read_bytes()
        require(hashlib.sha256(raw).hexdigest()==ALL_SOURCES[key][1],'Pinned source changed')
        docs[key]=json.loads(raw) if name.endswith('.json') else raw.decode()
    old={k:docs[k] for k in previous.SOURCES}
    current={k:docs[k] for k in SOURCES}
    template='\n'.join('{{'+key+'}}' for key in table_text(build(old,current)))
    content=artifacts(old,current,template)
    for name in OUTPUTS:
        raw=content[name].encode()
        require(raw==(root/'expected'/name).read_bytes(),'Aggregate mismatch: '+name)
        if not args.check:
            dest=root/'replayed'/name;dest.parent.mkdir(exist_ok=True);dest.write_bytes(raw)
    print(json.dumps(dict(status='completed_development_aggregate_replay_verified',
        sources=14,tables=6,temporal_contrast_metric_rows=18,
        exact_outputs=list(OUTPUTS),training_receipts_reverified=False,
        new_training=False,new_inference=False,new_bootstrap=False,
        independent_confirmation=False,submission_ready=False)))

if __name__=='__main__':main()
'''
    return text.encode()


def members():
    docs = manuscript.load_sources()
    template = (ROOT/manuscript.OUTPUT/'manuscript.template.md').read_text()
    content = manuscript.artifacts(*docs, template)
    items = {'README.md': README.encode(), 'reproduce.py': standalone()}
    for key, (path, digest) in SOURCES.items():
        raw = (ROOT/manuscript.BASE/path).read_bytes()
        if sha(raw) != digest:
            raise ValueError('Source changed: '+key)
        items[INPUTS[key]] = raw
    for name in OUTPUTS:
        raw = content[name].encode()
        if raw != (ROOT/manuscript.OUTPUT/name).read_bytes():
            raise ValueError('Saved manuscript aggregate changed: '+name)
        items['expected/'+name] = raw
    screen(items)
    manifest = dict(schema=2,scope='aggregate_replay_only',
        members={k:dict(bytes=len(v),sha256=sha(v)) for k,v in sorted(items.items())},
        no_raw_rows=True,no_checkpoints=True,new_training=False,new_bootstrap=False,
        independent_confirmation=False,venue_anonymity_certified=False,submission_ready=False)
    items['MANIFEST.json'] = (json.dumps(manifest,indent=2,allow_nan=False)+'\n').encode()
    return items


def archive(items):
    if set(items) != ALLOWED:
        raise ValueError('Archive allowlist mismatch')
    screen(items)
    stream = io.BytesIO()
    with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as zipped:
        for name,raw in sorted(items.items()):
            info = zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0))
            info.create_system=3;info.external_attr=0o100644<<16
            info.compress_type=zipfile.ZIP_DEFLATED
            zipped.writestr(info,raw,compresslevel=9)
    return stream.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true');args=parser.parse_args()
    items=members();raw=archive(items)
    with tempfile.TemporaryDirectory(prefix='m3w-aggregate-v2-') as temp:
        root=Path(temp)
        with zipfile.ZipFile(io.BytesIO(raw)) as zipped:
            for name in zipped.namelist():
                path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(zipped.read(name))
        for check in (True,False):
            command=[sys.executable,'-I',str(root/'reproduce.py')]+(['--check'] if check else [])
            run=subprocess.run(command,cwd=root,capture_output=True,text=True,timeout=20)
            if run.returncode:
                raise RuntimeError(run.stderr)
            result=json.loads(run.stdout)
        assert all((root/'replayed'/name).read_bytes()==items['expected/'+name] for name in OUTPUTS)
    target=PRIVATE/'review_bundle.zip'
    receipt=dict(result_source='fresh_run_isolated_aggregate_replay_of_cached_verified_statistics',
        archive=str(target.relative_to(ROOT)),archive_bytes=len(raw),archive_sha256=sha(raw),
        members={k:dict(bytes=len(v),sha256=sha(v)) for k,v in sorted(items.items())},
        source_evidence_sha256=sha((ROOT/manuscript.OUTPUT/'evidence.json').read_bytes()),
        linked_fit_receipts_verified_during_export=216,linked_readout_receipts_verified_during_export=72,
        isolated_python=True,python_version=sys.version.split()[0],dependencies='standard_library_only',
        exact_output_bytes=True,known_identifiers_scan_passed=True,
        full_experimental_reproducibility=False,venue_anonymity_certified=False,**result)
    public_receipt=PUBLIC/'verification.json'
    if args.check:
        assert target.read_bytes()==raw
        assert json.loads(public_receipt.read_text())==receipt
    else:
        PRIVATE.mkdir(parents=True,exist_ok=True);PUBLIC.mkdir(parents=True,exist_ok=True)
        target.write_bytes(raw)
        public_receipt.write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k!='members'}))


if __name__=='__main__':main()
