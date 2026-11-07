"""Make a small identity-stripped, independently executed ledger evidence bundle."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.build_m3w_blinded_reproduction import archive_bytes, clean_run, scan_files
from src.evaluation import causal_ledger_portable as portable

HOME = ROOT/'outputs/publication_readiness_2026_09/moving_record_joint_actions_v1/causal_ledger'
PUBLIC = HOME/'credit_transport'; ARCHIVE = ROOT/'data/stage_cvpr2027_experiments/causal_ledger_reproduction_v1/ledger_evidence.zip'
PINNED = {
    'allocation': (HOME,'69f26a9e64221759aeec6ab770e277671f107f78cd6ea497005c54009319bd9c','risk'),
    'bank': (HOME/'physical_control','6bdd9e49ef4fd165e2a01cdf9044777ace010d1e7524fceee6c90f9188aa9b56','risks'),
    'credit': (PUBLIC,'5f7ece052db47d75f19546cd43a4dc126a232c18d325ebaab0fe7578b65137ff','risks'),
}


def digest(data): return hashlib.sha256(data).hexdigest()
def encode(x): return (json.dumps(x,indent=2,allow_nan=False)+'\n').encode()


def make_evidence():
    contrasts = {}; policies = {}; credit = {}
    for study,(root,sha,field) in PINNED.items():
        raw = (root/'analysis.json').read_bytes()
        replay = json.loads((root/'replay_execution.json').read_text())
        if digest(raw)!=sha or replay['artifact']['sha256']!=sha:
            raise ValueError('Require verified frozen scalars')
        a = json.loads(raw); records = a['records']
        if len(records)!=102 or a['independent_confirmation']:
            raise ValueError('Only complete exposed development studies are exported')
        sites = sorted({r['source'] for r in records}); seeds = sorted({r['seed'] for r in records})
        groups = [[r for r in records if r['source']==s and r['seed']==k] for s in sites for k in seeds]
        for key,c in a['comparisons'].items():
            v = c['completion_lower_pp']; rows = [[sum(r['comparisons'][key][m] for r in g) for m in
                ('reference','known_delta','unknown_radius')] for g in groups]
            contrasts[study+'/'+key] = dict(source_seed_masses=[rows[i:i+3] for i in range(0,len(rows),3)],
                archived=[v['point'],*v['interval95']])
        for key,v in a[field].items():
            rows = [r['metrics'][key]['risk'] for r in records]
            policies[study+'/'+key] = dict(recording_views=[[r[k] for k in ('selected','completion_all_risk_upper',
                'completion_easy_risk_upper','known_easy_net_degradation')] for r in rows],
                archived={k:v[k] for k in ('switches','passes','undefined_easy','easy_over_2')})
        if study=='credit':
            for key,v in a['credit_totals'].items():
                credit[key] = dict(recording_views=[r[key] if isinstance(r[key],list) else [r[key]] for r in a['credit_audit']],
                                   archived=v if isinstance(v,list) else [v])
    return dict(scope='exposed_source_development_aggregate',budget=.02,contrasts=contrasts,policies=policies,
        credit=credit,bootstrap=np.random.default_rng(73107).integers(14,size=(3000,14)).tolist())


def make_files():
    data = make_evidence(); result = portable.reconstruct(data)
    outputs = {k:v.encode() for k,v in portable.outputs(result).items()}
    files = {'reproduce.py':Path(portable.__file__).read_bytes(),
        'README.md':(ROOT/'reproducibility/causal_ledger_evidence/README.md').read_bytes(),
        'evidence.json':(json.dumps(data,separators=(',',':'),allow_nan=False)+'\n').encode(),
        **{'expected/'+k:v for k,v in outputs.items()}}
    files['manifest.json'] = encode(dict(files={k:dict(sha256=digest(v),bytes=len(v)) for k,v in files.items()},
        expected_outputs={k:digest(v) for k,v in outputs.items()},independent_confirmation=False,
        full_training_reproduction=False,anonymity_certified=False))
    scan_files(files)
    return files, result


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--verify',action='store_true'); args = p.parse_args()
    files,result = make_files(); raw = archive_bytes(files)
    if args.verify:
        if ARCHIVE.read_bytes()!=raw:
            raise ValueError('Ledger archive differs from frozen source evidence')
    else:
        ARCHIVE.parent.mkdir(parents=True,exist_ok=True)
        with ARCHIVE.open('xb') as f: f.write(raw)
    receipt = dict(archive_sha256=digest(raw),archive_bytes=len(raw),members=len(files),
        comparisons=len(result['comparisons']),policies=len(result['risk']),credit_components=len(result['credit']),
        clean_run=clean_run(ARCHIVE),full_training_reproduction=False,independent_confirmation=False,submission_ready=False)
    p = PUBLIC/'portable_verification.json'; text = encode(receipt)
    if args.verify:
        if p.read_bytes()!=text:
            raise ValueError('Clean replay differs from stored receipt')
    else:
        with p.open('xb') as f: f.write(text)
    print(json.dumps(dict(bytes=len(raw),members=len(files),exit_code=receipt['clean_run']['exit_code'],
        comparisons=receipt['comparisons'],policies=receipt['policies'])))


if __name__=='__main__':
    main()
