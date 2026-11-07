"""Export and verify the identity-stripped calibration reporting bundle."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.build_m3w_blinded_reproduction import archive_bytes,clean_run,scan_files
from scripts.build_m3w_ledger_reproduction import digest,encode
from src.evaluation import causal_ledger_portable as portable

HOME = ROOT/'outputs/publication_readiness_2026_09/moving_record_joint_actions_v1/causal_ledger/veto_refit/guard'
ARCHIVE = ROOT/'data/stage_cvpr2027_experiments/calibration_reproduction_v1/calibration_evidence.zip'
RECEIPT = HOME/'size'/'portable_verification.json'
STUDIES = {'guard':HOME,'context':HOME/'context','size':HOME/'size'}


def evidence():
    contrasts,policies,sources = {},{},{}
    for name,folder in STUDIES.items():
        raw = (folder/'analysis.json').read_bytes()
        h = digest(raw)
        for phase in ('score','replay'):
            receipt = json.loads((folder/f'{phase}_execution.json').read_text())
            if receipt['artifact']['sha256']!=h:
                raise ValueError('Require an exact frozen scoring and replay receipt')
        a = json.loads(raw)
        r = a['records']
        if len(r)!=102 or a['independent_confirmation'] or a['deployment_changed']:
            raise ValueError('Complete development studies only')
        site = sorted({x['source'] for x in r})
        seed = sorted({x['seed'] for x in r})
        grouped = [[[x for x in r if x['source']==s and x['seed']==k] for k in seed] for s in site]
        for key,v in a['comparisons'].items():
            masses = [[[sum(x['comparisons'][key][m] for x in g) for m in
                ('reference','known_delta','unknown_radius')] for g in group] for group in grouped]
            s = v['completion_lower_pp']
            if s is not None:
                contrasts[f'{name}/{key}'] = dict(source_seed_masses=masses,archived=[s['point'],*s['interval95']])
        for key,v in a['risk'].items():
            rows = [x['metrics'][key]['risk'] for x in r]
            policies[f'{name}/{key}'] = dict(recording_views=[[x[k] for k in
                ('selected','completion_all_risk_upper','completion_easy_risk_upper',
                 'known_easy_net_degradation')] for x in rows],
                archived={k:v[k] for k in ('switches','passes','undefined_easy','easy_over_2')})
        sources[name] = h
    out = dict(scope='exposed_source_development_aggregate',budget=.02,contrasts=contrasts,policies=policies,
        credit={},bootstrap=np.random.default_rng(73107).integers(14,size=(3000,14)).tolist())
    return out,sources


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verify',action='store_true')
    args = p.parse_args()
    data,source = evidence()
    result = portable.reconstruct(data)
    outputs = {k:v.encode() for k,v in portable.outputs(result).items()}
    files = {'reproduce.py':Path(portable.__file__).read_bytes(),
        'README.md':(ROOT/'reproducibility/calibration_evidence/README.md').read_bytes(),
        'evidence.json':(json.dumps(data,separators=(',',':'),allow_nan=False)+'\n').encode(),
        **{'expected/'+k:v for k,v in outputs.items()}}
    files['manifest.json'] = encode(dict(files={k:dict(sha256=digest(v),bytes=len(v)) for k,v in files.items()},
        expected_outputs={k:digest(v) for k,v in outputs.items()},source_analysis_sha256=source,
        independent_confirmation=False,full_training_reproduction=False,anonymity_certified=False))
    scan_files(files)
    raw = archive_bytes(files)
    if args.verify:
        if ARCHIVE.read_bytes()!=raw:
            raise ValueError('Archive differs from frozen scalar evidence')
    else:
        ARCHIVE.parent.mkdir(parents=True,exist_ok=True)
        with ARCHIVE.open('xb') as f:
            f.write(raw)
    receipt = dict(archive_sha256=digest(raw),archive_bytes=len(raw),members=len(files),
        comparisons=len(result['comparisons']),policies=len(result['risk']),source_analysis_sha256=source,
        clean_run=clean_run(ARCHIVE),full_training_reproduction=False,independent_confirmation=False,
        submission_ready=False)
    text = encode(receipt)
    if args.verify:
        if RECEIPT.read_bytes()!=text:
            raise ValueError('Isolated replay differs from its recorded receipt')
    else:
        with RECEIPT.open('xb') as f:
            f.write(text)
    print(json.dumps(dict(bytes=len(raw),comparisons=receipt['comparisons'],
        policies=receipt['policies'],exit_code=receipt['clean_run']['exit_code'])))


if __name__=='__main__':
    main()
