"""Export a fitting-source scalar replay with no paths or site names."""

import argparse
import hashlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_review_replay as pack
from scripts import run_m3w_cv_admission as study


def build(out):
    score = study.OUT/'readout.json'
    r = study.read(score)
    z = study.read(study.PRIVATE/'recording_scores.json')
    if z['parent'] != study.sha(score) or r['a_labels_opened'] or r['independent_confirmation']:
        raise ValueError('Require fitting-only replay')
    rows = z['rows']
    roster = {v['recording']:v['source'] for v in rows}
    rec = {k:f'R{i:02d}' for i,k in enumerate(sorted(roster))}
    src = {k:f'S{i:02d}' for i,k in enumerate(sorted(set(roster.values())))}
    seeds = {k:i for i,k in enumerate(study.SEEDS,1)}
    packet = dict(schema_version=1,seeds=list(seeds.values()),roster={rec[k]:src[v] for k,v in roster.items()},
        recording_studies={'admission':pack.trim(rows,rec,src,seeds)},
        joint_cells={k:{str(s):{} for s in seeds.values()} for k in src.values()})
    expected = dict(admission={**r['summary'],'seeds':list(seeds.values())},joint_pair_contrasts={})
    script = Path(pack.replay.__file__).read_text().rsplit("if __name__ == '__main__':",1)[0]
    if script.count('(34, 14, 3)') != 1:
        raise ValueError('Unexpected replay template')
    script = script.replace('(34, 14, 3)','(24, 9, 3)') + '''
root = Path(__file__).resolve().parent
verify_manifest(root)
result = recompute(read(root/'statistics.json'))
assert result == read(root/'expected.json')
print(json.dumps(result['admission'], sort_keys=True))
'''
    files = {name:(pack.ROOT/name).read_bytes() for name in pack.FILES}
    files.update({'README.md':b'Run python replay.py with NumPy. Replays scalar agent and risk results only.\n',
        'replay.py':script.encode(),'statistics.json':pack.raw_json(packet),'expected.json':pack.raw_json(expected),
        'requirements.txt':f'numpy=={study.np.__version__}\n'.encode()})
    files['manifest.json']=pack.raw_json({k:hashlib.sha256(v).hexdigest() for k,v in files.items()})
    pack.write_zip(out,files)
    print(study.sha(out))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=study.PRIVATE/'anonymous_cv_admission.zip')
    build(p.parse_args().out)
