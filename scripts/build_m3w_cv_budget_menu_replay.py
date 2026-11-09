"""Export anonymous capacity counts and source bootstrap without raw trajectories."""
import argparse
import hashlib
import inspect
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_cv_budget_menu_factorial as study
from scripts import build_m3w_review_replay as pack


def build(path):
    ex,np = study.exp,study.np
    public = study.OUT/'nested_verification.json'
    expected = ex.read(public)
    raw = study.OUT/'readout.json'
    if (expected['parent']['sha256']!=ex.sha(raw) or not expected['labels_hindsight_only']
            or expected['a_labels_opened'] or expected['independent_confirmation']):
        raise ValueError('Need the registered development-only capacity result')
    d = study.base.fit_population(lambda **v:None)
    f = ex.read(ex.OUT/'freeze.json')
    lookup = {s:f'S{i:02d}' for i,s in enumerate(sorted(set(d['sites'])))}
    rows = []
    for i,c in enumerate(f['cells']):
        _,h = study.base.subset(d,c['identity']['held'])
        for rec in np.unique(h['recording']):
            ix = h['recording']==rec
            site, = np.unique(h['sites'][ix])
            z = ex.read(study.PRIVATE/f'nested_{i}_{rec}.json')
            if z['parent']!=ex.sha(raw):
                raise ValueError('Checkpoint registration changed')
            rows.append(dict(source=lookup[site],rows=int(ix.sum()),
                **{k:{j:v[j] for j in ('status','edits') if j in v} for k,v in z['results'].items()}))
    result = study.summarize(rows)
    if result!={k:expected[k] for k in result}:
        raise ValueError('Anonymous rows do not reproduce the published result')
    script = 'from collections import Counter\nfrom pathlib import Path\nimport hashlib\nimport json\nimport numpy as np\n'
    script += f'ARMS = {study.ARMS!r}\n\n'
    script += inspect.getsource(study.summarize)
    script += '''
root = Path(__file__).resolve().parent
manifest = json.loads((root/'manifest.json').read_text())
for name,sha in manifest.items():
    if hashlib.sha256((root/name).read_bytes()).hexdigest()!=sha:
        raise ValueError(f'Manifest mismatch: {name}')
actual = summarize(json.loads((root/'statistics.json').read_text()))
if actual!=json.loads((root/'expected.json').read_text()):
    raise ValueError('Capacity table did not replay')
print(json.dumps(actual,sort_keys=True))
'''
    files = {'README.md':b'Run python replay.py. Verifies anonymous scalar counts and source bootstrap only.\n',
        'replay.py':script.encode(),'statistics.json':pack.raw_json(rows),'expected.json':pack.raw_json(result),
        'requirements.txt':f'numpy=={np.__version__}\n'.encode()}
    files['manifest.json'] = pack.raw_json({k:hashlib.sha256(v).hexdigest() for k,v in files.items()})
    pack.write_zip(path,files)
    print(ex.sha(path))


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=study.PRIVATE/'anonymous_capacity.zip')
    build(p.parse_args().out)
