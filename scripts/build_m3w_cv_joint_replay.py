"""Package anonymous solver witness scalars for exact summary replay."""

import argparse
import hashlib
import inspect
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_cv_joint_feasibility as study
from scripts import build_m3w_review_replay as pack


def build(out):
    receipt = study.read(study.PRIVATE / 'recording_receipt.json')
    result = study.read(study.OUT / 'readout.json')
    if receipt['registration'] != result['registration'] or result['independent_confirmation']:
        raise ValueError('Require bound development-only readout')
    cells = receipt['cells']
    source = {k: f'S{i:02d}' for i, k in enumerate(sorted({r['source'] for r in cells}))}
    rec = {k: f'R{i:02d}' for i, k in enumerate(sorted({r['recording'] for r in cells}))}
    seed = {k: i for i, k in enumerate(study.SEEDS, 1)}
    rows = []
    for row in cells:
        w = row['witness']
        risk = None if w is None else {k: w['original_risk'][k] for k in (
            'completion_all_risk_upper', 'completion_easy_risk_upper', 'known_easy_net_degradation')}
        rows.append(dict(source=source[row['source']], recording=rec[row['recording']], cap=row['cap'],
                         seed=seed.get(row['seed']), status=row['status'], witness=None if w is None else
                         dict(edits=w['edits'], gain_pp=w['gain_pp'], risk=risk)))
    header = (f'import hashlib\nimport json\nfrom pathlib import Path\nimport numpy as np\n'
              f'CAPS = {study.CAPS!r}\nSTATES = {study.STATES!r}\n')
    check = '''
root = Path(__file__).resolve().parent
for name, digest in json.loads((root / 'manifest.json').read_text()).items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
rows = json.loads((root / 'records.json').read_text())
assert len(rows) == 272
for r in rows:
    if r['status'] != 'verified':
        continue
    w = r['witness']
    assert w['edits'] > 0 and min(w['gain_pp']) > 1e-6
    assert all(v is not None and v <= .02 for v in w['risk'].values())
result = summarize(rows)
assert result == json.loads((root / 'expected.json').read_text())
print(json.dumps(result, sort_keys=True))
'''
    script = header + inspect.getsource(study.summarize) + '\n' + inspect.getsource(study.mean_edits) + check
    files = {'README.md': b'Run python replay.py; NumPy only. Replays aggregate scalar evidence, not the solver.\n',
             'replay.py': script.encode(), 'records.json': pack.raw_json(rows),
             'expected.json': pack.raw_json(result['summary']),
             'requirements.txt': f'numpy=={study.np.__version__}\n'.encode()}
    files['manifest.json'] = pack.raw_json({k: hashlib.sha256(v).hexdigest() for k, v in files.items()})
    pack.write_zip(out, files)
    print(study.sha(out))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=study.PRIVATE / 'anonymous_joint_feasibility.zip')
    build(p.parse_args().out)
