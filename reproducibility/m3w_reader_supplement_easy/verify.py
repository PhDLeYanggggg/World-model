"""Verify the anonymous easy-risk table and its inherited evidence."""

from hashlib import sha256
import json
from pathlib import Path
import runpy
from tempfile import TemporaryDirectory
from zipfile import ZipFile

MEMBERS = {'README.md', 'verify.py', 'supplement.pdf', 'easy_risk.json', 'prior_evidence.zip'}


def check_easy(data, prior):
    if ((data['flights'], data['components'], data['seeds'], data['risk_limit']) != (160, 14, [17, 29, 43], .02)
            or set(data['results']) != set(prior) or not data['original_2pct_unchanged']
            or not data['original_scored_outcomes_reused'] or data['new_external_cohort_opened']
            or data['deployment_changed'] or data['fit']):
        raise ValueError('Unexpected retrospective risk inputs')
    out = {}
    for arm, v in data['results'].items():
        m, r = v['matched'], v['released']
        original, all_pass, releases = prior[arm]
        easy = m['also_full_grid_easy']
        lost = all_pass-easy
        undefined, violation = m['possible_undefined_easy'], m['possible_easy_violation']
        if ((m['views'], m['original_three'], m['original_three_and_full_grid_all'], r.get('views', 0))
                != (480, original, all_pass, releases)
                or not 0 <= easy <= all_pass or not max(undefined, violation) <= lost <= undefined+violation
                or not 0 <= undefined <= lost or not 0 <= violation <= lost
                or not easy <= m['full_grid_all_and_easy'] <= min(m['full_grid_all'], m['full_grid_easy'])
                or not m['full_grid_easy'] <= m['full_grid_easy_support'] <= 480
                or r.get('original_three', 0) != r.get('also_full_grid_easy', 0)):
            raise ValueError('Easy risk and inherited counts disagree')
        out[arm] = [original, all_pass, easy, releases]
    return out


def verify(root):
    root = Path(root)
    hashes = json.loads((root/'manifest.json').read_text())
    if set(hashes) != MEMBERS or {p.name for p in root.iterdir()} != MEMBERS | {'manifest.json'}:
        raise ValueError('Unexpected supplement members')
    for name, expect in hashes.items():
        p = root/name
        if p.is_symlink() or sha256(p.read_bytes()).hexdigest() != expect:
            raise ValueError('Changed supplement file')
    with TemporaryDirectory() as tmp, ZipFile(root/'prior_evidence.zip') as z:
        folder = Path(tmp)
        names = z.namelist()
        if len(set(names)) != len(names) or any(Path(n).name != n or n in ('', '.', '..') for n in names):
            raise ValueError('Invalid inherited archive')
        for n in names:
            (folder/n).write_bytes(z.read(n))
        prior = runpy.run_path(str(folder/'verify.py'))['verify'](folder)
        easy = json.loads((root/'easy_risk.json').read_text())
        if sha256((folder/'risk.json').read_bytes()).hexdigest() != easy['binding']['all_risk_readout_sha256']:
            raise ValueError('Easy and all-edited readouts are not paired')
    return dict(inherited=prior, full_grid_easy_risk=check_easy(easy, prior['full_grid_all_risk']))


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2, allow_nan=False))
