"""Verify the anonymous full-grid cross-dataset evidence and inherited tables."""

from hashlib import sha256
import json
from pathlib import Path
import runpy
from tempfile import TemporaryDirectory
from zipfile import ZipFile

MEMBERS = {'README.md', 'verify.py', 'supplement.pdf', 'wepdtof_risk.json', 'prior_evidence.zip'}


def check_wep(data):
    arms = data['arms']
    if ((data['clips'], data['sources'], data['seeds'], data['limit']) != (15, 13, [17, 29, 43], .02)
            or set(arms) != {'fixed_joint', 'fixed_independent', 'fixed_uniform', 'fixed_no_pair'}
            or not data['original_rule_unchanged'] or not data['original_scored_outcomes_reused']
            or any(data[k] for k in ('new_confirmation', 'external_tuning', 'fit', 'deployment_changed'))):
        raise ValueError('Unexpected WEPDTOF cohort or evaluation rule')
    result = {}
    for arm, v in arms.items():
        if ((v['views'], v['edited_views'], v['edits']) != (45, 36, 366)
                or not 0 <= v['full_risk_pass'] <= v['full_easy_support'] <= 36
                or not 0 <= v['original_and_full_risk_pass'] <= min(v['original_risk_pass'], v['full_risk_pass'])
                or not 0 <= v['full_pass_utility_excluded'] <= min(v['full_risk_pass'], v['utility_excluded'])
                or not 0 <= v['utility_guaranteed'] <= 36-v['utility_excluded']
                or not 0 <= v['full_grid_dual_guaranteed'] <= min(v['full_risk_pass']-v['full_pass_utility_excluded'],
                                                                 v['utility_guaranteed'])):
            raise ValueError('Inconsistent full-grid sensitivity counts')
        result[arm] = {k: v[k] for k in ('edited_views', 'original_risk_pass', 'full_risk_pass',
                                        'full_pass_utility_excluded', 'full_grid_dual_guaranteed')}
    return result


def verify(root):
    root = Path(root)
    hashes = json.loads((root/'manifest.json').read_text())
    if set(hashes) != MEMBERS or {p.name for p in root.iterdir()} != MEMBERS | {'manifest.json'}:
        raise ValueError('Unexpected supplement members')
    for name, digest in hashes.items():
        p = root/name
        if p.is_symlink() or sha256(p.read_bytes()).hexdigest() != digest:
            raise ValueError('Changed supplement file')
    data = json.loads((root/'wepdtof_risk.json').read_text())
    result = check_wep(data)
    with TemporaryDirectory() as tmp, ZipFile(root/'prior_evidence.zip') as z:
        folder = Path(tmp)
        names = z.namelist()
        if len(set(names)) != len(names) or any(Path(n).name != n or n in ('', '.', '..') for n in names):
            raise ValueError('Invalid inherited archive')
        for n in names:
            (folder/n).write_bytes(z.read(n))
        prior = runpy.run_path(str(folder/'verify.py'))['verify'](folder)
    return dict(inherited=prior, wepdtof_full_grid=result)


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2, allow_nan=False))
