"""Bind the paper's frozen WEPDTOF readout to an anonymous scalar replay."""

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from math import isclose
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reproducibility.wepdtof_scalar_replay.verify import ARMS, CONTRASTS, SEEDS

PUBLIC = ROOT/'outputs/publication_readiness_2026_09'
ADMIN = ROOT/'data/stage_cvpr2027_experiments/submission_admin'
SOURCE = 'wepdtof_external_transfer_v1/external_readout.json'
TEMPLATE = ROOT/'reproducibility/wepdtof_scalar_replay'
DRAFT = ADMIN.parent/'evidence_manuscript_v166/build/source_bindings.json'
RATIOS = ('completion_all_risk_upper', 'completion_easy_risk_upper', 'known_easy_net_degradation')
ARM_FIELDS = ('ade_px', 'fde_px', 'lower_vs_cv', 'lower_vs_ols4', 'p95_ade_px', 'edit_rate')


def digest(blob):
    return sha256(blob).hexdigest()


def json_bytes(obj):
    return (json.dumps(obj, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()


def extract(source):
    groups = defaultdict(dict)
    for row in source['rows']:
        key = row['clip']
        group = groups[row['source']]
        if key in group:
            raise ValueError('Duplicate original clip within source')
        group[key] = []
        if sorted(map(int, row['seeds'])) != SEEDS:
            raise ValueError('A clip lacks the original three seeds')
        for seed in SEEDS:
            cell = row['seeds'][str(seed)]
            arms = {}
            for arm in ARMS:
                r = cell['arms'][arm]
                risk = r['risk']
                arms[arm] = {**{k: r[k] for k in ARM_FIELDS}, 'risk': [risk[k] for k in RATIOS],
                             'edits': risk['selected'], 'unknown_edits': risk['unknown_selected']}
            group[key].append(dict(seed=seed, cv_mass=cell['reference_mass'], arms=arms,
                                   paired={k: cell['comparisons'][k] for k in CONTRASTS}))
    return dict(groups=[[clips[c] for c in sorted(clips)] for _, clips in sorted(groups.items())])


def resamples():
    return {str(n): np.random.default_rng(82317).integers(0, n, size=3000*n).tolist() for n in (10, 13)}


def check_close(got, expected):
    if isinstance(expected, dict):
        for k, v in expected.items():
            check_close(got[k], v)
    elif isinstance(expected, list):
        for a, b in zip(got, expected, strict=True):
            check_close(a, b)
    elif expected is None or isinstance(expected, (int, str)):
        if got != expected:
            raise ValueError(f'Frozen scalar differs: {got} vs {expected}')
    elif got is None or not isclose(got, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f'Frozen scalar differs: {got} vs {expected}')


def package(output, receipt):
    blob = (PUBLIC/SOURCE).read_bytes()
    bindings = json.loads(DRAFT.read_text())['source_bindings']
    if digest(blob) != bindings[SOURCE]:
        raise ValueError('Readout does not match the manuscript evidence')
    source = json.loads(blob)
    if (not source['external_scoring_completed'] or not source['original_2pct'] or
            source['external_tuning'] or source['dronecrowd_confirmation_access']):
        raise ValueError('Require the original frozen external readout')
    items = {name: (TEMPLATE/name).read_bytes() for name in ('README.md', 'verify.py')}
    items['evidence.json'] = json_bytes(extract(source))
    items['resamples.json'] = json_bytes(resamples())
    items['manifest.json'] = json_bytes(dict(members={k: digest(v) for k, v in items.items()},
        scope='anonymous scalar replay; no forecasts, tracks, images, row-level labels or checkpoints'))
    for name, item in items.items():
        if re.search(rb'/Users/|/home/|/cephfs/|github\.com|kcl\.ac\.uk|yangyue|PhDLeYang|jewelry_store', item, re.I):
            raise ValueError(f'Unmasked identity or source path in {name}')
    with TemporaryDirectory() as tmp:
        t = Path(tmp)
        for k, v in items.items():
            (t/k).write_bytes(v)
        result = json.loads(subprocess.run([sys.executable, '-I', '-B', '-S', str(t/'verify.py')], cwd=t,
                            check=True, capture_output=True, text=True, timeout=45).stdout)
    old = source['summary']
    check_close(result, dict(sources=old['sources'], views=old['recording_seed_views'],
        arms={k: {f: old['arms'][k][f] for f in (*ARM_FIELDS, 'edits', 'risk_passes', 'dual_passes', 'undefined_easy')}
              for k in ARMS}, matched=old['matched_comparisons']))
    taxonomy = json.loads((PUBLIC/'wepdtof_external_transfer_v1/failure_breakdown.json').read_text())
    if taxonomy['parent']['sha256'] != digest(blob):
        raise ValueError('Failure table is bound to a different source')
    for k, r in result['arms'].items():
        if k in taxonomy['per_arm']:
            a = taxonomy['per_arm'][k]
            check_close(r, dict(reasons=a['reasons'], unknown_edits=a['wholly_unknown_edits'],
                                positive_ols4=a['lower_positive_ols4']))
    output.parent.mkdir(parents=True, exist_ok=True)
    pending = output.with_suffix('.pending.zip')
    with ZipFile(pending, 'w', compression=ZIP_DEFLATED) as zip_file:
        for k, v in sorted(items.items()):
            entry = ZipInfo(k, (1980, 1, 1, 0, 0, 0))
            entry.compress_type = ZIP_DEFLATED
            zip_file.writestr(entry, v)
    pending.replace(output)
    result.update(source_sha256=digest(blob), archive_sha256=digest(output.read_bytes()),
                  derived_support_not_exported_as_input=True, offline_stdlib=True,
                  original_2pct=True, no_new_label_access=True, independent_confirmation=False)
    if receipt:
        receipt.parent.mkdir(parents=True, exist_ok=True)
        data = json_bytes(result)
        if receipt.exists() and receipt.read_bytes() != data:
            raise ValueError('Existing replay receipt differs')
        pending = receipt.with_suffix('.pending')
        pending.write_bytes(data)
        pending.replace(receipt)
    print(json.dumps(dict(matched=result['matched'], fixed_joint=result['arms']['fixed_joint']['reasons'],
                          archive_sha256=result['archive_sha256']), indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ADMIN/'wepdtof_scalar_replay.zip')
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    package(args.output, args.receipt)
