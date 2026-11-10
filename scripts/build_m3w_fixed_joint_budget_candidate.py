"""Bind the movement-matched sensitivity analysis to v205 and its anonymous replay."""

import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reproducibility.focused_review_supplement.verify import run_replay, unpack
from reproducibility.opendd_fixed_joint_budget_supplement import verify
from scripts import build_m3w_fixed_joint_release_candidate as prior
from scripts import build_m3w_vehicle_direction_candidate as builder
from scripts.build_m3w_exact_exchange_review_supplement import check

archive, bundle, main = prior.archive, prior.bundle, prior.main
OUT = prior.OUT.parent/'evidence_manuscript_v205_fixed_joint_budget'
PUBLIC = main.PUBLIC/OUT.name
STUDY = main.PUBLIC/'opendd_fixed_joint_budget_v1'
TEMPLATE = archive.ROOT/'reproducibility/opendd_fixed_joint_budget_supplement'
DEST = bundle.LOCAL/'review_supplement_v37.zip'


def inputs():
    raw = {k: (STUDY/f'{k}.json').read_bytes() for k in ('registration', 'mask_freeze', 'readout')}
    raw['protocol'] = (STUDY/'protocol.md').read_bytes()
    h = {k: main.digest(v) for k, v in raw.items()}
    f, r, reg = (json.loads(raw[k]) for k in ('mask_freeze', 'readout', 'registration'))
    if (f['registration'] != h['registration'] or r['registration'] != h['registration']
            or f['outcomes_read'] or f['optimal_witnesses'] != 252 or len(f['packets']) != 126
            or r['mask_freeze_sha256'] != h['mask_freeze'] or r['directions'] != 126
            or not r['original_quota_reproduced'] or not r['original_2pct'] or not f['exact_count']
            or r['deployment_changed'] or r['independent_confirmation'] or r['new_training']
            or r['new_external_labels_read'] or reg['budget_tolerance'] != .01
            or reg['inputs'][str((STUDY/'protocol.md').relative_to(archive.ROOT))] != h['protocol']):
        raise ValueError('Require the frozen movement-matched analysis')
    return r, h


def revise(text, r):
    anchor = 'release comparison.\n'
    if text.count(anchor) != 1:
        raise ValueError('Expected the fixed-joint release paragraph from v204')
    m = r['summary']['movement_matched']['contrasts']['supported_merit']
    observed, lower = (m[k] for k in ('ade_available_pp', 'ade_lower_pp'))
    debit = m['missing_debit_pp']['point']
    paragraph = (f'Matching both total and query-weighted movement to within 1\\%\n'
        f'leaves an observed gain of {observed["point"]:+.6f} pp {prior.interval(observed, 6)}, but a\n'
        f'conservative gain of {lower["point"]:+.6f} pp {prior.interval(lower, 6)}.\n'
        f'The paired missing-label debit is {debit:.6f} pp. Matching movement\n'
        'therefore does not match the label coverage of that movement.\n')
    return text.replace(anchor, anchor+paragraph, 1), (f'{abs(lower["point"]):.6f}', f'{debit:.6f}')


def package():
    r, hashes = inputs()
    paper = bundle.read_receipt(OUT.name)
    pdf = paper['artifacts']['manuscript.pdf']
    prev = json.loads((prior.PUBLIC/'review_receipt.json').read_bytes())
    if paper['source_readout_sha256'] != hashes or paper['prior_manuscript_sha256'] != prev['companion_pdf_sha256']:
        raise ValueError('Manuscript and replay evidence mismatch')
    bundle.checked(OUT/'manuscript.pdf', pdf)
    data = {k: r[k] for k in ('summary', 'directions', 'locations', 'seeds')}
    data['draws'] = np.random.default_rng(12773).integers(7, size=(3000, 7)).tolist()
    parts = {k: (TEMPLATE/k).read_bytes() for k in ('README.md', 'verify.py')}
    parts['protocol.md'] = bundle.checked(STUDY/'protocol.md', hashes['protocol'])
    parts['budget.json'] = main.as_json(data)
    if any(archive.PRIVATE_MARKERS.search(v) for v in parts.values()):
        raise ValueError('Identifying content in anonymous addendum')
    parts['prior_evidence.zip'] = bundle.checked(prior.DEST, prev['archive_sha256'])
    if set(parts) != verify.MEMBERS:
        raise ValueError('Unexpected replay contents')
    parts['manifest.json'] = main.as_json({k: main.digest(v) for k, v in parts.items()})
    expected = {mode: {**p, 'contrasts': {a: {k: {m: v[m] for m in ('point', 'ci95')}
        for k, v in vs.items()} for a, vs in p['contrasts'].items()}} for mode, p in r['summary'].items()}
    with TemporaryDirectory(prefix='opendd-fixed-joint-budget-review-') as tmp:
        root = Path(tmp)
        for n in ('first.zip', 'second.zip'):
            archive.encode(root/n, parts)
        raw = (root/'first.zip').read_bytes()
        if raw != (root/'second.zip').read_bytes():
            raise ValueError('Nondeterministic archive')
        unpack(root/'first.zip', root/'unpacked')
        if not check(run_replay(root/'unpacked')['opendd_fixed_joint_budget'], expected):
            raise ValueError('Anonymous budget replay differs from the real readout')
    row = dict(archive_sha256=main.digest(raw), archive_bytes=len(raw), companion_pdf_sha256=pdf,
        prior_bundle_sha256=prev['archive_sha256'], source_readout_sha256=hashes['readout'],
        standard_library_replay=True, includes_raw_labels=False, uploaded=False, submission_ready=False)
    for p, v in ((DEST, raw), (PUBLIC/'review_receipt.json', main.as_json(row))):
        if p.exists() and p.read_bytes() != v:
            raise FileExistsError('Refusing to replace a frozen candidate')
        if not p.exists():
            with p.open('xb') as f:
                f.write(v)
    print(row)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compiler', type=Path, required=True)
    a = p.parse_args()
    builder.build(OUT, a.compiler.resolve(), PUBLIC/'receipt.json', parent_dir=prior.OUT,
                  read_inputs=inputs, revise_source=revise)
    package()
