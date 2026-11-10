"""Build the v197 support-ranking comparison and its anonymous replay."""

import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reproducibility.focused_review_supplement.verify import run_replay, unpack
from reproducibility.swap_support_supplement import verify
from scripts import build_m3w_realized_swap_value_candidate as prior
from scripts.build_m3w_exact_exchange_review_supplement import check

builder, archive, bundle, main = prior.builder, prior.archive, prior.bundle, prior.main
OUT = prior.OUT.parent/'evidence_manuscript_v197_swap_support'
RECEIPT = main.PUBLIC/OUT.name/'receipt.json'
STUDIES = {k: main.PUBLIC/f'opendd_swap_support{suffix}_v1' for k, suffix in (('ranking', ''), ('allocation', '_allocation'))}
PACK = main.PUBLIC/'swap_support_review_supplement_v1'
DEST = bundle.LOCAL/'review_supplement_v28.zip'
TEMPLATE = archive.ROOT/'reproducibility/swap_support_supplement'


def inputs():
    hashes, data = {}, {}
    for k, d in STUDIES.items():
        raw = {a: (d/a).read_bytes() for a in ('readout.json', 'freeze.json', 'registration.json', 'protocol.md')}
        h = {a: main.digest(v) for a, v in raw.items()}
        r, f = (json.loads(raw[a]) for a in ('readout.json', 'freeze.json'))
        if (r['freeze'] != h['freeze.json'] or r['key'] != f['key'] or r['key'] != h['registration.json']
                or not r['original_2pct'] or r['external_outcomes_read'] or r['independent_confirmation']
                or r['deployment_changed']):
            raise ValueError('Unexpected development role or freeze')
        hashes.update({f'{k}/{a}': v for a, v in h.items()})
        data[k] = r['summary']
    if not data['ranking']['allocation_screen'] or data['ranking']['views'] != 8814:
        raise ValueError('The fixed ranking screen did not pass')
    return data, hashes


def revise(source, d):
    marker = 'The observed improvement does not determine the full-grid effect.'
    if source.count(marker) != 1:
        raise ValueError('Missing fixed action paragraph')
    r, a = d['ranking'], d['allocation']
    old = r['means']['relative']['auc']['v196_fused']
    new = r['means']['relative']['auc']['footprint_sign']
    rank_lo, rank_hi = r['paired']['relative_footprint_sign_minus_v196_fused']['ci95']
    c = a['counts']
    value = a['paired']['joint_minus_trajectory']['relative_lower']
    lo, hi = value['ci95']
    text = rf'''

\paragraph{{Predict support separately.}}
Training separate support and observed-effect targets, with a
past-only footprint, improves the relative positive-gain AUC from
{old:.3f} to {new:.3f} on {r['views']:,} nominated OpenDD actions
(paired location interval [{rank_lo:+.3f}, {rank_hi:+.3f}] for the change).
At the original {c['swaps']}-swap budget, individual and combined sign
ranking find {c['trajectory_strict']} and {c['joint_strict']} qualifying swaps, compared with
{c['opportunity_strict']} for the opportunity control. The direct combined-minus-individual
conservative relative gain is \({value['point']:+.6f}\)
[{lo:+.6f}, {hi:+.6f}] in native units.'''
    return source.replace(marker, marker+text, 1), (f'{old:.3f} to {new:.3f}', f'{r["views"]:,}')


def package():
    data, hashes = inputs()
    parent, draft = bundle.read_receipt(prior.PACK.name), bundle.read_receipt(OUT.name)
    pdf = draft['artifacts']['manuscript.pdf']
    if draft['source_readout_sha256'] != hashes or draft['prior_manuscript_sha256'] != parent['companion_pdf_sha256']:
        raise ValueError('Wrong paper or review archive parent')
    bundle.checked(OUT/'manuscript.pdf', pdf)
    rank, alloc = data['ranking'], data['allocation']
    sites = sorted(next(iter(rank['paired'].values()))['by_site'])
    means = lambda v: [v['by_site'][s] for s in sites]
    small = lambda v: {k: v[k] for k in ('point', 'ci95')}
    panel = dict(rank_effects={k: means(v) for k, v in rank['paired'].items()},
        allocation_effects={k: {m: means(v) for m, v in z.items()} for k, z in alloc['paired'].items()},
        allocation_counts=alloc['counts'], rank_draws=np.random.default_rng(210).integers(7, size=(3000, 7)).tolist(),
        allocation_draws=np.random.default_rng(199).integers(7, size=(3000, 7)).tolist())
    expected = dict(rank_effects={k: small(v) for k, v in rank['paired'].items()},
        allocation_effects={k: {m: small(v) for m, v in z.items()} for k, z in alloc['paired'].items()},
        allocation_counts=alloc['counts'], allocation_screen=rank['allocation_screen'])
    parts = {k: (TEMPLATE/k).read_bytes() for k in ('README.md', 'verify.py')}
    for k, d in STUDIES.items():
        parts[k+'_protocol.md'] = bundle.checked(d/'protocol.md', hashes[f'{k}/protocol.md'])
    parts['support.json'] = main.as_json(panel)
    if any(archive.PRIVATE_MARKERS.search(p) for p in parts.values()):
        raise ValueError('Identifying material in review data')
    parts['prior_evidence.zip'] = bundle.checked(prior.DEST, parent['archive_sha256'])
    if set(parts) != verify.MEMBERS:
        raise ValueError('Unexpected review roster')
    parts['manifest.json'] = main.as_json({k: main.digest(v) for k, v in parts.items()})
    with TemporaryDirectory(prefix='swap-support-review-') as tmp:
        root = Path(tmp)
        for k in ('first.zip', 'second.zip'):
            archive.encode(root/k, parts)
        raw = (root/'first.zip').read_bytes()
        if raw != (root/'second.zip').read_bytes():
            raise ValueError('Nondeterministic review archive')
        unpack(root/'first.zip', root/'unpacked')
        if not check(run_replay(root/'unpacked')['swap_support'], expected):
            raise ValueError('The grouped estimates do not replay')
    row = dict(archive_sha256=main.digest(raw), archive_bytes=len(raw), prior_bundle_sha256=parent['archive_sha256'],
        companion_pdf_sha256=pdf, source_readouts={k: hashes[f'{k}/readout.json'] for k in STUDIES},
        standard_library_replay=True, includes_raw_labels=False, includes_checkpoints=False,
        uploaded=False, submission_ready=False)
    for p, content in {DEST: raw, PACK/'receipt.json': main.as_json(row)}.items():
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists() and p.read_bytes() != content:
            raise FileExistsError('Refusing to replace a different review artifact')
        if not p.exists():
            with p.open('xb') as f:
                f.write(content)
    print(row)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compiler', type=Path, required=True)
    a = p.parse_args()
    builder.build(OUT, a.compiler.resolve(), RECEIPT, parent_dir=prior.OUT, read_inputs=inputs, revise_source=revise)
    package()
