"""Build the fixed-nomination value comparison and review replay."""

import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reproducibility.focused_review_supplement.verify import run_replay, unpack
from reproducibility.realized_swap_value_supplement import verify
from scripts import build_m3w_recipient_query_mask_candidate as prior
from scripts.build_m3w_exact_exchange_review_supplement import check

builder, archive, bundle, main = prior.builder, prior.archive, prior.bundle, prior.main
OUT = prior.OUT.parent/'evidence_manuscript_v196_realized_swap_value'
RECEIPT = main.PUBLIC/OUT.name/'receipt.json'
STUDY = main.PUBLIC/'opendd_realized_swap_value_v1'
PACK = main.PUBLIC/'realized_swap_value_review_supplement_v1'
DEST = bundle.LOCAL/'review_supplement_v27.zip'
TEMPLATE = archive.ROOT/'reproducibility/realized_swap_value_supplement'


def inputs():
    raw = {k: (STUDY/k).read_bytes() for k in ('readout.json', 'freeze.json', 'registration.json', 'protocol.md')}
    hashes = {k: main.digest(v) for k, v in raw.items()}
    d, f = (json.loads(raw[k]) for k in ('readout.json', 'freeze.json'))
    if (d['freeze'] != hashes['freeze.json'] or d['key'] != f['key'] or d['key'] != hashes['registration.json']
            or not f['decisions_frozen'] or not d['original_2pct'] or d['independent_confirmation']
            or d['external_outcomes_read'] or d['deployment_changed']):
        raise ValueError('Expected the frozen, scored value comparison')
    return d, hashes


def revise(source, d):
    marker = 'joint forecasting.'
    anchor = 'better opportunity coverage is not better\n'+marker
    if source.count(anchor) != 1:
        raise ValueError('Unexpected query mask paragraph')
    s = d['summary']
    n = s['counts']
    c = s['paired']['joint_minus_opportunity']
    observed, lo, hi = (c[k]['point'] for k in ('relative_observed', 'relative_lower', 'relative_upper'))
    t = rf'''

\paragraph{{Score the nominated swap directly.}}
At the same {n['swaps']} swaps, learning signed individual gains gives
{n['trajectory_strict']} qualifying swaps, or {n['joint_strict']} when adding half the predicted
relative gain, versus {n['opportunity_strict']} for the opportunity head.
The combined rule's observed relative contribution versus the
opportunity head is \({observed:+.6f}\) native units, but its
missing-label bounds are [{lo:+.6f}, {hi:+.6f}].
The observed improvement does not determine the full-grid effect.'''
    return source.replace(anchor, anchor+t, 1), (f'{abs(lo):.6f}', f'{hi:+.6f}')


def package():
    d, hashes = inputs()
    parent, draft = bundle.read_receipt(prior.PACK.name), bundle.read_receipt(OUT.name)
    pdf = draft['artifacts']['manuscript.pdf']
    if draft['source_readout_sha256'] != hashes or draft['prior_manuscript_sha256'] != parent['companion_pdf_sha256']:
        raise ValueError('Wrong draft or review archive parent')
    bundle.checked(OUT/'manuscript.pdf', pdf)
    s = d['summary']
    sites = sorted(s['paired']['joint_minus_opportunity']['relative_lower']['by_site'])
    panel = dict(counts=s['counts'], draws=np.random.default_rng(199).integers(7, size=(3000, 7)).tolist(),
                 location_means={a: {k: [v['by_site'][j] for j in sites] for k, v in m.items()}
                                 for a, m in s['paired'].items()})
    expected = dict(counts=s['counts'], paired={a: {k: {t: v[t] for t in ('point', 'ci95')}
                    for k, v in m.items()} for a, m in s['paired'].items()})
    items = {k: (TEMPLATE/k).read_bytes() for k in ('README.md', 'verify.py')}
    items['protocol.md'] = bundle.checked(STUDY/'protocol.md', hashes['protocol.md'])
    items['realized_value.json'] = main.as_json(panel)
    if any(archive.PRIVATE_MARKERS.search(b) for b in items.values()):
        raise ValueError('Identifying review data')
    items['prior_evidence.zip'] = bundle.checked(prior.DEST, parent['archive_sha256'])
    if set(items) != verify.MEMBERS:
        raise ValueError('Unexpected replay roster')
    items['manifest.json'] = main.as_json({k: main.digest(v) for k, v in items.items()})
    with TemporaryDirectory(prefix='realized-swap-review-') as tmp:
        root = Path(tmp)
        for name in ('first.zip', 'second.zip'):
            archive.encode(root/name, items)
        data = (root/'first.zip').read_bytes()
        if data != (root/'second.zip').read_bytes():
            raise ValueError('Nondeterministic review archive')
        unpack(root/'first.zip', root/'unpacked')
        if not check(run_replay(root/'unpacked')['realized_swap_value'], expected):
            raise ValueError('Direct paired intervals did not replay')
    row = dict(archive_sha256=main.digest(data), archive_bytes=len(data), prior_bundle_sha256=parent['archive_sha256'],
               companion_pdf_sha256=pdf, source_readout_sha256=hashes['readout.json'], standard_library_replay=True,
               includes_raw_labels=False, includes_checkpoints=False, uploaded=False, submission_ready=False)
    for p, content in {DEST: data, PACK/'receipt.json': main.as_json(row)}.items():
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
    args = p.parse_args()
    builder.build(OUT, args.compiler.resolve(), RECEIPT, parent_dir=prior.OUT, read_inputs=inputs, revise_source=revise)
    package()
