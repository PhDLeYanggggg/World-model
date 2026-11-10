"""Build the v195 query-budget comparison and its anonymous replay."""

import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reproducibility.focused_review_supplement.verify import run_replay, unpack
from reproducibility.recipient_query_mask_supplement import verify
from scripts import build_m3w_recipient_eligibility_candidate as prior
from scripts.build_m3w_exact_exchange_review_supplement import check

builder, archive, bundle, main = prior.builder, prior.archive, prior.bundle, prior.main
OUT = prior.OUT.parent/'evidence_manuscript_v195_recipient_query_mask'
RECEIPT = main.PUBLIC/OUT.name/'receipt.json'
STUDY = main.PUBLIC/'opendd_recipient_query_mask_v1'
PACK = main.PUBLIC/'recipient_query_mask_review_supplement_v1'
DEST = bundle.LOCAL/'review_supplement_v26.zip'
TEMPLATE = archive.ROOT/'reproducibility/recipient_query_mask_supplement'


def inputs():
    raw = {k: (STUDY/k).read_bytes() for k in ('readout.json', 'freeze.json', 'registration.json', 'protocol.md')}
    hashes = {k: main.digest(v) for k, v in raw.items()}
    d, f = (json.loads(raw[k]) for k in ('readout.json', 'freeze.json'))
    if (d['freeze'] != hashes['freeze.json'] or d['key'] != f['key'] or d['key'] != hashes['registration.json']
            or not f['decisions_frozen'] or not d['original_2pct'] or d['independent_confirmation']
            or d['external_outcomes_read'] or d['deployment_changed']):
        raise ValueError('Expected the frozen, scored development allocation')
    return d, hashes


def revise(source, d):
    marker = 'ranking, and ranking the remaining choices, are distinct failures.'
    if source.count(marker) != 1:
        raise ValueError('Missing recipient paragraph')
    n = d['summary']['counts']
    if n['hindsight_optimal']+n['hindsight_zero_quota'] != len(d['by_direction']):
        raise ValueError('The hindsight result is not a complete optimum')
    effect = d['summary']['paired']['query_mask_minus_archived_mask']['relative_lower']
    a, b = effect['ci95']
    t = rf'''
Reallocating the same {n['swaps']} swaps with an opportunity head, while
holding the recipient nomination fixed and spending 95--100\% of
the original swap movement, raises the qualifying count from
{n['archived_mask_qualifying']} to {n['query_mask_qualifying']}.
The same nominations permit {n['verified_hindsight_qualifying']} qualifying swaps in hindsight.
Relative-motion gain is \({effect['point']:+.6f}\) native units
[{a:+.6f}, {b:+.6f}]; better opportunity coverage is not better
joint forecasting.'''
    return source.replace(marker, marker+t, 1), (f"{abs(effect['point']):.6f}", 'qualifying swaps in hindsight')


def package():
    d, hashes = inputs()
    parent, draft = bundle.read_receipt(prior.PACK.name), bundle.read_receipt(OUT.name)
    pdf = draft['artifacts']['manuscript.pdf']
    if draft['source_readout_sha256'] != hashes or draft['prior_manuscript_sha256'] != parent['companion_pdf_sha256']:
        raise ValueError('Wrong draft or replay parent')
    bundle.checked(OUT/'manuscript.pdf', pdf)
    s = d['summary']
    sites = sorted(s['by_site_counts'])
    # Emit no recording IDs, model identifiers, or actual trajectories.
    panel = dict(sites=[{k: s['by_site_counts'][j].get(k, 0) for k in s['counts']} for j in sites],
                 budgets=[[r['counts']['swaps'], *[r['spend'][k] for k in verify.MASKS]] for r in d['by_direction']],
                 draws=np.random.default_rng(198).integers(7, size=(3000, 7)).tolist(),
                 location_means={c: {k: [v['by_site'][j] for j in sites] for k, v in m.items()}
                                 for c, m in s['paired'].items()})
    replay = verify.replay(panel)
    expected = dict(counts=s['counts'], spend=s['spend'], paired={c: {k: {x: v[x] for x in ('point', 'ci95')}
                    for k, v in m.items()} for c, m in s['paired'].items()})
    expected['query_minus_archived_count_rate_pp'] = replay['query_minus_archived_count_rate_pp']
    parts = {k: (TEMPLATE/k).read_bytes() for k in ('README.md', 'verify.py')}
    parts['protocol.md'] = bundle.checked(STUDY/'protocol.md', hashes['protocol.md'])
    parts['query_mask.json'] = main.as_json(panel)
    if any(archive.PRIVATE_MARKERS.search(b) for b in parts.values()):
        raise ValueError('Identifying material in review data')
    parts['prior_evidence.zip'] = bundle.checked(prior.DEST, parent['archive_sha256'])
    if set(parts) != verify.MEMBERS:
        raise ValueError('Unexpected replay roster')
    parts['manifest.json'] = main.as_json({k: main.digest(v) for k, v in parts.items()})
    with TemporaryDirectory(prefix='query-mask-review-') as tmp:
        root = Path(tmp)
        for name in ('first.zip', 'second.zip'):
            archive.encode(root/name, parts)
        data = (root/'first.zip').read_bytes()
        if data != (root/'second.zip').read_bytes():
            raise ValueError('Nondeterministic archive')
        unpack(root/'first.zip', root/'unpacked')
        if not check(run_replay(root/'unpacked')['query_mask'], expected):
            raise ValueError('Grouped results did not reproduce')
    row = dict(archive_sha256=main.digest(data), archive_bytes=len(data), prior_bundle_sha256=parent['archive_sha256'],
               companion_pdf_sha256=pdf, source_readout_sha256=hashes['readout.json'],
               query_minus_archived_count_rate_pp=expected['query_minus_archived_count_rate_pp'],
               standard_library_replay=True, includes_raw_labels=False, includes_checkpoints=False,
               uploaded=False, submission_ready=False)
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
