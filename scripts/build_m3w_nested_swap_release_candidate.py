"""Build the disjoint release comparison and anonymous replay."""

import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reproducibility.focused_review_supplement.verify import run_replay, unpack
from reproducibility.nested_swap_release_supplement import verify
from scripts import build_m3w_swap_support_candidate as prior
from scripts.build_m3w_exact_exchange_review_supplement import check

builder, archive, bundle, main = prior.builder, prior.archive, prior.bundle, prior.main
OUT = prior.OUT.parent/'evidence_manuscript_v198_nested_swap_release'
RECEIPT = main.PUBLIC/OUT.name/'receipt.json'
STUDY = main.PUBLIC/'opendd_nested_swap_release_v1'
PACK = main.PUBLIC/'nested_swap_release_review_supplement_v1'
DEST = bundle.LOCAL/'review_supplement_v29.zip'
TEMPLATE = archive.ROOT/'reproducibility/nested_swap_release_supplement'


def inputs():
    keys = ('protocol.md', 'registration.json', 'freeze.json', 'calibration_freeze.json', 'readout.json')
    raw = {k: (STUDY/k).read_bytes() for k in keys}
    h = {k: main.digest(v) for k, v in raw.items()}
    f, c, r = (json.loads(raw[k]) for k in keys[2:])
    if (r['calibration'] != h['calibration_freeze.json'] or c['freeze'] != h['freeze.json']
            or not f['key'] == c['key'] == r['key'] == h['registration.json'] or not c['original_2pct']
            or not r['original_2pct'] or r['external_outcomes_read'] or r['independent_confirmation']
            or r['deployment_changed'] or len(c['rules']) != 126):
        raise ValueError('The disjoint release study did not bind its evidence')
    return (r, c), h


def revise(source, data):
    marker = r'\section{What Transfers to Other Recordings?}'
    if source.count(marker) != 1:
        raise ValueError('Missing external section')
    s = data[0]['summary']
    a, b = s['sign'], s['magnitude']
    text = rf'''\paragraph{{Calibrate on a different location.}}
We fit on five OpenDD locations, calibrate on a sixth and score on
a seventh. Sign and magnitude release {a['released']} and {b['released']} views;
{a['strict']} and {b['strict']} pass risk and beat both references and the incumbent.
The additive control passes risk and both references in
{s['additive']['dual']} of {s['additive']['released']} releases. These counts are not matched.

'''
    return source.replace(marker, text+marker, 1), (f"{s['additive']['dual']} of {s['additive']['released']}",
                                                   f"{a['released']} and {b['released']} views")


def package():
    (data, calibration), hashes = inputs()
    parent, draft = bundle.read_receipt(prior.PACK.name), bundle.read_receipt(OUT.name)
    pdf = draft['artifacts']['manuscript.pdf']
    if draft['source_readout_sha256'] != hashes or draft['prior_manuscript_sha256'] != parent['companion_pdf_sha256']:
        raise ValueError('Wrong review or paper parent')
    bundle.checked(OUT/'manuscript.pdf', pdf)
    s = data['summary']
    sites = sorted(s['additive']['effects']['lower_cv_pp']['by_site'])
    seeds = sorted({x['seed'] for x in calibration['rules'].values()})
    lookup = {(x['held'], x['calibration'], x['seed']): x['arms'] for x in calibration['rules'].values()}
    directions = []
    for row in data['by_direction']:
        i = row['identity']
        h, c, seed = (i[k] for k in ('held', 'donor_cal', 'seed'))
        arms = {a: {k: row['arms'][a][k] for k in verify.COUNTS} for a in verify.ARMS}
        for a, out in arms.items():
            out['calibration_count'] = lookup[h, c, seed][a]['calibration_recordings']
        directions.append(dict(roles=[sites.index(h), sites.index(c), seeds.index(seed)], arms=arms))
    panel = dict(directions=directions, draws=np.random.default_rng(211).integers(7, size=(3000, 7)).tolist(),
        effects={a: {k: [v['by_site'][x] for x in sites] for k, v in c['effects'].items()} for a, c in s.items()})
    expected = {a: dict(c, effects={k: {t: v[t] for t in ('point', 'ci95')} for k, v in c['effects'].items()})
                for a, c in s.items()}
    parts = {k: (TEMPLATE/k).read_bytes() for k in ('README.md', 'verify.py')}
    parts['protocol.md'] = bundle.checked(STUDY/'protocol.md', hashes['protocol.md'])
    parts['nested_release.json'] = main.as_json(panel)
    if any(archive.PRIVATE_MARKERS.search(p) for p in parts.values()):
        raise ValueError('Identifying review material')
    parts['prior_evidence.zip'] = bundle.checked(prior.DEST, parent['archive_sha256'])
    if set(parts) != verify.MEMBERS:
        raise ValueError('Unexpected review archive members')
    parts['manifest.json'] = main.as_json({k: main.digest(v) for k, v in parts.items()})
    with TemporaryDirectory(prefix='nested-swap-review-') as tmp:
        root = Path(tmp)
        for n in ('first.zip', 'second.zip'):
            archive.encode(root/n, parts)
        raw = (root/'first.zip').read_bytes()
        if raw != (root/'second.zip').read_bytes():
            raise ValueError('Nondeterministic review archive')
        unpack(root/'first.zip', root/'unpacked')
        if not check(run_replay(root/'unpacked')['nested_swap_release'], expected):
            raise ValueError('Release evidence does not replay')
    row = dict(archive_sha256=main.digest(raw), archive_bytes=len(raw), prior_bundle_sha256=parent['archive_sha256'],
               companion_pdf_sha256=pdf, source_readout_sha256=hashes['readout.json'], standard_library_replay=True,
               includes_raw_labels=False, includes_checkpoints=False, uploaded=False, submission_ready=False)
    for p, content in {DEST: raw, PACK/'receipt.json': main.as_json(row)}.items():
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists() and p.read_bytes() != content:
            raise FileExistsError('Refusing to replace another review artifact')
        if not p.exists():
            with p.open('xb') as f:
                f.write(content)
    print(row)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compiler', type=Path, required=True)
    arg = p.parse_args()
    builder.build(OUT, arg.compiler.resolve(), RECEIPT, parent_dir=prior.OUT, read_inputs=inputs, revise_source=revise)
    package()
