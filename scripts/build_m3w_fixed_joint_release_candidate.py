"""Build v204 with the matched OpenDD fixed-proposal release experiment."""

import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reproducibility.focused_review_supplement.verify import run_replay, unpack
from reproducibility.opendd_fixed_joint_release_supplement import verify
from scripts import build_m3w_songdo_score_ranking_candidate as prior
from scripts.build_m3w_exact_exchange_review_supplement import check

archive, bundle, main = prior.archive, prior.bundle, prior.main
OUT = prior.OUT.parent/'evidence_manuscript_v204_fixed_joint_release'
PUBLIC = main.PUBLIC/OUT.name
STUDY = main.PUBLIC/'opendd_fixed_joint_release_v1'
TEMPLATE = archive.ROOT/'reproducibility/opendd_fixed_joint_release_supplement'
DEST = bundle.LOCAL/'review_supplement_v36.zip'


def inputs():
    raw = {k: (STUDY/f'{k}.json').read_bytes() for k in ('registration', 'score_freeze', 'calibration', 'readout')}
    raw['protocol'] = (STUDY/'protocol.md').read_bytes()
    h = {k: main.digest(v) for k, v in raw.items()}
    r, f, c, reg = (json.loads(raw[k]) for k in ('readout', 'score_freeze', 'calibration', 'registration'))
    if (r['registration'] != h['registration'] or f['registration'] != h['registration']
            or r['score_freeze_sha256'] != h['score_freeze'] or r['calibration_sha256'] != h['calibration']
            or c['score_freeze_sha256'] != h['score_freeze'] or f['outcome_fields_loaded']
            or r['directions'] != 126 or len(c['directions']) != 126 or len(f['packets']) != 126
            or not r['original_2pct'] or r['fit'] or r['external_outcomes_read'] or r['independent_confirmation']
            or r['deployment_changed'] or reg['files'][str((STUDY/'protocol.md').relative_to(archive.ROOT))] != h['protocol']):
        raise ValueError('Require the frozen fixed-joint development experiment')
    return r, h


def interval(row, digits):
    return '['+', '.join(f'{x:+.{digits}f}' for x in row['ci95'])+']'


def revise(text, r):
    anchor = r'\section{What Transfers to Other Recordings?}'
    if text.count(anchor) != 1:
        raise ValueError('Expected unchanged v203 external section')
    c = r['summary']['fixed_quarter']
    n = c['counts']['original']['selected']
    if any(v['selected'] != n for v in c['counts'].values()):
        raise ValueError('Require the same selection count')
    dual, ade, rel = (c['contrasts']['supported_merit'][k] for k in ('dual_rate_pp', 'ade_lower_pp', 'relative_lower_native'))
    a, b = (c['counts'][k]['dual'] for k in ('original', 'supported_merit'))
    paragraph = rf'''\paragraph{{Predict risk support with the proposal held fixed.}}
Using the same two edited paths per recording in seven OpenDD
locations, we add log predicted easy-reference availability to
the original merit. The causal support head excludes both the
calibration and held locations. At {n:,} matched recording views
the dual-pass count rises from {a} to {b} ({dual['point']:+.2f} pp,
95\% location interval {interval(dual, 2)}), but conservative
trajectory gain falls ({ade['point']:+.6f} pp {interval(ade, 6)}).
Conservative relative-motion gain also falls ({rel['point']:+.5f}
native units). Predicting support does not suffice to rank useful
replacements. The supplement reports the distinct calibrated
release comparison.

'''
    return text.replace(anchor, paragraph+anchor, 1), (f'{n:,} matched', f'{abs(ade["point"]):.6f}')


def package():
    r, hashes = inputs()
    candidate = bundle.read_receipt(OUT.name)
    pdf = candidate['artifacts']['manuscript.pdf']
    if candidate['source_readout_sha256'] != hashes:
        raise ValueError('Paper and evidence differ')
    bundle.checked(OUT/'manuscript.pdf', pdf)
    prev = json.loads((prior.PUBLIC/'review_receipt.json').read_bytes())
    if candidate['prior_manuscript_sha256'] != prev['companion_pdf_sha256']:
        raise ValueError('Unexpected inherited paper and replay')
    data = {k: r[k] for k in ('summary', 'directions', 'seeds')}
    data['draws'] = np.random.default_rng(12773).integers(7, size=(3000, 7)).tolist()
    parts = {k: (TEMPLATE/k).read_bytes() for k in ('README.md', 'verify.py')}
    parts['protocol.md'] = bundle.checked(STUDY/'protocol.md', hashes['protocol'])
    parts['release.json'] = main.as_json(data)
    if any(archive.PRIVATE_MARKERS.search(v) for v in parts.values()):
        raise ValueError('Identifying material in anonymous evidence')
    parts['prior_evidence.zip'] = bundle.checked(prior.DEST, prev['archive_sha256'])
    if set(parts) != verify.MEMBERS:
        raise ValueError('Unexpected replay contents')
    parts['manifest.json'] = main.as_json({k: main.digest(v) for k, v in parts.items()})
    expected = {mode: dict(counts=p['counts'], contrasts={a: {k: {m: v[m] for m in ('point', 'ci95')}
        for k, v in vs.items()} for a, vs in p['contrasts'].items()}) for mode, p in r['summary'].items()}
    with TemporaryDirectory(prefix='opendd-fixed-joint-review-') as tmp:
        root = Path(tmp)
        for name in ('first.zip', 'second.zip'):
            archive.encode(root/name, parts)
        raw = (root/'first.zip').read_bytes()
        if raw != (root/'second.zip').read_bytes():
            raise ValueError('Nondeterministic archive')
        unpack(root/'first.zip', root/'unpacked')
        if not check(run_replay(root/'unpacked')['opendd_fixed_joint_release'], expected):
            raise ValueError('Location replay did not reproduce the readout')
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
    prior.parent.prior.builder.build(OUT, a.compiler.resolve(), PUBLIC/'receipt.json', parent_dir=prior.OUT,
                                     read_inputs=inputs, revise_source=revise)
    package()
