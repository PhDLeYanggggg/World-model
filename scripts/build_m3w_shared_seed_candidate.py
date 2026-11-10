"""Build v208 with the shared-seed robustness check and anonymous replay."""

import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reproducibility.focused_review_supplement.verify import run_replay, unpack
from reproducibility.songdo_shared_seed_supplement import verify
from scripts import build_m3w_completion_signs_candidate as prior
from src.evaluation import m3w_completion_contrasts as stats

archive, bundle, main, builder = prior.archive, prior.bundle, prior.main, prior.builder
OUT = prior.OUT.parent/'evidence_manuscript_v208_shared_seed'
PUBLIC = main.PUBLIC/OUT.name
STUDY = main.PUBLIC/'songdo_shared_seed_completion_v1'
TEMPLATE = archive.ROOT/'reproducibility/songdo_shared_seed_supplement'
DEST = bundle.LOCAL/'review_supplement_v40.zip'


def inputs():
    files = dict(readout=STUDY/'readout.json', protocol=STUDY/'protocol.md',
                 original=main.PUBLIC/'songdo_calibrated_20261110/confirmation_receipt.json')
    raw = {k: v.read_bytes() for k, v in files.items()}
    r, orig = (json.loads(raw[k]) for k in ('readout', 'original'))
    b = r['binding']
    if (not r['original_2pct_unchanged'] or (r['flights'], r['components'], r['bootstrap_draws']) != (160, 14, 3000)
            or any(r[k] for k in ('numeric_future_coordinates_read', 'calibration_outcomes_read',
                                  'new_cohort_opened', 'training', 'deployment_changed'))
            or any(b[k] != orig['binding'][k] for k in ('registration_sha256', 'release_freeze_sha256',
                                                       'primary_score_sha256', 'flight_score_hash_root'))):
        raise ValueError('Require the original matched Songdo confirmation scores')
    return r, {k: main.digest(v) for k, v in raw.items()}


def revise(text, r):
    anchor = 'but the bootstrap interval of the upper endpoint still includes zero.'
    proof = 'vary independently: set each unknown target to the treatment forecast'
    if text.count(anchor) != 1 or text.count(proof) != 1:
        raise ValueError('Expected the original two-sided completion discussion')
    v = r['comparisons']['independent']
    upper = v['endpoints']['shared_upper']
    n = v['tighter_steps']
    a, b = (1000*x for x in upper['ci95'])
    p = rf'''
Requiring one missing target across seeds tightens {n} affected steps.
The shared upper endpoint is \({1000*upper['point']:+.6f}\)
[{a:+.3f}, {b:+.3f}] in \(10^{{-3}}\) pp. The sample sign and
source-level uncertainty are unchanged.'''
    text = text.replace(anchor, anchor+p, 1)
    text = text.replace(proof, 'vary by scored occurrence: set each unknown target to the treatment forecast', 1)
    return text, (f'{abs(1000*upper["point"]):.6f}', 'affected steps')


def package():
    r, h = inputs()
    paper = bundle.read_receipt(OUT.name)
    parent = json.loads((prior.PUBLIC/'review_receipt.json').read_bytes())
    pdf = paper['artifacts']['manuscript.pdf']
    if paper['source_readout_sha256'] != h or paper['prior_manuscript_sha256'] != parent['companion_pdf_sha256']:
        raise ValueError('Paper and source evidence mismatch')
    bundle.checked(OUT/'manuscript.pdf', pdf)
    data = dict(flights=r['flights'], seeds=r['seeds'], points={a: v['source_means'] for a, v in r['comparisons'].items()},
                draws=np.random.default_rng(1439).integers(14, size=(3000, 14)).tolist())
    parts = {k: (TEMPLATE/k).read_bytes() for k in ('README.md', 'verify.py')}
    parts.update({'protocol.md': bundle.checked(STUDY/'protocol.md', h['protocol']),
                  'completion.py': Path(stats.__file__).read_bytes(), 'shared.json': main.as_json(data)})
    if any(archive.PRIVATE_MARKERS.search(v) for v in parts.values()):
        raise ValueError('Identifying content in anonymous addendum')
    parts['prior_evidence.zip'] = bundle.checked(prior.DEST, parent['archive_sha256'])
    if set(parts) != verify.MEMBERS:
        raise ValueError('Unexpected supplement members')
    parts['manifest.json'] = main.as_json({k: main.digest(v) for k, v in parts.items()})
    with TemporaryDirectory(prefix='songdo-shared-seed-review-') as tmp:
        root = Path(tmp)
        for n in ('first.zip', 'second.zip'):
            archive.encode(root/n, parts)
        raw = (root/'first.zip').read_bytes()
        if raw != (root/'second.zip').read_bytes():
            raise ValueError('Nondeterministic review archive')
        unpack(root/'first.zip', root/'unpacked')
        if run_replay(root/'unpacked')['songdo_shared_seed'] != {a: v['endpoints'] for a, v in r['comparisons'].items()}:
            raise ValueError('Anonymous shared-seed replay differs from the original readout')
    receipt = dict(archive_sha256=main.digest(raw), archive_bytes=len(raw), companion_pdf_sha256=pdf,
        prior_bundle_sha256=parent['archive_sha256'], source_readout_sha256=h['readout'],
        standard_library_replay=True, includes_raw_labels=False, uploaded=False, submission_ready=False)
    for p, v in ((DEST, raw), (PUBLIC/'review_receipt.json', main.as_json(receipt))):
        if p.exists() and p.read_bytes() != v:
            raise FileExistsError('Refusing to replace a frozen review artifact')
        if not p.exists():
            with p.open('xb') as f:
                f.write(v)
    print(receipt)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compiler', type=Path, required=True)
    a = p.parse_args()
    builder.build(OUT, a.compiler.resolve(), PUBLIC/'receipt.json', parent_dir=prior.OUT,
                  read_inputs=inputs, revise_source=revise)
    package()
