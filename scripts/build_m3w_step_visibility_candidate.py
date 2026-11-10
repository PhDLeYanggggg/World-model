"""Build and verify the v206 step-observability ablation."""

import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reproducibility.focused_review_supplement.verify import run_replay, unpack
from reproducibility.opendd_step_visibility_supplement import verify
from scripts import build_m3w_fixed_joint_budget_candidate as prior
from scripts.build_m3w_exact_exchange_review_supplement import check

archive, bundle, main, builder = prior.archive, prior.bundle, prior.main, prior.builder
OUT = prior.OUT.parent/'evidence_manuscript_v206_step_visibility'
PUBLIC = main.PUBLIC/OUT.name
STUDY = main.PUBLIC/'opendd_step_visibility_rank_v1'
TEMPLATE = archive.ROOT/'reproducibility/opendd_step_visibility_supplement'
DEST = bundle.LOCAL/'review_supplement_v38.zip'


def inputs():
    raw = {k: (STUDY/f'{k}.json').read_bytes() for k in ('registration', 'freeze', 'readout')}
    raw['protocol'] = (STUDY/'protocol.md').read_bytes()
    h = {k: main.digest(v) for k, v in raw.items()}
    f, r, reg = (json.loads(raw[k]) for k in ('freeze', 'readout', 'registration'))
    if (f['registration'] != h['registration'] or r['registration'] != h['registration']
            or f['held_outcomes_loaded'] or r['freeze_sha256'] != h['freeze']
            or (r['training_fits'], r['optimizer_updates'], r['directions']) != (63, 12600, 126)
            or (f['fits'], f['optimizer_updates']) != (63, 12600) or not r['original_2pct']
            or any(r[k] for k in ('external_outcomes_read', 'independent_confirmation', 'deployment_changed'))
            or reg['inputs'][str((STUDY/'protocol.md').relative_to(archive.ROOT))] != h['protocol']):
        raise ValueError('Require the frozen step-observability study')
    return r, h


def revise(text, r):
    anchor = 'therefore does not match the label coverage of that movement.\n'
    if text.count(anchor) != 1:
        raise ValueError('Expected the v205 matched movement paragraph')
    c = r['summary']['contrasts']['learned_minus_profile']
    ade = c['ade_lower_pp']
    interval = prior.prior.interval
    p = rf'''
\paragraph{{Predict which forecast steps will be labeled.}}
An eight-step history head reduces missing-label Brier score by
{abs(c['brier_delta']['point']):.5f} and displacement-weighted debit MAE by
{abs(c['debit_mae_delta']['point']):.5f} against training-site step averages.
Nevertheless, penalizing the frozen score by predicted debit loses
{abs(ade['point']):.6f} pp of conservative trajectory gain versus the same
penalty using step averages (95\% location interval {interval(ade, 6)}).
Both selection counts and movement budgets are matched. A better
prediction of missing steps did not produce better replacements.
'''
    return text.replace(anchor, anchor+p, 1), (f'{abs(ade["point"]):.6f}', f'{abs(c["brier_delta"]["point"]):.5f}')


def package():
    r, h = inputs()
    paper = bundle.read_receipt(OUT.name)
    prev = json.loads((prior.PUBLIC/'review_receipt.json').read_bytes())
    pdf = paper['artifacts']['manuscript.pdf']
    if paper['source_readout_sha256'] != h or paper['prior_manuscript_sha256'] != prev['companion_pdf_sha256']:
        raise ValueError('Paper and supplement do not match')
    bundle.checked(OUT/'manuscript.pdf', pdf)
    d = dict(summary=r['summary'], directions=r['directions'], seeds=r['seeds'], fits=r['training_fits'],
             draws=np.random.default_rng(12773).integers(7, size=(3000, 7)).tolist())
    parts = {k: (TEMPLATE/k).read_bytes() for k in ('README.md', 'verify.py')}
    parts['protocol.md'] = bundle.checked(STUDY/'protocol.md', h['protocol'])
    parts['visibility.json'] = main.as_json(d)
    if any(archive.PRIVATE_MARKERS.search(v) for v in parts.values()):
        raise ValueError('Identifying addendum content')
    parts['prior_evidence.zip'] = bundle.checked(prior.DEST, prev['archive_sha256'])
    if set(parts) != verify.MEMBERS:
        raise ValueError('Unexpected review members')
    parts['manifest.json'] = main.as_json({k: main.digest(v) for k, v in parts.items()})
    expected = {**r['summary'], 'contrasts': {a: {k: {m: v[m] for m in ('point', 'ci95')}
        for k, v in vs.items()} for a, vs in r['summary']['contrasts'].items()}}
    with TemporaryDirectory(prefix='step-visibility-review-') as tmp:
        root = Path(tmp)
        for n in ('first.zip', 'second.zip'):
            archive.encode(root/n, parts)
        raw = (root/'first.zip').read_bytes()
        if raw != (root/'second.zip').read_bytes():
            raise ValueError('Nondeterministic review archive')
        unpack(root/'first.zip', root/'unpacked')
        if not check(run_replay(root/'unpacked')['opendd_step_visibility'], expected):
            raise ValueError('Anonymous visibility replay differs from the original readout')
    receipt = dict(archive_sha256=main.digest(raw), archive_bytes=len(raw), companion_pdf_sha256=pdf,
        prior_bundle_sha256=prev['archive_sha256'], source_readout_sha256=h['readout'],
        standard_library_replay=True, includes_raw_labels=False, uploaded=False, submission_ready=False)
    for p, v in ((DEST, raw), (PUBLIC/'review_receipt.json', main.as_json(receipt))):
        if p.exists() and p.read_bytes() != v:
            raise FileExistsError('Refusing to replace a different frozen candidate')
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
