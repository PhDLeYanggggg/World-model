"""Build the WEPDTOF history attribution and anonymous scalar replay."""

import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reproducibility.focused_review_supplement.verify import run_replay, unpack
from reproducibility.wepdtof_history_factorial_supplement import verify
from scripts import build_m3w_easy_reference_release_candidate as prior
from scripts.build_m3w_exact_exchange_review_supplement import check

builder, archive, bundle, main = prior.builder, prior.archive, prior.bundle, prior.main
OUT = prior.OUT.parent/'evidence_manuscript_v200_wepdtof_history'
RECEIPT = main.PUBLIC/OUT.name/'receipt.json'
STUDY = main.PUBLIC/'wepdtof_history_factorial_v1'
PACK = main.PUBLIC/'wepdtof_history_review_supplement_v1'
DEST = bundle.LOCAL/'review_supplement_v31.zip'
TEMPLATE = archive.ROOT/'reproducibility/wepdtof_history_factorial_supplement'
ORIGINAL = main.PUBLIC/'wepdtof_external_transfer_v1'


def inputs():
    files = {k: (STUDY/k).read_bytes() for k in ('protocol.md', 'readout.json')}
    r = json.loads(files['readout.json'])
    hashes = {k: main.digest(v) for k, v in files.items()}
    orig = main.digest((ORIGINAL/'external_readout.json').read_bytes())
    if (r['binding']['old_readout'] != orig or r['binding']['protocol'] != hashes['protocol.md']
            or r['independent_confirmation'] or r['reserved_outcomes_read'] or r['deployment_changed']
            or not r['original_2pct'] or r['units'] != 'image_pixels' or len(r['by_clip']) != 15):
        raise ValueError('Wrong retrospective WEPDTOF evidence')
    return r, hashes


def revise(source, data):
    marker = r'\paragraph{WEPDTOF frozen evaluation.}'
    tail = 'resolve the sign, and the optimistic endpoint is not a risk pass.'
    if source.count(marker) != 1 or source.count(tail) != 1:
        raise ValueError('Missing WEPDTOF paragraph')
    s = data['summary']
    n = s['rows']['scored_histories']
    a, b = (s['contributions'][m]['total']['point'] for m in ('cv', 'ols4'))
    text = rf'''{marker}
Matched joint yields \WepExtFixedDual{{}}/\WepExtViews{{}} dual passes.
Its lower-bound gain over independent is \WepExtFixedIndependent{{}} pp
(\WepExtMatchedSources{{}}/\WepExtSources{{}} sources); completion bounds
[\WepCompletionLower{{}}, \WepCompletionUpper{{}}] pp leave the sign unresolved.
Retrospectively, annotation histories lower CV and OLS4 ADE by
{a:.2f} and {b:.2f} pixels on {n} matched track-windows.
Refitting the OLS4 intercept from detections does not reliably help.'''
    return (source[:source.index(marker)]+text+source[source.index(tail)+len(tail):],
            (f'{a:.2f} and {b:.2f} pixels', f'{n} matched'))


def package():
    data, hashes = inputs()
    parent, draft = bundle.read_receipt(prior.PACK.name), bundle.read_receipt(OUT.name)
    pdf = draft['artifacts']['manuscript.pdf']
    if draft['source_readout_sha256'] != hashes or draft['prior_manuscript_sha256'] != parent['companion_pdf_sha256']:
        raise ValueError('Inconsistent paper and supplement inputs')
    bundle.checked(OUT/'manuscript.pdf', pdf)
    rows, s = data['by_clip'], data['summary']
    sites = sorted({r['source'] for r in rows.values()})
    arms = ('detector', 'annotation_anchor', 'annotation_velocity', 'annotation_both')
    clips = []
    for r in rows.values():
        p = dict(source=sites.index(r['source']), roster=r['roster'], stable_histories=r['stable_histories'],
                 risk_pass=r['refit_risk']['supported'])
        p.update({m: [r['factorial'][m][a]['ade'] for a in arms] for m in ('cv', 'ols4')})
        p['refit_gains'] = {k: r['full_roster']['ols4'][k]-r['full_roster']['ols4_refit'][k] for k in ('ade', 'fde', 'p95')}
        clips.append(p)
    draws = np.random.default_rng(82317).integers(13, size=(3000, 13)).tolist()
    def estimate(v):
        return {k: v[k] for k in ('point', 'ci95')}
    expected = {m: {k: estimate(v) for k, v in s['contributions'][m].items()} for m in ('cv', 'ols4')}
    expected.update(gap=estimate(s['detector_minus_annotation_velocity_gap']),
                    refit={k: estimate(v) for k, v in s['refit_gain_vs_ols4'].items()},
                    risk_passes=s['refit_risk']['supported'])
    parts = {k: (TEMPLATE/k).read_bytes() for k in ('README.md', 'verify.py')}
    parts['protocol.md'] = bundle.checked(STUDY/'protocol.md', hashes['protocol.md'])
    parts['clips.json'] = main.as_json(dict(clips=clips, draws=draws))
    if any(archive.PRIVATE_MARKERS.search(p) for p in parts.values()):
        raise ValueError('Identifying review material')
    parts['prior_evidence.zip'] = bundle.checked(prior.DEST, parent['archive_sha256'])
    if set(parts) != verify.MEMBERS:
        raise ValueError('Unexpected supplement contents')
    parts['manifest.json'] = main.as_json({k: main.digest(v) for k, v in parts.items()})
    with TemporaryDirectory(prefix='wepdtof-history-review-') as tmp:
        root = Path(tmp)
        for n in ('first.zip', 'second.zip'):
            archive.encode(root/n, parts)
        raw = (root/'first.zip').read_bytes()
        if raw != (root/'second.zip').read_bytes():
            raise ValueError('Nondeterministic review archive')
        unpack(root/'first.zip', root/'unpacked')
        if not check(run_replay(root/'unpacked')['wepdtof_history_factorial'], expected):
            raise ValueError('History factorial replay does not match the real readout')
    row = dict(archive_sha256=main.digest(raw), archive_bytes=len(raw), prior_bundle_sha256=parent['archive_sha256'],
               companion_pdf_sha256=pdf, source_readout_sha256=hashes['readout.json'], standard_library_replay=True,
               includes_raw_labels=False, includes_checkpoints=False, uploaded=False, submission_ready=False)
    for p, content in {DEST: raw, PACK/'receipt.json': main.as_json(row)}.items():
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists() and p.read_bytes() != content:
            raise FileExistsError('Refusing to replace a frozen review artifact')
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
