"""Build the step-budget paper candidate and an anonymous replay addendum."""

import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reproducibility.focused_review_supplement.verify import run_replay, unpack
from reproducibility.songdo_step_budget_supplement import verify
from scripts import build_m3w_vehicle_stgcnn_review_supplement as prior
from scripts.build_m3w_exact_exchange_review_supplement import check

builder, archive, bundle, main = prior.prior.builder, prior.archive, prior.bundle, prior.main
OUT = prior.prior.OUT.parent/'evidence_manuscript_v201_songdo_step_budget'
RECEIPT = main.PUBLIC/OUT.name/'receipt.json'
STUDY = main.PUBLIC/'songdo_step_budget_v1'
PACK = main.PUBLIC/'songdo_step_budget_review_supplement_v1'
DEST = bundle.LOCAL/'review_supplement_v33.zip'
TEMPLATE = archive.ROOT/'reproducibility/songdo_step_budget_supplement'


def inputs():
    raw = {k: (STUDY/k).read_bytes() for k in ('protocol.md', 'freeze.json', 'readout.json')}
    h = {k: main.digest(v) for k, v in raw.items()}
    f, r = (json.loads(raw[k]) for k in ('freeze.json', 'readout.json'))
    if (r['freeze'] != h['freeze.json'] or r['binding'] != f['binding']
            or r['binding']['protocol'] != h['protocol.md'] or f['future_coordinates_read']
            or not r['diagonal_reproduced'] or not r['original_2pct']
            or r['new_cohort_opened'] or r['calibration_outcomes_read'] or r['independent_confirmation']
            or r['deployment_changed'] or r['flights'] != 160):
        raise ValueError('Wrong retrospective step-budget evidence')
    return r, h


def revise(source, data):
    start = r'\paragraph{Hold the vehicles fixed.}'
    end = r'\paragraph{How much does the Songdo menu leave unused?}'
    if source.count(start) != 1 or source.count(end) != 1:
        raise ValueError('Require the original matched-path discussion')
    q = data['variants']['matched_steps']['direct']['lower_pp']
    if q['point'] is None or data['unmatched_views']:
        raise ValueError('The main text assumes the whole paired roster is supported')
    a, b, c = [v*1000. for v in (q['point'], *q['ci95'])]
    retained = data['retained']['movement_retained']['point']*100.
    text = rf'''\paragraph{{Match recipients and displacement.}}
With recipients fixed, joint paths gain \(+0.014\) [-0.045, +0.066]
on independent recipients and \(+0.004\) [-0.002, +0.012] on joint recipients
(\(10^{{-3}}\) pp). Relative-motion lower bounds are negative on both
recipient sets. In a sensitivity check, matching displacement at each
step retains {retained:.1f}\% of the original movement. Joint minus
independent gives {a:+.3f} [{b:+.3f}, {c:+.3f}] in the same units.
The intervals resample sources. Neither check changes release.

'''
    return source[:source.index(start)]+text+source[source.index(end):], (f'{retained:.1f}%', f'{a:+.3f}')


def package():
    data, hashes = inputs()
    parent, draft = bundle.read_receipt(prior.PACK.name), bundle.read_receipt(OUT.name)
    pdf = draft['artifacts']['manuscript.pdf']
    if draft['source_readout_sha256'] != hashes or draft['prior_manuscript_sha256'] != parent['companion_pdf_sha256']:
        raise ValueError('Paper and replay evidence mismatch')
    bundle.checked(OUT/'manuscript.pdf', pdf)
    v = data['variants']
    panel = dict(flights=data['flights'], views={a: x['evaluated_views'] for a, x in v.items()},
        source_means={a: {k: z['source_means'] for k, z in x['direct'].items()} for a, x in v.items()},
        retained={k: z['source_means'] for k, z in data['retained'].items()},
        draws=np.random.default_rng(1439).integers(14, size=(3000, 14)).tolist())
    def summary(v):
        return {k: {p: x[p] for p in ('point', 'ci95')} for k, x in v.items()}
    expected = dict(direct={a: summary(x['direct']) for a, x in v.items()}, retention=summary(data['retained']),
                    matched_views=v['matched_steps']['evaluated_views'])
    parts = {k: (TEMPLATE/k).read_bytes() for k in ('README.md', 'verify.py')}
    parts['protocol.md'] = bundle.checked(STUDY/'protocol.md', hashes['protocol.md'])
    parts['profiles.json'] = main.as_json(panel)
    if any(archive.PRIVATE_MARKERS.search(p) for p in parts.values()):
        raise ValueError('Identifying review material')
    parts['prior_evidence.zip'] = bundle.checked(prior.DEST, parent['archive_sha256'])
    if set(parts) != verify.MEMBERS:
        raise ValueError('Unexpected supplement contents')
    parts['manifest.json'] = main.as_json({k: main.digest(v) for k, v in parts.items()})
    with TemporaryDirectory(prefix='songdo-step-budget-review-') as tmp:
        root = Path(tmp)
        for n in ('first.zip', 'second.zip'):
            archive.encode(root/n, parts)
        raw = (root/'first.zip').read_bytes()
        if raw != (root/'second.zip').read_bytes():
            raise ValueError('Nondeterministic review archive')
        unpack(root/'first.zip', root/'unpacked')
        if not check(run_replay(root/'unpacked')['songdo_step_budget'], expected):
            raise ValueError('Anonymous step-budget replay disagrees with results')
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
    args = p.parse_args()
    builder.build(OUT, args.compiler.resolve(), RECEIPT, parent_dir=prior.prior.OUT,
                  read_inputs=inputs, revise_source=revise)
    package()
