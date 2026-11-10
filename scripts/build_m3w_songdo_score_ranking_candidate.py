"""Build the paper and anonymous replay for the Songdo score-rank diagnosis."""

import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reproducibility.focused_review_supplement.verify import run_replay, unpack
from reproducibility.songdo_score_ranking_supplement import verify
from scripts import audit_m3w_songdo_score_ranking as study
from scripts import build_m3w_focused_evidence_v202 as parent
from scripts import build_m3w_songdo_failure_case_supplement as cases

archive, bundle, main = cases.archive, cases.bundle, cases.main
OUT = parent.OUT.parent/'evidence_manuscript_v203_songdo_score_ranking'
PUBLIC = main.PUBLIC/OUT.name
DEST = bundle.LOCAL/'review_supplement_v35.zip'
TEMPLATE = archive.ROOT/'reproducibility/songdo_score_ranking_supplement'


def inputs():
    raw = (study.OUT/'readout.json').read_bytes()
    result = json.loads(raw)
    evidence, frozen, binding, scalar = study.archived.verified_archive(study.archived.ARCHIVE)
    if (any(result[k] != v for k, v in binding.items()) or result['new_labels_read']
            or result['threshold_changed'] or result['new_confirmation'] or result['deployment_changed']
            or study.rank.diagnostic(evidence, frozen, study.old) != result['arms']):
        raise ValueError('Expected the verified retrospective ranking result')
    protocol = bundle.checked(study.OUT/'protocol.md', result['protocol_sha256'])
    hashes = {k: main.digest(v) for k, v in dict(readout=raw, protocol=protocol, evidence=scalar).items()}
    return result, hashes, dict(zip(('readout.json', 'protocol.md', 'rank_evidence.json'), (raw, protocol, scalar)))


def revise(text, result):
    anchor = 'deploying the joint rule.\n'
    c = result['arms']['joint']
    if text.count(anchor) != 1 or any(v['released'] or v['first_block'] != {'undefined_only': 1}
            or v['first_block_undefined']['completion_easy_risk_upper'] != 1 for v in c['calibration'].values()):
        raise ValueError('Expected the joint rule to have no admissible calibration prefix')
    rows = c['confirmation']
    delta, risk, undefined = (rows[k]['ranked_minus_uniform'] for k in ('dual', 'measured_violation', 'undefined_risk'))
    a, b = (100*v for v in delta['ci95'])
    q = c['equal_budget_views']
    n = [c['calibration'][str(s)]['risk_supported'] for s in result['seeds']]
    paragraph = rf'''
\paragraph{{Why the joint rule did not release.}}
For each seed, the top-ranked joint calibration proposal lacked a
selected easy reference, although {n[0]}, {n[1]}, and {n[2]} other proposals passed
risk. On the same frozen confirmation proposals, taking the top
quarter within each source and seed selects {q} views. Relative to
uniform choice at the same count, this changes the dual-pass rate by
{100*delta['point']:+.2f} pp (95\% source interval [{a:+.2f}, {b:+.2f}]).
Measured violations fall by {-100*risk['point']:.2f} pp but undefined
risk rises by {100*undefined['point']:.2f} pp. The score avoids some
measured failures without reliably selecting more admissible gains.
This full-list diagnostic does not change the release threshold.
'''
    return text.replace(anchor, anchor+paragraph, 1), (f'{q} views', f'{100*undefined["point"]:.2f} pp')


def package():
    _, hashes, parts = inputs()
    paper = bundle.read_receipt(OUT.name)
    pdf = paper['artifacts']['manuscript.pdf']
    if paper['source_readout_sha256'] != hashes:
        raise ValueError('Paper and supplementary inputs differ')
    bundle.checked(OUT/'manuscript.pdf', pdf)
    prev = json.loads((cases.STUDY/'review_receipt.json').read_bytes())
    parts.update({k: (TEMPLATE/k).read_bytes() for k in ('README.md', 'verify.py')})
    parts['frozen_score_ranking.py'] = Path(study.rank.__file__).read_bytes()
    parts['replay_songdo.py'] = Path(study.old.__file__).read_bytes()
    if any(archive.PRIVATE_MARKERS.search(v) for v in parts.values()):
        raise ValueError('Identifying material in the anonymous addendum')
    parts['prior_evidence.zip'] = bundle.checked(cases.DEST, prev['archive_sha256'])
    if set(parts) != verify.MEMBERS:
        raise ValueError('Unexpected supplement members')
    parts['manifest.json'] = main.as_json({k: main.digest(v) for k, v in parts.items()})
    with TemporaryDirectory(prefix='songdo-score-review-') as tmp:
        root = Path(tmp)
        for n in ('first.zip', 'second.zip'):
            archive.encode(root/n, parts)
        raw = (root/'first.zip').read_bytes()
        if raw != (root/'second.zip').read_bytes():
            raise ValueError('Nondeterministic review archive')
        unpack(root/'first.zip', root/'unpacked')
        replayed = run_replay(root/'unpacked')['songdo_score_ranking']
        if replayed != json.loads(parts['readout.json'])['arms']:
            raise ValueError('Anonymous ranking replay disagrees')
    receipt = dict(archive_sha256=main.digest(raw), archive_bytes=len(raw), companion_pdf_sha256=pdf,
        prior_bundle_sha256=prev['archive_sha256'], source_readout_sha256=hashes['readout'],
        standard_library_replay=True, includes_raw_labels=False, uploaded=False, submission_ready=False)
    for p, v in ((DEST, raw), (PUBLIC/'review_receipt.json', main.as_json(receipt))):
        if p.exists() and p.read_bytes() != v:
            raise FileExistsError('Refusing to replace a frozen candidate artifact')
        if not p.exists():
            with p.open('xb') as f:
                f.write(v)
    print(receipt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler', type=Path, required=True)
    args = parser.parse_args()
    parent.prior.builder.build(OUT, args.compiler.resolve(), PUBLIC/'receipt.json', parent_dir=parent.OUT,
                              read_inputs=lambda: inputs()[:2], revise_source=revise)
    package()
