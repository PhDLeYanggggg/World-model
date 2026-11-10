"""Bind decision-control scalars and offline verification to manuscript v177."""

import json
from math import isclose
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reproducibility.focused_review_supplement.verify import run_replay, unpack
from reproducibility.opendd_decision_supplement import verify
from scripts import package_m3w_combined_review_supplement as bundle


main, archive = bundle.main, bundle.archive
OUT = archive.PUBLIC/'decision_review_supplement_v1'
DEST = bundle.LOCAL/'review_supplement_v5.zip'
TEMPLATE = archive.ROOT/'reproducibility/opendd_decision_supplement'
PAIR = 'opendd_equal_spend_preference_v1'
CONDITIONAL = 'opendd_risk_conditional_utility_v1'


def export(pair, conditional):
    if set(pair['by_site']) != set(conditional['by_site']):
        raise ValueError('Learning controls use different held sites')
    groups = []
    for name in sorted(pair['by_site']):
        group = []
        for p, c in zip(pair['by_site'][name], conditional['by_site'][name], strict=True):
            if (p['seed'] != c['seed'] or p['scores']['frozen'] != c['scores']['frozen']
                    or p['scores']['pairwise'] != c['scores']['original_pairwise']):
                raise ValueError('Learning controls use different frozen plans')
            scores = c['scores'] | {'pointwise': p['scores']['pointwise']}
            group.append(dict(views=scores['frozen']['views'], scores={k: {f: s[f] for f in verify.FIELDS} for k, s in scores.items()},
                              paired_conservative_gain_pp=c['effects'][verify.PRIMARY]['conservative_gain_pp']))
        groups.append(group)
    return dict(locations=groups)


def check(readout, pair, conditional):
    arms = conditional['summary']['arms'] | {'pointwise': pair['summary']['pointwise']}
    if readout['arms'] != {k: {f: s[f] for f in verify.FIELDS} for k, s in arms.items()}:
        raise ValueError('Scalar replay does not match the fitted score counts')
    old = conditional['summary']['comparisons'][verify.PRIMARY]
    for k, v in readout['primary'].items():
        if not all(isclose(x, y, abs_tol=1e-11) for x, y in zip(
                [v['point'], *v['interval95']], [old[k]['point_pp'], *old[k]['ci95_pp']], strict=True)):
            raise ValueError('Scalar replay changed a paired interval')


def package():
    old = bundle.read_receipt('allocation_review_supplement_v1')
    draft = bundle.read_receipt('evidence_manuscript_v177_decision_ablations')
    v4 = bundle.checked(bundle.LOCAL/'review_supplement_v4.zip', old['archive_sha256'])
    binding = draft['decision_ablation_readouts_sha256']
    pair, conditional = (json.loads(bundle.checked(archive.PUBLIC/name/'readout.json', binding[name]))
                         for name in (PAIR, CONDITIONAL))
    items = {k: (TEMPLATE/k).read_bytes() for k in ('README.md', 'verify.py')}
    items |= {'allocation_evidence.zip': v4, 'decision_controls.json': main.as_json(export(pair, conditional))}
    if set(items) != verify.MEMBERS or any(archive.PRIVATE_MARKERS.search(v) for k, v in items.items()
                                           if k != 'allocation_evidence.zip'):
        raise ValueError('Identifying or unexpected content in the anonymous supplement')
    items['manifest.json'] = main.as_json({k: main.digest(v) for k, v in items.items()})
    with TemporaryDirectory(prefix='decision-review-') as temp:
        root = Path(temp)
        for path in (root/'first.zip', root/'second.zip'):
            archive.encode(path, items)
        raw = (root/'first.zip').read_bytes()
        if raw != (root/'second.zip').read_bytes():
            raise ValueError('The reviewer archive is not deterministic')
        unpack(root/'first.zip', root/'unpacked')
        replay = run_replay(root/'unpacked')
        check(replay['decision_controls'], pair, conditional)
    receipt = dict(archive_sha256=main.digest(raw), archive_bytes=len(raw), prior_bundle_sha256=old['archive_sha256'],
        companion_pdf_sha256=draft['artifacts']['manuscript.pdf'], readout_sha256=binding, standard_library_replay=True,
        deterministic=True, includes_checkpoints=False, includes_raw_labels=False, uploaded=False, submission_ready=False)
    for p, v in {DEST: raw, OUT/'receipt.json': main.as_json(receipt)}.items():
        if p.exists() and p.read_bytes() != v:
            raise FileExistsError(f'Refusing to replace {p.name}')
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            with p.open('xb') as f:
                f.write(v)
    print(receipt)


if __name__ == '__main__':
    package()
