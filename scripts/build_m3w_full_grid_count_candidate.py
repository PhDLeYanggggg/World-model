"""Build v212 and its anonymous fixed-spend replay."""

import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_exchange_exposition_candidate as prior
from scripts import build_m3w_reader_supplement as reader
from reproducibility.m3w_full_grid_count_replay.verify import FIELDS, MEMBERS, replay

OUT = prior.OUT.with_name('evidence_manuscript_v212_count_grid')
PUBLIC = prior.PUBLIC.with_name(OUT.name)
READOUT = PUBLIC.parent/'opendd_full_grid_count_motion_v1/readout.json'
TEMPLATE = reader.TEMPLATE.with_name('m3w_full_grid_count_replay')

OLD_RESULT = r'''Capping the sum of edited agents' peak displacements from CV at the
online total leaves 210/606 (equal-location difference
+14.68 pp, 95\% location bootstrap
[+5.69, +24.44]).
Under both constraints, maximizing the frozen count-logit increments
without labels yields 60/606 (+0.39 pp;
95\% interval [0.00, +1.03]). This
comparison uses the complete score sequence, not an online policy.'''


def inputs():
    raw = READOUT.read_bytes()
    data = json.loads(raw)
    if (not data['hindsight_only'] or not data['original_2pct'] or data['new_external_outcome_access']
            or data['training'] or data['deployment_changed'] or data['summary']['original_unresolved_after_search']):
        raise ValueError('Expected the verified exposed-data count analysis')
    subset = lambda row: {k: row.get(k, 0) for k in FIELDS}
    evidence = dict(counts=subset(data['summary']), locations=[list(map(subset, v)) for v in data['by_location'].values()],
                    contrasts=data['contrasts'], draws=np.random.default_rng(817).integers(7, size=(3000, 7)).tolist())
    replay(evidence)
    return evidence, dict(opendd_count_grid=reader.digest(raw))


def revise(text, d):
    method = 'witness with the original scorer. Separately, the arbitrary-path'
    controls = 'at the same caps. None beats the frozen scores.'
    if any(text.count(anchor) != 1 for anchor in (OLD_RESULT, method, controls)):
        raise ValueError('Expected the v211 fixed-budget experiment and solver description')
    n = d['counts']
    a = d['contrasts']['full_grid_minus_blind_full_grid']
    lo, hi = a['ci95_pp']
    update = (f"Capping the sum of peak displacements at the online total leaves\n"
              f"{n['original_verified_after_search']}/606 feasible views under the original rule and\n"
              f"{n['full_grid']}/606 under full-grid risk. Enumeration resolves two cases\n"
              f"the earlier solver left uncertain. The frozen score allocation\n"
              f"obtains {n['blind_full_grid']}/606 with full-grid risk (hindsight gap\n"
              f"{a['point_pp']:+.2f} pp, 95\\% location interval [{lo:+.2f}, {hi:+.2f}]).\n"
              "The score allocation uses the whole score sequence, not outcomes.")
    text = text.replace(OLD_RESULT, update, 1)
    text = text.replace(controls, 'at the same caps. The frozen scores give 60 original-rule passes.', 1)
    text = text.replace(method, r'''witness with the original scorer. Each nominated prefix also has
fixed risk excesses and a count of mandatory easy edits, so
Eq.~\ref{eq:full-grid-risk} adds linear constraints to the same
search. Separately, the arbitrary-path''', 1)
    return text, ('212/606', '210/606', '55/606', '24.37')


def package():
    data, hashes = inputs()
    main = json.loads((PUBLIC/'receipt.json').read_bytes())
    parent = json.loads((prior.PUBLIC/'review_receipt.json').read_bytes())
    pdf = main['artifacts']['manuscript.pdf']
    if main['source_readout_sha256'] != hashes or main['prior_manuscript_sha256'] != parent['companion_pdf_sha256']:
        raise ValueError('Main and supplemental evidence do not match')
    reader.checked(OUT/'manuscript.pdf', pdf)
    old = reader.checked(reader.ADMIN/'review_supplement_v45.zip', parent['archive_sha256'])
    supplement = reader.checked(reader.ADMIN/'review_supplement_v44.pdf', parent['reader_pdf_sha256'])
    parts = {k: (TEMPLATE/k).read_bytes() for k in ('README.md', 'verify.py')}
    parts.update({'supplement.pdf': supplement, 'count_motion.json': reader.as_json(data), 'prior_evidence.zip': old})
    if set(parts) != MEMBERS or reader.PRIVATE.search(parts['count_motion.json'].decode()):
        raise ValueError('Unexpected archive members or identifying content')
    parts['manifest.json'] = reader.as_json({k: reader.digest(v) for k, v in parts.items()})
    payload = reader.archive_bytes(parts)
    with TemporaryDirectory(prefix='v212-replay-') as tmp:
        p = Path(tmp)/'review.zip'
        p.write_bytes(payload)
        reader.unpack(p, Path(tmp)/'replay')
        if reader.run_replay(Path(tmp)/'replay')['opendd_full_grid_count_motion'] != replay(data):
            raise ValueError('Anonymous replay differs from the count study')
    receipt = dict(companion_pdf_sha256=pdf, reader_pdf_sha256=parent['reader_pdf_sha256'], reader_pdf_reused='v44',
                   archive_sha256=reader.digest(payload), archive_bytes=len(payload), prior_archive_sha256=parent['archive_sha256'],
                   source_readout_sha256=hashes, standard_library_replay=True, includes_raw_labels=False,
                   includes_checkpoints=False, original_2pct=True, uploaded=False, submission_ready=False)
    for p, raw in ((reader.ADMIN/'review_supplement_v46.zip', payload), (PUBLIC/'review_receipt.json', reader.as_json(receipt))):
        if p.exists() and p.read_bytes() != raw:
            raise FileExistsError('Refusing to overwrite a frozen review artifact')
        if not p.exists():
            with p.open('xb') as f:
                f.write(raw)
    print(receipt)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compiler', type=Path, required=True)
    args = p.parse_args()
    prior.builder.build(OUT, args.compiler.resolve(), PUBLIC/'receipt.json', parent_dir=prior.OUT,
                        read_inputs=inputs, revise_source=revise)
    package()
