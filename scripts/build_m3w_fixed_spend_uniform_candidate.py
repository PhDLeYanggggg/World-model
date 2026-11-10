"""Build the manuscript and anonymous replay with the OpenDD uniform control."""

import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_full_grid_count_candidate as prior
from scripts import build_m3w_reader_supplement as reader
from scripts.build_m3w_full_grid_candidate import builder
from reproducibility.m3w_fixed_spend_uniform_replay.verify import MEMBERS, replay

OUT = prior.OUT.with_name('evidence_manuscript_v213_fixed_spend_uniform')
PUBLIC = prior.PUBLIC.with_name(OUT.name)
READOUT = PUBLIC.parent/'opendd_fixed_spend_uniform_v1/readout.json'
TEMPLATE = reader.TEMPLATE.with_name('m3w_fixed_spend_uniform_replay')


def inputs():
    raw = READOUT.read_bytes()
    d = json.loads(raw)
    if not d['original_2pct'] or d['new_external_outcome_access'] or d['training'] or d['deployment_changed']:
        raise ValueError('Expected the exposed-development uniform control')
    evidence = dict(summary=d['summary'], locations=list(d['by_location'].values()), contrasts=d['contrasts'],
                    draws=np.random.default_rng(817).integers(7, size=(3000, 7)).tolist())
    replay(evidence)
    return evidence, dict(opendd_fixed_spend_uniform=reader.digest(raw))


def revise(text, d):
    anchor = 'The score allocation uses the whole score sequence, not outcomes.'
    if text.count(anchor) != 1:
        raise ValueError('Expected v212 fixed-spend exposition')
    s = d['summary']
    lo, hi = d['contrasts']['frozen_minus_uniform']['location_ci95_pp']
    delta = d['contrasts']['frozen_minus_uniform']['point_bounds_pp'][0]
    uniform = f"{s['expected_passes_low']:.2f}"
    paragraph = (f"Uniform legal allocation at this same spend gives {uniform}/606\n"
                 f"expected passes. The frozen allocation has no clear advantage\n"
                 f"({delta:+.2f} pp across locations, 95\\% interval [{lo:+.2f}, {hi:+.2f}]).\n"
                 "We count affordable prefix vectors by suffix credit and remaining\n"
                 "edits; we enumerate or draw uniform ranks and reject plans above\n"
                 "the motion cap. Both controls use the whole recording's score\n"
                 "or eligibility sequence, not outcomes; neither is an online policy.")
    return text.replace(anchor, paragraph, 1), (f'{uniform}/606', f'{abs(lo):.2f}', f'{hi:.2f}', 'no clear advantage')


def package():
    data, hashes = inputs()
    main = json.loads((PUBLIC/'receipt.json').read_bytes())
    parent = json.loads((prior.PUBLIC/'review_receipt.json').read_bytes())
    pdf = main['artifacts']['manuscript.pdf']
    if main['source_readout_sha256'] != hashes or main['prior_manuscript_sha256'] != parent['companion_pdf_sha256']:
        raise ValueError('Main and supplement use different evidence')
    reader.checked(OUT/'manuscript.pdf', pdf)
    old = reader.checked(reader.ADMIN/'review_supplement_v46.zip', parent['archive_sha256'])
    supplement = reader.checked(reader.ADMIN/'review_supplement_v44.pdf', parent['reader_pdf_sha256'])
    parts = {k: (TEMPLATE/k).read_bytes() for k in ('README.md', 'verify.py')}
    parts.update({'supplement.pdf': supplement, 'fixed_spend_uniform.json': reader.as_json(data), 'prior_evidence.zip': old})
    if set(parts) != MEMBERS or reader.PRIVATE.search(parts['fixed_spend_uniform.json'].decode()):
        raise ValueError('Unexpected archive contents')
    parts['manifest.json'] = reader.as_json({k: reader.digest(v) for k, v in parts.items()})
    payload = reader.archive_bytes(parts)
    with TemporaryDirectory(prefix='v213-replay-') as tmp:
        p = Path(tmp)/'review.zip'
        p.write_bytes(payload)
        reader.unpack(p, Path(tmp)/'replay')
        if reader.run_replay(Path(tmp)/'replay')['opendd_fixed_spend_uniform'] != replay(data):
            raise ValueError('Anonymous replay differs from the fixed-spend readout')
    receipt = dict(companion_pdf_sha256=pdf, reader_pdf_sha256=parent['reader_pdf_sha256'], reader_pdf_reused='v44',
                   archive_sha256=reader.digest(payload), archive_bytes=len(payload), prior_archive_sha256=parent['archive_sha256'],
                   source_readout_sha256=hashes, standard_library_replay=True, includes_raw_labels=False,
                   includes_checkpoints=False, original_2pct=True, uploaded=False, submission_ready=False)
    for path, raw in ((reader.ADMIN/'review_supplement_v47.zip', payload), (PUBLIC/'review_receipt.json', reader.as_json(receipt))):
        if path.exists() and path.read_bytes() != raw:
            raise FileExistsError('Refusing to overwrite a frozen review artifact')
        if not path.exists():
            with path.open('xb') as f:
                f.write(raw)
    print(receipt)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compiler', type=Path, required=True)
    args = p.parse_args()
    builder.build(OUT, args.compiler.resolve(), PUBLIC/'receipt.json', parent_dir=prior.OUT,
                  read_inputs=inputs, revise_source=revise)
    package()
