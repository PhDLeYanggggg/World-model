"""Add the OpenDD relative-trajectory comparison to the anonymous manuscript."""

import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_fixed_spend_uniform_candidate as prior
from scripts import build_m3w_reader_supplement as reader
from scripts.build_m3w_full_grid_candidate import builder
from reproducibility.m3w_backbone_relative_replay.verify import MEMBERS, replay

OUT = prior.OUT.with_name('evidence_manuscript_v214_backbone_relative')
PUBLIC = prior.PUBLIC.with_name(OUT.name)
READOUT = PUBLIC.parent/'opendd_backbone_relative_v1/readout.json'
TEMPLATE = reader.TEMPLATE.with_name('m3w_backbone_relative_replay')


def inputs():
    raw = READOUT.read_bytes()
    d = json.loads(raw)
    if (not d['original_2pct_unchanged'] or not d['original_absolute_scores_reproduced']
            or d['new_external_outcomes_opened'] or d['training'] or d['deployment_changed'] or d['cells'] != 21):
        raise ValueError('Expected the frozen OpenDD development readout')
    groups = [[d['by_site_seed'][f'{s}_{r}'] for r in d['seeds']] for s in d['sites']]
    data = dict(**{k: d[k] for k in ('absolute', 'paired', 'support')}, locations=groups,
                draws=np.random.default_rng(3183).integers(7, size=(3000, 7)).tolist())
    replay(data)
    return data, dict(opendd_backbone_relative=reader.digest(raw))


TABLE_HEAD = r'''\centering\small
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lrrrrr@{}}
\toprule
Forecast & ADE & FDE & P95 & CV gain & Risk \\'''
TABLE_ROWS = {
    'cv': 'CV & 5.161 & 15.183 & 12.497 & ref. & --',
    'ols4': 'OLS4 & 6.124 & 16.645 & 14.579 & -27.68 & 0',
    'completion': 'Completion-trained & 4.104 & 13.715 & 9.720 & +11.54 & 14',
    'comoving_peers': r'EqMotion~\cite{eqmotion} & 4.510 & 13.914 & 10.087 & +6.19 & 5',
    'social_peers': r'Social-STGCNN~\cite{socialstgcnn} & 4.701 & 14.620 & 11.292 & +2.33 & 3',
}
CAPTION = 'use available labels in release coordinates. CV gain is conservative,'
PARAGRAPH = 'adaptations, not comparisons of the original benchmark scores.'
SCOPE = 'Neither graph model exceeds the\ncompletion-trained forecaster on these measures.'


def revise(text, data):
    if any(text.count(k) != 1 for k in (TABLE_HEAD, *TABLE_ROWS.values(), CAPTION, PARAGRAPH, SCOPE)):
        raise ValueError('Expected the original matched OpenDD backbone table')
    text = text.replace(TABLE_HEAD, TABLE_HEAD.replace(r'\small', r'\footnotesize')
                        .replace('3pt', '2.5pt').replace('lrrrrr', 'lrrrrrr')
                        .replace('P95 &', 'P95 & Rel. &'), 1)
    markers = []
    for key, old in TABLE_ROWS.items():
        value = f"{data['absolute'][key]['relative_ade']['point']:.3f}"
        fields = old.split(' & ')
        fields.insert(4, value)
        text = text.replace(old, ' & '.join(fields), 1)
        markers.append(value)
    text = text.replace(CAPTION, 'use available labels in release coordinates. Rel. is the ADE of\n'
                        'relative-position vectors on common labeled steps. Graph rows use\n'
                        'neighbors. CV gain is conservative,', 1)
    text = text.replace(SCOPE, 'Neither neighbor-input adaptation exceeds the\n'
                        'completion-trained forecaster on these measures.', 1)
    rows = [data['paired'][k] for k in ('eq_neighbors_minus_ego', 'social_neighbors_minus_ego')]
    a, b = (f"{r['observed']['point']:+.3f}" for r in rows)
    lo, hi = (f"[{r['lower']['point']:+.3f}, {r['upper']['point']:+.3f}]" for r in rows)
    q = data['support']['observed_queries']
    best = data['absolute']['comoving_ego']['relative_ade']['point']
    result = (f'\nOn {q:,} queries with a labeled pair-step, the observed relative-ADE\n'
              f'gain from neighbors is {a} for EqMotion and {b} for Social-STGCNN.\n'
              f'The missing-step envelopes for those gains are {lo} and {hi}\n'
              'on the full pair roster, in release coordinates. A positive gain means\n'
              f'smaller error. Ego-only EqMotion has the lowest available relative ADE ({best:.3f}).\n'
              'Relative-vector error is not a collision metric.')
    text = text.replace(PARAGRAPH, PARAGRAPH+result, 1)
    return text, (*markers, f'{q:,}', 'Rel.')


def package():
    data, hashes = inputs()
    main = json.loads((PUBLIC/'receipt.json').read_bytes())
    parent = json.loads((prior.PUBLIC/'review_receipt.json').read_bytes())
    pdf = main['artifacts']['manuscript.pdf']
    if main['source_readout_sha256'] != hashes or main['prior_manuscript_sha256'] != parent['companion_pdf_sha256']:
        raise ValueError('Main and supplement do not bind the same parent')
    reader.checked(OUT/'manuscript.pdf', pdf)
    old = reader.checked(reader.ADMIN/'review_supplement_v47.zip', parent['archive_sha256'])
    supplement = reader.checked(reader.ADMIN/'review_supplement_v44.pdf', parent['reader_pdf_sha256'])
    parts = {k: (TEMPLATE/k).read_bytes() for k in ('README.md', 'verify.py')}
    parts.update({'supplement.pdf': supplement, 'backbone_relative.json': reader.as_json(data), 'prior_evidence.zip': old})
    if set(parts) != MEMBERS or reader.PRIVATE.search(parts['backbone_relative.json'].decode()):
        raise ValueError('Unexpected or identifying archive contents')
    parts['manifest.json'] = reader.as_json({k: reader.digest(v) for k, v in parts.items()})
    payload = reader.archive_bytes(parts)
    with TemporaryDirectory(prefix='v214-replay-') as tmp:
        p = Path(tmp)/'review.zip'
        p.write_bytes(payload)
        reader.unpack(p, Path(tmp)/'replay')
        if reader.run_replay(Path(tmp)/'replay')['opendd_backbone_relative'] != replay(data):
            raise ValueError('Anonymous replay differs from the vehicle readout')
    receipt = dict(companion_pdf_sha256=pdf, reader_pdf_sha256=parent['reader_pdf_sha256'], reader_pdf_reused='v44',
                   archive_sha256=reader.digest(payload), archive_bytes=len(payload), prior_archive_sha256=parent['archive_sha256'],
                   source_readout_sha256=hashes, standard_library_replay=True, includes_raw_labels=False,
                   includes_checkpoints=False, original_2pct=True, uploaded=False, submission_ready=False)
    for path, raw in ((reader.ADMIN/'review_supplement_v48.zip', payload), (PUBLIC/'review_receipt.json', reader.as_json(receipt))):
        if path.exists() and path.read_bytes() != raw:
            raise FileExistsError('Refusing to overwrite a frozen review artifact')
        if not path.exists():
            with path.open('xb') as f:
                f.write(raw)
    print(receipt)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compiler', required=True, type=Path)
    args = p.parse_args()
    builder.build(OUT, args.compiler.resolve(), PUBLIC/'receipt.json', parent_dir=prior.OUT,
                  read_inputs=inputs, revise_source=revise)
    package()
