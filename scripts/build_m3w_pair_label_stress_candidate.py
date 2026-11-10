"""Build the equal-coverage figure, paper update and anonymous replay."""

import argparse
from io import BytesIO
import json
from math import ceil
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
from reportlab.lib import colors
from reportlab.pdfgen import canvas

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_backbone_relative_candidate as prior
from scripts import build_m3w_reader_supplement as reader
from scripts.build_m3w_full_grid_candidate import builder
from reproducibility.m3w_pair_label_stress_replay.verify import MEMBERS, replay

OUT = prior.OUT.with_name('evidence_manuscript_v215_pair_label_stress')
PUBLIC = prior.PUBLIC.with_name(OUT.name)
READOUT = PUBLIC.parent/'opendd_pair_label_stress_v1/readout.json'
TEMPLATE = reader.TEMPLATE.with_name('m3w_pair_label_stress_replay')
FIGURE = 'equal_coverage.pdf'
ANCHOR = 'Relative-vector error is not a collision metric.'
CONTRASTS = {
    'eq_neighbors_minus_ego': ('EqMotion', 'Neighbors vs. ego only'),
    'social_neighbors_minus_ego': ('Social-STGCNN', 'Neighbors vs. ego only'),
    'completion_minus_cv': ('Completion-trained', 'Forecast vs. CV'),
}


def inputs():
    raw = READOUT.read_bytes()
    d = json.loads(raw)
    if (not d['all_bounds_cover_known_gain'] or not d['original_2pct'] or d['new_external_targets_opened']
            or d['training'] or d['deployment_changed'] or d['cells'] != 21):
        raise ValueError('Expected the verified OpenDD masking readout')
    data = dict(support=d['support'], results=d['results'],
                draws=np.random.default_rng(3183).integers(7, size=(3000, 7)).tolist())
    replay(data)
    return data, dict(opendd_pair_label_stress=reader.digest(raw))


def revise(text, data):
    if text.count(ANCHOR) != 1:
        raise ValueError('Expected the v214 relative-error paragraph')
    case = data['results']['eq_neighbors_minus_ego']
    masked = case['masked']['3']
    d = masked['first_minus_last']
    p = f"{d['point']:+.3f}"
    lo, hi = (f'{x:+.3f}' for x in d['ci95'])
    first, last = (f"{masked['arms'][arm]['observed']['point']:+.3f}" for arm in ('first', 'last'))
    paragraph = (f"\nOn 3,711 complete queries, hiding the first or last three steps\n"
                 f"gives EqMotion neighbor gains of {first} and {last} at identical\n"
                 f"coverage (paired difference {p}, 95\\% location interval\n"
                 f"[{lo}, {hi}]). With no masking the gain is {case['full']['point']:+.3f}.\n"
                 "The supplement compares all masks against the complete scores.")
    return text.replace(ANCHOR, ANCHOR+paragraph, 1), ('3,711 complete queries', p)


def figure(data):
    colors_by_arm = dict(first=colors.HexColor('#1E5B82'), last=colors.HexColor('#AE572B'))
    ink, faint = colors.HexColor('#202527'), colors.HexColor('#D5D9D9')
    rows = data['results']
    limits = [abs(r['full']['point']) for r in rows.values()]
    limits.extend(abs(x) for r in rows.values() for m in r['masked'].values() for a in ('first', 'last')
                  for x in m['arms'][a]['observed']['ci95'])
    limit = max(1, ceil(max(limits)))
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=(612, 792), invariant=1)
    c.setTitle('Equal coverage, different missing steps')
    c.setAuthor('')
    c.setFillColor(ink)
    c.setFont('Helvetica-Bold', 21)
    c.drawString(42, 743, 'Equal coverage, different missing steps')
    c.setFont('Helvetica', 10)
    c.drawString(42, 722, 'OpenDD development: 3,711 originally complete queries across seven locations.')
    c.drawString(42, 708, 'Forecasts are fixed. Every mask at a given severity hides the same pair-step count.')
    left, right = 192, 560

    def x(v):
        return left+(v+limit)*(right-left)/(2*limit)

    for n, top in ((3, 665), (6, 435)):
        c.setFillColor(ink)
        c.setFont('Helvetica-Bold', 13)
        c.drawString(42, top, f'{n} of 12 steps hidden ({int((12-n)/12*100)}% retained)')
        ys = [top-39, top-85, top-131]
        base = top-157
        for tick in np.linspace(-limit, limit, 5):
            c.setStrokeColor(faint if tick else colors.HexColor('#82898A'))
            c.setLineWidth(.7)
            c.line(x(tick), base, x(tick), ys[0]+17)
            c.setFont('Helvetica', 9)
            c.setFillColor(ink)
            c.drawCentredString(x(tick), base-12, f'{tick:g}')
        for y, (key, labels) in zip(ys, CONTRASTS.items(), strict=True):
            c.setFillColor(ink)
            c.setFont('Helvetica-Bold', 10)
            c.drawString(42, y+3, labels[0])
            c.setFont('Helvetica', 9)
            c.drawString(42, y-11, labels[1])
            row = rows[key]
            for arm, shift in (('first', 10), ('last', -10)):
                z = row['masked'][str(n)]['arms'][arm]['observed']
                lo, hi = z['ci95']
                c.setStrokeColor(colors_by_arm[arm])
                c.setFillColor(colors_by_arm[arm])
                c.setLineWidth(1.1)
                c.line(x(lo), y+shift, x(hi), y+shift)
                c.circle(x(z['point']), y+shift, 3.4, stroke=0, fill=1)
            d = x(row['full']['point'])
            p = c.beginPath()
            p.moveTo(d-4, y)
            p.lineTo(d, y+4)
            p.lineTo(d+4, y)
            p.lineTo(d, y-4)
            p.close()
            c.setFillColor(ink)
            c.drawPath(p, stroke=0, fill=1)
        c.setFont('Helvetica', 9)
        c.drawCentredString((left+right)/2, base-27, 'Relative ADE gain (release coordinates; higher is better)')
    for xx, arm, label in ((42, 'first', 'Hide first steps'), (200, 'last', 'Hide last steps')):
        c.setFillColor(colors_by_arm[arm])
        c.circle(xx+4, 224, 3.4, stroke=0, fill=1)
        c.setFillColor(ink)
        c.setFont('Helvetica', 10)
        c.drawString(xx+14, 220, label)
    c.drawString(355, 220, 'Diamond: complete-label gain')
    c.setFont('Helvetica', 9)
    c.drawString(42, 191, 'Bars show paired 95% location intervals. The diamond uses the original complete labels.')
    c.drawString(42, 176, 'These are controlled masks, not estimates of the dataset\'s natural missingness process.')
    c.showPage()
    c.save()
    return buf.getvalue()


def package():
    data, hashes = inputs()
    main = json.loads((PUBLIC/'receipt.json').read_bytes())
    parent = json.loads((prior.PUBLIC/'review_receipt.json').read_bytes())
    pdf = main['artifacts']['manuscript.pdf']
    if main['source_readout_sha256'] != hashes or main['prior_manuscript_sha256'] != parent['companion_pdf_sha256']:
        raise ValueError('Main and supplement do not bind the same source')
    reader.checked(OUT/'manuscript.pdf', pdf)
    old = reader.checked(reader.ADMIN/'review_supplement_v48.zip', parent['archive_sha256'])
    supplement = reader.checked(reader.ADMIN/'review_supplement_v44.pdf', parent['reader_pdf_sha256'])
    chart = figure(data)
    parts = {k: (TEMPLATE/k).read_bytes() for k in ('README.md', 'verify.py')}
    parts.update({'supplement.pdf': supplement, FIGURE: chart, 'pair_label_stress.json': reader.as_json(data),
                  'prior_evidence.zip': old})
    if set(parts) != MEMBERS or reader.PRIVATE.search(parts['pair_label_stress.json'].decode()):
        raise ValueError('Unexpected or identifying replay contents')
    parts['manifest.json'] = reader.as_json({k: reader.digest(v) for k, v in parts.items()})
    payload = reader.archive_bytes(parts)
    with TemporaryDirectory(prefix='v215-replay-') as tmp:
        p = Path(tmp)/'review.zip'
        p.write_bytes(payload)
        reader.unpack(p, Path(tmp)/'replay')
        if reader.run_replay(Path(tmp)/'replay')['opendd_pair_label_stress'] != replay(data):
            raise ValueError('Anonymous replay differs from the registered masking study')
    receipt = dict(companion_pdf_sha256=pdf, reader_pdf_sha256=parent['reader_pdf_sha256'], reader_pdf_reused='v44',
                   figure_sha256=reader.digest(chart), archive_sha256=reader.digest(payload), archive_bytes=len(payload),
                   prior_archive_sha256=parent['archive_sha256'], source_readout_sha256=hashes,
                   standard_library_replay=True, includes_raw_labels=False, includes_checkpoints=False,
                   original_2pct=True, uploaded=False, submission_ready=False)
    for path, raw in ((OUT/FIGURE, chart), (reader.ADMIN/'review_supplement_v49.zip', payload),
                      (PUBLIC/'review_receipt.json', reader.as_json(receipt))):
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
