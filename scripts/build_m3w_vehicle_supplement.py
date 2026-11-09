"""Build two standalone evidence pages from frozen OpenDD vehicle studies."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory

from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_RIGHT
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT/'outputs/publication_readiness_2026_09'
LOCAL = ROOT/'data/stage_cvpr2027_experiments/submission_admin'
OUT = PUBLIC/'vehicle_controls_supplement_v1'
SOURCE = {
    'forecast': ('opendd_vehicle_eqmotion_v1', '8352ad67d169350d4383b6a55bbe29d661d8ba90ee48b9ea2af952ca5b2d7b75'),
    'completion': ('opendd_pair_completion_v1', '878a3096d90b7891b1b765a04c036a3b0477be07b563ecccb9cbbf67841264f5'),
    'support': ('opendd_support_rerank_v1', '48ac6dfc70af81f1d71ce8856721e17ca95a8c93258e48dc26c25e1e9503f5eb'),
}
INK, MUTED, RULE, BLUE = (colors.HexColor(x) for x in ('#161d25', '#424c58', '#d1d7dd', '#235772'))
LEFT, WIDTH = 56, 500


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def evidence():
    data = {}
    for name, (directory, pinned) in SOURCE.items():
        p = PUBLIC/directory/'readout.json'
        if digest(p) != pinned:
            raise ValueError('Readout differs from the bound evidence: '+name)
        r = data[name] = json.loads(p.read_text())
        external = r.get('external_outcomes_read', r.get('external_outcomes_opened'))
        if external is not False or r['independent_confirmation'] or not r['original_2pct_unchanged']:
            raise ValueError('Unexpected evidence roles: '+name)
        freeze = r['binding']['freeze'] if name == 'forecast' else r['freeze_sha256']
        if digest(p.with_name('freeze.json')) != freeze:
            raise ValueError('Readout does not bind the frozen choices: '+name)
    f, c, s = (data[k] for k in SOURCE)
    if (len(f['by_site_seed']) != 21 or c['directions'] != 126 or s['directions'] != 126
            or any(x['matched'] != 8874 for r in (c, s) for x in r['summary'].values())):
        raise ValueError('Expected the fixed paired vehicle rosters')
    return data


def para(pdf, text, y, *, size=11, lead=15, color=INK, bold=False, x=LEFT, width=WIDTH, space=9):
    p = Paragraph(text, ParagraphStyle('line', fontName='Helvetica-Bold' if bold else 'Helvetica',
                                     fontSize=size, leading=lead, textColor=color))
    _, h = p.wrap(width, 600)
    p.drawOn(pdf, x, y-h)
    return y-h-space


def table(pdf, data, widths, y):
    styles = [
        ParagraphStyle('first', fontName='Helvetica', fontSize=9.5, leading=12, textColor=INK, alignment=TA_LEFT),
        ParagraphStyle('number', fontName='Helvetica', fontSize=9.5, leading=12, textColor=INK, alignment=TA_RIGHT),
    ]
    rows = [[Paragraph(f'<b>{v}</b>' if i == 0 else str(v), styles[j > 0])
             for j, v in enumerate(row)] for i, row in enumerate(data)]
    t = Table(rows, colWidths=widths, hAlign='LEFT')
    t.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                          ('TOPPADDING', (0, 0), (-1, -1), 5), ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                          ('LEFTPADDING', (0, 0), (-1, -1), 5), ('RIGHTPADDING', (0, 0), (-1, -1), 5),
                          ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#eff2f5')),
                          ('LINEBELOW', (0, 0), (-1, 0), .65, RULE), ('LINEBELOW', (0, -1), (-1, -1), .65, RULE)]))
    _, h = t.wrap(WIDTH, 600)
    t.drawOn(pdf, LEFT, y-h)
    return y-h-10


def ci(x, digits=2):
    return f"{x['mean']:+.{digits}f} [{x['ci95'][0]:+.{digits}f}, {x['ci95'][1]:+.{digits}f}]"


def header(pdf, kicker, title):
    y = para(pdf, kicker, 740, size=9, lead=10, bold=True, color=BLUE, space=9)
    return para(pdf, title, y, size=22, lead=25, bold=True, space=14)


def footer(pdf, page, y):
    if y < 61:
        raise ValueError('Content runs into the footer')
    pdf.setStrokeColor(RULE)
    pdf.line(LEFT, 49, LEFT+WIDTH, 49)
    pdf.setFont('Helvetica', 9)
    pdf.setFillColor(MUTED)
    pdf.drawString(LEFT, 34, 'Vehicle controls and selection ablations')
    pdf.drawRightString(LEFT+WIDTH, 34, str(page))
    pdf.showPage()


def forest(pdf, rows, y):
    x0, x1 = 245, 551
    ticks = [-20, -15, -10, -5, 0]
    def x(v):
        if not -22 <= v <= 2:
            raise ValueError('Forecast contrast outside the figure scale')
        return x0+(x1-x0)*(v+22)/24
    pdf.setFont('Helvetica', 9.5)
    for t in ticks:
        pdf.setStrokeColor(RULE if t else MUTED)
        pdf.setLineWidth(.55 if t else .9)
        pdf.line(x(t), y-99, x(t), y+3)
        pdf.setFillColor(MUTED)
        pdf.drawCentredString(x(t), y-112, str(t))
    for i, (label, r) in enumerate(rows):
        at = y-17-31*i
        pdf.setFillColor(INK)
        pdf.drawString(LEFT, at-3, label)
        pdf.setFillColor(colors.HexColor('#b1c2ce'))
        for v in r['sites']:
            pdf.circle(x(v), at, 2, stroke=0, fill=1)
        a, b = map(x, r['ci95'])
        pdf.setStrokeColor(BLUE)
        pdf.setLineWidth(1.4)
        pdf.line(a, at, b, at)
        for z in (a, b):
            pdf.line(z, at-3, z, at+3)
        pdf.setFillColor(BLUE)
        pdf.circle(x(r['mean']), at, 3.1, stroke=0, fill=1)
    return para(pdf, 'Conservative gain (percentage points of observed CV error)', y-117,
                size=9, lead=11, color=MUTED, x=x0, width=x1-x0, space=12)


def forecast_page(pdf, f):
    y = header(pdf, 'A. MATCHED FORECAST CONTROL', 'Do additional neighbors improve the forecast?')
    y = para(pdf, 'On the same 141,783 OpenDD vehicle histories, we compare constant velocity (CV), '
             'four-point least squares (OLS4), the existing bounded vehicle forecaster, and two fixed-output '
             'EqMotion adaptations. Every fit uses the same observed eight and requested twelve steps. '
             'The EqMotion variants differ only in whether the three nearest observed neighbors are supplied.', y)
    y = para(pdf, 'Table A1. Native errors on available steps (ADE) and endpoints (FDE).', y,
             size=10, lead=13, bold=True, space=5)
    labels = [('CV', 'cv'), ('OLS4', 'ols4'), ('Existing vehicle forecast', 'completion'),
              ('EqMotion, ego only', 'comoving_ego'), ('EqMotion + neighbors', 'comoving_peers')]
    rows = [['Forecast', 'ADE', 'FDE', 'p95 ADE', 'Conservative gain vs CV, pp']]
    for name, key in labels:
        m = f['metrics'][key]
        rows.append([name, *(f"{m[k]['mean']:.3f}" for k in ('known_ade_native', 'fde_available_native', 'p95_ade_native')),
                     'reference' if key == 'cv' else ci(m['cv_gain']['lower'])])
    y = table(pdf, rows, [154, 55, 60, 64, 167], y)
    y = para(pdf, '<b>Direct paired contrast.</b> Ego-only EqMotion reduces observed error relative to the '
             'existing forecaster but loses after accounting for missing-step disagreement. The neighbor '
             'variant loses on observed steps in all seven locations.', y)
    y = forest(pdf, [(name, f['paired'][key]['lower']) for name, key in (
        ('Ego only vs existing forecast', 'comoving_ego_minus_completion'),
        ('Neighbors vs existing forecast', 'comoving_peers_minus_completion'),
        ('Neighbors vs ego only', 'comoving_peers_minus_comoving_ego'))], y)
    y = para(pdf, 'Figure A1. Small dots denote location means, larger dots the macro mean; bars give 95% '
             'paired location intervals. Each location is averaged over three seeds.', y,
             size=9.5, lead=12, color=MUTED)
    y = para(pdf, 'The EqMotion comparison uses the author core of Xu et al. (CVPR 2023) with one output '
             'fixed before fitting. It is not a comparison to published best-of-20 scores.', y,
             size=9.5, lead=12, color=MUTED)
    footer(pdf, 1, y)


def selection_page(pdf, c, s):
    y = header(pdf, 'B. MATCHED INTERVENTION ABLATIONS', 'What changes when the edit budget does not?')
    y = para(pdf, 'All arms select two edits at the same fixed query and peak displacement. The study '
             'uses 493 distinct recordings and 8,874 repeated recording views across three seeds and six '
             'calibration directions. These are development locations, not independent external scenes.', y)
    y = para(pdf, 'Table B1. Direct differences at the same two edits and radius.', y, size=10, lead=13,
             bold=True, space=5)
    r = [
        ('Learned support vs additive', s['intervals']['additive_support_minus_additive']),
        ('Learned support vs shuffled support', s['intervals']['additive_support_minus_additive_shuffled']),
        ('Factored debit vs equal-capacity direct', c['intervals']['direct_aux']),
    ]
    rows = [['Paired comparison', 'Observed gain, pp', 'Conservative gain, pp']]
    rows.extend([name, ci(v['ade_available_pp'], 6), ci(v['ade_lower_pp'], 6)] for name, v in r)
    y = table(pdf, rows, [208, 142, 150], y)
    y = para(pdf, 'Each bound uses the same observed CV error mass. The lower bound subtracts forecast '
             'disagreement on unavailable steps. It is not the error measured on those steps. '
             'The 95% intervals resample seven locations, not individual repeated views.', y,
             size=9.5, lead=12, color=MUTED)
    y = para(pdf, 'Table B2. Qualification and release are different outcomes.', y,
             size=10, lead=13, bold=True, space=5)
    rows = [['Rule', 'Qualified / 8,874', 'Released views', 'Qualified releases', 'All releases qualify']]
    for label, name in (('Archived additive', 'additive'), ('Archived joint', 'joint'),
                        ('Direct + auxiliary', 'direct_aux'), ('Factored debit', 'factorized')):
        v = c['summary'][name]
        rows.append([label, f"{v['matched_dual']:,}", v['released'], v['released_dual'], f"{v['all_dual']}/{v['active']}"])
    y = table(pdf, rows, [148, 86, 78, 88, 100], y)
    y = para(pdf, 'A qualifying view meets all original risk checks and has positive conservative gain '
             'against both CV and OLS4. The final column counts active directions. No edits or an undefined '
             'easy reference do not count as a pass.', y, size=9.5, lead=12, color=MUTED)
    y = para(pdf, '<b>Interpretation.</b> Support reranking does not beat its shuffled control on '
             'conservative utility. Factoring the missing-step debit increases the number of releases but '
             'worsens the matched trajectory and relative-motion comparisons. The evidence supports '
             'keeping CV as the fallback; it does not support a new release policy.', y)
    footer(pdf, 2, y)


def render(path, data):
    pdf = canvas.Canvas(str(path), pagesize=(612, 792), invariant=True, pageCompression=1)
    pdf.setAuthor('')
    pdf.setTitle('Vehicle controls and selection ablations')
    forecast_page(pdf, data['forecast'])
    selection_page(pdf, data['completion'], data['support'])
    pdf.save()


def verify(path):
    p = PdfReader(path)
    if (len(p.pages) != 2 or p.metadata.author or any(page.get('/Annots') for page in p.pages)):
        raise ValueError('Expected two anonymous, link-free pages')
    content = [page.extract_text() for page in p.pages]
    expected = [('141,783', '4.104', '+11.54', '+6.19', 'EqMotion'),
                ('8,874', '-0.002534', '-0.001171', '3/25', '6/23')]
    if not all(all(t in page for t in terms) for page, terms in zip(content, expected, strict=True)):
        raise ValueError('Expected metrics missing from the rendered text layer')
    if re.search(r'https?://|github|/Users/|yangyue|King.s College|\bLe Yang\b', '\n'.join(content), re.I):
        raise ValueError('Identity or URL in supplement')


def build(output):
    data = evidence()
    with TemporaryDirectory() as d:
        a, b = (Path(d)/name for name in ('first.pdf', 'second.pdf'))
        render(a, data)
        render(b, data)
        verify(a)
        if digest(a) != digest(b):
            raise ValueError('Supplement is not byte-reproducible')
        if output.exists() and digest(output) != digest(a):
            raise FileExistsError('Refusing to overwrite a different supplement')
        if not output.exists():
            output.write_bytes(a.read_bytes())
    receipt = dict(pdf_sha256=digest(output), pages=2, source_readouts={k: v[1] for k, v in SOURCE.items()},
                   cached_verified_readouts=True, deterministic_pdf=True, extracted_metrics_verified=True,
                   no_author_metadata_or_link_annotations=True, external_outcomes_opened=False, upload_performed=False,
                   manuscript_changed=False, submission_ready=False)
    OUT.mkdir(exist_ok=True)
    (OUT/'receipt.json').write_text(json.dumps(receipt, indent=2, sort_keys=True)+'\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=LOCAL/'vehicle_controls_supplement.pdf')
    build(parser.parse_args().output)
