"""Add an internal evidence guide without modifying the source supplement."""

import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from pypdf.annotations import Link
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle


INDEX = (
    ('Candidate forecasts and matched controls', 'Forecast Bank and Selection Controls'),
    ('Recording risk versus paired accuracy', 'Recording Risk and Accuracy Are Different Estimands'),
    ('Pair geometry and source influence', 'Relative-Path Error and Source Influence'),
    ('OpenDD development: ranking at a fixed budget', 'Ranking at a Fixed Intervention Budget'),
    ('MovingDroneCrowd++: OLS4 feasibility bounds', 'OLS-anchored path capacity at fixed edit counts'),
    ('WEPDTOF: frozen external comparison', 'One-shot transfer on reserved WEPDTOF sources'),
    ('Learning query plans from recording supervision', 'Learning query plans from verified recording plans'),
    ('Source roles and replay', 'Sealed Evaluation and Replay'),
)
INK = HexColor('#171e25')
GRAY = HexColor('#4a515a')
RULE = HexColor('#cdd2d6')
MARGIN, WIDTH = 62, 488
STYLE = ParagraphStyle('body', fontName='Times-Roman', fontSize=12, leading=16, textColor=INK)


def destinations(reader):
    found = {}
    for item in reader.outline:
        if isinstance(item, list):
            continue
        if item.title in found:
            raise ValueError(f'Ambiguous supplement heading: {item.title}')
        found[item.title] = reader.get_destination_page_number(item)
    for _, title in INDEX:
        if found.get(title, -1) < 0:
            raise ValueError(f'Missing valid PDF destination: {title}')
    return found


def paragraph(c, s, y):
    p = Paragraph(s, STYLE)
    _, h = p.wrap(WIDTH, 500)
    p.drawOn(c, MARGIN, y-h)
    return y-h-14


def header(c, label, title):
    c.setFillColor(GRAY)
    c.setFont('Helvetica-Bold', 10)
    c.drawString(MARGIN, 722, label)
    c.setFillColor(INK)
    c.setFont('Times-Bold', 29)
    c.drawString(MARGIN, 681, title)


def footer(c, page):
    c.setStrokeColor(RULE)
    c.line(MARGIN, 58, MARGIN+WIDTH, 58)
    c.setFillColor(GRAY)
    c.setFont('Helvetica', 9)
    c.drawString(MARGIN, 42, 'Supplementary material')
    c.drawRightString(MARGIN+WIDTH, 42, str(page))
    c.showPage()


def guide(lookup, study):
    output = BytesIO()
    c = canvas.Canvas(output, pagesize=letter, invariant=1)
    c.setTitle('Supplementary material: evidence guide')
    c.setAuthor('')
    header(c, 'EVIDENCE GUIDE', 'Where to find the results')
    y = paragraph(c, 'The main paper asks whether it helps to replace a motion baseline '
                  'when the number and size of edits are held fixed. The supplement separates '
                  'what the available forecasts can do in hindsight from what a past-only rule '
                  'can recover. The links below go to the relevant analysis.', 644)
    links = []
    for label, title in INDEX:
        page = lookup[title]+2
        c.setStrokeColor(RULE)
        c.line(MARGIN, y+8, MARGIN+WIDTH, y+8)
        c.setFillColor(INK)
        c.setFont('Helvetica', 11)
        c.drawString(MARGIN+2, y-10, label)
        c.drawRightString(MARGIN+WIDTH-2, y-10, str(page+1))
        links.append((0, (MARGIN, y-20, MARGIN+WIDTH, y+6), page))
        y -= 37
    links.append((0, (MARGIN, y-28, MARGIN+WIDTH, y+4), 1))
    c.setStrokeColor(RULE)
    c.line(MARGIN, y+8, MARGIN+WIDTH, y+8)
    c.setFont('Helvetica-Bold', 11)
    c.drawString(MARGIN+2, y-10, 'Additional matched ablation: direct pair supervision')
    c.drawRightString(MARGIN+WIDTH-2, y-10, '2')
    paragraph(c, 'Page numbers above refer to this PDF. The original supplement follows '
              'the addendum and keeps its own section and page numbering.', y-42)
    footer(c, 1)

    header(c, 'DEVELOPMENT ABLATION', 'Does the pair score add value?')
    y = paragraph(c, 'All three arms search the same forecast paths at exactly the same count '
                  'and displacement in each query. The additive arm scores edits separately. '
                  'One joint arm imitates full recording plans; the other learns the extra '
                  'reduction in pair-relative error beyond the two single edits.', 644)
    arms = ('matched_independent', 'imitation', 'joint')
    labels = ('Additive', 'Imitation pair score', 'Direct pair-effect score')
    rows = [['Policy', 'Risk passes', 'Risk and both baselines']]
    for key, label in zip(arms, labels, strict=True):
        parts = [g['arms'][key] for g in study['groups'].values()]
        rows.append([label, f"{sum(a['risk_pass'] for a in parts)} / 606",
                     f"{sum(a['dual_pass'] for a in parts)} / 606"])
    table = Table(rows, colWidths=[207, 116, 165], rowHeights=[27]*len(rows))
    table.setStyle(TableStyle([('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                               ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                               ('FONTSIZE', (0, 0), (-1, -1), 10),
                               ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
                               ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                               ('LINEABOVE', (0, 0), (-1, 0), .75, INK),
                               ('LINEBELOW', (0, 0), (-1, 0), .5, RULE),
                               ('LINEBELOW', (0, -1), (-1, -1), .75, INK)]))
    _, h = table.wrap(WIDTH, 500)
    table.drawOn(c, MARGIN, y-h)
    lower = study['contrasts']['lower_pp']
    a, b = lower['ci95']
    y = paragraph(c, f"The direct head gains {lower['mean']:+.5f} percentage points over the "
                  f'additive control after charging for missing steps. The 95% location bootstrap '
                  f'interval is [{a:+.5f}, {b:+.5f}]. It changes '
                  f"{study['totals']['direct_differs_additive']} of {study['totals']['queries_2_slots']:,} "
                  'two-edit decisions but does not give a reliable gain.', y-h-20)
    probes = [p for g in study['groups'].values() for p in g['probes'].values() if p is not None]
    wins = sum(p['r2_vs_zero'] > 0 for p in probes)
    y = paragraph(c, f'On held pair-target probes, the direct head beats predicting zero '
                  f'in {wins} of {len(probes)} supported location-seed views. The target is '
                  'therefore not reliably recovered from these features. This comparison uses '
                  '21 fits, three seeds and 202 exposed OpenDD recordings. The intervals summarize '
                  'six estimable locations; they are not independent confirmation. The 2% '
                  'risk rule is unchanged.', y)
    paragraph(c, 'Risk and utility are separate tests. The right column requires the original '
              'recording-risk checks and strictly positive conservative gains over both CV and '
              'OLS4. The three policies have 3,690 edits each.', y)
    footer(c, 2)
    c.save()
    output.seek(0)
    return output, links


def build(source, metrics, output):
    if output.exists() or output.resolve() == source.resolve():
        raise FileExistsError('Refuse to overwrite either supplement')
    original = PdfReader(source)
    lookup = destinations(original)
    study = json.loads(metrics.read_text())
    if sum(g['views'] for g in study['groups'].values()) != 606 or not study['original_2pct']:
        raise ValueError('Unexpected ablation protocol')
    front, links = guide(lookup, study)
    # Clone the document so that named references inside the existing PDF survive.
    writer = PdfWriter(clone_from=original)
    for i, page in enumerate(PdfReader(front).pages):
        writer.insert_page(page, index=i)
    writer.add_outline_item('Evidence guide', 0)
    writer.add_outline_item('Direct pair-effect ablation', 1)
    for page, rect, target in links:
        link = writer.add_annotation(page, Link(rect=rect, target_page_index=target, border=[0, 0, 0]))
        link['/Dest'][0] = writer.pages[target].indirect_reference
    writer.add_metadata({'/Title': 'Supplementary material', '/Author': '', '/Subject': ''})
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix('.pending')
    writer.write(tmp)
    checked = PdfReader(tmp)
    def annotations(page):
        return page['/Annots'] if '/Annots' in page else []

    for a, b in zip(original.pages, checked.pages[2:], strict=True):
        if a.extract_text() != b.extract_text() or len(annotations(a)) != len(annotations(b)):
            raise ValueError('Original content or annotation counts changed')
    original_names, final_names = original.named_destinations, checked.named_destinations
    if set(original_names) != set(final_names) or any(
            checked.get_destination_page_number(final_names[k]) != original.get_destination_page_number(v)+2
            for k, v in original_names.items()):
        raise ValueError('Named references changed')
    shifted = destinations(checked)
    if len(annotations(checked.pages[0])) != len(links) or any(
            shifted[title] != page+2 for title, page in lookup.items()) or (
            shifted.get('Evidence guide'), shifted.get('Direct pair-effect ablation')) != (0, 1):
        raise ValueError('Supplement navigation is incomplete')
    for note, (_, _, target) in zip(annotations(checked.pages[0]), links, strict=True):
        if note.get_object()['/Dest'][0] != checked.pages[target].indirect_reference:
            raise ValueError('Evidence guide link points to the wrong page')
    tmp.replace(output)
    return dict(pages=len(checked.pages), original_pages=len(original.pages), guide_links=len(links),
                sha256=sha256(output.read_bytes()).hexdigest())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('metrics', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.metrics, args.output), indent=2))
