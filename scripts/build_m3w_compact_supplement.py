"""Build a short, anonymous reading companion for the unchanged paper."""

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from pypdf import PdfReader
from reportlab.pdfgen import canvas

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import package_m3w_main_claims as main
from scripts import build_m3w_vehicle_supplement as layout

OUT = main.PUBLIC/'compact_review_supplement_v1'
ADMIN = main.PRIVATE/'submission_admin'
DRAFT_SHA = '3541d2c7cc29b04ea8db3c74151a6850530d43ac4a071f3e8dc7add15210e6ad'
ARCHIVE = 'reviewer_evidence.zip'
TITLE = 'When Should a Neural Forecast Replace a Motion Baseline?'
SOURCES = dict(
    external='wepdtof_external_transfer_v1/external_readout.json',
    external_replay='wepdtof_scalar_replay_v1/readout.json',
    moving='moving_record_peak_allocation_v1/readout.json',
    ceiling='opendd_vehicle_relative_ceiling_v1/readout.json',
    anchor='opendd_vehicle_anchor_rank_v1/readout.json',
    paired='opendd_vehicle_anchor_rank_v1/paired_readout.json',
    forecast='opendd_kinematic_blend_v1/readout.json',
    replay='reviewer_evidence_v1/receipt.json',
)
HEADINGS = ('Reading the evidence', 'Scoring with partial labels',
            'One-shot evaluation on WEPDTOF', 'Can a joint ranker use the available headroom?',
            'Forecast controls and offline replay')
FORBIDDEN = re.compile(r'/Users/|/home/|/cephfs/|https?://|github\.com|kcl\.ac\.uk|'
                       r'PhDLeYang|Le\s+Yang|yangyue', re.I)


def load(root, rel):
    p = root/rel
    return json.loads(p.read_text()), layout.digest(p)


def evidence():
    data, bindings = {}, {}
    for k, rel in SOURCES.items():
        data[k], bindings[rel] = load(main.PUBLIC, rel)
    draft_pdf = main.DRAFT/'manuscript.pdf'
    archive = ADMIN/ARCHIVE
    if layout.digest(draft_pdf) != DRAFT_SHA or len(PdfReader(draft_pdf).pages) != 8:
        raise ValueError('Expected the unchanged eight-page main manuscript')
    if layout.digest(archive) != data['replay']['archive_sha256']:
        raise ValueError('Anonymous companion archive has changed')
    with TemporaryDirectory(prefix='reviewer-replay-') as tmp, ZipFile(archive) as z:
        z.extractall(tmp)
        out = subprocess.run([sys.executable, '-I', '-B', '-S', 'verify.py'], cwd=tmp, check=True,
                             capture_output=True, text=True, timeout=45)
    replay = json.loads(out.stdout)
    if replay['headline'] != data['replay']['headline']:
        raise ValueError('Offline replay disagrees with its verified receipt')
    main.verify_draft(replay['headline'], main.DRAFT)
    original = main.export()
    moving_sources = {r['identity']['source'] for r in data['moving']['rows']}
    if (len(original['opendd']), len(original['moving']), len(original['wepdtof']),
            len(moving_sources)) != (7, 24, 13, 9):
        raise ValueError('Manuscript cohort changed')
    data['headline'] = replay['headline']
    external = data['external']
    if (not external['external_scoring_completed'] or external['external_tuning']
            or external['dronecrowd_confirmation_access'] or not external['original_2pct']
            or layout.digest(main.PUBLIC/'wepdtof_external_transfer_v1/augmented_freeze.json') != external['forecast_freeze']
            or data['external_replay']['source_sha256'] != bindings[SOURCES['external']]):
        raise ValueError('External study is not the frozen readout')
    a, p, c, f = (data[k] for k in ('anchor', 'paired', 'ceiling', 'forecast'))
    for r in (a, p, c, f):
        if not r['original_2pct_unchanged'] or r['external_outcomes_read'] or r['independent_confirmation']:
            raise ValueError('Follow-up evidence must be exposed development only')
    if (a['freeze'] != p['freeze']
            or a['freeze'] != layout.digest(main.PUBLIC/'opendd_vehicle_anchor_rank_v1/freeze.json')
            or f['freeze_sha256'] != layout.digest(main.PUBLIC/'opendd_kinematic_blend_v1/freeze.json')
            or a['totals']['joint']['full_win'] != c['totals']['joint_relative_win']
            or a['totals']['relative']['count'] != c['totals']['views']
            or p['active']['relative']['full_win'] != a['totals']['relative']['admitted_full_win']):
        raise ValueError('Development comparisons are no longer aligned')
    return data, bindings


def interval(stat, digits=3):
    point = stat['mean'] if 'mean' in stat else stat['estimate']
    a, b = stat['ci95']
    return f'{point:+.{digits}f} [{a:+.{digits}f}, {b:+.{digits}f}]'


class Page:
    def __init__(self, pdf, number, heading):
        self.pdf = pdf
        self.number = number
        pdf.bookmarkPage(f's{number}')
        pdf.addOutlineEntry(heading, f's{number}', level=0)
        self.y = layout.header(pdf, f'SUPPLEMENTARY MATERIAL  /  S{number}', heading)

    def text(self, value, *, note=False):
        self.y = layout.para(self.pdf, value, self.y, size=10 if note else 11,
                             lead=13 if note else 15, color=layout.MUTED if note else layout.INK)

    def sub(self, value):
        self.y = layout.para(self.pdf, value, self.y-4, size=11, lead=13, bold=True, space=5)

    def table(self, rows, widths):
        self.y = layout.table(self.pdf, rows, widths, self.y)

    def close(self):
        if self.y < 64:
            raise ValueError(f'Supplementary page {self.number} overlaps footer')
        pdf = self.pdf
        pdf.setStrokeColor(layout.RULE)
        pdf.line(layout.LEFT, 49, layout.LEFT+layout.WIDTH, 49)
        pdf.setFillColor(layout.MUTED)
        pdf.setFont('Helvetica', 9)
        pdf.drawString(layout.LEFT, 34, 'Forecast replacement under a matched intervention budget')
        pdf.drawRightString(layout.LEFT+layout.WIDTH, 34, f'S{self.number}')
        pdf.showPage()


def overview(p, d):
    x, m, w = (d['headline'][k] for k in ('opendd', 'moving', 'wepdtof'))
    p.text(f'<b>{TITLE}</b>')
    p.text('A forecaster can improve average error and still be a poor replacement for constant '
           'velocity (CV) in many recordings. We keep the candidate paths, requested number of edits '
           'and their allowed displacement comparable. The controls ask whether the policy chooses '
           'better recipients, not whether it can edit more forecasts.')
    p.sub('Three tests answer different questions')
    p.table([
        ['Evidence', 'Evaluation roster', 'Headline result'],
        ['OpenDD fixed-budget development', f'{x["recordings"]} recordings in {x["locations"]} locations',
         f'{x["hindsight_passes"]}/{x["views"]} hindsight witnesses; {x["past_only_passes"]}/{x["views"]} past-only passes. '
         f'{x["unresolved"]} unresolved hindsight view.'],
        ['MovingDroneCrowd++ capacity', f'{m["recordings"]} recordings in 9 development sources',
         f'{m["ols4_excluded"]}/{m["views"]} views cannot beat OLS4 even after relaxing path shape.'],
        ['WEPDTOF frozen external comparison', f'{w["recordings"]} clips in {w["sources"]} source groups',
         f'{w["fixed_joint_dual_passes"]}/{w["views"]} fixed-count joint passes on both risk and accuracy.'],
    ], [184, 136, 180])
    p.text('All three studies use seeds 17, 29 and 43. A recording-seed view repeats the same recording, '
           'so it is a count of evaluations, not an independent sample. OpenDD vehicle follow-ups on '
           'S4 and S5 use a separate, larger roster; their counts are not added to the 606 views.')
    p.sub('What each control changes')
    p.text('CV makes no replacements. OLS4 fits velocity to the last four observed positions. '
           'The independent rule ranks edits separately; the shared-family rule uses one candidate '
           'family in a query. The no-pair control keeps the joint search and drops the interaction '
           'term. Outcome-aware search is a diagnostic, never an inference rule.')
    p.text('S2 defines the risk and accuracy calculations; S3 reports the frozen external test. '
           'S4 gives the matched vehicle ablation and S5 the forecast controls and replay instructions.', note=True)


def scoring(p, _):
    p.text('Each query uses eight observed positions and requests twelve future steps on its dataset grid. '
           'At inference, path selection uses observed history and frozen forecasters. Future labels and '
           'validity masks are used for supervision, scoring and the explicitly label-aware capacity tests, '
           'not as inputs to the past-only policy.')
    p.sub('Recording-level risk')
    p.text('Let b be the CV mean displacement error on a row\'s available future steps. A row is easy '
           'if 0 &lt; b &le; the fitted cutoff. For each selected row with labels, positive harm is '
           'max(candidate error minus b, 0). For a wholly unobserved selected row, charge the maximum '
           'candidate-to-CV displacement as a worst-case debit.')
    p.table([
        ['Check', 'Numerator', 'Denominator'],
        ['All selected, positive harm', 'Observed positive harm + wholly unobserved debit',
         'Observed CV error on selected rows'],
        ['Easy selected, positive harm', 'Observed easy positive harm + the same debit',
         'Observed CV error on selected easy rows'],
        ['Entire recording, signed easy', 'Observed signed candidate-minus-CV change on easy rows',
         'Observed CV error on all easy rows'],
    ], [138, 174, 188])
    p.text('All three ratios must be defined and at most 2%. An empty or zero denominator does not '
           'pass. No-edit CV is an accuracy reference, not a successful edited-risk test.')
    p.sub('Paired accuracy and relative motion')
    p.text('Weight actors so each query has equal mass. Divide each direct gain by the same observed '
           'CV error mass. In the original row-based analysis, missing-row charges use maximum '
           'disagreement on wholly unobserved rows; partial rows keep their available-label ADE. '
           'In the OpenDD and WEPDTOF stepwise comparisons, debit disagreement at every missing step.')
    p.text('For a missing position y, the reverse triangle inequality bounds the loss difference by '
           'the distance between the two predictions. Apply it to the relative displacement of '
           'two actors if either label is missing. These are conservative comparisons, not '
           'imputed positions or collision rates. Resample sources after averaging seeds; '
           'bootstrap intervals and missing-label bounds describe different uncertainty.')
    p.text('The supplement reports aggregate statistics from existing benchmarks. It includes no '
           'video frames, subject identifiers or person-level paths. Image-pixel results and '
           'OpenDD release coordinates are kept separate.', note=True)


def external(p, d):
    r = d['external']['summary']
    e = d['external_replay']['arms']['fixed_joint']
    p.text('The camera-only inputs and four fixed-count proposal rules were frozen before the '
           'source-disjoint WEPDTOF [3] outcome readout. Only the exposed pilot camera was used for '
           'development. The matched arms are utility controls, not certified release policies.')
    rows = [['Rule', 'Edits', 'Risk pass / 45', 'Dual pass / 45', 'Lower gain vs CV, pp']]
    for label, arm in (('Strict joint', 'joint'), ('Independent', 'fixed_independent'),
                       ('Shared family', 'fixed_uniform'), ('No pair term', 'fixed_no_pair'),
                       ('Joint', 'fixed_joint')):
        a = r['arms'][arm]
        rows.append([label, a['edits'], a['risk_passes'], a['dual_passes'], interval(a['lower_vs_cv'])])
    p.table(rows, [125, 51, 83, 79, 162])
    j = r['arms']['fixed_joint']
    p.text(f'The fixed-count joint arm has observed ADE {j["ade_px"]["estimate"]:.2f} px, FDE '
           f'{j["fde_px"]["estimate"]:.2f} px and p95 ADE {j["p95_ade_px"]["estimate"]:.2f} px. '
           f'Its conservative gain against OLS4 is {interval(j["lower_vs_ols4"], 2)} pp, despite a small '
           'positive gain against CV. A dual pass requires all risk checks and positive lower gains against both references.')
    counts = e['reasons']
    p.text(f'Of {r["recording_seed_views"]} joint views, {counts["no_intervention"]} have no eligible edit, '
           f'{counts["undefined_easy_reference"]} edited views have no estimable easy ratio, '
           f'{counts["violates_original_risk"]} fail a measured risk check, and {counts["risk_pass"]} pass risk. '
           'None of those risk passes beats both references.')
    p.sub('Direct comparisons at the same edit count')
    for label, key in (('independent', 'fixed_independent'), ('shared family', 'fixed_uniform'),
                       ('no pair term', 'fixed_no_pair')):
        c = r['matched_comparisons'][key]
        p.text(f'Joint minus {label}: <b>{interval(c, 5)} pp</b> '
               f'({c["estimable_sources"]}/{c["total_sources"]} source groups).')
    p.text('The remaining source groups are not assigned zero. Strict proposals make no edits; '
           'their edited-policy comparisons are not estimable. All intervals here use 3,000 paired '
           'draws of original source groups.', note=True)


def headroom(p, d):
    c, a, b = d['ceiling']['totals'], d['anchor']['totals'], d['paired']
    p.text('This follow-up uses 493 distinct vehicle recordings across seven exposed OpenDD '
           'locations. Each matched view has exactly two edits at the same query and peak radii. '
           'The 8,874 views repeat three seeds and six calibration directions.')
    p.sub('Label-aware ceiling vs choices made from history')
    p.table([
        ['Test', 'Qualifying views'],
        ['Hindsight: original risk and positive gain vs both motion references', f'{c["any_dual"]:,}'],
        ['Also no conservative trajectory loss and better relative motion vs additive', f'{c["any_relative_win"]:,}'],
        ['Also keeps at least one additive recipient', f'{c["any_keep_recipient_relative_win"]:,}'],
        ['Learned unrestricted joint: meets the full condition', a['joint']['full_win']],
        ['Learned anchored head with CV/OLS4 targets: full condition', a['dual_only']['full_win']],
        ['Learned anchored head with relative targets: full condition', a['relative']['full_win']],
    ], [380, 120])
    v = b['relative_minus']['dual_only']
    p.text('The two anchored heads share the model size, donor pairs and updates. Adding '
           f'anchor-relative targets gives {interval(v["ade_available_pp"], 6)} pp of available-step '
           f'gain but {interval(v["ade_lower_pp"], 6)} pp of conservative gain directly against the '
           f'CV/OLS4-only head. The conservative relative-motion contrast is '
           f'{interval(v["relative_lower_native"], 5)} in release coordinates.')
    r = b['active']['relative']
    p.text(f'After calibration, {r["admitted"]} held replacements are admitted; {r["risk_pass"]} pass '
           f'original risk and {r["full_win"]} meets the full condition. No active direction qualifies '
           'on every replacement. CV remains the vehicle fallback.')
    p.text('The exact oracle adds the changes to the edges incident on each edited actor and corrects '
           'their shared edge once. It is a check on a fixed bank, not a model that can see the future. '
           'This tests forecast allocation, unlike ScePT [2], which learns to generate scene-consistent paths.', note=True)


def forecast(p, d):
    f = d['forecast']['summary']
    p.text('Forecast quality is separate from the choice to intervene. These forecasts share the same '
           '141,783 past-eligible vehicle histories from 500 recordings at seven exposed locations. '
           'Fitted models use the same three seeds and six donor locations per held location.')
    rows = [['Forecast', 'ADE', 'FDE', 'p95 ADE', 'Lower gain vs CV, pp']]
    for label, key in (('CV', 'cv'), ('OLS4', 'ols4'), ('Median velocity', 'median4'),
                       ('Smoothed velocity', 'smooth_velocity'), ('Decaying acceleration', 'decaying_acceleration'),
                       ('Decaying turn', 'decaying_turn'), ('Fitted kinematic mixture', 'blend'),
                       ('Bounded learned forecast', 'incumbent')):
        r = f[key]
        rows.append([label, *(f'{r[k]["mean"]:.3f}' for k in ('known_ade_native', 'fde_available_native',
                                                           'p95_ade_native')),
                     'Reference' if key == 'cv' else interval(r['cv_gain']['lower'], 2)])
    p.table(rows, [169, 53, 55, 61, 162])
    p.text('Errors are in OpenDD release coordinates. The learned forecast beats the simple controls '
           'on conservative gain on this roster, not on an independently confirmed deployment test. '
           'The mixture retains more risk passes but neither meets the full recording requirement.')
    p.sub('Offline verification')
    p.text(f'Extract <b>{ARCHIVE}</b> and run <font name="Courier">python -I -B -S verify.py</font>. '
           'It checks all archive member hashes and replays the headline counts, WEPDTOF risk decisions '
           'from saved ratios, and source bootstrap comparisons. It needs no repository or weights. '
           'It does not reproduce raw tracking, forecaster training or the new vehicle follow-ups.')
    p.sub('References and disclosure')
    p.text('[1] A. Mao, M. Mohri, and Y. Zhong. Regression with Multi-Expert Deferral. ICML, 2024. '
           'Expert deferral is established; our comparison fixes the multi-agent edit budget.<br/>'
           '[2] Y. Chen, B. Ivanovic, and M. Pavone. ScePT: Scene-Consistent, Policy-Based Trajectory '
           'Predictions for Planning. CVPR, 2022.<br/>'
           '[3] O. Tezcan, Z. Duan, M. Cokbas, P. Ishwar, and J. Konrad. WEPDTOF: A Dataset and '
           'Benchmark Algorithms for In-the-Wild People Detection and Tracking from Overhead Fisheye Cameras. '
           'WACV, 2022.', note=True)
    p.text('OpenAI Codex assisted with code, evidence checks, analysis, drafting and preparation of '
           'this supplement. The author is responsible for its content.', note=True)


def render(path, data):
    pdf = canvas.Canvas(str(path), pagesize=(612, 792), invariant=1)
    pdf.setTitle('Forecast replacement: supplementary material')
    pdf.setAuthor('')
    for k, (name, draw) in enumerate(zip(HEADINGS, (overview, scoring, external, headroom, forecast),
                                         strict=True), 1):
        p = Page(pdf, k, name)
        draw(p, data)
        p.close()
    pdf.save()


def verify(path, data=None):
    pdf = PdfReader(path)
    if len(pdf.pages) != len(HEADINGS) or len(pdf.outline) != len(HEADINGS):
        raise ValueError('Missing page or navigation entry')
    for i, (page, bookmark) in enumerate(zip(pdf.pages, pdf.outline, strict=True)):
        text = ' '.join(page.extract_text().split())
        if (HEADINGS[i] not in text or bookmark.title != HEADINGS[i] or
                pdf.get_destination_page_number(bookmark) != i or page.get('/Annots') or FORBIDDEN.search(text)):
            raise ValueError('Unexpected content, external annotation or link on page '+str(i+1))
    if pdf.metadata.author or FORBIDDEN.search(str(pdf.metadata)):
        raise ValueError('Unexpected metadata')
    if data is not None:
        h, a, p = (data[k] for k in ('headline', 'anchor', 'paired'))
        o, m, w = (h[k] for k in ('opendd', 'moving', 'wepdtof'))
        required = {
            0: (f'{o["hindsight_passes"]}/{o["views"]}', f'{o["past_only_passes"]}/{o["views"]}',
                f'{m["ols4_excluded"]}/{m["views"]}', f'{w["fixed_joint_dual_passes"]}/{w["views"]}'),
            2: (interval(data['external']['summary']['matched_comparisons']['fixed_independent'], 5),),
            3: (f'{a["totals"]["relative"]["full_win"]}',
                interval(p['relative_minus']['dual_only']['ade_lower_pp'], 6)),
            4: (interval(data['forecast']['summary']['incumbent']['cv_gain']['lower'], 2),),
        }
        for i, phrases in required.items():
            text = ' '.join(pdf.pages[i].extract_text().split())
            if not all(s in text for s in phrases):
                raise ValueError('Bound evidence missing from page '+str(i+1))


def build(output, receipt):
    data, source = evidence()
    with TemporaryDirectory(prefix='compact-review-') as d:
        a, b = (Path(d)/f'{k}.pdf' for k in 'ab')
        render(a, data)
        render(b, data)
        verify(a, data)
        blob = a.read_bytes()
        if blob != b.read_bytes():
            raise ValueError('PDF is not deterministic')
    if output.exists() and output.read_bytes() != blob:
        raise FileExistsError('Refusing to overwrite a different PDF')
    output.parent.mkdir(parents=True, exist_ok=True)
    pending = output.with_suffix('.pending')
    pending.write_bytes(blob)
    pending.replace(output)
    r = dict(pages=len(HEADINGS), bookmarks=len(HEADINGS), sha256=layout.digest(output), sources=source,
             builder_sha256=layout.digest(Path(__file__)),
             manuscript_sha256=DRAFT_SHA, replay_archive_sha256=data['replay']['archive_sha256'],
             fresh_run='layout build and clean-directory replay of paper headline counts',
             cached_verified='bound external and development readouts', no_new_outcome_access=True,
             manuscript_unchanged=True, submitted=False, submission_ready=False)
    if receipt:
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_bytes(main.as_json(r))
    print(json.dumps(r, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ADMIN/'compact_review_supplement_v1.pdf')
    p.add_argument('--receipt', type=Path)
    a = p.parse_args()
    build(a.output, a.receipt)
