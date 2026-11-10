"""Build the anonymous Songdo addendum and updated reviewer bundle."""

import argparse
from collections import defaultdict
from hashlib import sha256
from io import BytesIO
import json
from math import isclose
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
from reportlab.pdfgen import canvas

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reproducibility.songdo_review_supplement import replay_songdo as replay
from scripts import build_m3w_vehicle_supplement as layout
from scripts import export_m3w_songdo_confirmation as songdo
from scripts import package_m3w_combined_review_supplement as old


ROOT, LOCAL = songdo.ROOT, old.LOCAL
TEMPLATE = ROOT/'reproducibility/songdo_review_supplement'
OUT = old.archive.PUBLIC/'songdo_review_supplement_v1'
ARMS, SEEDS = replay.ARMS, replay.SEEDS
RISK = ('completion_all_risk_upper', 'completion_easy_risk_upper', 'known_easy_net_degradation')


def extract(run, expected):
    reg, rh = songdo.secondary.load(run/'registration.json')
    freeze, fh = songdo.secondary.load(run/'freeze.json')
    cuts, ch = songdo.secondary.load(run/'calibration_cutoffs.json')
    if (rh, fh, ch) != tuple(expected['binding'][k] for k in (
            'registration_sha256', 'release_freeze_sha256', 'calibration_cutoffs_sha256')):
        raise ValueError('Local receipts differ from the published scoring lineage')
    by_role = {k: defaultdict(list) for k in ('calibration', 'confirmation')}
    confirmation_hashes = []
    for flight, (role, group) in sorted(reg['flight_roles'].items()):
        v, vh = songdo.secondary.load(run/role/(Path(flight).stem+'.json'))
        if (v['flight'], v['role'], v['registration_sha256'], v['release_freeze_sha256'],
                v['calibration_cutoffs_sha256']) != (flight, role, rh, fh if role == 'confirmation' else None, ch):
            raise ValueError('Unregistered per-flight score')
        if role == 'confirmation':
            confirmation_hashes.append(vh)
        cells = []
        for seed in SEEDS:
            cell = v['scores'][str(seed)]
            m = cell['matched']
            data = dict(seed=seed, arms=None)
            if m['matched_eligible']:
                data['arms'] = {a: dict(score=cell['scores'][a], risk=[m['arms'][a]['risk'][k] for k in RISK],
                                       gains=[m['arms'][a][k]['lower_pp'] for k in ('vs_cv', 'vs_ols4')]) for a in ARMS}
            if role == 'confirmation':
                data['contrasts'] = {a: {k: m['matched_contrasts'][a][k] for k in replay.PAIR} for a in ARMS[:-1]}
                data['accuracy'] = {a: {k: m['arms'][a][k] for k in replay.ABSOLUTE} for a in ('cv', 'ols4', *ARMS)}
            cells.append(data)
        by_role[role][group].append(cells)
    if sha256(''.join(sorted(confirmation_hashes)).encode()).hexdigest() != expected['binding']['flight_score_hash_root']:
        raise ValueError('The individual confirmation receipts changed')
    return dict(**{k: [g[n] for n in sorted(g)] for k, g in by_role.items()},
                native_cutoffs=[cuts['easy_cut'], cuts['hard_cut']], calibration_cutoff_examples=cuts['support_rows'],
                source_resamples=np.random.default_rng(1439).integers(14, size=(3000, 14)).tolist(),
                original_risk_limit=.02, threshold_freeze_sha256=fh, score_sha256=expected['binding']['primary_score_sha256'],
                calibration_thresholds=freeze['thresholds'])


def check_result(actual, expected, calibration):
    if actual['thresholds'] != calibration['thresholds'] or actual['calibration_counts'] != calibration['calibration_counts']:
        raise ValueError('Independent calibration replay differs from the pre-outcome freeze')
    for arm in ARMS:
        if actual['matched_risk'][arm] != expected['matched'][arm]['risk'] or actual['actual_risk'][arm] != expected['released'][arm]['risk']:
            raise ValueError('The original per-view risk and dual checks did not reproduce')

    def compare(a, b):
        for k in ('defined_seed_views', 'estimable_components'):
            if a[k] != b[k]:
                raise ValueError('The independent replay changed the estimation support')
        if a['point'] is None and b['point'] is None:
            return
        if a['point'] is None or b['point'] is None or not all(isclose(x, y, abs_tol=1e-12, rel_tol=1e-12)
                for x, y in zip((a['point'], *a['ci95']), (b['point'], *b['ci95']), strict=True)):
            raise ValueError('The independent replay changed the point or interval')

    for arm in ARMS[:-1]:
        for k in replay.PAIR:
            compare(actual['contrasts'][arm][k], expected['matched_joint_minus'][arm][k])
    for a in ('cv', 'ols4', *ARMS):
        for k in replay.ABSOLUTE:
            compare(actual['matched'][a][k], expected['matched'][a]['metrics'][k])
            if a in ARMS:
                compare(actual['actual'][a][k], expected['released'][a]['metrics'][k])


def addendum(result):
    buf = BytesIO()
    pdf = canvas.Canvas(buf, pagesize=(612, 792), invariant=1)
    pdf.setAuthor('')
    pdf.setCreator('Anonymous')
    pdf.setTitle('Songdo confirmation: matched proposals and release')
    y = layout.header(pdf, 'SUPPLEMENTARY MATERIAL', 'Songdo: proposals and release')
    y = layout.para(pdf, 'Thresholds fit on 40 flights in three source components were frozen before scoring '
                    '160 flights in 14 separate linked components. The three seeds repeat those flights. '
                    'Every matched proposal makes two edits at the same radius on one query.', y)
    names = dict(independent='Independent', additive='Additive', same_family='Same family', joint='Joint')
    rows = [['Joint minus', 'Conservative gain (pp)', '95% source bootstrap (pp)']]
    for k in ARMS[:-1]:
        r = result['contrasts'][k]['lower_pp']
        rows.append([names[k], f"{r['point']:+.6f}", f"[{r['ci95'][0]:+.6f}, {r['ci95'][1]:+.6f}]"])
    y = layout.table(pdf, rows, [178, 140, 182], y)
    y = layout.para(pdf, 'The matched comparison does not establish an advantage over independent '
                    'or additive selection. The same-family contrast is positive but small. Intervals '
                    'resample source components after averaging seeds and flights, not individual forecasts.', y)
    rows = [['Policy', 'Proposal risk / 480', 'Released / 480', 'Released risk / edits', 'Released dual / edits']]
    for k in ARMS:
        p, a = (result[s][k] for s in ('matched_risk', 'actual_risk'))
        r, n = a['original_risk_pass'], a['edited']
        rows.append([names[k], p['original_risk_pass'], n, f'{r}/{n}' if n else 'Not applicable',
                     f"{a['dual_pass']}/{n}" if n else 'Not applicable'])
    y = layout.table(pdf, rows, [103, 96, 88, 108, 105], y)
    y = layout.para(pdf, 'The frozen joint, additive and same-family policies return CV on every flight-seed '
                    'view. Fallback is not a risk pass. The independent policy released 15 views; only five '
                    'passed all original risk checks and four also beat both references.', y)
    cv, ols = (result['matched'][k]['available_ade_native']['point'] for k in ('cv', 'ols4'))
    y = layout.para(pdf, f'Available-step ADE: CV {cv:.6f}, OLS4 {ols:.6f} (native release units). '
                    'The study uses eight observed and twelve predicted steps; no physical safety claim.', y, size=10, lead=13)
    y = layout.para(pdf, 'Data: Fonod et al., Songdo Traffic v2 (Zenodo:17924857). '
                    'The accompanying scalar replay checks threshold fitting, matched contrasts and release counts.',
                    y, size=9.5, lead=12, color=layout.MUTED)
    if y < 60:
        raise ValueError('Addendum content overlaps footer')
    pdf.showPage()
    pdf.save()
    raw = buf.getvalue()
    text = old.check_pdf(raw, 1)
    if not all(k in text for k in ('-0.000442', '0.000237', '5/15', '4/15', '160 flights')):
        raise ValueError('Addendum evidence did not render')
    return raw


def manifest(files):
    return old.main.as_json({k: old.main.digest(v) for k, v in files.items()})


def build(destination):
    expected = songdo.export(songdo.RUN, songdo.PUBLIC)
    if expected != json.loads((songdo.PUBLIC/'confirmation_receipt.json').read_text()):
        raise ValueError('Local scores do not reproduce the public receipt')
    calibration = json.loads((songdo.PUBLIC/'calibration_receipt.json').read_text())
    evidence = extract(songdo.RUN, expected)
    result = replay.replay(evidence)
    check_result(result, expected, calibration)
    pdf = addendum(result)
    prior = json.loads((old.OUT/'verification.json').read_text())
    paper = old.read_receipt('evidence_manuscript_v170_songdo')
    old.check_pdf(old.checked(old.main.PRIVATE/'evidence_manuscript_v170_songdo/manuscript.pdf',
                              paper['artifacts']['manuscript.pdf']), 8)
    with TemporaryDirectory(prefix='songdo-supplement-') as temp:
        p = Path(temp)
        songdo_items = {'verify.py': (TEMPLATE/'replay_songdo.py').read_bytes(),
                        'README.md': b'Run python -I -B -S verify.py after extraction.\n',
                        'evidence.json': json.dumps(evidence, separators=(',', ':'), allow_nan=False).encode()}
        songdo_items['manifest.json'] = manifest(songdo_items)
        old.archive.encode(p/'songdo_replay.zip', songdo_items)
        score_archive = (p/'songdo_replay.zip').read_bytes()
        old.check_replay(score_archive)
        items = {k: (TEMPLATE/k).read_bytes() for k in ('README.md', 'verify.py')}
        items.update({'songdo_confirmation.pdf': pdf, 'songdo_replay.zip': score_archive,
                      'original_evidence.zip': old.checked(LOCAL/'review_supplement_v1.zip', prior['sha256'])})
        items['manifest.json'] = manifest(items)
        a, b = p/'a.zip', p/'b.zip'
        old.archive.encode(a, items)
        old.archive.encode(b, items)
        raw = a.read_bytes()
        if raw != b.read_bytes():
            raise ValueError('The updated supplementary archive is not deterministic')
        clean = p/'clean'
        with ZipFile(a) as z:
            z.extractall(clean)
        r = subprocess.run([sys.executable, '-I', '-B', '-S', 'verify.py'], cwd=clean,
                           check=True, capture_output=True, text=True, timeout=240)
        cold = json.loads(r.stdout)
    if cold['songdo'] != result or cold['original']['headline'] != prior['headline']:
        raise ValueError('Clean offline replay differs from frozen evidence')
    receipt = dict(archive_sha256=old.main.digest(raw), archive_bytes=len(raw), pdf_sha256=old.main.digest(pdf),
                   main_pdf_sha256=paper['artifacts']['manuscript.pdf'], prior_bundle_sha256=prior['sha256'],
                   calibration_freeze_sha256=evidence['threshold_freeze_sha256'],
                   primary_score_sha256=evidence['score_sha256'], flights=result['flights'],
                   source_components=result['source_components'], views=result['views'],
                   joint_minus_independent_lower_pp=result['contrasts']['independent']['lower_pp'],
                   original_released_risk=result['actual_risk'], anonymous_offline_replay_pass=True,
                   deterministic_zip=True, no_new_label_access=True, uploaded=False, submission_ready=False)
    files = {destination: raw, destination.with_name('songdo_confirmation_supplement_v1.pdf'): pdf,
             OUT/'receipt.json': old.main.as_json(receipt)}
    if any(p.exists() and p.read_bytes() != v for p, v in files.items()):
        raise FileExistsError('Refusing to overwrite a different artifact')
    for p, v in files.items():
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            with p.open('xb') as f:
                f.write(v)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=LOCAL/'review_supplement_v2.zip')
    build(p.parse_args().output)
