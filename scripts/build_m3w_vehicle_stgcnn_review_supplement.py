"""Package the matched OpenDD public baseline without updating the main PDF."""

import argparse
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np
from pypdf import PdfReader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reproducibility.focused_review_supplement.verify import run_replay, unpack
from reproducibility.vehicle_stgcnn_supplement import verify
from scripts import build_m3w_wepdtof_history_candidate as prior
from scripts.build_m3w_exact_exchange_review_supplement import check

archive, bundle, main = prior.archive, prior.bundle, prior.main
STUDY = main.PUBLIC/'opendd_vehicle_stgcnn_v1'
PACK = main.PUBLIC/'vehicle_stgcnn_review_supplement_v1'
DEST = bundle.LOCAL/'review_supplement_v32.zip'
TEMPLATE = archive.ROOT/'reproducibility/vehicle_stgcnn_supplement'


def review_table(metrics, pairs):
    lines = ['\n## Matched OpenDD development comparison\n',
             '| Arm | Available ADE | Available FDE | p95 ADE | Conservative CV gain (pp) | Original risk passes |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    names = dict(social_peers='Social-STGCNN + neighbors', social_ego='Social-STGCNN ego only',
                 completion='Completion-trained incumbent', cv='CV', ols4='OLS4')
    for a, label in names.items():
        m = metrics[a]
        scalars = [m[k]['mean'] for k in (*verify.KEYS[:2], 'p95_ade_native', 'cv_gain_lower_pp')]
        risk = 'Undefined without edits' if a == 'cv' else str(m['risk_passes'])
        lines.append(f"| {label} | {' | '.join(f'{x:.3f}' for x in scalars)} | {risk} |")
    lines.extend(('', 'The two direct Social-STGCNN-minus-EqMotion contrasts below use',
                  'a common CV denominator and pair-specific missing-label distances.', '',
                  '| Matched input | Conservative gain (pp) | 95% paired location interval |',
                  '| --- | ---: | ---: |'))
    for a, name in zip(verify.BACKBONES, ('Ego only', 'With neighbors'), strict=True):
        v = pairs[a]['lower']
        lo, hi = v['ci95']
        lines.append(f"| {name} | {v['point']:+.3f} | [{lo:+.3f}, {hi:+.3f}] |")
    lines.extend(('', 'Errors use OpenDD release coordinates. This does not claim a new',
                  'independent test, an official pedestrian benchmark result, or a',
                  'deployed selector. Each interval resamples seven locations after',
                  'averaging the three training seeds.', ''))
    return '\n'.join(lines).encode()


def appendix(root, compiler, metrics, paired, contrast):
    text = (TEMPLATE/'forecast_controls.tex').read_text()
    labels = dict(cv='CV', ols4='OLS4', completion='Vehicle incumbent',
                  social_ego='Social-STGCNN ego', social_peers='Social-STGCNN + neighbors')
    rows = []
    for a, label in labels.items():
        v = metrics[a]
        values = [f"{v[k]['mean']:.3f}" for k in (*verify.KEYS[:2], 'p95_ade_native', 'cv_gain_lower_pp')]
        risk = 'n/a' if a == 'cv' else str(v['risk_passes'])
        rows.append(' & '.join((label, *values, risk))+r'\\')
    pairs = []
    for a, label in zip(verify.BACKBONES, ('Ego only', 'With neighbors'), strict=True):
        x = paired[a]['lower']
        lo, hi = x['ci95']
        pairs.append(f"{label} & {x['point']:+.3f} & [{lo:+.3f}, {hi:+.3f}]"+r'\\')
    x = contrast['social_peers_minus_ego']
    lo, hi = x['lower']['ci95']
    sentence = (f"Neighbors change the observed gain over the ego-only Social-STGCNN by "
                f"{x['observed']['mean']:+.2f} points, but the direct conservative contrast is "
                f"{x['lower']['mean']:+.2f} points [{lo:+.2f}, {hi:+.2f}].")
    for k, value in {'MODEL_ROWS': '\n'.join(rows), 'PAIR_ROWS': '\n'.join(pairs), 'NEIGHBOR_SENTENCE': sentence}.items():
        text = text.replace(f'%%{k}%%', value)
    (root/'forecast_controls.tex').write_text(text)
    subprocess.run([str(compiler), '-X', 'compile', '--untrusted', '--only-cached', 'forecast_controls.tex'],
                   check=True, cwd=root, env={**os.environ, 'SOURCE_DATE_EPOCH': '1794268800'})
    binary = (root/'forecast_controls.pdf').read_bytes()
    pdf = PdfReader(BytesIO(binary))
    document = ' '.join(p.extract_text() for p in pdf.pages)
    if (not 1 <= len(pdf.pages) <= 2 or archive.PRIVATE_MARKERS.search(document.encode())
            or (pdf.metadata and pdf.metadata.get('/Author')) or 'Social-STGCNN' not in document
            or 'Risk passes' not in document):
        raise ValueError('Invalid anonymous supplement PDF')
    return binary


def run(compiler):
    source = STUDY/'readout.json'
    raw = source.read_bytes()
    data = json.loads(raw)
    cross = (STUDY/'backbone_pairs.json').read_bytes()
    pairs = json.loads(cross)
    if (data['binding']['freeze'] != main.digest((STUDY/'freeze.json').read_bytes()) or
            not data['original_2pct'] or data['independent_confirmation'] or
            data['deployment_changed'] or data['external_outcomes_opened']):
        raise ValueError('Require the original fixed vehicle development score')
    if (pairs['binding']['social'] != main.digest(raw) or not pairs['original_reference_metrics_reproduced']
            or pairs['external_outcomes_opened'] or pairs['deployment_changed'] or pairs['cells'] != 21
            or set(pairs['by_site_seed']) != set(data['by_site_seed'])):
        raise ValueError('Require the direct EqMotion comparison on the same roster')
    parent, draft = bundle.read_receipt(prior.PACK.name), bundle.read_receipt(prior.OUT.name)
    pdf = draft['artifacts']['manuscript.pdf']
    bundle.checked(prior.OUT/'manuscript.pdf', pdf)
    if pdf != parent['companion_pdf_sha256']:
        raise ValueError('Inherited evidence and PDF do not match')
    sites = sorted({k.rsplit('_', 1)[0] for k in data['by_site_seed']})
    seeds = sorted({k.rsplit('_', 1)[1] for k in data['by_site_seed']})
    cells = []
    for k, r in data['by_site_seed'].items():
        site, seed = k.rsplit('_', 1)
        arms = {}
        for a, m in r['metrics'].items():
            out = {t: m[t] for t in (*verify.KEYS[:4], 'risk_passes', 'recordings')}
            out.update(cv_gain_lower_pp=m['cv_gain_known_lower_pp'][1], ols_gain_lower_pp=m['ols_gain_known_lower_pp'][1])
            arms[a] = out
        backbones = {a: {j: v[j] for j in ('observed', 'lower')} for a, v in pairs['by_site_seed'][k].items()}
        cells.append(dict(site=sites.index(site), seed=seeds.index(seed), arms=arms, paired=r['paired'],
                          backbones=backbones))
    panel = dict(cells=cells, draws=np.random.default_rng(3183).integers(7, size=(3000, 7)).tolist())
    parts = {k: (TEMPLATE/k).read_bytes() for k in ('README.md', 'verify.py')}
    parts['README.md'] += review_table(data['metrics'], pairs['paired'])
    parts['protocol.md'] = (STUDY/'protocol.md').read_bytes()
    parts['opendd_cells.json'] = main.as_json(panel)
    if any(archive.PRIVATE_MARKERS.search(p) for p in parts.values()):
        raise ValueError('Identifying review material')
    parts['prior_evidence.zip'] = bundle.checked(prior.DEST, parent['archive_sha256'])
    with TemporaryDirectory(prefix='vehicle-stgcnn-review-') as tmp:
        root = Path(tmp)
        parts['forecast_controls.pdf'] = appendix(root, compiler, data['metrics'], pairs['paired'], data['paired'])
        if set(parts) != verify.MEMBERS:
            raise ValueError('Unexpected anonymous review members')
        parts['manifest.json'] = main.as_json({k: main.digest(v) for k, v in parts.items()})
        for n in ('first.zip', 'second.zip'):
            archive.encode(root/n, parts)
        binary = (root/'first.zip').read_bytes()
        if binary != (root/'second.zip').read_bytes():
            raise ValueError('Nondeterministic review archive')
        unpack(root/'first.zip', root/'unpacked')
        expected = dict(metrics=data['metrics'], paired=data['paired'], backbone=pairs['paired'])
        if not check(run_replay(root/'unpacked')['opendd_vehicle_stgcnn'], expected):
            raise ValueError('Anonymous baseline replay disagrees with the readout')
    rec = dict(archive_sha256=main.digest(binary), archive_bytes=len(binary), prior_bundle_sha256=parent['archive_sha256'],
               companion_pdf_sha256=pdf, readout_sha256=main.digest(raw), paired_readout_sha256=main.digest(cross),
               appendix_sha256=main.digest(parts['forecast_controls.pdf']),
               standard_library_replay=True,
               includes_raw_labels=False, includes_checkpoints=False, uploaded=False, submission_ready=False)
    for p, content in {DEST: binary, DEST.with_suffix('.pdf'): parts['forecast_controls.pdf'],
                       PACK/'receipt.json': main.as_json(rec)}.items():
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists() and p.read_bytes() != content:
            raise FileExistsError('Refusing to replace a frozen review artifact')
        if not p.exists():
            with p.open('xb') as f:
                f.write(content)
    print(rec)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compiler', type=Path, default=Path('/Applications/ChatGPT.app/Contents/Resources/tectonic/tectonic'))
    run(p.parse_args().compiler.resolve())
