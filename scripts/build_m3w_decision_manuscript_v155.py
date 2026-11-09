"""Bind the Moving initial-forecast experiment and fixed-peak direction audit."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v154 as previous
from scripts.build_m3w_decision_manuscript_v131 import package_supplement

BASE, PACK = previous.BASE, previous.PACK
PUBLIC = previous.PUBLIC.parent / 'evidence_manuscript_v155'
DEFAULT = previous.DEFAULT.parent / PUBLIC.name


def estimate(stats):
    lo, hi = stats['interval95']
    return f'{stats["point"]:+.3f} [{lo:+.3f}, {hi:+.3f}]'


def build(out, style, archive, part):
    root = PUBLIC.parent
    src = root / 'moving_dual_record_forecast_v1'
    files = [src / k for k in ('registration.json', 'freeze.json', 'readout.json', 'matched_directions.json')]
    g, f, r, a = [json.loads(p.read_text()) for p in files]
    h = BASE.cal.digest
    if (f['registration'] != h(files[0]) or r['freeze'] != h(files[1]) or a['parent']['sha256'] != h(files[2])
            or not f['before_held_scoring'] or not g['select_only_on_calibration']
            or not a['after_readout_diagnostic'] or not a['no_amplification']
            or any(not z['original_2pct'] or z['a_labels_opened'] or z['confirmation_access'] for z in (g, r, a))
            or any(z['independent_confirmation'] for z in (r, a)) or r['deployment_changed']):
        raise ValueError('Require frozen development-only forecasts and fixed-budget diagnostics')
    s, m = r['summary'], a['summary']
    values = dict(Fits=sum(len(c['fits']) for c in f['cells']), Updates=f'{f["optimizer_updates"]:,}',
                  Strict=r['strict_selected'], Cells=len(f['cells']), Views=s['views'], Sources=s['source_groups'],
                  Records=s['recordings'], Bootstrap=f'{s["bootstrap_draws"]:,}',
                  Edits=f'{m["risk"]["half_anchor"]["switches"]:,}', AnchorDual=a['dual_pass']['half_anchor'],
                  RawUpper=r['single_query_upper']['diagnostic']['excluded'],
                  AnchorUpper=r['single_query_upper']['half_anchor']['excluded'])
    for k, name in (('half_anchor', 'Anchor'), ('ols_only', 'OLS'), ('dual_risk', 'Risk'), ('diagnostic', 'Screen')):
        values[name + 'Risk'] = s['risk'][k]['supported_views']
        values[name + 'Dual'] = r['dual_pass'][k]
        values[name + 'MatchedDual'] = a['dual_pass'][k]
        if k != 'half_anchor':
            effect = s['effects'][k + '_vs_ols4']
            values[name + 'Known'] = estimate(effect['known_pp'])
            values[name + 'Lower'] = estimate(effect['completion_lower_pp'])
            values[name + 'Matched'] = estimate(m['effects'][k + '_vs_anchor']['completion_lower_pp'])
    doc = previous.build(out, style, archive, part)
    if part == 'main':
        tex, defs = out / 'manuscript.tex', out / 'evidence_values.tex'
        key, receipt = 'build_bindings', 'source_bindings.json'
        text = BASE.replace_once(tex.read_text(),
            'The bound separates restricted opportunity from ranking failure (supplement).',
            'A matched risk-aware forecast also adds no pass at fixed recipients and radii (supplement).')
    else:
        tex, defs = out / 'supplement.tex', out / 'supplement_values.tex'
        key, receipt = 'files', 'supplement_bindings.json'
        frag = PUBLIC / 'dual_record_forecast.tex'
        files.append(frag)
        anchor = r'\section*{LLM Usage Statement}'
        text = BASE.replace_once(tex.read_text(), anchor, frag.read_text() + '\n' + anchor)
    tex.write_text(text)
    defs.write_text(defs.read_text() + '\n' + '\n'.join(
        rf'\newcommand{{\MovingInitial{k}}}{{{v}}}' for k, v in values.items()) + '\n')
    doc['source_bindings'].update({str(p.relative_to(root)): h(p) for p in files})
    doc[key].update({p.name: h(p) for p in (tex, defs)})
    (out / receipt).write_text(json.dumps(doc, indent=2, sort_keys=True) + '\n')
    return doc


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('part', choices=('main', 'supplement'))
    p.add_argument('--source-zip', type=Path)
    p.add_argument('--style', type=Path, default=BASE.v22.STYLE)
    p.add_argument('--archive', type=Path, default=BASE.v22.supp.base.REPLAY)
    a = p.parse_args()
    out = DEFAULT / ('build' if a.part == 'main' else 'supplement')
    doc = build(out, a.style, a.archive, a.part)
    if a.source_zip and a.part == 'main':
        PACK.main_package.package(out, a.source_zip, doc['build_bindings'])
    elif a.source_zip:
        package_supplement(out, a.source_zip, doc)
