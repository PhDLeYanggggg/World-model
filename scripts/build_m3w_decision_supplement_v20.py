"""Build an anonymous evidence supplement from checksum-bound scalar readouts."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import build_m3w_decision_manuscript_v13 as main
from scripts import build_m3w_decision_supplement_v14 as v14

base = v14.v12
PUBLIC = base.HOME/'evidence_manuscript_v20'
DEFAULT = base.DATA/'evidence_manuscript_v20/supplement'
STUDIES = ('native_pair_blend_direction_v1', 'native_recording_blend_feasibility_v1',
           'native_prefix_velocity_v1', 'native_completion_motion_blend_v1')
PAYLOAD = ('supplement.tex', 'supplement_values.tex')


def checked_counts(direction, feasibility, prefix, blend):
    if (direction['matched_switches'], feasibility['switches'], prefix['matched_switches'],
            blend['common_substitutions']) != (9675,)*4:
        raise ValueError('All studies must use the same registered switch count')
    if not (direction['hindsight_only'] and feasibility['hindsight_only']
            and prefix['inference_uses_past_only']):
        raise ValueError('The result roles must distinguish oracle from causal')
    records = feasibility['by_recording_seed']
    counts = Counter(row['status'] for row in records)
    witnesses = feasibility['feasible_witnesses']
    checked = {(r['recording'], r['seed']) for r in records if r['status'] == 'checked_feasible'}
    if (len(records) != feasibility['views'] or counts != feasibility['counts']
            or sum(row['switches'] for row in records) != feasibility['switches']
            or len({(row['recording'], row['seed']) for row in records}) != len(records)
            or checked != {(w['recording'], w['seed']) for w in witnesses}
            or len(witnesses) != len(checked)
            or not all(r['risk']['supported'] for r in records if r['status'] == 'checked_feasible')):
        raise ValueError('Recording feasibility partition does not replay')
    return counts


def macros(direction, feasibility, prefix, blend):
    counts = checked_counts(direction, feasibility, prefix, blend)
    fields = {}
    for k, name in (('shared_hindsight', 'Shared'), ('separate_hindsight', 'Separate')):
        row = direction['counts'][k]
        if sum(row.values()) != direction['active_queries'] or row['unresolved']:
            raise ValueError('Incomplete pair directional result')
        fields['Pair'+name] = f'{row["verified_gain"]:,}'
    risk = prefix['matched']['risk']
    for k, name in (('fixed_prefix', 'Fixed'), ('query_prefix', 'Query'), ('agent_prefix', 'Agent'), ('gain', 'Gain')):
        if risk[k]['required_views'] != feasibility['views'] or risk[k]['switches'] != prefix['matched_switches']:
            raise ValueError('Expected full matched roster for the prefix comparison')
        fields['PrefixPass'+name] = risk[k]['supported_views']
    rows = []
    for scope, name, b in (
            ('matched', 'Fixed', 'fixed_prefix'), ('matched', 'Query mean', 'query_prefix'),
            ('matched', 'Gain ranking', 'gain'),
            ('whole_roster', 'OLS4', 'history_ols4_velocity'),
            ('whole_roster', 'CV', 'constant_velocity_causal_fd')):
        if name in ('Fixed', 'OLS4'):
            label = 'Matched recipients' if name == 'Fixed' else 'Whole roster'
            rows.append(r'\multicolumn{3}{@{}l}{\emph{'+label+r'}} \\')
        e = prefix[scope]['effects'][f'agent_prefix_vs_{b}']

        def cell(v):
            lo, hi = v['interval95']
            return f'${v["point"]:+.3f}\\;[{lo:+.3f}, {hi:+.3f}]$'

        rows.append(' & '.join([name, cell(e['known_pp']), cell(e['completion_lower_pp'])])+r' \\')
    fields.update(PrefixRows='\n'.join(rows), PairActive=f'{direction["active_queries"]:,}',
        FeasibilityViews=feasibility['views'], FeasibilitySupported=counts['checked_feasible'],
        FeasibilityExcluded=counts['risk_infeasible_with_fixed_recipients'],
        FeasibilityUndefined=counts['structural_support_absent'],
        FeasibilityUnknown=feasibility['exclusion_channels']['unknown_radius_alone_excludes_easy'],
        FeasibilityPair=feasibility['feasible_with_positive_pair_ray'],
        FeasibilityOLS=feasibility['feasible_with_positive_pair_vs_ols'],
        AlphaObserved=f'{blend["alpha_quantiles"]["observed_fit"][1]:.3f}',
        AlphaCompletion=f'{blend["alpha_quantiles"]["completion_fit"][1]:.3f}')
    return '\n'.join(f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in fields.items())+'\n'


def build(out, archive, style):
    bindings = {}
    studies = [main.verified_analysis(name, bindings) for name in STUDIES]
    new = macros(*studies)
    v14.build(out, archive, style)
    tex, values = (out/name for name in PAYLOAD)
    block = PUBLIC/'supplement_diagnostics.tex'
    key = r'\section{Prior Work and Claim Boundary}'
    source = tex.read_text()
    if source.count(key) != 1:
        raise ValueError('Expected one insertion point')
    key_ref = r'\bibitem{crc}'
    if source.count(key_ref) != 1:
        raise ValueError('Expected the previous risk-control reference')
    text = source.replace(key, block.read_text()+'\n\n'+key)
    prior = 'Selective regression can worsen performance within particular'
    if text.count(prior) != 1:
        raise ValueError('Expected the previous selective-regression discussion')
    text = text.replace(prior, ('Multi-expert classification deferral can exhibit expert\n'
        'identification and underfitting problems~\\cite{liu}. Its\n'
        'classification analysis does not identify the cause of\n'
        'failure for joint continuous forecasts under partial labels.\n'+prior))
    ref = (r'\bibitem{liu} S. Liu, Y. Cao, L. Feng, B. An, and L. Ong.'+'\n'
           'When more experts hurt: Underfitting in multi-expert learning\n'
           r'to defer. In \emph{ICML}, pages 75507--75531, 2026.'+'\n\n')
    tex.write_text(text.replace(key_ref, ref+key_ref).replace(r'\begin{thebibliography}{4}',
                                                          r'\begin{thebibliography}{5}'))
    values.write_text(values.read_text()+new)
    receipt = dict(development_only=True, external_scores_read=False, source_bindings=bindings,
        base_aggregate_sha256=base.REPLAY_SHA, style_sha256=base.sha(style),
        diagnostic_fragment_sha256=base.sha(block), files={k: base.sha(out/k) for k in PAYLOAD})
    (out/'supplement_bindings.json').write_text(json.dumps(receipt, sort_keys=True, indent=2)+'\n')
    return receipt


def package(out, destination, receipt):
    if set(receipt['files']) != set(PAYLOAD):
        raise ValueError('Unexpected files in supplement source manifest')
    data = {}
    for name in PAYLOAD:
        p = out/name
        if base.sha(p) != receipt['files'][name]:
            raise ValueError('Supplement source changed after assembly')
        data[name] = p.read_bytes()
    data['README.txt'] = (b'Anonymous working supplement. Supply cvpr.sty from the CVPR 2027 template '
                          b'and compile supplement.tex. This archive contains no independent results.\n')
    with ZipFile(destination, 'w') as zipped:
        for name, value in sorted(data.items()):
            item = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            item.external_attr = 0o644 << 16
            zipped.writestr(item, value, compress_type=ZIP_DEFLATED)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, default=base.REPLAY)
    p.add_argument('--style', type=Path, default=base.STYLE)
    p.add_argument('--output-dir', type=Path, default=DEFAULT)
    p.add_argument('--source-zip', type=Path)
    a = p.parse_args()
    doc = build(a.output_dir, a.archive, a.style)
    if a.source_zip:
        package(a.output_dir, a.source_zip, doc)
