"""Build the v14 companion supplement from the frozen development risk replay."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import build_m3w_decision_supplement_v12 as v12

PUBLIC = v12.HOME/'evidence_manuscript_v14'
DEFAULT = v12.DATA/'evidence_manuscript_v14/supplement'
SHA = '45215e45296439ffd3c0143cf5ce9551640cf148df7a892bbe1dc1abb4107049'


def macros(audit):
    risk_rows = []
    for key, label in (
            ('gain', 'Gain ranking'), ('support_first', 'Support first'),
            ('shortfall', 'Risk shortfall'), ('student_nomination', 'Student, fixed path'),
            ('student_full', 'Student, full bank'), ('student_screen', 'Screened student')):
        a = audit['risk'][key]
        p = a['disjoint_recording_patterns']
        other = ('debit_exceeds', 'undefined', 'debit_exceeds+undefined')
        failed_known = sum(n for k, n in p.items() if 'observed_exceeds' in k.split('+'))
        fields = [label, f'{a["switches"]:,}', f'{a["unknown_switches"]:,}', p.get('pass', 0),
                  failed_known, *(p.get(k, 0) for k in other)]
        if sum(fields[3:]) != audit['views']:
            raise ValueError('Disjoint checks do not cover all frozen views')
        risk_rows.append(' & '.join(map(str, fields))+r' \\')
    utility_rows = []
    for key, label in (('student_nomination_vs_gain', 'Student, fixed path'),
                       ('student_full_vs_gain', 'Student, full bank')):
        a = audit['utility'][key]
        lower, known, upper = (a[k] for k in ('completion_lower_pp', 'known_pp', 'completion_upper_pp'))
        lo, hi = lower['influence']['leave_one_source_out_range']
        parts = [label, f'{known["point"]:+.3f}', f'$[{lower["point"]:+.3f}, {upper["point"]:+.3f}]$',
                 str(lower['influence']['positive_sources']), f'$[{lo:+.3f}, {hi:+.3f}]$']
        utility_rows.append(' & '.join(parts)+r' \\')
    values = dict(FrozenRiskRows='\n'.join(risk_rows), FrozenUtilityRows='\n'.join(utility_rows))
    return '\n'.join("\\newcommand{\\"+k+"}{"+v+"}" for k, v in values.items())+'\n'


def build(out, archive, style):
    evidence = PUBLIC/'risk_attribution.json'
    if v12.sha(evidence) != SHA:
        raise ValueError('Expected the verified frozen risk attribution')
    audit = v12.read(evidence)
    if (not audit['scorer_replay_exact'] or audit['external_access'] or
            audit['inference_or_gates_changed'] or audit['independent_confirmation']):
        raise ValueError('Not the original exposed replay')
    v12.build(out, archive, style)
    tex = out/'supplement.tex'
    source = tex.read_text()
    before = r'\section{Prior Work and Claim Boundary}'
    if source.count(before) != 1:
        raise ValueError('Expected one insertion point in the v12 supplement')
    block = (PUBLIC/'supplement_risk_attribution.tex').read_text()
    tex.write_text(source.replace(before, block+'\n\n'+before))
    values = out/'supplement_values.tex'
    values.write_text(values.read_text()+macros(audit))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, default=v12.REPLAY)
    p.add_argument('--style', type=Path, default=v12.STYLE)
    p.add_argument('--output-dir', type=Path, default=DEFAULT)
    args = p.parse_args()
    build(args.output_dir, args.archive, args.style)
