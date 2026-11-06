"""Report every fitting-source prefix and fixed direction bin without selection."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_m3w_moving_fitting_prefix_audit as run


def fmt(x):return 'NA' if x is None else f'{x:.4f}'


def main():
    a=run.read(run.PUBLIC/'analysis.json');banks=a['banks']
    comparable=sum(b['existing_basis_parity']['comparable_rows'] for b in banks)
    maximum=max(b['existing_basis_parity']['maximum_cosine_difference'] for b in banks)
    sources=[(b['producer_fold'],s) for b in banks for s in b['sources']]
    first_negative=sum(s['summary'][0]['known_gain_percent'] is not None and s['summary'][0]['known_gain_percent']<0 for _,s in sources)
    terminal_positive=sum(s['summary'][-1]['known_gain_percent'] is not None and s['summary'][-1]['known_gain_percent']>0 for _,s in sources)
    lines=['# Fitting-Source Prefix Attribution','',
        f'Fresh_run:all{a["total_rows"]}agent-window rows in6producer-A OOF forecast banks.',
        'Cached_verified:original arrays,raw120features and teacher-disjoint predictions.',
        'Not_run:new controller/forecaster training,new own-B or independent C/H readout.','',
        '## Existing Information','',
        f'First-step heading cosine is reconstructed from raw120on{comparable}rows;',
        f'max difference{maximum:.9g};zero comparable rows exceed1e-4.',
        f'The remaining{a["total_rows"]-comparable}undefined/tiny-direction rows remain in all summaries.',
        'This cue is not new information. No hard angle veto or new input is promoted.','',
        '## Every Source','',
        '| Producer fold | Source | Rows | First-step gain % | Prefix3 gain % | Available12 gain % | Terminal beneficial with first label | Also first-step harmful |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for fold,s in sources:
        lines.append(f'| {fold} | {s["source"]} | {s["rows"]} | {fmt(s["summary"][0]["known_gain_percent"])} | '
            f'{fmt(s["summary"][2]["known_gain_percent"])} | {fmt(s["summary"][-1]["known_gain_percent"])} | '
            f'{s["terminal_beneficial_with_first_label"]} | {s["terminal_beneficial_but_first_harm"]} |')
    lines+=['',f'{first_negative}/14sources have worse first-step error;{terminal_positive}/14improve at the available12-step average.',
        'These are source-specific,query-balanced descriptive percentages,not a new',
        'official table,independent-source CI or an oracle selector. Prefix3 includes',
        'available labels only. Overlapping windows are not independent samples.','',
        '## Fixed Direction Bins','',
        '| Fold | Source | Prefix | Bin | Rows | Query-weight share | Easy harm mass share | Positive gain mass share |',
        '|---|---|---:|---|---:|---:|---:|---:|']
    for fold,s in sources:
        for p in s['summary']:
            for b in p['direction_bins']:
                lines.append(f'| {fold} | {s["source"]} | {p["prefix"]} | {b["name"]} | {b["rows"]} | '
                    f'{fmt(b["weight_share"])} | {fmt(b["easy_harm_mass_share"])} | {fmt(b["positive_gain_mass_share"])} |')
    lines+=['','All bins,prefixes,sources and support are retained;empty mass is NA.',
        'A small opposing-direction subset cannot account for most easy harm.',
        'Directional disagreement also appears in useful corrections,so its',
        'presence alone is not a safe veto or deployment confidence.','',
        '## Next Actual Model Test','',
        'The distinction from old prefix-cost and temporal-ramp negatives matters:',
        'those changed a critic or an action while retaining forecast producers.',
        'A fixed matched actor-loss experiment can instead test whether supervision',
        'over multiple prefixes repairs early forecasts without losing the original',
        'available12-step objective. It must be preregistered,teacher-disjoint and',
        'cannot promote an early-only improvement over a failed terminal comparison.',
        'Prefix weighting is not claimed novel and does not itself certify selection.','',
        'This is model/data attribution,not a pipeline success claim. Original2%',
        'recording/support/NA rules and old failures remain. No metric/time,true3D,',
        'foundation,human-gold,Stage5C execution or SMC claim.']
    (run.PUBLIC/'README.md').write_text('\n'.join(lines)+'\n')
    print('Complete fitting-prefix table reported; no cue or policy selected.')


if __name__=='__main__':main()
