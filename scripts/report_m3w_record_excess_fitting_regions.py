"""C-only optimization diagnosis, never held-model selection or calibration."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import audit_m3w_record_excess_fitting_regions as run
from scripts.report_m3w_moving_harm_refinement import text_once


def main():
    source, public = run.source, run.PUBLIC
    s, v = source.read(public/'analysis.json'), source.read(public/'verification.json')
    if (not v['all72cells_and36C_cost_arrays_exact']
            or v['analysis_sha256'] != source.sha(public/'analysis.json')):
        raise ValueError('Complete actual C-region replay first')
    lines = ['# Selection Transport Within The Fitting Data', '',
        'Fresh36Ccost arrays and72model-by-region cells replay exactly. No new',
        'Hprediction or Hscore preceded freezing this diagnostic and its trigger.',
        'These are pooled observed fitting masses across dependent repeated views,',
        'not independent validation, calibration, macro-source utility or new FDE.', '',
        '|Cost model|Cselected region|B actual/pred|H actual/pred|R actual/pred|ER actual/pred|EH actual/pred|',
        '|---|---|---:|---:|---:|---:|---:|']
    for model, regions in s['values'].items():
        for region, value in regions.items():
            lines.append('|'+model+'|'+region+'|'+'|'.join('not_estimable' if value['actual_over_predicted'][k] is None
                else f'{value["actual_over_predicted"][k]:.6f}' for k in ('B','H','R','ER','EH'))+'|')
    lines += ['', f'Predeclared C-only follow-up trigger:{s["C_only_adaptive_followup_trigger"]}; '
        f'components:{s["trigger_moments"]}.', '',
        'The same new head can fit the frozen teacher region much better and still',
        'be optimistic on the cases its own selector chooses. This within-C result',
        'cannot be explained solely by a C-to-H domain change, but it does not',
        'identify all causal mechanisms or establish learnability on unseen scenes.',
        'It licenses only the predefined one-factor training-mask follow-up, not',
        'an Hthreshold/weight search or a deployment claim.', '',
        'All18models were already frozen; the Cdiagnostic changed no weights or',
        'official data roles. Unknown contributions, every empty view and the',
        'original2%contract remain. Original24group primary is not_estimable;',
        '23supported groups are exposed development. No independent confirmation,',
        'metric/seconds/physical-safety/true3D/foundation/human-gold claim, Stage5C',
        'execution or SMC.', '']
    text_once(public/'readout.md', '\n'.join(lines))
    print(source.json.dumps(dict(complete=True, C_trigger=s['C_only_adaptive_followup_trigger'], new_H_prediction=False)))


if __name__ == '__main__':
    main()
