"""Keep fixed-set support failures and unavailable complete-view intervals visible."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import audit_m3w_selected_reference as run
from scripts.report_m3w_moving_harm_refinement import text_once


def main():
    s, public = run.source, run.PUBLIC
    result = s.read(public / 'analysis.json')
    verification = s.read(public / 'verification.json')
    if (not verification['all4176selected_reference_rows_and_metrics_exact']
            or verification['analysis_sha256'] != s.sha(public / 'analysis.json')):
        raise ValueError('Complete exact cached scalar replay first')
    rows = [r for ref in result['receipts'] for r in s.read(s.checked(ref))['rows']]
    lines = ['# Reference Errors On Fixed Selected Sets', '',
        'All4176original arm/policy/recording views retained. Counts match exactly;',
        'original float64 reference masses match rtol1e-12/atol1e-9. Fresh18old neural',
        'cost arrays and108new reference arrays match their earlier hashes. Cached',
        'choice ledgers are verified, not reoptimized. Full8.5MiBper-record aggregate',
        'ledgers stay local under groups/; public hashes and compact summaries are',
        'versioned, not those derived ledgers. Scalar replay is exact, with',
        'cached reference vectors; it is not a second fresh inference.', '',
        '|Joint-free cost arm|Active /348|Active without observed ER|Those with labels for all selected|Old/direct/hurdle/C-selected positive ER on unsupported|',
        '|---|---:|---:|---:|---|']
    for arm in run.ARMS:
        c = result['counts'][arm]['joint_free']
        n = c['positive_prediction_without_observed_support']
        lines.append(f'|{arm}|{c["active"]}|{c["active_without_observed_ER"]}|'
            f'{c["active_without_ER_all_selected_have_some_label"]}|'
            + '/'.join(str(n[m]) for m in run.MODELS) + '|')
    lines.extend(['', 'No predictor removes the unsupported views by merely supplying a positive',
        'mean. This is not proof of theoretical impossibility, and empty reference',
        'is not a measured zero risk. The new head predicts ER, not R; even a better',
        'ER predictor could not repair all-reference R errors by construction.', '',
        '## Post-Hoc Conditional Description, Not A Primary Or Safety Pass', '',
        'All-view loss intervals remain not_estimable because some required selected',
        'sets are empty. The following means condition on nonempty selected known',
        'rows solely to describe failure direction. No CI or complete-roster gain is',
        'claimed; excluded counts stay explicit and the original NA results remain.', '',
        '|Arm|Defined /348|Selected-minus-old row MSE reduction|Selected-minus-old record-mass squared-error reduction|',
        '|---|---:|---:|---:|'])
    conditional = {}
    for arm in run.ARMS:
        take = [r for r in rows if r['arm'] == arm and r['policy'] == 'joint_free']
        defined = [r for r in take if r['models']['old_neural']['known_row_normalized_MSE'] is not None]
        values = {loss: float(s.np.mean([r['models']['old_neural'][loss] - r['models']['C_selected'][loss]
            for r in defined])) for loss in ('known_row_normalized_MSE', 'known_record_normalized_mass_squared_error')}
        conditional[arm] = dict(defined=len(defined), required=len(take), conditional_descriptive_only=True, **values)
        lines.append(f'|{arm}|{len(defined)}|' + '|'.join(f'{v:+.7g}' for v in values.values()) + '|')
    lines.extend(['', 'In the shared-policy selected region, lower row MSE coexists with worse',
        'record-mass error. Neither this conditional description nor the prior',
        'full-roster MSE comparison establishes a safe controller. A controlled',
        'aggregation-loss experiment is warranted; it is not permission to adjust',
        'Hthresholds, drop unsupported recordings or change the original2%contract.', '',
        'All23supported source groups are exposed development; original24primary',
        'remains not_estimable. Reserved roles remain closed. No trajectory or new',
        'policy was run. No metric/seconds/physical-safety/true3D/foundation/human-gold',
        'claim; Stage5C and SMC off.', ''])
    s.once(public / 'conditional_description.json', s.stable(conditional))
    text_once(public / 'readout.md', '\n'.join(lines))
    text_once(public / 'operation_zh.md', '''# 固定选择区域复核

```bash
.venv-pytorch/bin/python scripts/audit_m3w_selected_reference.py run
.venv-pytorch/bin/python scripts/audit_m3w_selected_reference.py verify
.venv-pytorch/bin/python scripts/report_m3w_selected_reference.py
```

run重新推理原神经头和新参考头，并核对原哈希；verify复用已核验私有参考向量，
仅重算标量。均不重新优化MILP或改变选择。空视图保留为不可估，不计为安全零。
表中非空视图均值只是事后条件描述，不是完整主要结果或部署收益。
私有缓存、权重和原数据不得提交。无新训练，Stage5C/SMC关闭。
完整逐录像聚合账本保留在本地groups/并单独忽略，不提交8.5MiB派生账本。
''')
    print(s.json.dumps(dict(complete=True, views=len(rows), diagnostic_only=True, conditional=conditional)))


if __name__ == '__main__':
    main()
