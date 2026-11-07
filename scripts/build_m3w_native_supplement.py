"""Typeset an existing-estimand supplement without reading outcome arrays."""
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_m3w_native_paper_v10 import latex_document, sha, write_json

PUBLIC = ROOT/'outputs/publication_readiness_2026_09/evidence_manuscript_v10'
BUILD = ROOT/'data/stage_cvpr2027_experiments/evidence_manuscript_v10/supplement_build'
SOURCES = (
    'scripts/run_m3w_moving_prefix_actor.py',
    'src/evaluation/m3w_forecast_cost_bounds.py',
    'tests/test_m3w_native_bound_estimands.py',
    'scripts/build_m3w_native_paper_v10.py',
)


def main():
    template = PUBLIC/'supplementary_math.md'
    text = template.read_text()
    if 'not a bound on complete-grid ADE' not in text or 'rules remain unchanged' not in text:
        raise ValueError('Keep the existing estimand and unchanged protocol scope explicit')
    tex = latex_document(text)
    (PUBLIC/'supplementary_math.tex').write_text(tex)
    BUILD.mkdir(parents=True, exist_ok=True)
    (BUILD/'main.tex').write_text(tex)
    style = BUILD.parent/'paper_build/cvpr.sty'
    shutil.copyfile(style, BUILD/'cvpr.sty')
    write_json(PUBLIC/'supplementary_math_bindings.json', dict(
        result_source='fresh_derivation_typesetting_and_synthetic_tests_not_new_real_data_result',
        source_hashes={p: sha(ROOT/p) for p in SOURCES},
        markdown_sha256=sha(template), tex_sha256=sha(PUBLIC/'supplementary_math.tex'),
        builder_sha256=sha(Path(__file__)), cvpr_style_sha256=sha(style),
        registered_metric_changed=False, registered_training_changed=False,
        new_outcome_readout=False, independent_confirmation=False,
        submission_ready=False, stage5c_executed=False, smc_enabled=False))
    print('Existing-estimand supplement prepared; no real labels opened or metric changed.')


if __name__ == '__main__':
    main()
