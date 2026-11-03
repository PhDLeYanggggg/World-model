"""Forward empty-support adapter; original source experiment is preserved."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_moving_source_forecast as base
from src.evaluation.m3w_moving_forecast_support import fitting_design_with_empty_groups,qualify_supported_analysis

PUBLIC=ROOT/'outputs/publication_readiness_2026_09/moving_source_forecast_supported_v1'
PRIVATE=ROOT/'data/stage_cvpr2027_experiments/moving_source_forecast_supported_v1'
CONFIG=ROOT/'configs/m3w_moving_source_forecast_supported_v1.json'
PARENT_PUBLIC,PARENT_PRIVATE=base.PUBLIC,base.PRIVATE
FILES=(*base.FILES,Path(__file__),CONFIG,PUBLIC/'protocol.md',
       ROOT/'src/evaluation/m3w_moving_forecast_support.py',ROOT/'tests/test_m3w_moving_forecast_support.py')


def main():
    old_cfg,admission=base.register()
    cfg=json.loads(CONFIG.read_text());extra={'name','structurally_empty_groups','supported_groups_expected',
        'declared_groups_retained','source_rows_unchanged','full_declared_primary_status','supported_group_contrast_role'}
    if {k:v for k,v in cfg.items() if k not in extra}!={k:v for k,v in old_cfg.items() if k!='name'}:
        raise ValueError('Only explicit structural support qualification may change')
    population=json.loads((PARENT_PUBLIC/'population.json').read_text())
    declared=set(admission['proposed_within_fit_threefold_groups']);empty=set(cfg['structurally_empty_groups'])
    if (declared-set(population['groups'])!=empty or len(population['groups'])!=23 or population['rows']!=195286
            or population['complete_rows']!=136295 or population['known_rows']!=190570):
        raise ValueError('Keep original all-row population and exact observed support defect')
    value=dict(bindings={str(p.relative_to(ROOT)):base.sha(p) for p in FILES},configuration=cfg,
        parent_registration_sha256=base.sha(PARENT_PUBLIC/'registration.json'),
        preserved_population_sha256=base.sha(PARENT_PUBLIC/'population.json'),
        original_prepare_log_sha256=base.sha(PARENT_PRIVATE/'prepare.log'),
        original_prepare_pid=77627,original_prepare_exit_code=1,new_model_updates_before_repair=0,
        declared_groups=sorted(declared),prediction_supported_groups=population['groups'],empty_groups=sorted(empty),
        source_roles_and_all_rows_unchanged=True,original_full_group_primary_passed=False,
        supported_metrics_secondary_only=True,stage5c_executed=False,smc_enabled=False)
    original_fit,original_evaluate,original_once=base.fit_design,base.evaluate,base.once
    base.PUBLIC,base.PRIVATE,base.CONFIG,base.FILES=PUBLIC,PRIVATE,CONFIG,FILES
    def register():
        original_once(PUBLIC/'registration.json',base.stable(value))
        return cfg,admission
    base.register=register
    base.fit_design=lambda data,groups,errors:fitting_design_with_empty_groups(original_fit,data,groups,errors,empty)
    def evaluate(*args,**kwargs):
        def write(path,body):
            if path==PUBLIC/'analysis.json':body=qualify_supported_analysis(body,declared,empty)
            return original_once(path,body)
        base.once=write
        try:
            raw=original_evaluate(*args,**kwargs)
        finally:
            base.once=original_once
        return qualify_supported_analysis(raw,declared,empty)
    base.evaluate=evaluate
    base.main()


if __name__=='__main__':main()
