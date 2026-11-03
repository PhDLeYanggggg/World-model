"""Verify actual fitted checkpoint roles after the zero-update wrapper failure."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_interaction_feature_cost as run
import numpy as np
import torch


def main():
    torch.set_num_threads(1)
    reg = json.loads(run.REG.read_text())
    freeze = json.loads((run.PUBLIC/'training_freeze.json').read_text())
    receipt = json.loads((run.parent.prior.SOURCE/'receipt.json').read_text())
    site_path = run.parent.prior.SOURCE/'sites.npy'
    if run.sha(site_path)!=receipt['arrays']['sites']:
        raise ValueError('SOURCE row-site identity changed')
    sites = np.load(site_path,mmap_mode='r',allow_pickle=False)
    jobs = {j['key']:j for j in reg['jobs']}
    rows = []
    for ref in freeze['fits']:
        meta = json.loads(run.checked(ref).read_text())
        job = jobs[meta['job']['key']]
        state = run.api.base.base.read_checkpoint(run.checked(meta['checkpoint']))
        ids = state['fit_ids']
        expected = set(job['critic_sites'])
        if (state['identity']['registration_sha256']!=run.sha(run.REG)
                or state['identity']['job']!=job or state['identity']['feature_arm']!=meta['arm']
                or set(sites[ids])!=expected or set(state['source_sites'])!=expected
                or set(state['preprocess']['source_names'])!=expected
                or set(sites[ids])&set(job['producer_sites']+job['evaluation_sites'])
                or run.parent.prior.frozen.fingerprint(ids)!=job['fit_scope']['ids_sha256']
                or state['preprocess']['fitting_rows']!=len(ids)
                or state['preprocess']['mean'].shape!=(508,) or state['preprocess']['std'].shape!=(508,)):
            raise ValueError('Actual checkpoint fitting/preprocessing role mismatch')
        rows.append(dict(job=job['key'],arm=meta['arm'],seed=state['seed'],fit_rows=len(ids),
                         fit_ids_sha256=job['fit_scope']['ids_sha256'],source_sites=sorted(expected),
                         normalizer_sites=state['preprocess']['source_names'],checkpoint=meta['checkpoint'],
                         producer_and_evaluation_excluded=True))
        del state
    if len(rows)!=36 or len({(r['job'],r['arm']) for r in rows})!=36:
        raise ValueError('Every current fitted checkpoint required')
    result = dict(result_source='fresh_actual_checkpoint_role_audit_not_new_fit_or_held_policy_score',pid=os.getpid(),
                  registration=run.artifact(run.REG),training_freeze=run.artifact(run.PUBLIC/'training_freeze.json'),
                  auditor=run.artifact(Path(__file__)),checkpoints=rows,all36_fit_and_normalizer_roles_G_exact=True,
                  future_label_arrays_opened_by_this_auditor=False,initial_failed_H_preprocessing_discarded=True,
                  independent_confirmation=False,deployment_changed=False)
    run.once(run.PUBLIC/'actual_fit_role_audit.json',result)
    print(json.dumps(dict(pid=os.getpid(),all36_fit_and_normalizer_roles_G_exact=True)))


if __name__=='__main__':
    main()
