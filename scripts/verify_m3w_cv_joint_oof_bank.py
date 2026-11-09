"""Audit all donor checkpoints and replay a fixed block of each prediction file."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_cv_joint_oof_bank as study


def main():
    torch, np = study.torch, study.np
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    reg, data, groups, settings, arch, ade = study.setup(lambda **_: None)
    parent = study.OUT / 'freeze.json'
    freeze = study.read(parent)
    if freeze['registration'] != reg or freeze['fits'] != 18 or freeze['a_labels_accessed']:
        raise ValueError('Require a complete unexposed donor bank')
    rows = []
    for cell in freeze['cells']:
        r = cell['identity']
        d = study.design(data['sites'], groups[r['fold']], ade)
        ck = torch.load(study.checked(cell['checkpoint']), map_location='cpu', weights_only=False)
        np.testing.assert_array_equal(ck['train_ids'], d['train_ids'])
        np.testing.assert_array_equal(ck['factors'], d['factors'])
        ids = d['held_ids']
        if ck['identity'] != r or ck['step'] != study.STEPS or ck['draws'][ids].any():
            raise ValueError('Donor training source or checkpoint changed')
        net = study.SourceForecaster(arch[r['family']], 4)
        net.load_state_dict(ck['model'])
        with np.load(study.checked(cell['forecast']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], data['ids'][ids])
            np.testing.assert_array_equal(z['prediction'][:128], study.predict(net, {'geometry': data['geometry']}, ids[:128], 128))
        rows.append(dict(fold=r['fold'], family=r['family'], seed=r['seed'], checked_forecast_rows=128,
                         training_draws=int(ck['draws'].sum()), held_source_training_draws=0))
    study.once(study.OUT / 'verification.json', study.stable(dict(parent=study.artifact(parent),
        verified_checkpoints=len(rows), exact_block_rows=sum(x['checked_forecast_rows'] for x in rows),
        same_donor_paths_checked_by_hash=True, a_labels_opened=False, independent_confirmation=False, cells=rows)))
    print(dict(checkpoints=len(rows), exact_rows=sum(x['checked_forecast_rows'] for x in rows)))


if __name__ == '__main__':
    main()
