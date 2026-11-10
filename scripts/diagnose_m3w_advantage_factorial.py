"""Post-freeze diagnostics only; never choose a model or a deployment threshold."""

from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_advantage_factorial as trial
from scripts.audit_m3w_incumbent_slate import probabilities

run, np, torch = trial.run, trial.np, trial.torch


def compact(rows, arm):
    if not rows:
        return dict(slates=0)
    z = np.asarray(rows, dtype=float)
    # Columns are predicted gain, observed gain, keep/chosen full-grid pass,
    # switch, oracle pass, and absolute-head keep/chosen sigmoid scores.
    result = dict(slates=len(z), switches=int(z[:, 4].sum()),
        keep_passes=int(z[:, 2].sum()), chosen_passes=int(z[:, 3].sum()),
        rescues=int((z[:, 1] > 0).sum()), destructions=int((z[:, 1] < 0).sum()),
        oracle_passes=int(z[:, 5].sum()), predicted_gain_mean=float(z[:, 0].mean()),
        observed_gain_mean=float(z[:, 1].mean()),
        gain_mae=float(np.abs(z[:, 0]-z[:, 1]).mean()),
        gain_mse=float(np.square(z[:, 0]-z[:, 1]).mean()))
    if arm.endswith('absolute'):
        result['uncalibrated_sigmoid_diagnostics'] = dict(
            keep=probabilities(list(zip(z[:, 6], z[:, 2], strict=True))),
            chosen=probabilities(list(zip(z[:, 7], z[:, 3], strict=True))))
    else:
        result['score_semantics'] = 'tanh_score_difference_not_a_probability'
    return result


@torch.inference_mode()
def observations(net, arm, x, y, valid):
    scores = net(x)
    choice = trial.model.choose(scores, valid)
    index = torch.arange(len(x))
    keep, chosen = y[:, 0], y[index, choice]
    if arm.endswith('advantage'):
        gain = torch.tanh(scores[index, choice]-scores[:, 0])
        keep_score = chosen_score = torch.zeros_like(gain)
    else:
        keep_score, chosen_score = scores[:, 0].sigmoid(), scores[index, choice].sigmoid()
        gain = chosen_score-keep_score
    values = (gain, chosen.float()-keep.float(), keep, chosen, choice != 0,
              (y & valid).any(1), keep_score, chosen_score)
    return torch.stack([v.float() for v in values], 1).tolist()


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    audit = run.read(trial.OUT/'audit.json')
    freeze = run.read(trial.OUT/'head_freeze.json')
    inner = run.read(trial.OUT/'inner_freeze.json')
    selection = run.read(run.checked(freeze['selection']))
    rows = {role: {arm: [] for arm in trial.ARMS} for role in ('train', 'held')}
    losses = {role: {arm: [] for arm in trial.ARMS} for role in ('inner', 'outer')}
    coverage, by_cell = Counter(), {}
    empty, unresolved = [], []
    for key, item in inner['cells'].items():
        cell = run.read(run.checked(item))
        for held, value in cell['validation'].items():
            support = trial.recording_gain(value['arms'][trial.ARMS[0]], value['audit'])
            coverage.update({k: support[k] for k in (
                'recordings', 'exchange_slates', 'no_exchange', 'unresolved_recordings')})
            entry = dict(outer=held, validation=value['site'], seed=cell['seed'], **support)
            if not support['exchange_slates']:
                empty.append(entry)
            if support['unresolved_recordings']:
                unresolved.append(entry)
        for arm, meta in cell['models'].items():
            z = torch.load(run.checked(meta['artifact']), map_location='cpu', weights_only=True)
            losses['inner'][arm].append(z['trace'])
    for args, _ in trial.previous.matched.context_cells():
        held, seed = args[:2]
        key = f'{held}_{seed}'
        cell = freeze['cells'][key]
        nets = {}
        for arm, meta in cell['models'].items():
            nets[arm] = trial.study.load(meta,
                lambda: trial.model.Selector(arm, meta['identity']['normalization']), 400)
            z = torch.load(run.checked(meta['artifact']), map_location='cpu', weights_only=True)
            losses['outer'][arm].append(z['trace'])
        train = {arm: [] for arm in trial.ARMS}
        held_rows = {arm: [] for arm in trial.ARMS}
        for item in cell['models'][trial.ARMS[0]]['identity']['data'].values():
            meta = run.read(run.checked(item))
            with np.load(run.checked(meta['artifact']), allow_pickle=False) as z:
                x, y, valid = (torch.from_numpy(z[k]) for k in ('x', 'full', 'valid'))
            for start in range(0, len(x), 128):
                for arm, net in nets.items():
                    train[arm].extend(observations(net, arm, x[start:start+128],
                                                  y[start:start+128], valid[start:start+128]))
        picks = run.read(run.checked(cell['choices']))['picks']
        d, bank, menu, cut = args[3], args[4], args[5], args[8]
        for slate, frozen_choice in zip(trial.previous.prior.slate_rows(args), picks, strict=True):
            ai, r, anchor, pairs, aligned, feat = slate
            if aligned is None:
                assert frozen_choice is None
                continue
            x = trial.prior.model.with_incumbent(feat[None])
            plans = [anchor, *(trial.old.model.exchanged(anchor, *p) for p in pairs)]
            labels = []
            for plan in plans:
                prediction, score = trial.old.measure(d, ai, bank, menu[3], r, plan, cut)
                full = trial.previous.prior.prior.certificate(
                    bank[0][ai], prediction, d['y'][ai], d['valid'][ai], cut[0], score)[0]
                labels.append(full)
            y = torch.tensor([labels], dtype=torch.bool)
            valid = torch.ones_like(y)
            for arm, net in nets.items():
                with torch.inference_mode():
                    assert int(trial.model.choose(net(x), valid)) == frozen_choice[arm]
                held_rows[arm].extend(observations(net, arm, x, y, valid))
        by_cell[key] = {role: {a: compact(v, a) for a, v in source.items()}
                        for role, source in (('train', train), ('held', held_rows))}
        for role, source in (('train', train), ('held', held_rows)):
            for arm in trial.ARMS:
                rows[role][arm].extend(source[arm])
        print(key, flush=True)
    trace = {role: {a: dict(models=len(v), steps=[100, 200, 300, 400],
                           mean=np.mean(v, axis=0).tolist(), min=np.min(v, axis=0).tolist(),
                           max=np.max(v, axis=0).tolist()) for a, v in arms.items()}
             for role, arms in losses.items()}
    result = dict(code=run.artifact(Path(__file__)), audit=run.artifact(trial.OUT/'audit.json'),
        registration=audit['registration'], selection=run.artifact(trial.OUT/'selection.json'),
        label='full_grid_pass_difference', normalization='training_only',
        result_source='fresh_run_postfreeze_diagnostic', evidence_role='exposed_development',
        train_role='in_sample_repeated_donors_not_independent', held_role='101_frozen_eligible_views',
        counts={role: {a: compact(v, a) for a, v in arms.items()} for role, arms in rows.items()},
        inner_repeated_coverage=dict(coverage), empty_validation_folds=empty,
        unresolved_validation_folds=unresolved, loss_trace_summary=trace, per_cell=by_cell,
        selected_families={k: v['selected'] for k, v in selection['choices'].items()},
        new_training_updates=0, no_threshold_search=True, deployment_changed=False,
        new_external_outcome_access=False)
    run.once(trial.OUT/'generalization_diagnostic.json', run.stable(result))
    print(run.stable(dict(counts=result['counts'], coverage=dict(coverage))))


if __name__ == '__main__':
    main()
