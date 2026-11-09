"""Measure held marginal-risk prediction and risk failure decomposition."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import train_m3w_cv_prefix_state as study

old, base, prior, np, torch = study.old, study.base, study.prior, study.np, study.torch
FIELDS = ('gain_mae', 'all_excess_mae', 'easy_excess_mae', 'signed_easy_mae', 'geometry_mae')


def assess(nets, packet, d):
    pred = study.predictions(nets, packet)
    out = {}
    for i, p in enumerate(packet):
        keep = (p['counts'] > 0) & (p['counts'] == p['anchor_count'])
        if not keep.any():
            continue
        y = p['y'][keep]
        errors = np.stack([np.abs(pred[k][i][keep]-y) for k in study.MODES])
        v = np.column_stack((errors[:, :, :2].mean(2).sum(1), errors[:, :, 2:].sum(1)))
        s = str(d['sites'][p['at'][0]])
        z = out.setdefault(s, np.zeros((len(study.MODES), len(FIELDS)+1)))
        z[:, :-1] += v
        z[:, -1] += len(y)
    return {k: z[:, :-1] / z[:, -1:] for k, z in out.items()}


def nets_for(cell, width):
    out = {}
    for k, f in cell['fits'].items():
        x = torch.load(old.checked(f['artifact']), map_location='cpu', weights_only=True)
        if x['identity'] != f['identity'] or x['step'] != study.STEPS:
            raise ValueError('Frozen prefix model identity changed')
        net = prior.model.head(width, 6)
        net.load_state_dict(x['net'])
        out[k] = net.eval()
    return out


def risks(rows):
    result = {}
    for k in study.ARMS:
        rs = [r['metrics'][k]['risk'] for r in rows]
        known = [all(x[z] is not None and x[z] <= .02
            for z in ('known_all_risk', 'known_easy_risk', 'known_easy_net_degradation')) for x in rs]
        result[k] = dict(known_only_passes=sum(known),
            missing_debit_rejections=sum(a and not x['supported'] for a, x in zip(known, rs, strict=True)),
            easy_positive_over=sum(x['completion_easy_risk_upper'] is not None
                and x['completion_easy_risk_upper'] > .02 for x in rs),
            easy_undefined=sum(x['completion_easy_risk_upper'] is None for x in rs),
            wholly_unknown_edits=sum(x['unknown_selected'] for x in rs))
    return result


def main():
    torch.set_num_threads(4)
    f, r = (study.OUT / k for k in ('freeze.json', 'readout.json'))
    freeze, result = map(old.read, (f, r))
    if (result['freeze'] != old.sha(f) or not freeze['before_held_scoring'] or not result['original_2pct']
            or result['a_labels_opened'] or result['independent_confirmation']):
        raise ValueError('Require frozen development scores')
    reg = old.read(study.OUT / 'registration.json')
    shape = old.read(old.checked(reg['parents'][0]))
    base.register()
    d = base.fit_population(lambda **_: None)
    stats = {}
    for cell, parent in zip(freeze['cells'], shape['cells'], strict=True):
        i = cell['identity']
        if i != parent['identity']:
            raise ValueError('Parent roles changed')
        _, h = base.subset(d, i['held'])
        past, _ = study.study.parent.parent.context.context(h)
        net = study.study.loader.load(parent['fits'][1], past.shape[1]+37)
        *_, packet = study.prepare(h, net, cell['cut'][0])
        study.shadow_packet(packet, train=True)
        nets = nets_for(cell, packet[0]['state_x'].shape[1])
        for k, v in assess(nets, packet, h).items():
            stats[k, i['seed']] = v
    sites = sorted({s for s, _ in stats})
    if len(sites) != 9 or len(stats) != 27:
        raise ValueError('Need all held source-seed prediction cells')
    x = np.asarray([[stats[k, s] for s in base.SEEDS] for k in sites])
    draw = np.random.default_rng(73107).integers(len(sites), size=(3000, len(sites)))
    acc = {k: dict(masked=float(x[:, :, 0, j].mean()), state=float(x[:, :, 1, j].mean()),
        masked_minus_state=prior.geometry.bootstrap(x[:, :, 0, j]-x[:, :, 1, j], draw))
        for j, k in enumerate(FIELDS)}
    rows = study.evaluate(freeze['cells'], d, lambda **_: None, export=True)
    roster = {x['recording']: x['source'] for x in rows}
    if prior.prefix.grouped.grouped_readout(rows, roster, base.SEEDS) != result['summary']:
        raise ValueError('Frozen readout changed')
    out = dict(parent=old.artifact(r), code=old.artifact(Path(__file__)), prediction=acc, risk=risks(rows),
        diagnostic_only=True, source_seed_cells=27, original_2pct=True,
        a_labels_opened=False, independent_confirmation=False)
    old.once(study.OUT / 'prediction_audit.json', old.stable(out))
    print(out, flush=True)


if __name__ == '__main__':
    main()
