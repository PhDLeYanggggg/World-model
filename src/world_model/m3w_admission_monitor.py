"""Non-intervening training telemetry, verified against frozen scientific state."""
from contextlib import contextmanager
import copy
import hashlib
from pathlib import Path

import torch
from src.world_model import m3w_admission_cost as frozen


def parts(prediction, target, real, rms, **kwargs):
    primary = frozen.parent.ORIGINAL_LOSS(prediction, target, real, rms, **kwargs)['total']
    upper = frozen.parent.score_penalty(prediction, target, real, rms, underprediction_weight=4.)
    extra, _ = frozen.admission_penalty(prediction, target, real, rms)
    with torch.no_grad():
        p = prediction.detach().to(torch.float64)
        gain = p[..., 0] - p[..., 1]
        admitted = (real & (gain > 0) & (p[..., 2] > 0) & (p[..., 3] > 0)
            & (p[..., 1] <= .02*p[..., 2]) & (p[..., 4] <= .02*p[..., 3]))
        chosen = gain.masked_fill(~admitted, -torch.inf).argmax(1)
        selected = admitted.any(1)
    ix = torch.arange(len(prediction), device=prediction.device)
    pred, truth = prediction[ix, chosen], target.detach()[ix, chosen]
    q_pred = pred[:, [1, 4]] - .02*pred[:, [2, 3]]
    q_true = truth[:, [1, 4]] - .02*truth[:, [2, 3]]
    scale = torch.sqrt(rms[[1, 4]].square() + (.02*rms[[2, 3]]).square())
    false = selected[:, None] & (q_true > 0)
    penalty = torch.relu((q_true-q_pred)/scale).square()*false
    channel = penalty.mean(0)/2
    torch.testing.assert_close(channel.sum(), extra)
    mask = false[:, 1] & (pred[:, 1] > 0)
    fractions = (pred[mask, 4]/pred[mask, 1]).detach()
    return dict(base=primary, upper=upper, aux=extra, aux_all=channel[0], aux_easy=channel[1]), dict(
        selected_rows=int(selected.sum()), false_all_rows=int(false[:, 0].sum()),
        false_easy_rows=int(false[:, 1].sum()),
        false_easy_gate_fraction_mean=float(fractions.mean()) if len(fractions) else None,
        false_easy_gate_fraction_under_001=int((fractions < .01).sum()),
        false_easy_gate_fraction_defined=len(fractions))


def vector(loss, parameters):
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1)
        for p, g in zip(parameters, gradients)]).detach()


def cosine(left, right):
    den = float(left.norm()*right.norm())
    return float(left@right)/den if den > 0 else None


class Monitor:
    def __init__(self, arm, settings, saved=None):
        if arm not in ('upper_score', 'admitted_score'):
            raise ValueError('The actual upper/admission trajectory is required')
        self.arm, self.settings = arm, settings
        self.model, self.last_ix, self.known = None, None, None
        self.state = (copy.deepcopy(saved) if saved is not None else dict(step=0, rows=[],
            draw_hash='0'*64, row_draws=0, unknown_draws=0, unknown_supervised_draws=0,
            selected_rows=0, false_all_rows=0, false_easy_rows=0,
            steps_aux_positive=0, first_aux_positive_step=None,
            summed_losses={k: 0. for k in ('base', 'upper', 'aux', 'aux_all', 'aux_easy')}))

    def snapshot(self):
        return copy.deepcopy(self.state)

    def observe(self, prediction, target, real, rms, actual, **kwargs):
        values, info = parts(prediction, target, real, rms, **kwargs)
        expected = values['base'] + values['upper']
        if self.arm == 'admitted_score':
            expected = expected + values['aux']
        if not torch.equal(actual, expected):
            raise ValueError('Telemetry must decompose the exact unmodified training loss')
        self.state['step'] += 1
        step = self.state['step']
        self.state['draw_hash'] = hashlib.sha256(bytes.fromhex(self.state['draw_hash']) + self.last_ix.tobytes()).hexdigest()
        self.state['row_draws'] += len(self.last_ix)
        unknown = ~self.known[self.last_ix]
        supervised = real.any(1)
        self.state['unknown_draws'] += int(unknown.sum())
        self.state['unknown_supervised_draws'] += int((unknown & supervised).sum())
        if (unknown & supervised).any():
            raise ValueError('Unknown futures cannot supervise the monitor or original loss')
        for key in ('selected_rows', 'false_all_rows', 'false_easy_rows'):
            self.state[key] += info[key]
        for key, value in values.items():
            self.state['summed_losses'][key] += float(value.detach())
        if values['aux'].detach() > 0:
            self.state['steps_aux_positive'] += 1
            if self.state['first_aux_positive_step'] is None:
                self.state['first_aux_positive_step'] = step
        if step == 1 or step % self.settings['heartbeat_every'] == 0 or step == self.settings['steps']:
            parameters = list(self.model.parameters())
            previous = [None if p.grad is None else p.grad.detach().clone() for p in parameters]
            gs = {key: vector(values[key], parameters) for key in ('base', 'upper', 'aux_all', 'aux_easy')}
            anchor, auxiliary = gs['base'] + gs['upper'], gs['aux_all'] + gs['aux_easy']
            for parameter, before in zip(parameters, previous):
                if before is None:
                    if parameter.grad is not None:
                        raise ValueError('Telemetry accumulated a training gradient')
                elif not torch.equal(parameter.grad, before):
                    raise ValueError('Telemetry modified a training gradient')
            norm = float(anchor.norm())
            row = dict(step=step, losses={k: float(v.detach()) for k, v in values.items()}, **info,
                gradient_norms={k: float(g.norm()) for k, g in gs.items()}, anchor_gradient_norm=norm,
                auxiliary_over_anchor_gradient=float(auxiliary.norm())/norm if norm else None,
                easy_auxiliary_over_anchor_gradient=float(gs['aux_easy'].norm())/norm if norm else None,
                auxiliary_anchor_cosine=cosine(auxiliary, anchor),
                easy_auxiliary_anchor_cosine=cosine(gs['aux_easy'], anchor),
                actual_auxiliary_used=self.arm == 'admitted_score')
            self.state['rows'].append(row)


@contextmanager
def installed(monitor):
    active = frozen.parent.active
    base, detector = active.completion.base, active.detector
    old_ctor, old_draw, old_save = detector.DetectorCostHead, base.draw_batch, base.save_checkpoint
    with frozen.installed(monitor.arm):
        old_loss = base.continuous_loss
        def ctor(*args, **kwargs):
            model = old_ctor(*args, **kwargs)
            monitor.model = model
            return model
        def draw(*args, **kwargs):
            ix = old_draw(*args, **kwargs)
            monitor.last_ix = ix
            return ix
        def loss(prediction, target, real, rms, **kwargs):
            terms = old_loss(prediction, target, real, rms, **kwargs)
            monitor.observe(prediction, target, real, rms, terms['total'], **kwargs)
            return terms
        def save(path, state):
            if state['step'] != monitor.state['step']:
                raise ValueError('Atomic checkpoint and telemetry step must align')
            old_save(path, dict(state, admission_monitor=monitor.snapshot()))
        detector.DetectorCostHead, base.draw_batch = ctor, draw
        base.continuous_loss, base.save_checkpoint = loss, save
        try:
            yield
        finally:
            detector.DetectorCostHead, base.draw_batch = old_ctor, old_draw
            base.continuous_loss, base.save_checkpoint = old_loss, old_save


def fit(data, *, objective_arm, **kwargs):
    path = Path(kwargs['path'])
    saved = None
    if path.exists() and kwargs.get('resume'):
        checkpoint = frozen.parent.active.completion.base.read_checkpoint(path)
        saved = checkpoint['admission_monitor']
        if saved['step'] != checkpoint['step'] or saved['draw_hash'] != checkpoint['draw_hash']:
            raise ValueError('Resume telemetry and original scientific state differ')
    monitor = Monitor(objective_arm, kwargs['settings'], saved)
    monitor.known = data['known']
    with installed(monitor):
        return frozen.parent.active.fit(data, **kwargs)
