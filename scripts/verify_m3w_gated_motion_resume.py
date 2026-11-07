"""Verify exact real-data checkpoint recovery for one registered training arm."""
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_gated_motion_residual as m

torch, np, run = m.torch, m.np, m.run


def main():
    _, jobs = m.register()
    full = m.prev.receipt(m.PUBLIC, 'train')
    f = next(v for v in full['fits'] if (v['identity']['seed'], v['identity']['arm']) == (17, 'gated'))
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    start = time.monotonic()

    def beat(**v):
        event = dict(phase='verify_resume', seconds=time.monotonic()-start, **v)
        m.fit_source.source.atomic(m.PRIVATE/'resume_probe_heartbeat.json', event)
        if v.get('state') == 'training':
            print(event, flush=True)

    data = m.fit_source.fitting_only(jobs[0]['identity'], beat)
    g = data['geometry'][:, :16].reshape(-1, 8, 2)
    rms = np.maximum(np.sqrt(np.square(np.diff(g, axis=1)).sum(2).mean(1)), .05)
    fold = dict(train_ids=np.arange(len(g)), held_ids=np.empty(0, dtype=int),
                groups=data['groups'], factors=1./rms)
    directory = m.PRIVATE/'resume_probe'
    cp = directory/'checkpoint.pt'
    torch.manual_seed(17)
    model = m.GatedMotionResidual(True)
    config = dict(seed=17, settings=m.SETTINGS, identity=f['identity'], directory=directory, heartbeat=beat)
    if not cp.exists():
        first = m.native.fit_trial(model, data, fold, stop_at=150, **config)
        if first['step'] != 150 or first['complete']:
            raise ValueError('Expected an incomplete real training checkpoint')
    resumed = m.native.fit_trial(m.GatedMotionResidual(True), data, fold, resume=True, **config)
    if not resumed['complete']:
        raise ValueError('Resume did not complete the fixed training budget')
    old, new = [torch.load(p, map_location='cpu', weights_only=False)
                for p in (run.checked(f['checkpoint']), cp)]
    keys = ('identity', 'step', 'seed', 'model', 'optimizer', 'sampler_rng', 'torch_rng', 'draws', 'losses')
    for k in keys:
        m.fit_source.source.exact(old[k], new[k])
    receipt = m.PUBLIC/'resume_verification.json'
    value = dict(seed=17, arm='gated', checkpoint_at=150, final_step=900,
                 state_fields_exact=list(keys), real_fit_rows=len(g),
                 before_development_scoring=False, external_access=False,
                 original_checkpoint_sha256=f['checkpoint']['sha256'])
    run.once(receipt, run.stable(value))
    print(f'Checkpoint recovery matched all {len(keys)} registered state fields.', flush=True)


if __name__ == '__main__':
    main()
