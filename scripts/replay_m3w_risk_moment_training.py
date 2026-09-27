"""Full fixed-first training replay, including its pilot/resume boundary."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_european_risk_moment_crossfit as run
from scripts.replay_m3w_dimensionless_training import exact


def main():
    run.torch.set_num_threads(4); run.torch.set_num_interop_threads(1)
    cfg, data, jobs, identity = run.load()
    c = next(run.contexts(data, jobs, identity))
    held = sorted(set(data['sites'][c['ids']]))[0]
    name, ids, _, pos, y, pr, hid = run.prepare(c, data, identity, held)
    rec = run.done_record(run.PRIVATE/'heads'/name/'complete.json', hid)
    home = run.PRIVATE/'training_replay'/name
    if (home/'checkpoint.pt').exists(): raise FileExistsError('Replay exists; verify without retraining')
    kwargs = dict(seed=hid['seed'], task='all', settings=cfg['head_training'], identity=hid,
        directory=home, heartbeat=lambda **kw: run.beat(head='verification_'+name, **kw))
    run.parent.head.fit(c['x'][pos], y, data['sites'][ids], c['envelope'][pos], pr, stop_at=100, **kwargs)
    _, fit = run.parent.head.fit(c['x'][pos], y, data['sites'][ids], c['envelope'][pos], pr, resume=True, **kwargs)
    a = run.torch.load(home/'checkpoint.pt', map_location='cpu', weights_only=False)
    b = run.torch.load(ROOT/rec['artifacts']['checkpoint']['path'], map_location='cpu', weights_only=False)
    for key in ('identity','settings','task','seed','preprocess','mean_envelope','model','optimizer',
                'sampler_rng','torch_rng','draws','step','trace'): exact(a[key], b[key])
    run.immutable_json(run.PUBLIC/'training_replay.json', dict(result_source='fresh_verification_only_training',
        head=name, steps=fit['step'], fit_seconds=fit['seconds'], replay_pid=run.os.getpid(),
        parameters_optimizer_rng_draws_losses_exact=True, checkpoint=run.artifact(home/'checkpoint.pt'),
        reference=rec['artifacts']['checkpoint'], new_independent_model=False, cold_raw_rebuild=False))


if __name__ == '__main__': main()
