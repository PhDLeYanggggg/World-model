"""Full first-fit resume equivalence and future-free real inference replay."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_european_causal_descriptor_refit as run


def main():
    p = argparse.ArgumentParser(); p.add_argument('--phase', choices=['fit', 'causal'], required=True); args = p.parse_args()
    run.base.torch.set_num_threads(4); run.base.torch.set_num_interop_threads(1)
    cfg, data, jobs, oid, pid, pbound, bound, identity = run.load()
    if args.phase == 'fit':
        c = next(run.base.floor_api.contexts(data, jobs, oid))
        name, fit, held, y, pr, old, mse, cs, control, u, ident, rec = run.prepare(c, data, 0, pid, pbound, bound, identity)
        path = run.PRIVATE/'heads'/name/'checkpoint.pt.gz'
        original = run.api.read_checkpoint(path)
        home = run.PRIVATE/'replay_first_fit'
        model, info = run.api.fit(c['x'][fit], u[fit], y, data['sites'][c['ids'][fit]], c['env'][fit], pr,
            seed=cs['seed'], settings=cfg['head_training'], identity=ident, directory=home,
            heartbeat=lambda **kw: run.beat(head='first_fit_replay', **kw), resume=(home/'checkpoint.pt.gz').exists())
        replay = run.api.read_checkpoint(home/'checkpoint.pt.gz')
        for k in ('model','optimizer','sampler_rng','torch_rng','draws','trace','initial_model','descriptor_preprocess','preprocess'):
            run.api.exact(original[k], replay[k])
        run.api.assert_matched(replay, cs)
        run.base.immutable_json(run.PUBLIC/'training_replay.json', dict(exact=True, full_fits=1,
            optimizer_rng_draws_loss_trace_exact=True, continuous_matches_pilot_resumed=True,
            reference=run.base.artifact(path), replay=run.base.artifact(home/'checkpoint.pt.gz')))
    else:
        causal = {k: data[k] for k in ('sites','history','origin','geometry','recordings','frames')}
        c = next(run.base.floor_api.contexts(causal, jobs, oid))
        name, fit, held, ridge, old, ref = run.base.parent.inherited(c, causal, 0, pid)
        state = run.api.read_checkpoint(run.PRIVATE/'heads'/name/'checkpoint.pt.gz')
        assert state['identity']['experiment'] == identity
        doc = run.done(run.PRIVATE/'heads'/name/'complete.json', state['identity'])
        pr, upr = state['preprocess'], state['descriptor_preprocess']
        u = run.api.descriptors(causal['geometry'][c['ids']], c['floor'], c['prediction'], c['x'], pr)
        prediction = run.api.predict(run.model_from(state), c['x'][held], u[held], c['env'][held], pr, upr)
        assert run.base.inter.array_hash(prediction) == doc['prediction_sha256']
        with np.load(run.base.PRIVATE/'heads'/name/'scores.npz', allow_pickle=False) as z: control = z['scores'].copy()
        with np.load(run.base.parent.PRIVATE/'heads'/(name+'_mse')/'scores.npz', allow_pickle=False) as z: mse = z['scores'].copy()
        run.write_arrays(run.PRIVATE/'heads'/name/'decisions.npz', run.actions(c, causal, held, old, mse, control, prediction), True)
        original = run.base.api.initialize(pr, cfg['head_training']['width'], state['seed'], state['mean_envelope'])
        model = run.api.initialize(pr, cfg['head_training']['width'], state['seed'], state['mean_envelope'])
        x = run.base.torch.from_numpy(np.clip((c['x'][held[:64]].astype(float)-pr['mean'])/pr['std'], -pr['clip'], pr['clip']).astype(np.float32))
        v = run.base.torch.from_numpy(np.clip((u[held[:64]].astype(float)-upr['mean'])/upr['std'], -upr['clip'], upr['clip']).astype(np.float32))
        e = run.base.torch.from_numpy((c['env'][held[:64]]/pr['cost_scale']).astype(np.float32))
        run.api.exact(original(x,e), model(x,e,v))
        run.base.immutable_json(run.PUBLIC/'causal_replay.json', dict(exact=True, groups=1,
            future_fields_removed=True, real_initial_function_equal=True, descriptor_features=len(run.api.FEATURES)))
    print(json.dumps(dict(exact=True, phase=args.phase)))


if __name__ == '__main__': main()
