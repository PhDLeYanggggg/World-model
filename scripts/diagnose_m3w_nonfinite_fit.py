"""Replay one failed fit in isolation; preserve the real training checkpoint."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'mps'], required=True)
    parser.add_argument('--max-steps', type=int, default=100)
    args = parser.parse_args()
    if platform.system() == 'Darwin' and platform.machine() != 'arm64':
        raise SystemExit('arm64 required before Torch import')
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
    import torch
    from src.evaluation.m3w_experiment_contract import ExperimentContract, file_digest
    from src.world_model.m3w_supervised_intervention import (
        ContractForecastDataset, build_forecaster, collate_forecasts,
        forecast_smooth_l1, model_dependencies)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    checkpoint = ROOT / 'data/stage_cvpr2027_experiments/8to12_eqmotion_v5/seed17_hold2/latest.pt'
    digest = file_digest(checkpoint)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    identity = state['identity']
    contract = ExperimentContract(json.loads((ROOT / 'configs/m3w_8to12_continuous_context_v5.json').read_text()), ROOT)
    if (identity['protocol_sha256'] != contract.digest
            or identity['source_code_sha256'] != file_digest(ROOT / 'src/world_model/m3w_supervised_intervention.py')):
        raise ValueError('Changed replay identity')
    dataset = ContractForecastDataset(contract, identity['fit_recordings'], purpose='fit', baseline_name=identity['baseline_name'])
    model = build_forecaster(identity['architecture']).to(args.device)
    if model_dependencies(model) != identity['model_dependencies']:
        raise ValueError('Changed author core')
    model.load_state_dict(state['model'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=identity['settings']['learning_rate'])
    optimizer.load_state_dict(state['optimizer'])
    torch.set_rng_state(state['torch_rng'])
    generator = torch.Generator()
    generator.set_state(state['sampler_rng'])
    order, cursor, trace, failure = state['order'], state['cursor'], [], None

    def finite_summary(tensor):
        t = tensor.detach().cpu()
        finite = torch.isfinite(t)
        return {'shape': list(t.shape), 'nonfinite': int((~finite).sum()),
                'max_abs_finite': float(t[finite].abs().max()) if finite.any() else None}

    def inspect_outputs(output):
        if isinstance(output, torch.Tensor):
            return [finite_summary(output)]
        if isinstance(output, (tuple, list)):
            return [s for item in output for s in inspect_outputs(item)]
        return []

    for step in range(state['step'], args.max_steps):
        if cursor == len(order):
            order, cursor = torch.randperm(len(dataset), generator=generator), 0
        ids = order[cursor:cursor + identity['settings']['batch_size']].tolist()
        batch = collate_forecasts([dataset[i] for i in ids])
        inputs = {k: v.to(args.device) for k, v in batch['inputs'].items()}
        optimizer.zero_grad(set_to_none=True)
        output = model(inputs)
        try:
            loss = forecast_smooth_l1(output, batch['target'].to(args.device), batch['target_mask'].to(args.device))
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        except (ValueError, RuntimeError) as exc:
            failure = {'attempted_step': step + 1, 'error': str(exc),
                'inputs': {k: finite_summary(v) for k, v in inputs.items() if v.is_floating_point()},
                'target': finite_summary(batch['target']), 'prediction': finite_summary(output),
                'identities': batch['identities'], 'module_trace': []}
            handles = []
            for name, module in model.named_modules():
                if name:
                    def hook(_, __, value, name=name):
                        failure['module_trace'].append({'module': name, 'outputs': inspect_outputs(value)})
                    handles.append(module.register_forward_hook(hook))
            with torch.no_grad():
                model(inputs)
            for handle in handles:
                handle.remove()
            cpu_model = deepcopy(model).cpu().double()
            cpu_inputs = {k: v.cpu().double() if v.is_floating_point() else v.cpu() for k, v in inputs.items()}
            with torch.no_grad():
                failure['same_weights_cpu_float64_forward'] = finite_summary(cpu_model(cpu_inputs))
            local = ROOT / 'data/stage_cvpr2027_experiments/nonfinite_fit_diagnostic'
            local.mkdir(parents=True, exist_ok=True)
            torch.save({'model': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                'batch_indices': ids, 'attempted_step': step + 1, 'source_checkpoint_sha256': digest},
                local / f'{args.device}_pre_failure.pt')
            break
        optimizer.step()
        trace.append({'step': step + 1, 'loss': float(loss.detach().cpu()), 'gradient_norm': float(norm.detach().cpu())})
        cursor += len(ids)
    if file_digest(checkpoint) != digest:
        raise ValueError('Original checkpoint unexpectedly changed')
    report = {'result_source': 'fresh_run_isolated_failed_fit_replay_not_completed_training',
        'protocol_sha256': contract.digest, 'source_checkpoint_sha256': digest,
        'device': args.device, 'torch_version': str(torch.__version__), 'trace': trace, 'failure': failure,
        'original_checkpoint_unchanged': True, 'development_labels_used': False,
        'script_sha256': file_digest(Path(__file__))}
    out = ROOT / 'outputs/publication_readiness_2026_09/8to12_public_predictors_v5'
    out.mkdir(parents=True, exist_ok=True)
    (out / f'nonfinite_replay_{args.device}.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'device': args.device, 'completed_steps': len(trace), 'failure_step': failure['attempted_step'] if failure else None,
                      'float64_forward': failure['same_weights_cpu_float64_forward'] if failure else None}))


if __name__ == '__main__':
    main()
