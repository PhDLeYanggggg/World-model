"""Pinned author EqMotion core with a past-only, fixed-head K=1 interface.

This is an adapted deterministic control, not a reproduction of best-of-20
paper metrics. The original trainer, preprocessing and pretrained weights are
never imported. Third-party source stays in the ignored local source directory.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import platform
import sys
import types

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Use arm64 .venv-pytorch before importing Torch')

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / 'configs/m3w_eqmotion_source.json'
INPUT_KEYS = {'history', 'history_mask', 'neighbors', 'neighbor_mask',
              'baseline', 'prediction_time', 'request_mask'}


def verified_source(source_directory=None):
    """Check every pinned byte before any downloaded Python is executed."""
    spec_bytes = SPEC.read_bytes()
    spec = json.loads(spec_bytes)
    directory = Path(source_directory or ROOT / spec['destination']).resolve()
    contents, hashes = {}, {}
    for relative, expected in spec['files'].items():
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory):
            raise ValueError('EqMotion source path escapes its directory')
        if not path.is_file():
            raise ValueError('Pinned EqMotion source missing; run scripts/fetch_m3w_eqmotion_source.py')
        content = path.read_bytes()
        blob = hashlib.sha1(f'blob {len(content)}\0'.encode() + content).hexdigest()
        if len(content) != expected['size'] or blob != expected['git_blob_sha1']:
            raise ValueError(f'EqMotion source identity mismatch: {relative}')
        contents[relative] = content
        hashes[relative] = hashlib.sha256(content).hexdigest()
    return contents, {'repository': spec['repository'], 'commit': spec['commit'],
                      'license': spec['license'], 'files_sha256': hashes,
                      'manifest_sha256': hashlib.sha256(spec_bytes).hexdigest(),
                      'adapter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def load_author_core(source_directory=None):
    contents, identity = verified_source(source_directory)
    namespace = '_m3w_eqmotion_' + identity['manifest_sha256']
    if namespace + '.model_t' not in sys.modules:
        package = types.ModuleType(namespace)
        package.__path__ = []
        sys.modules[namespace] = package
        try:
            for name in ('gcl_t', 'model_t'):
                relative = f'eth_ucy/{name}.py'
                tree = ast.parse(contents[relative], filename=relative)
                if name == 'model_t':
                    imports = [node for node in ast.walk(tree)
                               if isinstance(node, ast.ImportFrom) and node.module == 'eth_ucy.gcl_t']
                    if len(imports) != 1:
                        raise ValueError('Unexpected official core import layout')
                    # Isolate the official module name; model arithmetic is unchanged.
                    imports[0].module = namespace + '.gcl_t'
                module = types.ModuleType(namespace + '.' + name)
                module.__package__ = namespace
                module.__file__ = relative
                exec(compile(tree, relative, 'exec'), module.__dict__)
                sys.modules[module.__name__] = module
        except Exception:
            for key in (namespace, namespace + '.gcl_t', namespace + '.model_t'):
                sys.modules.pop(key, None)
            raise
    return sys.modules[namespace + '.model_t'].EqMotion, identity


class EqMotionFixedHead(nn.Module):
    """One head fixed before fitting/evaluation; never choose it using labels."""
    def __init__(self, *, history_steps, prediction_steps, hidden_nf, channels,
                 layers, fixed_head, source_directory=None):
        super().__init__()
        dims = (history_steps, prediction_steps, hidden_nf, channels, layers, fixed_head)
        if any(type(x) is not int for x in dims) or min(dims[:-1]) < 1 or history_steps < 2:
            raise ValueError('Positive integer EqMotion dimensions required')
        if hidden_nf % 2 or not 0 <= fixed_head < 20:
            raise ValueError('Even hidden width and a fixed head in [0, 19] required')
        core, self.source_identity = load_author_core(source_directory)
        self.history_steps, self.prediction_steps, self.fixed_head = history_steps, prediction_steps, fixed_head
        self.core = core(history_steps, 0, hidden_nf, history_steps, channels, prediction_steps,
                         device='cpu', n_layers=layers, recurrent=True)
        self.evaluation_semantics = {'emitted_candidates': 1, 'author_parallel_heads': 20,
                                    'head_selection': 'fixed_before_fit_no_label_access',
                                    'scope': 'adapted_K1_not_paper_minADE20_minFDE20'}
        for i in range(20):
            if i != fixed_head:
                self.core._modules[f'head_{i}'].requires_grad_(False)
                self.core.predict_head[i].requires_grad_(False)

    def prepare_context(self, inputs):
        if set(inputs) != INPUT_KEYS:
            raise ValueError('Only declared causal input fields are accepted')
        history, neighbors = inputs['history'], inputs['neighbors']
        hm, nm = inputs['history_mask'], inputs['neighbor_mask']
        request, times = inputs['request_mask'], inputs['prediction_time']
        b, k, d = history.shape
        if (k != self.history_steps or d != 3 or neighbors.ndim != 4 or neighbors.shape[0] != b
                or neighbors.shape[2:] != (k, 3) or hm.shape != (b, k) or nm.shape != neighbors.shape[:3]
                or inputs['baseline'].shape != (b, self.prediction_steps, 2)
                or request.shape != (b, self.prediction_steps) or times.shape != request.shape
                or any(m.dtype != torch.bool for m in (hm, nm, request))):
            raise ValueError('EqMotion fixed-grid input shape/mask mismatch')
        if not hm.all() or not request.all():
            raise ValueError('EqMotion requires full observed ego history and a fixed request grid')
        if (not torch.isfinite(history).all() or not torch.isfinite(neighbors[nm]).all()
                or not torch.isfinite(times).all() or not torch.isfinite(inputs['baseline']).all()):
            raise ValueError('Nonfinite observed context/request')
        if (history[..., 2] > 0).any() or (neighbors[..., 2][nm] > 0).any():
            raise ValueError('Context contains a post-current timestamp')
        step = 1. / self.prediction_steps
        expected_past = torch.arange(1 - k, 1, device=history.device, dtype=history.dtype) * step
        expected_future = torch.arange(1, self.prediction_steps + 1, device=history.device, dtype=history.dtype) * step
        if (not torch.allclose(history[..., 2], expected_past.expand(b, -1), atol=1e-6, rtol=1e-6)
                or not torch.allclose(times, expected_future.expand(b, -1), atol=1e-6, rtol=1e-6)):
            raise ValueError('EqMotion requires the same uniform observation/request stride; no interpolation')
        aligned = torch.isclose(neighbors[..., 2], history[:, None, :, 2], atol=1e-6, rtol=1e-6).all(-1)
        eligible = nm.all(-1) & aligned
        positions, counts = [], []
        for i in range(b):
            observed = torch.cat((history[i:i+1, :, :2], neighbors[i, eligible[i], :, :2]), 0)
            counts.append(len(observed))
            positions.append(torch.cat((observed, observed.new_zeros(neighbors.shape[1] + 1 - len(observed), k, 2)), 0))
        x = torch.stack(positions)
        velocity = torch.zeros_like(x)
        velocity[:, :, 1:] = x[:, :, 1:] - x[:, :, :-1]
        velocity[:, :, 0] = velocity[:, :, 1]
        return {'x': x, 'velocity': velocity, 'speed': velocity.norm(dim=-1).detach(),
                'num_valid': torch.tensor(counts, dtype=torch.long, device=x.device),
                'eligible_neighbors': eligible}

    def forward(self, inputs):
        context = self.prepare_context(inputs)
        forecasts, _ = self.core(context['speed'], context['x'], context['velocity'], context['num_valid'])
        return forecasts[:, 0, self.fixed_head]
