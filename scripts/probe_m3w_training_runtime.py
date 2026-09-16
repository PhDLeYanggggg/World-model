"""Actual backward/optimizer/checkpoint test; not a forecasting experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--checkpoint-every", type=int, default=60)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--recording", default="ucy_students03")
    return parser.parse_args()


def main():
    args = parse_args()
    if platform.system() == "Darwin" and platform.machine() != "arm64":
        raise SystemExit("Refusing Rosetta/x86_64 before importing Torch. Use .venv-pytorch/bin/python.")
    if any(x < 1 for x in (args.threads, args.steps, args.rows, args.batch_size, args.checkpoint_every)):
        raise SystemExit("Counts must be positive")
    if not args.run_id.replace("_", "").replace("-", "").isalnum():
        raise SystemExit("run-id must be a safe filename")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = str(args.threads)
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    from src.data_unification.m3w_causal_recordings import RecordingWindows, PROTOCOL_STEPS

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(1701)
    local_dir = ROOT / "data/stage_cvpr2027_causal/runtime" / args.run_id
    report_dir = ROOT / "outputs/publication_readiness_2026_09"
    local_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = local_dir / "latest.pt"
    if checkpoint.exists() and not args.resume:
        raise SystemExit("Run exists; use --resume or a new run-id")
    # Allocate on the requested backend; no device resource inventory or silent CPU fallback.
    device = torch.device(args.device)
    torch.zeros(1, device=device)
    start = time.perf_counter()
    reader = RecordingWindows(ROOT / "data/stage_cvpr2027_causal" / args.recording)
    eligible = np.flatnonzero(reader.index["protocol"] == PROTOCOL_STEPS)
    ids = eligible[np.linspace(0, len(eligible) - 1, min(args.rows, len(eligible)), dtype=int)]
    inputs, targets = [], []
    for idx in ids:
        past = reader.get_inputs(int(idx))
        history = past["history_xy"]
        # Reconstruct two masked PAST positions; get_labels is deliberately never called.
        neighbors = past["neighbor_xy"][:, :6]
        masks = past["neighbor_mask"][:, :6]
        context = (neighbors * masks[..., None]).sum(axis=1) / np.maximum(masks.sum(axis=1, keepdims=True), 1)
        token = np.concatenate([history[:6], context], axis=0)
        inputs.append(token)
        targets.append(history[6:].reshape(-1))
    x = torch.tensor(np.asarray(inputs), dtype=torch.float32)
    y = torch.tensor(np.asarray(targets), dtype=torch.float32)
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise RuntimeError("Nonfinite engineering data")
    loader = DataLoader(TensorDataset(x, y), batch_size=args.batch_size,
                        shuffle=False, num_workers=0, pin_memory=False)
    batches = list(loader)
    data_seconds = time.perf_counter() - start
    signature = hashlib.sha256(x.numpy().tobytes() + y.numpy().tobytes()).hexdigest()

    class Probe(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Linear(2, 64)
            self.position = nn.Parameter(torch.zeros(1, x.shape[1], 64))
            layer = nn.TransformerEncoderLayer(64, 4, 128, dropout=0.0, batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
            self.head = nn.Linear(64, 4)

        def forward(self, values):
            return self.head(self.encoder(self.embed(values) + self.position).mean(1))

    model = Probe().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    config = {k: v for k, v in vars(args).items() if k not in ("resume", "steps")}
    losses, elapsed, first_step = [], 0.0, 0
    if args.resume:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state["config"] != config or state["data_sha256"] != signature:
            raise RuntimeError("Resume config/data identity changed")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        for values in optimizer.state.values():
            for key, value in values.items():
                if torch.is_tensor(value):
                    values[key] = value.to(device)
        torch.set_rng_state(state["rng"])
        first_step, losses, elapsed = state["step"], state["losses"], state["training_seconds"]
    if first_step >= args.steps:
        raise SystemExit("Checkpoint already reaches requested steps; request more steps")

    def sync():
        if args.device == "mps":
            torch.mps.synchronize()

    def update(net, opt, step):
        bx, by = batches[step % len(batches)]
        opt.zero_grad(set_to_none=True)
        prediction = net(bx.to(device))
        loss = nn.functional.smooth_l1_loss(prediction, by.to(device))
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite loss")
        loss.backward()
        norm = nn.utils.clip_grad_norm_(net.parameters(), 1.0, error_if_nonfinite=True)
        opt.step()
        return float(loss.detach().cpu()), float(norm.detach().cpu())

    sync()
    training_start = time.perf_counter()
    heartbeat = local_dir / "heartbeat.jsonl"
    maximum_norm = 0.0
    for step in range(first_step, args.steps):
        loss, norm = update(model, optimizer, step)
        losses.append(loss)
        maximum_norm = max(maximum_norm, norm)
        if (step + 1) % 20 == 0 or step + 1 == args.steps:
            item = {"pid": os.getpid(), "step": step + 1, "target_steps": args.steps,
                    "loss": loss, "elapsed_seconds": elapsed + time.perf_counter() - training_start}
            with heartbeat.open("a") as stream:
                stream.write(json.dumps(item) + "\n")
            print(json.dumps(item), flush=True)
        if (step + 1) % args.checkpoint_every == 0 or step + 1 == args.steps:
            sync()
            state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "rng": torch.get_rng_state(), "step": step + 1, "losses": losses,
                     "config": config, "data_sha256": signature,
                     "training_seconds": elapsed + time.perf_counter() - training_start}
            temp = checkpoint.with_suffix(".tmp")
            torch.save(state, temp)
            os.replace(temp, checkpoint)
    sync()
    training_seconds = elapsed + time.perf_counter() - training_start
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    restored = Probe().to(device)
    restored.load_state_dict(state["model"])
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=3e-4)
    restored_optimizer.load_state_dict(state["optimizer"])
    original_loss, _ = update(model, optimizer, args.steps)
    restored_loss, _ = update(restored, restored_optimizer, args.steps)
    delta = max(float((a - b).abs().max().detach().cpu()) for a, b in zip(model.parameters(), restored.parameters()))
    resume_ok = delta <= (1e-6 if args.device == "mps" else 1e-8)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mib = rss / (1024 ** 2 if sys.platform == "darwin" else 1024)
    artifact = {
        "result_source": "fresh_run", "scope": "engineering_training_runtime_only",
        "pid": os.getpid(), "python": sys.executable, "architecture": platform.machine(),
        "torch_version": str(torch.__version__), "device": args.device,
        "torch_threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
        "dataloader_workers": loader.num_workers, "config": config, "data_sha256": signature,
        "source_points_sha256": reader.metadata["artifacts"]["points.npy"]["sha256"],
        "unique_engineering_rows": len(ids), "steps": args.steps, "resumed_from_step": first_step,
        "future_labels_read": False, "official_split_assigned": False,
        "objective": "masked historical position reconstruction, not forecasting",
        "parameters": sum(p.numel() for p in model.parameters()),
        "data_preparation_seconds": data_seconds, "training_seconds_including_checkpoints": training_seconds,
        "training_steps_per_second": args.steps / training_seconds,
        "training_rows_per_second": sum(len(batches[i % len(batches)][0]) for i in range(args.steps)) / training_seconds,
        "first20_mean_loss": float(np.mean(losses[:20])), "last20_mean_loss": float(np.mean(losses[-20:])),
        "all_loss_finite": bool(np.isfinite(losses).all()), "max_gradient_norm_current_invocation": maximum_norm,
        "peak_rss_mib": peak_mib, "checkpoint": str(checkpoint.relative_to(ROOT)),
        "resume_optimizer_step_loss_difference": abs(original_loss - restored_loss),
        "resume_optimizer_step_max_parameter_difference": delta, "resume_optimizer_step_pass": resume_ok,
        "runtime_checks_pass": bool(resume_ok and np.isfinite(losses).all()),
        "scientific_model_lift_claim": False, "submission_ready": False,
        "torch_build_config": torch.__config__.show(),
        "limits": "Short runtime test, not a 12-hour stability guarantee; timing includes checkpoint IO and scalar synchronization.",
    }
    (report_dir / f"runtime_{args.run_id}.json").write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps({k: artifact[k] for k in ("steps", "device", "torch_threads", "runtime_checks_pass", "training_rows_per_second", "peak_rss_mib")}), flush=True)
    if not artifact["runtime_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
