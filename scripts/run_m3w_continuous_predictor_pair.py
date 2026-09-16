"""Run both registered predictors on the repaired continuous-identity protocol."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    for family, config, device in (
        ('eqmotion', 'm3w_eqmotion_8to12_pruned_v4.json', 'mps'),
        ('transformer', 'm3w_transformer_8to12_v3.json', 'cpu'),
    ):
        print(f'Starting or resuming continuous-context {family}; not complete', flush=True)
        subprocess.run([sys.executable, str(ROOT / 'scripts/run_m3w_8to12_development.py'),
            '--protocol', str(ROOT / 'configs/m3w_8to12_continuous_context_v5.json'),
            '--config', str(ROOT / 'configs' / config),
            '--study-dir', str(ROOT / f'data/stage_cvpr2027_experiments/8to12_{family}_v5'),
            '--device', device, '--threads', '4'], cwd=ROOT, check=True)
        subprocess.run([sys.executable, str(ROOT / 'scripts/summarize_m3w_8to12_development.py'),
            '--protocol', str(ROOT / 'configs/m3w_8to12_continuous_context_v5.json'),
            '--study-dir', str(ROOT / f'data/stage_cvpr2027_experiments/8to12_{family}_v5'),
            '--report-dir', str(ROOT / f'outputs/publication_readiness_2026_09/8to12_{family}_v5')],
            cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
