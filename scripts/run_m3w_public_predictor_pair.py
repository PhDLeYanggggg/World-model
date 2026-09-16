"""Sequential fixed-budget public/local comparison with child resume and logs."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    for family, config, device in (
        ('eqmotion', 'm3w_eqmotion_8to12_pruned_v4.json', 'mps'),
        ('transformer', 'm3w_transformer_8to12_v3.json', 'cpu'),
    ):
        print(f'Starting or resuming {family}; no completed-result claim', flush=True)
        subprocess.run([sys.executable, str(ROOT / 'scripts/run_m3w_8to12_development.py'),
            '--protocol', str(ROOT / 'configs/m3w_8to12_public_predictors_v4.json'),
            '--config', str(ROOT / 'configs' / config),
            '--study-dir', str(ROOT / f'data/stage_cvpr2027_experiments/8to12_{family}_v4'),
            '--device', device, '--threads', '4'], cwd=ROOT, check=True)
        subprocess.run([sys.executable, str(ROOT / 'scripts/summarize_m3w_8to12_development.py'),
            '--protocol', str(ROOT / 'configs/m3w_8to12_public_predictors_v4.json'),
            '--study-dir', str(ROOT / f'data/stage_cvpr2027_experiments/8to12_{family}_v4'),
            '--report-dir', str(ROOT / f'outputs/publication_readiness_2026_09/8to12_{family}_v4')],
            cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
