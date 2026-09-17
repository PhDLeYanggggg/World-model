"""Run both frozen residual arms through every seed, OOF head and development test."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    for mode in ('residual_skip', 'motion_bounded'):
        print(f'Starting or resuming {mode}; no completed-result claim', flush=True)
        subprocess.run([sys.executable, str(ROOT/'scripts/run_m3w_8to12_development.py'),
            '--protocol', str(ROOT/'configs/m3w_8to12_residual_parameterization_v7.json'),
            '--config', str(ROOT/f'configs/m3w_transformer_8to12_{mode}_v7.json'),
            '--study-dir', str(ROOT/f'data/stage_cvpr2027_experiments/8to12_{mode}_v7'),
            '--device', 'cpu', '--threads', '4'], cwd=ROOT, check=True)
        subprocess.run([sys.executable, str(ROOT/'scripts/summarize_m3w_8to12_development.py'),
            '--protocol', str(ROOT/'configs/m3w_8to12_residual_parameterization_v7.json'),
            '--study-dir', str(ROOT/f'data/stage_cvpr2027_experiments/8to12_{mode}_v7'),
            '--report-dir', str(ROOT/f'outputs/publication_readiness_2026_09/8to12_{mode}_v7')],
            cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
