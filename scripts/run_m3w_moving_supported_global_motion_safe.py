"""Initialize checkpoint directories without changing the registered experiment."""
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_m3w_moving_supported_global_motion as original


def prepare_directories(private):
    path=Path(private)/'allocation';path.mkdir(parents=True,exist_ok=True)
    return path


def main():
    for p in (Path(__file__),original.PUBLIC/'inference_runtime_repair.md'):
        body=subprocess.run(['git','show','HEAD:'+str(p.relative_to(ROOT))],cwd=ROOT,
            capture_output=True,check=True).stdout
        if original.hashlib.sha256(body).hexdigest()!=original.sha(p):
            raise ValueError('Commit the runtime-only repair before execution')
    prepare_directories(original.PRIVATE)
    original.main()


if __name__=='__main__':main()
