"""Resume the frozen study with an aggregate event-prior JSON serialization fix."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_joint_harm as run


def serializable_prior(value, original):
    if isinstance(value, dict) and isinstance(value.get('event_prior'), run.source.np.ndarray):
        value = dict(value, event_prior=value['event_prior'].tolist())
    return original(value)


def main():
    paths = (Path(__file__), ROOT / 'tests/test_m3w_joint_harm_runtime.py', run.PUBLIC / 'runtime_fix.md')
    for path in paths:
        body = run.source.subprocess.run(['git', 'show', 'HEAD:'+str(path.relative_to(ROOT))],
            cwd=ROOT, capture_output=True, check=True).stdout
        if run.source.hashlib.sha256(body).hexdigest() != run.source.sha(path):
            raise ValueError('Commit the narrow runtime fix before unchanged resume')
    run.source.once(run.PUBLIC / 'runtime_patch.json', dict(original_registration=run.source.artifact(run.REG),
        bindings={str(p.relative_to(ROOT)): run.source.sha(p) for p in paths},
        change='event_prior ndarray to identical nested JSON list in fit receipt only',
        training_objective_changed=False, original_scientific_files_changed=False))
    original = run.source.stable
    run.source.stable = lambda value: serializable_prior(value, original)
    try: run.main()
    finally: run.source.stable = original


if __name__ == '__main__': main()
