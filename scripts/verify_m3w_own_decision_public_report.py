"""Fresh-process report reconstruction with private Python input reads denied."""
import builtins
import hashlib
import io
import json
import os
from pathlib import Path
import runpy
import time

ROOT=Path(__file__).resolve().parents[1];PUBLIC=ROOT/'outputs/publication_readiness_2026_09/european_own_decision_cost_v1'
PNG=ROOT/'data/stage_cvpr2027_experiments/european_own_decision_cost_v1/own_decision_readout.png'
SCRIPT=ROOT/'scripts/report_m3w_own_decision_cost.py'


def main():
    start=time.monotonic();attempts=[];po,fo,ioo=Path.open,builtins.open,io.open
    def guard(path,mode):
        if isinstance(path,(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(path)).absolute()
            if any(p.is_relative_to(ROOT/name) for name in ('data','external_data')):
                if p==PNG and 'w' in mode:return
                attempts.append(True);raise AssertionError('Public report must not read private inputs')
    def path_open(path,mode='r',*a,**kw):guard(path,mode);return po(path,mode,*a,**kw)
    def file_open(path,mode='r',*a,**kw):guard(path,mode);return fo(path,mode,*a,**kw)
    def io_open(path,mode='r',*a,**kw):guard(path,mode);return ioo(path,mode,*a,**kw)
    Path.open=path_open;builtins.open=file_open;io.open=io_open
    try:
        runpy.run_path(str(SCRIPT),run_name='__main__')
        value=dict(pid=os.getpid(),seconds=time.monotonic()-start,guard_installed_before_report_imports=True,
            private_input_read_attempts=len(attempts),permitted_private_output='own_decision_readout.png_write_only',
            scope='Python_Path_builtin_io_open_not_OS_process_isolation',full72_scalar_report_rebuilt=True,
            report_script_sha256=hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),summary_sha256=hashlib.sha256((PUBLIC/'summary.json').read_bytes()).hexdigest(),
            scientific_optimization_replayed=False)
        (PUBLIC/'public_rebuild_verification.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))
    finally:Path.open=po;builtins.open=fo;io.open=ioo


if __name__=='__main__':main()
