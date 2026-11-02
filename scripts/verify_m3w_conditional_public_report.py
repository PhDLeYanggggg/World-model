"""Rebuild public scalar results with private input reads denied before imports."""
import builtins
import hashlib
import io
import json
import os
from pathlib import Path
import runpy
import time
ROOT=Path(__file__).resolve().parents[1];PUBLIC=ROOT/'outputs/publication_readiness_2026_09/european_conditional_moment_v1'
PNG=ROOT/'data/stage_cvpr2027_experiments/european_conditional_moment_v1/conditional_moment.png'
SCRIPT=ROOT/'scripts/report_m3w_conditional_moment.py'


def main():
    start=time.monotonic();attempts=[];po,bo,ioo=Path.open,builtins.open,io.open
    def guard(path,mode):
        if isinstance(path,(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(path)).absolute()
            if any(p.is_relative_to(ROOT/n) for n in ('data','external_data')):
                if p==PNG and 'w' in mode:return
                attempts.append(str(p));raise AssertionError('Public reports cannot read private data or weights')
    def op(path,mode='r',*a,**kw):guard(path,mode);return po(path,mode,*a,**kw)
    def bp(path,mode='r',*a,**kw):guard(path,mode);return bo(path,mode,*a,**kw)
    def ip(path,mode='r',*a,**kw):guard(path,mode);return ioo(path,mode,*a,**kw)
    Path.open,builtins.open,io.open=op,bp,ip
    try:
        runpy.run_path(str(SCRIPT),run_name='__main__')
        value=dict(pid=os.getpid(),seconds=time.monotonic()-start,guard_installed_before_report_imports=True,private_input_read_attempts=len(attempts),
            scope='Python_Path_builtin_io_open_not_OS_isolation',full72_scalar_report_rebuilt=True,scientific_optimization_replayed=False,
            report_script_sha256=hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),summary_sha256=hashlib.sha256((PUBLIC/'summary.json').read_bytes()).hexdigest())
        (PUBLIC/'public_rebuild_verification.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))
    finally:Path.open,builtins.open,io.open=po,bo,ioo


if __name__=='__main__':main()
