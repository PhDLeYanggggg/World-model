"""Transport the frozen recovery registration without exceeding SSH argv limits."""
import io
import json
import subprocess
import tarfile

from scripts import manage_m3w_easy_harm_recovery_v2 as recovery


def payload(reg):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf,mode='w:gz') as tar:
        raw=(recovery.ROOT/recovery.CODE).read_bytes()
        entry=tarfile.TarInfo(recovery.CODE);entry.size=len(raw);tar.addfile(entry,io.BytesIO(raw))
    header=json.dumps(['recovery',recovery.manager.REMOTE,recovery.REG.read_text()]).encode()+b'\n'
    # SUBMIT itself is byte-identical to the committed registered implementation.
    command='import json,sys\nsys.argv=json.loads(sys.stdin.buffer.readline())\nexec('+repr(recovery.SUBMIT)+')'
    return command, header+buf.getvalue()


def main():
    reg=recovery.register()
    for rel in [*reg['bindings'],str(recovery.REG.relative_to(recovery.ROOT)),
                'scripts/submit_m3w_easy_harm_recovery_stream.py']:
        if subprocess.check_output(['git','show','HEAD:'+rel],cwd=recovery.ROOT)!=(recovery.ROOT/rel).read_bytes():
            raise ValueError('Commit frozen inputs and transport before submission')
    code, raw=payload(reg)
    result=recovery.manager.base.remote(code,[],raw,timeout=180)
    recovery.manager.base.once(recovery.HOME/'recovery_v2_submission.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
