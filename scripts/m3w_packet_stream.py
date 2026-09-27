"""One temporary SSH process for serial hash-verified M3W input packets."""
import hashlib
import json
import select
import shlex
import subprocess


RECEIVER = r'''
import hashlib,json,os,pathlib,re,sys
root=pathlib.Path(sys.argv[1])
assert json.loads((root/'.owner.json').read_text())['experiment']=='european_easy_hurdle_v1'
stream=sys.stdin.buffer
print(json.dumps({'ready':True}),flush=True)
while True:
    line=stream.readline(4097)
    if not line: break
    assert len(line)<=4096 and line.endswith(b'\n')
    header=json.loads(line);name=header['group'];size=header['bytes']
    assert re.fullmatch(r'[A-Za-z0-9_]+',name) and 0<size<=512*2**20
    chunks=[];left=size
    while left:
        chunk=stream.read(min(left,1024*1024))
        assert chunk,'Truncated packet'
        chunks.append(chunk);left-=len(chunk)
    content=b''.join(chunks);sha=hashlib.sha256(content).hexdigest()
    assert sha==header['sha256'],'Packet checksum mismatch'
    p=root/'inputs'/(name+'.npz');p.parent.mkdir(exist_ok=True)
    if p.exists(): assert hashlib.sha256(p.read_bytes()).hexdigest()==sha
    else:
        tmp=p.with_suffix('.tmp')
        with tmp.open('wb') as f: f.write(content);f.flush();os.fsync(f.fileno())
        os.replace(tmp,p)
    print(json.dumps({'group':name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}),flush=True)
'''


class PacketStream:
    def __init__(self, ssh, target):
        self.command=ssh+[shlex.join(['/usr/bin/python3','-c',RECEIVER,target])]
        self.process=None

    def receive(self):
        p=self.process
        if not select.select([p.stdout],[],[],60)[0]:
            raise TimeoutError('Packet acknowledgement timed out; preserve remote files and rehash before resume')
        line=p.stdout.readline()
        if not line:
            p.wait(timeout=10)
            raise RuntimeError(p.stderr.read().decode(errors='replace')[-1500:])
        return json.loads(line)

    def __enter__(self):
        self.process=subprocess.Popen(self.command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,bufsize=0)
        try:
            assert self.receive()=={'ready':True}
        except BaseException:
            self.__exit__(None,None,None)
            raise
        return self

    def send(self, name, payload):
        expected=dict(group=name,bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest())
        for part in ((json.dumps(expected)+'\n').encode(),payload):
            view=memoryview(part)
            while view:
                written=self.process.stdin.write(view)
                if not written:
                    raise BrokenPipeError('SSH input closed; preserve remote packets')
                view=view[written:]
        actual=self.receive()
        assert actual==expected
        return actual

    def __exit__(self, *args):
        if self.process is None:
            return
        p=self.process
        if not p.stdin.closed:
            p.stdin.close()
        try:
            p.wait(timeout=15)
        except subprocess.TimeoutExpired:
            # This is our transport only, never a scheduler or training process.
            p.terminate();p.wait(timeout=10)
        p.stdout.close();p.stderr.close()
