"""Render locally in memory and store small figure artifacts in the owned CREATE directory."""
import base64
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import plot_m3w_selected_pool_source as plot
from scripts import manage_m3w_selected_pool_create as manager


def main():
    rows,cfg=plot.load_source_rows();e=plot.build(rows,cfg)
    e.update(source_manifest_sha256=plot.run.digest(plot.run.PUBLIC/'source_manifest.json'),
        source_replay_sha256=plot.run.digest(plot.run.PUBLIC/'source_replay.json'),
        script_sha256=plot.run.digest(plot.__file__))
    svg,png=plot.render(e)
    files={'source_figure_data.json':(json.dumps(e,indent=2,allow_nan=False)+'\n').encode(),
        'source_mechanism.svg':svg.encode(),'source_mechanism_preview.png':png}
    assert sum(map(len,files.values()))<2*2**20
    receipt=dict(result_source=e['result_source'],source_groups=72,transfer_outcomes_read=False,
        local_artifacts_written=False,source_replay_sha256=e['source_replay_sha256'],
        store_script_sha256=plot.run.digest(__file__),
        artifacts={k:dict(bytes=len(v),sha256=hashlib.sha256(v).hexdigest()) for k,v in files.items()},
        remote_directory=manager.REMOTE+'/source_figure',scientific_design_changed=False)
    files['source_figure_receipt.json']=(json.dumps(receipt,indent=2)+'\n').encode()
    code=r'''
import base64,hashlib,json,pathlib,sys
p=json.loads(sys.stdin.read());root=pathlib.Path(p['root'])
assert root==pathlib.Path('/users/k24101830/m3w/european_selected_pool_v1')
assert json.loads((root/'.owner.json').read_text())['experiment']=='european_selected_pool_v1'
target=root/'source_figure';target.mkdir(exist_ok=True)
allowed={'source_figure_data.json','source_mechanism.svg','source_mechanism_preview.png','source_figure_receipt.json'}
assert set(p['files'])==allowed
total=0
for name,entry in p['files'].items():
    raw=base64.b64decode(entry['base64']);total+=len(raw);assert total<2*2**20
    assert hashlib.sha256(raw).hexdigest()==entry['sha256']
    path=target/name
    if path.exists():assert path.read_bytes()==raw
    else:
        with path.open('xb') as f:f.write(raw)
for name,entry in p['files'].items():
    assert hashlib.sha256((target/name).read_bytes()).hexdigest()==entry['sha256']
print(json.dumps({'verified_files':len(allowed),'bytes':total}))
'''
    result=manager.remote(code,dict(root=manager.REMOTE,files={k:dict(base64=base64.b64encode(v).decode(),
        sha256=hashlib.sha256(v).hexdigest()) for k,v in files.items()}))
    print(json.dumps(dict(receipt=receipt,remote_verification=result)))


if __name__=='__main__':main()
