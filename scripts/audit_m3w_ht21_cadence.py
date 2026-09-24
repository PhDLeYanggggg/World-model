"""Structural cadence sensitivity, not a choice of prediction evaluation clock."""
import argparse
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.fetch_m3w_ht21_annotations import ARCHIVE,PUBLIC,sha
from src.evaluation.m3w_ht21_intake import read_ground_truth,availability


def cadence_support(rows,frames,stride):
    if type(stride) is not int or stride<1:
        raise ValueError('Positive integer stride required')
    fields=('past_eligible','complete_future12','partial_future12','no_future12')
    total={f:0 for f in fields}
    for phase in range(stride):
        selected=rows[(rows[:,0].astype(int)-1)%stride==phase].copy()
        selected[:,0]=(selected[:,0]-1-phase)//stride+1
        coarse_frames=(frames-1-phase)//stride+1
        if coarse_frames<=0: continue
        result=availability(selected,coarse_frames,lengths=(8,))['8']
        for f in fields:total[f]+=result[f]
    return dict(raw_frame_stride=stride,history_span_raw_frames=7*stride,
        future_span_raw_frames=12*stride,all_anchor_phases_included=True,**total)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify',action='store_true');args=parser.parse_args()
    audit=json.loads((PUBLIC/'analysis.json').read_text())
    assert sha(ARCHIVE.read_bytes()) == audit['archive_sha256']
    records=[]
    with zipfile.ZipFile(ARCHIVE) as z:
        for r in audit['recordings']:
            if not r['ground_truth_present']: continue
            b=z.read(f"HT21Labels/{r['supplied_split']}/{r['sequence']}/gt/gt.txt")
            assert sha(b)==r['ground_truth_sha256']
            rows=read_ground_truth(b,r['metadata'])
            records.append(dict(sequence=r['sequence'],cadences=[cadence_support(rows,r['metadata']['frames'],s) for s in (1,5,10)]))
    result=dict(result_source='fresh_run',records=records,source_analysis_sha256=sha((PUBLIC/'analysis.json').read_bytes()),
        script_sha256=sha(Path(__file__).read_bytes()),tests_sha256=sha((ROOT/'tests/test_m3w_ht21_cadence.py').read_bytes()),
        evaluation_protocol_selected=False,metrics_computed=False,assumed_annotation_keyframes=False,
        future_outcomes_used_for_input_eligibility=False,independent_samples_claimed=False)
    path=PUBLIC/'cadence_support.json'
    if args.verify:
        assert result==json.loads(path.read_text())
        print('cached_verified: cadence support exact')
    else:
        with path.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(records,indent=2))


if __name__=='__main__':main()
