"""Independent CSV/set membership recount; never reads detector predictions."""
from bisect import bisect_right
from collections import defaultdict
import csv
import io
import json
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.fetch_m3w_ht21_annotations import ARCHIVE,PUBLIC,sha


def recount(tracks,length,stride):
    counts=dict(past_eligible=0,complete_future12=0,partial_future12=0,no_future12=0)
    endpoints={str(h):0 for h in (10,25,50,100)}
    for frames in tracks.values():
        ordered=sorted(frames)
        run={}
        for f in ordered:
            run[f]=run.get(f-stride,0)+1
            if run[f]<length:continue
            counts['past_eligible']+=1
            support=sum(f+j*stride in frames for j in range(1,13))
            category='complete_future12' if support==12 else 'partial_future12' if support else 'no_future12'
            counts[category]+=1
            for h in (10,25,50,100):endpoints[str(h)]+=int(f+h in frames)
    return counts,endpoints


def main():
    audit=json.loads((PUBLIC/'analysis.json').read_text())
    cadence=json.loads((PUBLIC/'cadence_support.json').read_text())
    assert sha(ARCHIVE.read_bytes())==audit['archive_sha256']
    checked=0
    with zipfile.ZipFile(ARCHIVE) as z:
        for recording in audit['recordings']:
            if not recording['ground_truth_present']:continue
            name=recording['sequence'];path=f'HT21Labels/train/{name}/gt/gt.txt'
            raw=z.read(path);assert sha(raw)==recording['ground_truth_sha256']
            tracks=defaultdict(set);keys=set();all_ids=set();classes=defaultdict(int);visible_static=0
            for row in csv.reader(io.StringIO(raw.decode())):
                assert len(row)==9
                frame,agent=int(row[0]),int(row[1]);key=(frame,agent)
                assert key not in keys;keys.add(key);all_ids.add(agent)
                klass=int(row[7]);classes[str(klass)]+=1
                if klass in (1,2,4) and float(row[6])>0 and float(row[8])>0:
                    tracks[agent].add(frame);visible_static+=int(klass==2)
            report=recording['audit']
            assert len(keys)==report['rows'] and len(all_ids)==report['track_ids']
            assert dict(classes)==report['row_class_counts'] and visible_static==report['visible_static_rows']
            for length in (8,16,32,64):
                counts,ends=recount(tracks,length,1)
                expected=report['availability_stride1'][str(length)]
                assert all(v==expected[k] for k,v in counts.items()) and ends==expected['endpoint_available']
                checked+=1
            found=next(r for r in cadence['records'] if r['sequence']==name)
            for item in found['cadences']:
                counts,_=recount(tracks,8,item['raw_frame_stride'])
                assert all(v==item[k] for k,v in counts.items());checked+=1
            print(json.dumps(dict(sequence=name,recount_pass=True,window_groups_checked=checked)),flush=True)
    result=dict(all_checks_passed=True,result_source='fresh_run_independent_arithmetic',
        analysis_sha256=sha((PUBLIC/'analysis.json').read_bytes()),
        cadence_sha256=sha((PUBLIC/'cadence_support.json').read_bytes()),
        verifier_sha256=sha(Path(__file__).read_bytes()),window_groups_checked=checked,
        csv_rows_checked=audit['aggregate']['ground_truth_rows'],same_agent=True,
        independent_research_replication=False,predictive_evaluation=False,scientific_admission=False)
    with (PUBLIC/'independent_verification.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
